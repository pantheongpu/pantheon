#include "../common/common.h"
#include <chrono>
#include <string>
#include <iostream>

// --- TLB AVALANCHE (MMU STRESS) ---
// Forces a 100% Translation Lookaside Buffer (TLB) miss rate by performing 
// pseudo-random jumps across the entire VRAM allocation, specifically targeting 
// boundaries larger than standard (4KB) and huge (2MB) pages. This chokes the 
// hardware page-table walkers.

// Simple Linear Congruential Generator (LCG) for fast, on-die random jumps
#define LCG_A 1664525
#define LCG_C 1013904223

// --- INITIALIZE MEMORY ---
// Only touch page-aligned entries; TLB jumps always land on page boundaries.
__global__ void init_tlb_memory_kernel(uint4* data, size_t n, size_t page_offset) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t num_pages = (n + page_offset - 1) / page_offset;

    for (size_t p = idx; p < num_pages; p += stride) {
        size_t i = p * page_offset;
        if (i < n) {
            uint32_t val = (uint32_t)i ^ 0xCAFEBABE;
            data[i] = make_uint4(val, ~val, val ^ 0x55555555, val ^ 0xAAAAAAAA);
        }
    }
}

// --- GOLDEN PASS ---
__global__ void golden_tlb_kernel(uint4* data, size_t n, unsigned int* golden_sink, int loops, size_t page_offset, unsigned int num_pages) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int acc = 0;
    unsigned int rng_state = (unsigned int)tid;
    // Select a page index first so every jump lands on an initialized,
    // page-aligned probe site; the previous `(rng * page_offset) % n` lost
    // alignment whenever the product wrapped past the buffer size.
    size_t curr_idx = ((size_t)((unsigned int)tid % num_pages)) * page_offset;
    if (curr_idx >= n) curr_idx = 0;

    for (int i = 0; i < loops; ++i) {
        acc += data[curr_idx].x;
        rng_state = (LCG_A * rng_state + LCG_C);
        curr_idx = ((size_t)(rng_state % num_pages)) * page_offset;
        if (curr_idx >= n) curr_idx = 0;
    }

    golden_sink[tid] = acc;
}

// --- MAIN STRESS KERNEL ---
__global__ void tlb_avalanche_kernel(uint4* data, size_t n, unsigned int* sink, int loops, int inject_error, size_t page_offset, unsigned int num_pages) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int acc = 0;
    unsigned int rng_state = (unsigned int)tid;
    size_t curr_idx = ((size_t)((unsigned int)tid % num_pages)) * page_offset;
    if (curr_idx >= n) curr_idx = 0;

    for (int i = 0; i < loops; ++i) {
        // 1. Force a read from the current page
        acc += data[curr_idx].x;

        // --- DYNAMIC FAULT INJECTION ---
        // Simulates an MMU page fault silently translating to the wrong physical memory address
        if (inject_error && tid == 1337 && i == loops - 1) {
            acc ^= 0xBADBEEF;
        }

        // 2. Generate the next random page jump
        rng_state = (LCG_A * rng_state + LCG_C);

        // 3. Jump to a random page-aligned probe site anywhere in the allocation
        curr_idx = ((size_t)(rng_state % num_pages)) * page_offset;
        if (curr_idx >= n) curr_idx = 0;
    }

    // Accumulate the read hashes
    sink[tid] += acc;
}

// --- VERIFICATION KERNEL ---
__global__ void verify_tlb_kernel(unsigned int* sink, unsigned int* golden_sink, unsigned int launches, size_t total_threads, unsigned int* err_count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < total_threads) {
        unsigned int exp = golden_sink[tid] * launches;
        unsigned int act = sink[tid];
        
        if (exp != act) {
            // Decoupled atomicAdd return value for MOCK platform compatibility
            if (*err_count < 5) {
                printf("[SDC FAULT][TLB_AVALANCHE] MMU Translation Error! TID: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n",
                       (unsigned long long)tid, exp, act, exp ^ act);
            }
            atomicAdd(err_count, 1);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);
    int mem_pct = atoi(argv[3]);

    // --- PANTHEON CONFIG KNOBS ---
    int block_size = 256;      
    int grid_size = 0;         // 0 = auto-calculate
    int kernel_loops = 20000;  // Number of pseudo-random page jumps per thread
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 0;      // 0=4KB Pages, 1=2MB Huge Pages, >1=Custom Stride

    bool verify_mode = false;
    int inject_error = 0;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--verify") verify_mode = true;
        if (std::string(argv[i]) == "--inject_error") inject_error = 1;
        if (std::string(argv[i]) == "--block_size" && i+1 < argc) block_size = atoi(argv[++i]);
        if (std::string(argv[i]) == "--grid_size" && i+1 < argc) grid_size = atoi(argv[++i]);
        if (std::string(argv[i]) == "--kernel_loops" && i+1 < argc) kernel_loops = atoi(argv[++i]);
        if (std::string(argv[i]) == "--warmup_iters" && i+1 < argc) warmup_iters = atoi(argv[++i]);
        if (std::string(argv[i]) == "--sync_mode" && i+1 < argc) sync_mode = atoi(argv[++i]);
        if (std::string(argv[i]) == "--init_pattern" && i+1 < argc) {
            if (!pantheon_parse_init_pattern(argv[++i], &init_pattern)) {
                std::cerr << "[PANTHEON] Unknown --init_pattern '" << argv[i] << "'." << std::endl;
                pantheon_print_init_patterns(std::cerr);
                return 1;
            }
        }
    }

    normalize_kernel_launch_config(block_size, grid_size, kernel_loops, warmup_iters, sync_mode);

    // --- 1. SET SYNC MODE ---
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__) || defined(__HIP_PLATFORM_HCC__)
    unsigned int sync_flag = hipDeviceScheduleBlockingSync;
    if (sync_mode == 0) sync_flag = hipDeviceScheduleSpin;
    else if (sync_mode == 1) sync_flag = hipDeviceScheduleYield;
    hipSetDeviceFlags(sync_flag);
#elif defined(__CUDACC__)
    unsigned int sync_flag = cudaDeviceScheduleBlockingSync;
    if (sync_mode == 0) sync_flag = cudaDeviceScheduleSpin;
    else if (sync_mode == 1) sync_flag = cudaDeviceScheduleYield;
    cudaSetDeviceFlags(sync_flag);
#else
    // MOCK PLATFORM: CPU execution doesn't require hardware scheduling flags
#endif

    CHECK(hipSetDevice(gpu_id));

    // To break the TLB, we need a MASSIVE memory footprint. 
    size_t free, total; CHECK(hipMemGetInfo(&free, &total));
    if (mem_pct > 99) mem_pct = 99;
    
    size_t alloc_size = (free * mem_pct) / 100;
    size_t num_elements = alloc_size / 16; 

    scale_warmup_for_large_alloc(warmup_iters, alloc_size);
    scale_kernel_loops_for_large_alloc(kernel_loops, alloc_size, 50);

    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    
    // --- 2. EXPLICIT OCCUPANCY ---
    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) {
        int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
        if (max_blocks_per_sm < 1) max_blocks_per_sm = 16; 
        num_blocks = prop.multiProcessorCount * max_blocks_per_sm;
        auto_grid = true;
    }
    
    size_t total_threads = (size_t)num_blocks * block_size;
    size_t sink_size = total_threads * sizeof(unsigned int);

    uint4* d_data; CHECK(hipMalloc(&d_data, alloc_size));
    unsigned int* d_sink; CHECK(hipMalloc(&d_sink, sink_size));

    CHECK(hipMemset(d_sink, 0, sink_size));

    // Resolve Page Jump Offset Strategy
    size_t page_offset;
    if (init_pattern == 0) {
        page_offset = 256;         // 4KB Standard Page = 256 uint4s
    } else if (init_pattern == 1) {
        page_offset = 131072;      // 2MB Huge Page = 131072 uint4s
    } else {
        page_offset = init_pattern; // Fuzzer custom offset
    }

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": TLB AVALANCHE (MMU Thrashing) | " 
              << mem_pct << "% VRAM..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << " (Page Stride: " << page_offset << " uint4s)" << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // Initialize page-aligned probe sites only.
    size_t num_pages = (num_elements + page_offset - 1) / page_offset;
    int init_grid = init_launch_grid_size(prop, num_pages, block_size);
    LAUNCH_KERNEL(init_tlb_memory_kernel, init_grid, block_size, d_data, num_elements, page_offset);
    CHECK(hipDeviceSynchronize());

    // --- GOLDEN PASS ---
    unsigned int* d_golden_sink = nullptr;

    if (verify_mode) {
        std::cout << "[PANTHEON] Generating expected MMU Translation baseline (Golden Pass)..." << std::endl;
        CHECK(hipMalloc(&d_golden_sink, sink_size));
        LAUNCH_KERNEL(golden_tlb_kernel, num_blocks, block_size, d_data, num_elements, d_golden_sink, kernel_loops, page_offset, (unsigned int)num_pages);
        CHECK(hipDeviceSynchronize());
    }

    // --- 4. WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup iterations..." << std::endl;
        for(int i = 0; i < warmup_iters; i++) {
            LAUNCH_KERNEL(tlb_avalanche_kernel, num_blocks, block_size, d_data, num_elements, d_sink, kernel_loops, inject_error, page_offset, (unsigned int)num_pages);
        }
        CHECK(hipDeviceSynchronize());
        
        // Reset accumulation sink so verification passes accurately
        CHECK(hipMemset(d_sink, 0, sink_size));
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t ops_performed = 0;
    unsigned int kernel_launches = 0;

    // --- 5. MAIN STRESS LOOP ---
    while (true) {
        LAUNCH_KERNEL(tlb_avalanche_kernel, num_blocks, block_size, d_data, num_elements, d_sink, kernel_loops, inject_error, page_offset, (unsigned int)num_pages);
        CHECK(hipDeviceSynchronize());
        
        kernel_launches++;
        
        // Each jump loads the 4-byte .x probe word; count what is consumed.
        ops_performed += (size_t)num_blocks * block_size * kernel_loops * 4;
        
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    
    // We expect the throughput here to be ABYSMAL. That means it's working.
    std::cout << "Throughput: " << (ops_performed / 1e9) / seconds << " GB/s" << std::endl;

    // --- 6. VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running MMU Translation State Verification Pass..." << std::endl;
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        int verify_grid = init_launch_grid_size(prop, total_threads, 256);
        LAUNCH_KERNEL(verify_tlb_kernel, verify_grid, 256, d_sink, d_golden_sink, kernel_launches, total_threads, d_err_count);
        CHECK(hipDeviceSynchronize());
        
        unsigned int h_err_count = 0;
        CHECK(hipMemcpy(&h_err_count, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));
        
        CHECK(hipFree(d_err_count));
        CHECK(hipFree(d_golden_sink));

        // Say so on success too. Silence is indistinguishable from a
        // verification that never ran, which is how a self-test that
        // could never fail went unnoticed in memory_pc_pingpong.
        std::cout << "Verification: " << (h_err_count ? "FAIL" : "PASS")
                  << " (" << h_err_count << " errors)" << std::endl;
        if (h_err_count > 0) {
            // CRITICAL: Exit with non-zero code to fail the CI step
            return 1;
        }
    }

    CHECK(hipFree(d_data)); 
    CHECK(hipFree(d_sink));
    return 0;
}
