#include "../common/common.h"
#include <vector>
#include <chrono>
#include <string>
#include <iostream>

// Linear Congruential Generator constants for full-period traversal
#define LCG_A 1664525
#define LCG_C 1013904223

// --- INITIALIZATION KERNEL ---
__global__ void init_lcg_kernel(size_t* data, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // data[i] = (A*i + C) % n
    for (size_t i = tid; i < n; i += stride) {
        data[i] = (LCG_A * i + LCG_C) & (n - 1);
    }
}

// --- GOLDEN PASS KERNEL ---
__global__ void golden_latency_kernel(size_t* data, size_t n, int loops, size_t* golden_sink) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx = tid & (n - 1); // Safe start index
    
    // Pure Pointer Chasing
    for (int i = 0; i < loops; ++i) {
        idx = data[idx];
    }
    golden_sink[tid] = idx;
}

// --- STRESS KERNEL ---
__global__ void latency_kernel(size_t* data, size_t n, int loops, size_t* sink, int inject_error) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx = tid & (n - 1);
    
    for (int i = 0; i < loops; ++i) {
        idx = data[idx];

        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && tid == 1337 && i == 500) {
            // Intentionally corrupt the pointer address, 
            // but keep it within array bounds to avoid a hard segfault
            idx = (idx ^ 0xDEADBEEF) & (n - 1);
        }
    }
    
    // Accumulate the final index. If an SDC occurs in ANY launch, 
    // it will poison this accumulator permanently.
    sink[tid] += idx;
}

// --- VERIFICATION KERNEL ---
__global__ void verify_latency_kernel(size_t* sink, size_t* golden_sink, unsigned int expected_launches, unsigned int* err_count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    size_t expected_val = golden_sink[tid] * expected_launches;
    size_t actual_val = sink[tid];
    
    if (actual_val != expected_val) {
        size_t xor_bits = expected_val ^ actual_val;
        
        printf("[SDC FAULT][CACHE_LATENCY] Pointer Chain Error! Thread: %llu | Exp: %llu | Act: %llu | XOR: 0x%llx\n",
               (unsigned long long)tid, (unsigned long long)expected_val, (unsigned long long)actual_val, (unsigned long long)xor_bits);

        // Report the error back to the host
        atomicAdd(err_count, 1);
    }
}

// Keep --inject_error deterministic even when a caller lowers kernel_loops
// below the in-kernel injection point, or when the CPU mock executes one
// logical thread.  Corrupting the result sink directly exercises the exact
// comparison used by the verification pass without risking an invalid pointer.
__global__ void inject_latency_sink_error(size_t* sink, size_t count) {
    if (count > 0 && blockIdx.x == 0 && threadIdx.x == 0) sink[0] ^= 1u;
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);
    int mem_pct = atoi(argv[3]);

    // --- PANTHEON CONFIG KNOBS ---
    int block_size = 256;      
    int grid_size = 0;         // 0 = auto-calculate
    int kernel_loops = 20000;  // Maps to inner pointer-chasing loops
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 0;      // Logged, but ignored for latency to prevent segfaults

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

    // 1. Calculate Main Buffer Size (The Walker)
    size_t free, total;
    CHECK(hipMemGetInfo(&free, &total));
    if (mem_pct > 99) mem_pct = 99;
    if (mem_pct < 1) mem_pct = 1;
    
    // Align to power of 2 for LCG algorithm
    size_t raw_size = (free * mem_pct) / 100;
    size_t num_elements = 1;
    while (num_elements * 2 * sizeof(size_t) <= raw_size) {
        num_elements *= 2;
    }
    size_t alloc_size = num_elements * sizeof(size_t);

    scale_warmup_for_large_alloc(warmup_iters, alloc_size);
    scale_kernel_loops_for_large_alloc(kernel_loops, alloc_size, 200);

    // 2. Allocate Main Buffer
    size_t* d_data; CHECK(hipMalloc(&d_data, alloc_size));

    // 3. EXPLICIT OCCUPANCY
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    
    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) { // Fallback to calculation if not provided by AI
        int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
        if (max_blocks_per_sm < 1) max_blocks_per_sm = 4; 
        num_blocks = prop.multiProcessorCount * max_blocks_per_sm;
        auto_grid = true;
    }

    size_t sink_size = num_blocks * block_size * sizeof(size_t);

    size_t* d_sink;
    CHECK(hipMalloc(&d_sink, sink_size));
    CHECK(hipMemset(d_sink, 0, sink_size)); // Must be zeroed for accumulation

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running CACHE LATENCY STRESS..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Mem Alloc (%): " << mem_pct << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << " (Overridden: Forced LCG Pointer Gen)" << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // 4. Initialize LCG Data Array
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Init Random Walk on " 
              << alloc_size / (1024*1024) << " MB (" << num_elements << " nodes)..." << std::endl;
    int init_grid = init_launch_grid_size(prop, num_elements, block_size);
    LAUNCH_KERNEL(init_lcg_kernel, init_grid, block_size, d_data, num_elements);
    CHECK(hipDeviceSynchronize());    

    // --- GOLDEN PASS ---
    size_t* d_golden_sink = nullptr;
    
    if (verify_mode) {
        std::cout << "[PANTHEON] Generating expected pointer chains (Golden Pass)..." << std::endl;
        CHECK(hipMalloc(&d_golden_sink, sink_size));
        LAUNCH_KERNEL(golden_latency_kernel, num_blocks, block_size, d_data, num_elements, kernel_loops, d_golden_sink);
        CHECK(hipDeviceSynchronize());
    }

    // --- WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup iterations..." << std::endl;
        for(int i = 0; i < warmup_iters; i++) {
            LAUNCH_KERNEL(latency_kernel, num_blocks, block_size, d_data, num_elements, kernel_loops, d_sink, inject_error);
        }
        CHECK(hipDeviceSynchronize());
        
        // Clear the sink accumulation so the telemetry loop has a fresh start for verification
        CHECK(hipMemset(d_sink, 0, sink_size));
    }

    // 5. Run Active Stress
    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    unsigned int kernel_launches = 0;
    
    while (true) {
        LAUNCH_KERNEL(latency_kernel, num_blocks, block_size, d_data, num_elements, kernel_loops, d_sink, inject_error);
        CHECK(hipDeviceSynchronize());
        
        kernel_launches++;

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    // --- VERIFICATION PASS ---
    int exit_code = 0;
    if (verify_mode) {
        std::cout << "[PANTHEON] Running Cache Pointer Chain Verification Pass..." << std::endl;

        if (inject_error) {
            LAUNCH_KERNEL(inject_latency_sink_error, 1, 1, d_sink, num_blocks * block_size);
            CHECK(hipDeviceSynchronize());
        }
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        LAUNCH_KERNEL(verify_latency_kernel, num_blocks, block_size, d_sink, d_golden_sink, kernel_launches, d_err_count);
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
            // Return a failing status after all device allocations are released.
            exit_code = 1;
        }
    }

    // A pointer hop is the useful unit of work for this dependent-load test.
    // Emit it through the common result contract so it is retained in local
    // reports and the performance database instead of being shown as N/A.
    double seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start_time
    ).count();
    double dependent_loads = static_cast<double>(kernel_launches)
        * static_cast<double>(num_blocks)
        * static_cast<double>(block_size)
        * static_cast<double>(kernel_loops);
    std::cout << "Throughput: " << dependent_loads / seconds
              << " dependent-loads/s" << std::endl;

    CHECK(hipFree(d_data));
    CHECK(hipFree(d_sink));
    return exit_code;
}
