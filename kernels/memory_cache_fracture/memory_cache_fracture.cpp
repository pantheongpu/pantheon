#include "../common/common.h"
#include <chrono>
#include <string>
#include <iostream>

// --- CACHE-LINE FRACTURING ---
// Forces massive uncoalesced reads. Every thread in the wavefront requests an address 
// separated by exactly 256 bytes, forcing 64 separate cache-line fetches per instruction.
__global__ void cache_fracture_kernel(uint4* data, size_t n, unsigned int* sink, int loops, int inject_error) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 256 byte stride per thread = 16 uint4s
    size_t thread_stride = 16; 
    size_t base_idx = (tid * thread_stride) % n;
    
    unsigned int acc = 0;
    
    for (int i = 0; i < loops; ++i) {
        // Physical fetch. Because of the 256B offset, the UMC cannot merge these requests.
        acc += load_nt(&data[base_idx]).x;
        
        // Shift the base index to prevent cache hits on the next loop
        base_idx = (base_idx + (blockDim.x * gridDim.x * thread_stride)) % n;

        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && tid == 1337 && i == 500) {
            // Intentionally corrupt the uncoalesced read accumulator
            acc += 0xBADBEEF; 
        }
    }

    // Accumulate into the thread's sink slot.
    sink[tid] += acc;
}

// --- VERIFICATION KERNEL ---
__global__ void verify_fracture_kernel(unsigned int* sink, size_t n, unsigned int expected_val, unsigned int* err_count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        unsigned int actual_val = sink[tid];
        
        if (actual_val != expected_val) {
            unsigned int xor_bits = expected_val ^ actual_val;
            
            printf("[SDC FAULT][CACHE_FRACTURE] Uncoalesced Read Error! TID: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n",
                   (unsigned long long)tid, expected_val, actual_val, xor_bits);
            
            // Report the error back to the host
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
    int kernel_loops = 10000;  // Maps to cache-line reads
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 0;      // 0=Zeroes, 1=Ones

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

    size_t free, total; CHECK(hipMemGetInfo(&free, &total));
    if (mem_pct > 99) mem_pct = 99;
    size_t alloc_size = (free * mem_pct) / 100;
    size_t num_elements = alloc_size / 16; 

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
    
    // --- 3. SET MEMORY INIT PATTERN ---
    if (init_pattern == 1) {
        CHECK(hipMemset(d_data, 0xFF, alloc_size)); 
    } else {
        CHECK(hipMemset(d_data, 0x00, alloc_size)); 
    }
    CHECK(hipMemset(d_sink, 0, sink_size)); // Sink must ALWAYS be zeroed for accumulation

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": CACHE-LINE FRACTURING (Queue Stress) | " 
              << mem_pct << "% VRAM..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- 4. WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup iterations..." << std::endl;
        for(int i = 0; i < warmup_iters; i++) {
            LAUNCH_KERNEL(cache_fracture_kernel, num_blocks, block_size, d_data, num_elements, d_sink, kernel_loops, inject_error);
        }
        CHECK(hipDeviceSynchronize());
        
        // Clear the sink accumulation so the telemetry loop has a fresh start for verification
        CHECK(hipMemset(d_sink, 0, sink_size));
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t reads_performed = 0;
    unsigned int kernel_launches = 0;

    // --- 5. ACTIVE LOOP ---
    while (true) {
        LAUNCH_KERNEL(cache_fracture_kernel, num_blocks, block_size, d_data, num_elements, d_sink, kernel_loops, inject_error);
        CHECK(hipDeviceSynchronize());
        
        kernel_launches++;

        reads_performed += (size_t)num_blocks * block_size * kernel_loops * 16; // 16 bytes per read
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    
    // Bandwidth will be incredibly low here, indicating successful queuing overload
    std::cout << "Throughput: " << (reads_performed / 1e9) / seconds << " GB/s" << std::endl;

    // --- 6. VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running Cache-Line Fracture Verification Pass..." << std::endl;
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        // Determine expected accumulator value based on the memory init pattern
        unsigned int expected_val = 0;
        if (init_pattern == 1) {
            unsigned int pattern_val = 0xFFFFFFFF;
            expected_val = pattern_val * kernel_loops * kernel_launches;
        }
        
        int verify_blocks = (total_threads + 255) / 256;
        LAUNCH_KERNEL(verify_fracture_kernel, verify_blocks, 256, d_sink, total_threads, expected_val, d_err_count);
        CHECK(hipDeviceSynchronize());
        
        unsigned int h_err_count = 0;
        CHECK(hipMemcpy(&h_err_count, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));
        CHECK(hipFree(d_err_count));

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
