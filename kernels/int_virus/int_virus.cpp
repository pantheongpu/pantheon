#include "../common/common.h"
#include <chrono>
#include <string>
#include <iostream>

// --- GOLDEN PASS KERNEL ---
// Generates the deterministic expected output for every thread
__global__ void golden_int_kernel(int iters, unsigned int* golden_sink) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned int a = tid ^ 0xDEADBEEF;
    unsigned int b = 0x55555555;
    unsigned int c = 0xAAAAAAAA;

    for(int i = 0; i < iters; ++i) {
        #pragma unroll 32
        for(int j = 0; j < 32; ++j) {
            a = (a * 1103515245) + 12345; 
            b = (b << 5) | (b >> 27);     
            c = a ^ b ^ c;                
            a = a + c;                    
        }
    }
    // Snapshot the final mixed state
    golden_sink[tid] = a ^ b ^ c;
}

// --- INTEGER VIRUS KERNEL ---
// Saturates the INT32 ALUs using bit-bashing, shifts, and XOR cascades.
// Stresses a completely different physical datapath than the FP32/FP16 tests.
__global__ void int_virus_kernel(int iters, unsigned int* sink, int inject_error) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Seed with thread ID to prevent compiler from optimizing out the loop
    unsigned int a = tid ^ 0xDEADBEEF;
    unsigned int b = 0x55555555;
    unsigned int c = 0xAAAAAAAA;

    for(int i = 0; i < iters; ++i) {
        #pragma unroll 32
        for(int j = 0; j < 32; ++j) {
            // Intense Integer Logic
            a = (a * 1103515245) + 12345; // Linear Congruential Generator math
            b = (b << 5) | (b >> 27);     // Bitwise Rotate
            c = a ^ b ^ c;                // XOR Cascade
            a = a + c;                    // Standard Integer Addition
        }

        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && tid == 1337 && i == 500) {
            // Induce a transient ALU fault
            a ^= 0xBADBEEF;
        }
    }

    // Accumulate the final state. If an SDC occurs in ANY launch, 
    // it will poison this accumulator permanently.
    sink[tid] += (a ^ b ^ c);
}

// --- VERIFICATION KERNEL ---
__global__ void verify_int_kernel(unsigned int* sink, unsigned int* golden_sink, unsigned int expected_launches, unsigned int* err_count, unsigned int init_offset) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate expected accumulated value. 
    // Unsigned integer overflow safely mirrors the wrap-around additions happening on the GPU.
    unsigned int expected_val = (golden_sink[tid] * expected_launches) + init_offset;
    unsigned int actual_val = sink[tid];
    
    if (actual_val != expected_val) {
        unsigned int xor_bits = expected_val ^ actual_val;
        
        printf("[SDC FAULT][INT_VIRUS] ALU Error! TID: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n",
               (unsigned long long)tid, expected_val, actual_val, xor_bits);

        // Report the error back to the host
        atomicAdd(err_count, 1);
    }
}

// Keep --inject_error deterministic when kernel_loops never reaches the
// in-kernel injection iteration (i == 500); corrupting one sink lane
// exercises the exact comparison used by the verification pass.
__global__ void inject_int_sink_error(unsigned int* sink, size_t count) {
    if (count > 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        sink[0] ^= 1u;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);

    // --- PANTHEON CONFIG KNOBS ---
    int block_size = 256;      
    int grid_size = 0;         // 0 = auto-calculate
    int kernel_loops = 10000;  // Maps directly to INT32 ALU 'iters'
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

    hipDeviceProp_t prop; 
    CHECK(hipGetDeviceProperties(&prop, gpu_id));

    // --- 2. EXPLICIT OCCUPANCY ---
    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) {
        // Adaptive Occupancy Strategy (with safety margin for register limits)
        int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
        if (max_blocks_per_sm > 4) max_blocks_per_sm -= 1; // Safety margin
        if (max_blocks_per_sm < 1) max_blocks_per_sm = 16; // Fallback
        num_blocks = prop.multiProcessorCount * max_blocks_per_sm;
        auto_grid = true;
    }

    // Dynamic Sink Allocation
    size_t total_threads = (size_t)num_blocks * block_size;
    size_t sink_size = total_threads * sizeof(unsigned int);
    
    unsigned int* d_sink; 
    CHECK(hipMalloc(&d_sink, sink_size));

    // --- 3. SET MEMORY INIT PATTERN ---
    if (init_pattern == 1) {
        CHECK(hipMemset(d_sink, 0xFF, sink_size)); // Sets integer baseline to 0xFFFFFFFF
    } else {
        CHECK(hipMemset(d_sink, 0x00, sink_size)); 
    }

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running INT VIRUS (Integer ALU Stress)..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- GOLDEN PASS ---
    unsigned int* d_golden_sink = nullptr;
    
    if (verify_mode) {
        std::cout << "[PANTHEON] Generating expected integer baselines (Golden Pass)..." << std::endl;
        CHECK(hipMalloc(&d_golden_sink, sink_size));
        LAUNCH_KERNEL(golden_int_kernel, num_blocks, block_size, kernel_loops, d_golden_sink);
        CHECK(hipDeviceSynchronize());
    }

    // --- 4. WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup iterations..." << std::endl;
        for(int i = 0; i < warmup_iters; i++) {
            LAUNCH_KERNEL(int_virus_kernel, num_blocks, block_size, kernel_loops, d_sink, inject_error);
        }
        CHECK(hipDeviceSynchronize());
        
        // Re-apply memory pattern so telemetry loop accumulates from the correct baseline
        if (init_pattern == 1) {
            CHECK(hipMemset(d_sink, 0xFF, sink_size)); 
        } else {
            CHECK(hipMemset(d_sink, 0x00, sink_size)); 
        }
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t ops_performed = 0;
    unsigned int kernel_launches = 0;
    
    // --- 5. ACTIVE LOOP ---
    while(true) {
        LAUNCH_KERNEL(int_virus_kernel, num_blocks, block_size, kernel_loops, d_sink, inject_error);
        CHECK(hipDeviceSynchronize());
        
        kernel_launches++;

        // 4 integer operations per unrolled loop * 32 unrolls
        ops_performed += (size_t)num_blocks * block_size * kernel_loops * 32 * 4;
        
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }
    
    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    
    // Output in TOPS (Tera Operations Per Second) since it's integer math, not floating point
    std::cout << "Throughput: " << (ops_performed / 1e12) / seconds << " TOPS" << std::endl;
    
    // --- 6. VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running INT32 Arithmetic Verification Pass..." << std::endl;

        if (inject_error && kernel_loops <= 500) {
            LAUNCH_KERNEL(inject_int_sink_error, 1, 1, d_sink, total_threads);
            CHECK(hipDeviceSynchronize());
        }
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        // Determine offset for expected math based on init pattern
        unsigned int init_offset = (init_pattern == 1) ? 0xFFFFFFFF : 0x00000000;

        int verify_blocks = (total_threads + 255) / 256;
        LAUNCH_KERNEL(verify_int_kernel, verify_blocks, 256, d_sink, d_golden_sink, kernel_launches, d_err_count, init_offset);
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

    CHECK(hipFree(d_sink));
    return 0;
}
