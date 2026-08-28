#include "../common/common.h"
#include <vector>
#include <chrono>
#include <string>
#include <iostream>

// --- FP64 (DOUBLE PRECISION) CHOKEHOLD ---
// Exposes the physical FP64 ALU limits of the GPU.
// Consumer cards will severely bottleneck here compared to Datacenter accelerators.
__global__ 
void fp64_virus_kernel(int iters, double* sink, int inject_error) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Register-locked 64-bit floats
    double a = 1.0;
    double b = 0.999; // Slightly less than 1 to prevent runaway divergence
    double c = 0.001;
    double d = -0.001;

    for(int i = 0; i < iters; ++i) {
        #pragma unroll 32
        for(int j = 0; j < 32; ++j) {
            // Fused Multiply-Add (Double Precision)
            #if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
                a = __builtin_fma(a, b, c);
                b = __builtin_fma(b, c, d);
                c = __builtin_fma(c, d, a);
                d = __builtin_fma(d, a, b);
            #else
                a = fma(a, b, c);
                b = fma(b, c, d);
                c = fma(c, d, a);
                d = fma(d, a, b);
            #endif
        }
        
        // Polarity Shock + Value Clamp
        // Keeps the 64-bit ALUs busy without hitting Infinity/NaN
        if ((i & 0xFF) == 0) {
            a = (a > 2.0 || a < -2.0) ? 1.0 : -a;
            b = (b > 2.0 || b < -2.0) ? 0.999 : -b;
        }

        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && tid == 1337 && i == 500) {
            // Intentionally corrupt the 64-bit FMA chain
            a += 100.0; 
        }
    }

    // Sink accumulator to prevent Dead Code Elimination (DCE)
    sink[tid] = a + b + c + d;
}

// --- VERIFICATION KERNEL (64-Bit Integer Comparison) ---
__global__ void verify_fp64_kernel(double* data, size_t n, unsigned long long expected_bits, unsigned int* err_count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Compare 64 bits directly using standard C++ pointer casting
        unsigned long long actual_bits = pantheon_bit_cast<unsigned long long>(data[tid]);

        if (actual_bits != expected_bits) {
            unsigned long long xor_bits = expected_bits ^ actual_bits;
            double actual_f = pantheon_bit_cast<double>(actual_bits);
            double expected_f = pantheon_bit_cast<double>(expected_bits);

            printf("[SDC FAULT][FP64_VIRUS] ALU Error! TID: %llu | Exp: %f (0x%016llx) | Act: %f (0x%016llx) | XOR: 0x%016llx\n",
                   (unsigned long long)tid, expected_f, expected_bits, actual_f, actual_bits, xor_bits);
            
            // Report the error back to the host
            atomicAdd(err_count, 1);
        }
    }
}

// Keep --inject_error deterministic when kernel_loops never reaches the
// in-kernel injection iteration (i == 500); corrupting one sink lane
// exercises the exact comparison used by the verification pass.
__global__ void inject_fp64_sink_error(double* sink, size_t count) {
    if (count > 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long bits = pantheon_bit_cast<unsigned long long>(sink[0]);
        sink[0] = pantheon_bit_cast<double>(bits ^ 1ull);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);

    // --- PANTHEON CONFIG KNOBS ---
    int block_size = 256;      
    int grid_size = 0;         // 0 = auto-calculate
    int kernel_loops = 10000;  // Maps directly to FP64 FMA 'iters'
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
        int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
        if (max_blocks_per_sm < 1) max_blocks_per_sm = 16; // Fallback
        num_blocks = prop.multiProcessorCount * max_blocks_per_sm;
        auto_grid = true;
    }

    size_t total_threads = (size_t)num_blocks * block_size;
    double* d_sink; 
    CHECK(hipMalloc(&d_sink, total_threads * sizeof(double)));

    // --- 3. SET MEMORY INIT PATTERN ---
    if (init_pattern == 1) {
        CHECK(hipMemset(d_sink, 0xFF, total_threads * sizeof(double))); 
    } else {
        CHECK(hipMemset(d_sink, 0x00, total_threads * sizeof(double))); 
    }

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running FP64 VIRUS (Double Precision Chokehold)..." << std::endl;
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
            LAUNCH_KERNEL(fp64_virus_kernel, num_blocks, block_size, kernel_loops, d_sink, inject_error);
        }
        CHECK(hipDeviceSynchronize());
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t ops_performed = 0;
    
    // --- 5. ACTIVE LOOP ---
    while(true) {
        LAUNCH_KERNEL(fp64_virus_kernel, num_blocks, block_size, kernel_loops, d_sink, inject_error);
        CHECK(hipDeviceSynchronize());
        
        // 4 instructions * 2 (FMA is 2 ops) = 8 ops per inner loop. 
        // We unroll 32 times, so 8 * 32 = 256 operations per outer loop iteration.
        ops_performed += total_threads * kernel_loops * 256;
        
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }
    
    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (ops_performed / 1e12) / seconds << " TFLOPS" << std::endl;
    
    // --- 6. VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running 64-Bit Arithmetic Verification Pass..." << std::endl;
        
        // Allocate error counter
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));

        double expected_val_f;
        // We assume thread 0 (index 0) is clean to use as our "Golden" reference value
        CHECK(hipMemcpy(&expected_val_f, d_sink, sizeof(double), hipMemcpyDeviceToHost));
        
        // Use 64-bit integer cast for the expected verification state to bypass NaN != NaN limits
        unsigned long long expected_val_bits = pantheon_bit_cast<unsigned long long>(expected_val_f);

        // The in-kernel injection point (i == 500) is unreachable when
        // kernel_loops is small; corrupt lane 0 after capturing the expected
        // value so the same comparison still trips deterministically.
        if (inject_error && kernel_loops <= 500) {
            LAUNCH_KERNEL(inject_fp64_sink_error, 1, 1, d_sink, total_threads);
            CHECK(hipDeviceSynchronize());
        }
        
        int verify_blocks = (total_threads + 255) / 256;
        LAUNCH_KERNEL(verify_fp64_kernel, verify_blocks, 256, d_sink, total_threads, expected_val_bits, d_err_count);
        CHECK(hipDeviceSynchronize());

        // Check for errors
        unsigned int h_err_count = 0;
        CHECK(hipMemcpy(&h_err_count, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));
        CHECK(hipFree(d_err_count));

        // Say so on success too. Silence is indistinguishable from a
        // verification that never ran, which is how a self-test that
        // could never fail went unnoticed in memory_pc_pingpong.
        std::cout << "Verification: " << (h_err_count ? "FAIL" : "PASS")
                  << " (" << h_err_count << " errors)" << std::endl;
        if (h_err_count > 0) {
            // CRITICAL: Exit with a non-zero code so Python knows the hardware failed
            return 1;
        }
    }
    
    CHECK(hipFree(d_sink));
    return 0;
}
