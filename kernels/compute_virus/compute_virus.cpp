#include "../common/common.h"
#include <vector>
#include <chrono>
#include <string>
#include <iostream>

// --- VOLTAGE VIRUS KERNEL (Stabilized Bit-Toggle) ---
__global__ 
void voltage_droop_kernel(int iters, float* sink, int inject_error) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = 1.0f; 
    float b = 0.999f; // Slightly less than 1 to prevent divergence
    float c = 0.001f;
    float d = -0.001f;

    for(int i = 0; i < iters; ++i) {
        #ifdef __HIP_PLATFORM_AMD__
            a = __builtin_fmaf(a, b, c);
            b = __builtin_fmaf(b, c, d);
        #else
            a = fmaf(a, b, c);
            b = fmaf(b, c, d);
        #endif
        
        // Polarity Shock + Value Clamp
        // If the value gets too large, we reset it to a known state
        // to keep the ALUs busy without hitting NaN.
        if ((i & 0xFF) == 0) {
            a = (a > 2.0f || a < -2.0f) ? 1.0f : -a;
            b = (b > 2.0f || b < -2.0f) ? 0.999f : -b;
        }
    }

    float final_val = a + b;
    
    // --- DYNAMIC FAULT INJECTION ---
    // If the Python flag was passed, flip a bit on thread 1337
    if (inject_error && tid == 1337) {
        // Cast the float to an integer to safely manipulate the raw bits
        unsigned int bits = pantheon_bit_cast<unsigned int>(final_val);
        // Flip the least significant bit in the mantissa using XOR
        bits ^= 0x00000001;
        // Cast back to float
        final_val = pantheon_bit_cast<float>(bits);
    }
    
    sink[tid] = final_val;
}

// --- VERIFICATION KERNEL (Bitwise Integer Comparison) ---
__global__ void verify_compute_kernel(float* data, size_t n, unsigned int expected_bits, unsigned int* err_count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Compare bits directly using standard C++ pointer casting to support MOCK builds
        unsigned int actual_bits = pantheon_bit_cast<unsigned int>(data[tid]);
        
        if (actual_bits != expected_bits) {
            unsigned int xor_bits = expected_bits ^ actual_bits;
            float actual_f = pantheon_bit_cast<float>(actual_bits);
            float expected_f = pantheon_bit_cast<float>(expected_bits);
            
            printf("[SDC FAULT][COMPUTE_VIRUS] ALU Error! TID: %llu | Exp: %f (0x%08x) | Act: %f (0x%08x) | XOR: 0x%08x\n",
                   (unsigned long long)tid, expected_f, expected_bits, actual_f, actual_bits, xor_bits);

            // Report the error back to the host
            atomicAdd(err_count, 1);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);

    // --- PANTHEON CONFIG KNOBS ---
    int block_size = 256;      
    int grid_size = 0;         // 0 = auto-calculate
    int kernel_loops = 20000;  // Maps directly to FMA 'iters'
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
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));

    // --- 2. EXPLICIT OCCUPANCY ---
    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) {
        int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
        if (max_blocks_per_sm < 1) max_blocks_per_sm = 1; // Fallback
        num_blocks = prop.multiProcessorCount * max_blocks_per_sm;
        auto_grid = true;
    }

    size_t total_threads = (size_t)num_blocks * block_size;
    size_t alloc_size = total_threads * sizeof(float);
    
    float* d_sink; CHECK(hipMalloc(&d_sink, alloc_size));
    
    // --- 3. SET MEMORY INIT PATTERN ---
    if (init_pattern == 1) {
        CHECK(hipMemset(d_sink, 0xFF, alloc_size)); 
    } else {
        CHECK(hipMemset(d_sink, 0x00, alloc_size)); 
    }

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running VOLTAGE DROOP VIRUS (ALU Hammer)..." << std::endl;
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
            LAUNCH_KERNEL(voltage_droop_kernel, num_blocks, block_size, kernel_loops, d_sink, inject_error);
        }
        CHECK(hipDeviceSynchronize());
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t ops_performed = 0;

    // --- 5. ACTIVE LOOP ---
    while(true) {
        LAUNCH_KERNEL(voltage_droop_kernel, num_blocks, block_size, kernel_loops, d_sink, inject_error);
        CHECK(hipDeviceSynchronize());
        
        // 4 operations per FMA loop iteration
        ops_performed += total_threads * kernel_loops * 4;
        
        if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (ops_performed / 1e12) / seconds << " TFLOPS" << std::endl;

    // --- 6. VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running Bitwise Arithmetic Verification Pass..." << std::endl;
        float expected_val_f;
        
        // Grab a clean value from thread 0 as our "Golden" standard (since we only infected 1337)
        CHECK(hipMemcpy(&expected_val_f, d_sink, sizeof(float), hipMemcpyDeviceToHost));
        unsigned int expected_val_bits = pantheon_bit_cast<unsigned int>(expected_val_f);

        // Allocate error counter
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));

        int verify_blocks = (total_threads + 255) / 256;
        LAUNCH_KERNEL(verify_compute_kernel, verify_blocks, 256, d_sink, total_threads, expected_val_bits, d_err_count);
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
