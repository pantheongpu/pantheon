#include "../common/common.h"
#include <vector>
#include <chrono>
#include <string>
#include <iostream>

// Include hardware-specific FP16 intrinsics
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
    #include <hip/hip_fp16.h>
#elif defined(__CUDACC__)
    #include <cuda_fp16.h>
#else
    // --- CI MOCK BUILD FALLBACKS ---
    // The CPU compiler lacks these specific FP16 hardware intrinsics.
    // We macro them to standard float operations just to pass the CI syntax check.
    #define half2 float
    #define __hfma2(a, b, c) ((a)*(b)+(c))
    #define __low2float(x) (x)
    #define __hneg2(x) (-(x))
#endif

// --- INCINERATOR KERNEL (FP16/Half2 + SRAM Stress) ---
// Smashes the dedicated FP16 math units while simultaneously hammering
// the L1/Shared Memory banks to create maximum localized thermal density.
__global__ 
void incinerator_kernel(int iters, float* sink, int inject_error) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t local_tid = threadIdx.x;
    
    // 1. Shared Memory Thrashing (LDS/SRAM)
    // Statically allocate max possible threads per block to support dynamic occupancy from AI fuzzer
    __shared__ float smem[1024]; 
    smem[local_tid] = (float)local_tid;
    __syncthreads();

    // 2. Packed FP16 (Half2) Registers
    // Removes 'volatile' and locks variables into ultra-fast FP16 datapaths.
    half2 a = __float2half2_rn(1.0f);
    half2 b = __float2half2_rn(0.999f);
    half2 c = __float2half2_rn(0.001f);
    half2 d = __float2half2_rn(-0.001f);

    for(int i = 0; i < iters; ++i) {
        
        // --- VECTOR UNIT STRESS (Packed FP16 FMA) ---
        #if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
            // AMD HIP construction for packed half2 math
            a = __hadd2(__hmul2(a, b), c);
            b = __hadd2(__hmul2(b, c), d);
            c = __hadd2(__hmul2(c, d), a);
            d = __hadd2(__hmul2(d, a), b);
        #else
            // NVIDIA Native Half2 FMA Instruction (__hfma2)
            a = __hfma2(a, b, c);
            b = __hfma2(b, c, d);
            c = __hfma2(c, d, a);
            d = __hfma2(d, a, b);
        #endif

        // --- SRAM STRESS (L1 Cache) ---
        // XOR index deliberately causes bank conflicts to generate heat
        int idx = local_tid ^ 1; 
        float val = smem[idx];
        smem[local_tid] = val + 0.0001f; 
        
        // --- POLARITY SHOCK & CLAMPING ---
        // FP16 overflows to Infinity extremely fast (max val is only ~65504).
        // Clamp the values to keep the math units busy without hitting the NaN trap.
        if ((i & 0xFF) == 0) {
            float a_float = __low2float(a);
            float b_float = __low2float(b);
            
            a = (a_float > 2.0f || a_float < -2.0f) ? __float2half2_rn(1.0f) : __hneg2(a);
            b = (b_float > 2.0f || b_float < -2.0f) ? __float2half2_rn(0.999f) : __hneg2(b);
            
            // Periodically reset shared memory to prevent float degradation
            smem[local_tid] = (float)local_tid;
        }
    }

    // --- DCE BYPASS & CONSENSUS SINK ---
    // We must reference smem to prevent the compiler from deleting the SRAM stress block.
    // However, we cannot add smem to the final value, or the Thread IDs will ruin the verification consensus.
    float final_val = __low2float(a) + __low2float(b);
    
    // This condition will physically never happen, but the compiler doesn't know that at compile-time.
    // This forces the compiler to execute all the SRAM reads/writes without poisoning our FP16 math.
    if (smem[local_tid] == 0xDEADBEEF) {
        final_val = 0.0f; 
    }

    // --- DYNAMIC FAULT INJECTION ---
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
__global__ void verify_incinerator_kernel(float* data, size_t n, unsigned int expected_bits, unsigned int* err_count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Compare bits directly using standard C++ pointer casting
        unsigned int actual_bits = pantheon_bit_cast<unsigned int>(data[tid]);
        
        if (actual_bits != expected_bits) {
            unsigned int xor_bits = expected_bits ^ actual_bits;
            float actual_f = pantheon_bit_cast<float>(actual_bits);
            float expected_f = pantheon_bit_cast<float>(expected_bits);
            
            printf("[SDC FAULT][COMPUTE_VIRUS_AGG] FP16/SRAM Error! TID: %llu | Exp: %f (0x%08x) | Act: %f (0x%08x) | XOR: 0x%08x\n",
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
    int kernel_loops = 20000;  // Maps directly to FP16 FMA 'iters'
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
    float* d_sink; CHECK(hipMalloc(&d_sink, total_threads * sizeof(float)));

    // --- 3. SET MEMORY INIT PATTERN ---
    if (init_pattern == 1) {
        CHECK(hipMemset(d_sink, 0xFF, total_threads * sizeof(float))); 
    } else {
        CHECK(hipMemset(d_sink, 0x00, total_threads * sizeof(float))); 
    }

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running INCINERATOR (FP16/Half2 + SRAM Stress)..." << std::endl;
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
            LAUNCH_KERNEL(incinerator_kernel, num_blocks, block_size, kernel_loops, d_sink, inject_error);
        }
        CHECK(hipDeviceSynchronize());
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t ops_performed = 0;

    // --- 5. ACTIVE LOOP ---
    while(true) {
        LAUNCH_KERNEL(incinerator_kernel, num_blocks, block_size, kernel_loops, d_sink, inject_error);
        CHECK(hipDeviceSynchronize());

        // A Half2 op is 2 calculations per instruction. FMA is 2 ops (Mul + Add).
        // We do 4 instructions * 2 (Half2) * 2 (FMA) = 16 operations per thread per loop.
        ops_performed += total_threads * kernel_loops * 16;

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (ops_performed / 1e12) / seconds << " TFLOPS" << std::endl;

    // --- 6. VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running Bitwise FP16 Verification Pass..." << std::endl;
        
        float expected_val_f;
        // Grab a clean value from thread 0 as our "Golden" standard
        CHECK(hipMemcpy(&expected_val_f, d_sink, sizeof(float), hipMemcpyDeviceToHost));
        
        // Host-side bit casting to prevent cross-boundary compiler errors
        unsigned int expected_val_bits = pantheon_bit_cast<unsigned int>(expected_val_f);
        
        // Allocate error counter
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        int verify_blocks = (total_threads + 255) / 256;
        LAUNCH_KERNEL(verify_incinerator_kernel, verify_blocks, 256, d_sink, total_threads, expected_val_bits, d_err_count);
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
