#include "../common/common.h"
#include "../common/fp16_shim.h"
#include <chrono>
#include <string>
#include <iostream>

// --- CROSS-PLATFORM WMMA SHIM ---
#ifdef __CUDACC__
    #include <mma.h>
    using namespace nvcuda;
    #define WMMA_SUPPORTED 1
#elif defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    // If the server has the rocwmma headers installed, compile the payload.
    // This bypasses the need for a brittle architecture whitelist.
    #if __has_include(<rocwmma/rocwmma.hpp>)
        #include <rocwmma/rocwmma.hpp>
        namespace wmma = rocwmma; // Map NVIDIA's namespace to AMD's
        #define WMMA_SUPPORTED 1
    #else
        #define WMMA_SUPPORTED 0
        #pragma message("rocWMMA header not found. Dummy kernel will be built.")
    #endif
#else
    #define WMMA_SUPPORTED 0
#endif

#if WMMA_SUPPORTED
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// --- GOLDEN PASS KERNEL ---
__global__ void golden_mma_kernel(int iters, float* golden_sink, int init_pattern) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Use init_pattern to dictate the starting sign of the matrix fragments
    float start_a = (init_pattern == 1) ? -1.01f : 1.01f;
    float start_b = (init_pattern == 1) ? -0.99f : 0.99f;

    wmma::fill_fragment(a_frag, __float2half(start_a));
    wmma::fill_fragment(b_frag, __float2half(start_b));
    wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < iters; ++i) {
        #pragma unroll 32
        for (int j = 0; j < 32; ++j) {
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        if ((i & 0xF) == 0) {
            wmma::fill_fragment(a_frag, __float2half(-start_a));
        } else if ((i & 0x1F) == 0) {
            wmma::fill_fragment(a_frag, __float2half(start_a));
        }
    }

    golden_sink[tid] = c_frag.x[0];
}

// --- MMA VIRUS KERNEL ---
__global__ void mma_virus_kernel(int iters, float* sink, int inject_error, int init_pattern) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    float start_a = (init_pattern == 1) ? -1.01f : 1.01f;
    float start_b = (init_pattern == 1) ? -0.99f : 0.99f;

    wmma::fill_fragment(a_frag, __float2half(start_a));
    wmma::fill_fragment(b_frag, __float2half(start_b));
    wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < iters; ++i) {
        #pragma unroll 32
        for (int j = 0; j < 32; ++j) {
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && tid == 1337 && i == 500) {
            c_frag.x[0] += 1000.0f;
        }
        
        if ((i & 0xF) == 0) {
            wmma::fill_fragment(a_frag, __float2half(-start_a));
        } else if ((i & 0x1F) == 0) {
            wmma::fill_fragment(a_frag, __float2half(start_a));
        }
    }

    sink[tid] = c_frag.x[0];
}

// --- VERIFICATION KERNEL ---
__global__ void verify_mma_kernel(float* sink, float* golden_sink, size_t n, unsigned int* err_count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float act = sink[tid];
        float exp = golden_sink[tid];
        
        // Cast to bits to avoid NaN evaluation weirdness and catch exact hardware flips
        unsigned int act_bits = pantheon_bit_cast<unsigned int>(act);
        unsigned int exp_bits = pantheon_bit_cast<unsigned int>(exp);

        if (act_bits != exp_bits) {
            unsigned int xor_bits = exp_bits ^ act_bits;
            printf("[SDC FAULT][MMA_VIRUS] Tensor Core Error! TID: %llu | Exp: %f (0x%08x) | Act: %f (0x%08x) | XOR: 0x%08x\n",
                   (unsigned long long)tid, exp, exp_bits, act, act_bits, xor_bits);
            atomicAdd(err_count, 1);
        }
    }
}

#else
__global__ void mma_virus_kernel(int iters, float* sink, int inject_error, int init_pattern) {
    // Dummy kernel to satisfy compiler on unsupported architectures
}
#endif

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);

    // --- PANTHEON CONFIG KNOBS ---
    int block_size = 256;      
    int grid_size = 0;         // 0 = auto-calculate
    int kernel_loops = 10000;  // Maps to WMMA 'iters'
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 0;      // 0=Positive Floats, 1=Negative Floats

    bool verify_mode = false;
    int inject_error = 0;

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

#if !WMMA_SUPPORTED
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Skipping MMA VIRUS (Hardware Matrix Cores or headers not available)." << std::endl;
    std::cout << "Throughput: 0.0 TFLOPS" << std::endl;
    return 0;
#else

    hipDeviceProp_t prop; 
    CHECK(hipGetDeviceProperties(&prop, gpu_id));

    // --- 2. EXPLICIT OCCUPANCY ---
    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) {
        int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
        if (max_blocks_per_sm > 4) max_blocks_per_sm -= 1; 
        if (max_blocks_per_sm < 1) max_blocks_per_sm = 8;  
        
        num_blocks = prop.multiProcessorCount * max_blocks_per_sm;
        auto_grid = true;
    }

    // Track thread-level sinks instead of just block-level
    size_t total_threads = (size_t)num_blocks * block_size;
    float* d_sink; 
    CHECK(hipMalloc(&d_sink, total_threads * sizeof(float)));

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running MMA VIRUS (Physical Tensor Cores)..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << " (0=Pos Matrix, 1=Neg Matrix)" << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- GOLDEN PASS ---
    float* d_golden_sink = nullptr;
    
    if (verify_mode) {
        std::cout << "[PANTHEON] Generating expected Tensor Core baseline (Golden Pass)..." << std::endl;
        CHECK(hipMalloc(&d_golden_sink, total_threads * sizeof(float)));
        LAUNCH_KERNEL(golden_mma_kernel, num_blocks, block_size, kernel_loops, d_golden_sink, init_pattern);
        CHECK(hipDeviceSynchronize());
    }

    // --- 4. WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup iterations..." << std::endl;
        for(int i = 0; i < warmup_iters; i++) {
            LAUNCH_KERNEL(mma_virus_kernel, num_blocks, block_size, kernel_loops, d_sink, inject_error, init_pattern);
        }
        CHECK(hipDeviceSynchronize());
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t ops_performed = 0;
    
    size_t flops_per_wmma = 8192;
    int warps_per_block = block_size / prop.warpSize;

    // --- 5. ACTIVE LOOP ---
    while(true) {
        LAUNCH_KERNEL(mma_virus_kernel, num_blocks, block_size, kernel_loops, d_sink, inject_error, init_pattern);
        CHECK(hipDeviceSynchronize());
        
        ops_performed += (size_t)num_blocks * warps_per_block * kernel_loops * 32 * flops_per_wmma;
        
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }
    
    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (ops_performed / 1e12) / seconds << " TFLOPS" << std::endl;
    
    // --- 6. VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running Tensor Core State Verification Pass..." << std::endl;
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        int verify_blocks = (total_threads + 255) / 256;
        LAUNCH_KERNEL(verify_mma_kernel, verify_blocks, 256, d_sink, d_golden_sink, total_threads, d_err_count);
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
#endif
}
