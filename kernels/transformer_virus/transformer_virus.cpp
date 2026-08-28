#include "../common/common.h"
#include <chrono>
#include <iostream>
#include <string>

// --- NATIVE COMPILER DETECTION ---
// Only include hardware headers if compiled by an actual GPU compiler
#if defined(__CUDACC__) || defined(__HIPCC__)
    #if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
        #include <hip/hip_fp16.h>
        #define PLATFORM_NVIDIA_HOPPER 0
        #define PLATFORM_NVIDIA_LEGACY 0
        // RDNA-family parts provide WMMA; CDNA Instinct parts provide MFMA.
        // The build system sets PANTHEON_AMD_WMMA_TARGET from the detected
        // architecture *family*, so no individual model number is hardcoded
        // here and a new member of either family needs no source change.
        #if defined(PANTHEON_AMD_WMMA_TARGET)
          #define PLATFORM_AMD_CDNA 0
          #if defined(PANTHEON_ENABLE_EXPERIMENTAL_WMMA) && __has_include(<rocwmma/rocwmma.hpp>)
            #include <rocwmma/rocwmma.hpp>
            namespace wmma= rocwmma;
            #define PLATFORM_AMD_WMMA 1
          #else
            #define PLATFORM_AMD_WMMA 0
          #endif
        #else
          #define PLATFORM_AMD_CDNA 1
          #define PLATFORM_AMD_WMMA 0
        #endif

        typedef float float4_t __attribute__((ext_vector_type(4)));
        typedef _Float16 half4_t __attribute__((ext_vector_type(4)));
    #else
        #include <cuda_fp16.h>
        #define PLATFORM_AMD_CDNA 0
        // WMMA is available on Volta and later, including Hopper and Blackwell.
        // Keep the Hopper classification for reporting/selection, but use the
        // supported CUDA WMMA interface rather than fragile handwritten WGMMA
        // PTX. The previous operand list was rejected by CUDA 12.8 ptxas.
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
            #include <mma.h>
            #define PLATFORM_NVIDIA_HOPPER 1
            #define PLATFORM_NVIDIA_LEGACY 0
        // Check for Volta (700) through Ada Lovelace (890)
        #elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
            #include <mma.h>
            #define PLATFORM_NVIDIA_HOPPER 0
            #define PLATFORM_NVIDIA_LEGACY 1
        #else
            #define PLATFORM_NVIDIA_HOPPER 0
            #define PLATFORM_NVIDIA_LEGACY 0
        #endif
    #endif

    // --- TENSOR VIRUS KERNEL (Real GPU) ---
    __global__ void transformer_virus_kernel(int iters, unsigned int* sink, int inject_now, int init_pattern) {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t local_tid = threadIdx.x;
        float final_val = 0.0f;
        
        float sign = (init_pattern == 1) ? -1.0f : 1.0f;

    #if PLATFORM_AMD_CDNA
        // --- PATH 1: AMD INSTINCT (MI200/MI300X/MI350X) ---
        // ROCm 7+ clang: float -> _Float16 in a braced initializer is ill-formed narrowing; cast explicitly.
        half4_t a = {static_cast<_Float16>(1.01f * sign), static_cast<_Float16>(1.01f * sign),
                     static_cast<_Float16>(1.01f * sign), static_cast<_Float16>(1.01f * sign)};
        half4_t b = {static_cast<_Float16>(0.99f * sign), static_cast<_Float16>(0.99f * sign),
                     static_cast<_Float16>(0.99f * sign), static_cast<_Float16>(0.99f * sign)};
        float4_t c = {0.001f * sign, 0.001f * sign, 0.001f * sign, 0.001f * sign};

        for (int i = 0; i < iters; i++) {
            #pragma unroll 16
            for (int j = 0; j < 16; j++) {
                c = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
            }
            if ((i & 0xFF) == 0) {
                c = (float4_t){0.001f * sign, 0.001f * sign, 0.001f * sign, 0.001f * sign};
                a[0] = -a[0]; 
            }
        }
        final_val = c[0];

    #elif PLATFORM_AMD_WMMA
	// --- PATH 18: AMD RDNA-family via rocWMMA ---
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> a;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;

	wmma::fill_fragment(a, __float2half(1.01f * sign));
	wmma::fill_fragment(b, __float2half(0.99f * sign));
	wmma::fill_fragment(c, 0.001f * sign);

	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> neg_a;
	wmma::fill_fragment(neg_a, __float2half(-1.01f * sign));

	for (int i = 0; i < iters; i++) {
	    #pragma unroll 16
	    for (int j = 0; j < 16; j++) {
		wmma::mma_sync(c, a, b, c);
	    }
	    if ((i & 0xFF) == 0) {
		wmma::fill_fragment(c, 0.001f * sign);
		if ((i & 0x100) == 0) a = neg_a;
		else wmma::fill_fragment(a, __float2half(1.01f * sign));
	    }
	}

	__shared__ float smem[2048];
	int warp_id = local_tid / 32;
	wmma::store_matrix_sync(&smem[warp_id * 256], c, 16, wmma::mem_col_major);
	__syncthreads();
	final_val = smem[warp_id * 256];

    #elif defined(__HIP_PLATFORM_AMD__) && defined(PANTHEON_AMD_WMMA_TARGET)
        // Keep release builds independent of optional rocWMMA headers. This
        // portable HIP path is selected whenever the target is a WMMA part and
        // rocWMMA is not available.
        float value = 0.001f * sign;
        const float multiplicand = 1.01f * sign;
        const float multiplier = 0.99f * sign;
        for (int i = 0; i < iters; i++) {
            #pragma unroll 16
            for (int j = 0; j < 16; j++) {
                value = fmaf(multiplicand, multiplier, value);
            }
            if ((i & 0xFF) == 0) {
                value = 0.001f * sign;
            }
        }
        final_val = value;

    #elif PLATFORM_NVIDIA_HOPPER || PLATFORM_NVIDIA_LEGACY
        // --- PATH 2: NVIDIA VOLTA AND LATER (including Hopper/Blackwell) ---
        // WMMA maps to native tensor-core instructions across these targets.
        using namespace nvcuda;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;

        wmma::fill_fragment(a, __float2half(1.01f * sign));
        wmma::fill_fragment(b, __float2half(0.99f * sign));
        wmma::fill_fragment(c, 0.001f * sign);
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> neg_a;
        wmma::fill_fragment(neg_a, __float2half(-1.01f * sign));

        for (int i = 0; i < iters; i++) {
            #pragma unroll 16
            for (int j = 0; j < 16; j++) {
                wmma::mma_sync(c, a, b, c);
            }
            if ((i & 0xFF) == 0) {
                wmma::fill_fragment(c, 0.001f * sign); 
                if ((i & 0x100) == 0) a = neg_a;
                else wmma::fill_fragment(a, __float2half(1.01f * sign));
            }
        }
        
        __shared__ float smem[2048];
        int warp_id = local_tid / 32;
        wmma::store_matrix_sync(&smem[warp_id * 256], c, 16, wmma::mem_col_major);
        __syncthreads();
        final_val = smem[warp_id * 256];

    #else
        // --- PORTABLE FALLBACK (unmatched architectures) ---
        // Without this branch an unrecognized target would leave final_val at
        // its 0.0f initializer and report plausible throughput for a kernel
        // that did no work at all.
        float value = 0.001f * sign;
        const float multiplicand = 1.01f * sign;
        const float multiplier = 0.99f * sign;
        for (int i = 0; i < iters; i++) {
            #pragma unroll 16
            for (int j = 0; j < 16; j++) {
                value = fmaf(multiplicand, multiplier, value);
            }
            if ((i & 0xFF) == 0) {
                value = 0.001f * sign;
            }
        }
        final_val = value;
    #endif

        // --- DYNAMIC FAULT INJECTION ---
        if (inject_now && tid == 1337) {
            final_val += 9999.0f;
        }

        // Bitcast and accumulate to prevent floating point healing and track state across launches
        unsigned int bits = pantheon_bit_cast<unsigned int>(final_val);
        sink[tid] += bits;
    }

#else
    // --- CI MOCK FALLBACK KERNEL (Standard CPU Compiler) ---
    __global__ void transformer_virus_kernel(int iters, unsigned int* sink, int inject_now, int init_pattern) {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        float sign = (init_pattern == 1) ? -1.0f : 1.0f;
        float final_val = (float)iters * sign;
        
        if (inject_now && tid == 1337) {
            final_val += 9999.0f;
        }
        
        unsigned int bits = pantheon_bit_cast<unsigned int>(final_val);
        sink[tid] += bits;
    }
#endif

// --- VERIFICATION KERNEL ---
__global__ void verify_tensor_kernel(unsigned int* sink, unsigned int* golden_sink, unsigned int launches, size_t n, unsigned int* err_count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        unsigned int exp = golden_sink[tid] * launches;
        unsigned int act = sink[tid];
        
        if (exp != act) {
            // Decoupled atomicAdd return value for MOCK platform compatibility
            if (*err_count < 5) {
                printf("[SDC FAULT][TRANSFORMER_VIRUS] Matrix Error! TID: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n",
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

    // --- PANTHEON CONFIG KNOBS ---
    int block_size = 256;      
    int grid_size = 0;         // 0 = auto-calculate
    int kernel_loops = 1000;   // Maps to math inner iterations
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 0;      // 0=Standard Polarity, 1=Inverted Polarity

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

    unsigned int* d_sink; 
    CHECK(hipMalloc(&d_sink, sink_size));
    CHECK(hipMemset(d_sink, 0, sink_size));

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running TRANSFORMER VIRUS (Matrix Core Hammer)..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << " (0=Standard Polarity, 1=Inverted Polarity)" << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;
    
    // --- GOLDEN PASS ---
    unsigned int* d_golden_sink = nullptr;

    if (verify_mode) {
        std::cout << "[PANTHEON] Generating expected Tensor/Matrix baseline (Golden Pass)..." << std::endl;
        CHECK(hipMalloc(&d_golden_sink, sink_size));

        // Wipe the VRAM clean so the baseline doesn't accumulate on top of garbage data!
        CHECK(hipMemset(d_golden_sink, 0, sink_size));

        // Execute 1 iteration of the virus kernel with inject_now = 0 to establish the baseline
        LAUNCH_KERNEL(transformer_virus_kernel, num_blocks, block_size, kernel_loops, d_golden_sink, 0, init_pattern);
        CHECK(hipDeviceSynchronize());
    }

    // --- 4. WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup iterations..." << std::endl;
        for(int i = 0; i < warmup_iters; i++) {
            LAUNCH_KERNEL(transformer_virus_kernel, num_blocks, block_size, kernel_loops, d_sink, 0, init_pattern);
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
    while(true) {
        int inject_now = (inject_error && kernel_launches == 50) ? 1 : 0;

        LAUNCH_KERNEL(transformer_virus_kernel, num_blocks, block_size, kernel_loops, d_sink, inject_now, init_pattern);
        CHECK(hipDeviceSynchronize());
        
        kernel_launches++;
        ops_performed += (total_threads / 32) * kernel_loops * 16 * 8192;
        
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }
    
    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (ops_performed / 1e12) / seconds << " TFLOPS" << std::endl;
    
    // --- 6. VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running Tensor/Matrix Verification Pass..." << std::endl;
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));

        int verify_blocks = (total_threads + 255) / 256;
        LAUNCH_KERNEL(verify_tensor_kernel, verify_blocks, 256, d_sink, d_golden_sink, kernel_launches, total_threads, d_err_count);
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
