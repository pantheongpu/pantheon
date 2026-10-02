#include "../common/common.h"
#include "../common/fp16_shim.h"
#include "../common/toggle_chaos.h"
#include "../common/vendor_gemm.h"
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

// Four operand tiles rotated through the fragments, eight MMAs per group,
// four independent accumulators. Four accumulators are what keeps the tensor
// pipe issuing: mma_sync into a single accumulator serialises on its own
// result, exactly as a single FMA chain does on the vector pipe.
#define MMA_TILES       4
#define MMA_TILE_ELEMS  (WMMA_M * WMMA_K)

// Fill the shared operand tiles. Identical in the golden and stress kernels,
// so a difference between them cannot creep in.
__device__ __forceinline__ void mma_fill_tiles(__half* tiles, int init_pattern) {
    for (int i = threadIdx.x; i < MMA_TILES * MMA_TILE_ELEMS; i += blockDim.x) {
        unsigned int h = pantheon_operand_hash(
            (unsigned int)i + (init_pattern == 1 ? 0x9e3779b9u : 0u));
        tiles[i] = __float2half(PANTHEON_OPERAND_VALUE(h));
    }
    __syncthreads();
}

// --- GOLDEN PASS KERNEL ---
template <int NA>
__device__ __forceinline__ void mma_run(int iters, int groups, int reuse, float* sink,
                                        int inject_error, int init_pattern, bool inject) {
    const int NB = 2;
    const int ACC = NA * NB;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ __half tiles[MMA_TILES * MMA_TILE_ELEMS];
    mma_fill_tiles(tiles, init_pattern);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a[NA];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b[NB];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c[ACC];

    #pragma unroll
    for (int k = 0; k < ACC; ++k) wmma::fill_fragment(c[k], 0.0f);

    int phase = 0;
    for (int i = 0; i < iters; ++i) {
        // `groups` operand reloads per iteration, `reuse` MMA passes each.
        // Raising reuse issues more MMAs per shared-memory load, which is
        // what a throughput-tuned GEMM wants; a furnace does not necessarily,
        // because the loads themselves burn power in the LSU and the shared
        // banks. Measured on a B200 the reload-heavy shape drew more, hence
        // the defaults, but both directions stay reachable.
        #pragma unroll
        for (int m = 1; m < NA; ++m) {
            wmma::load_matrix_sync(
                a[m], tiles + ((phase + m + 1) & (MMA_TILES - 1)) * MMA_TILE_ELEMS, WMMA_K);
        }
        wmma::load_matrix_sync(
            b[0], tiles + ((phase + NA + 1) & (MMA_TILES - 1)) * MMA_TILE_ELEMS, WMMA_N);

        #pragma unroll 4
        for (int g = 0; g < groups; ++g) {
            wmma::load_matrix_sync(
                a[0], tiles + ((phase + g + 0) & (MMA_TILES - 1)) * MMA_TILE_ELEMS, WMMA_K);
            wmma::load_matrix_sync(
                b[1], tiles + ((phase + g + 1) & (MMA_TILES - 1)) * MMA_TILE_ELEMS, WMMA_N);

            #pragma unroll 4
            for (int r = 0; r < reuse; ++r) {
                #pragma unroll
                for (int m = 0; m < NA; ++m) {
                    #pragma unroll
                    for (int n = 0; n < NB; ++n) {
                        wmma::mma_sync(c[m * NB + n], a[m], b[n], c[m * NB + n]);
                    }
                }
            }
        }
        phase = (phase + 1) & (MMA_TILES - 1);

        // --- DYNAMIC FAULT INJECTION ---
        // On the final iteration: with random operands the accumulator random
        // walks, so an injection mid-run can be walked back before it reaches
        // the sink.
        if (inject && inject_error && tid == 1337 && i == iters - 1) {
            c[0].x[0] += 1000.0f;
        }
    }

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < ACC; ++k) acc += c[k].x[0];
    sink[tid] = acc;
}

// The golden and stress passes are the same code with injection disabled, so
// the two cannot compute different things by accident.
template <int NA>
__global__ void golden_mma_kernel(int iters, int groups, int reuse, float* golden_sink, int init_pattern) {
    mma_run<NA>(iters, groups, reuse, golden_sink, 0, init_pattern, false);
}

template <int NA>
__global__ void mma_virus_kernel(int iters, int groups, int reuse, float* sink, int inject_error, int init_pattern) {
    mma_run<NA>(iters, groups, reuse, sink, inject_error, init_pattern, true);
}

// --mma_acc selects the instantiation, in one macro per pass so the golden,
// warmup and stress launches cannot drift apart.
#define MMA_LAUNCH_GOLDEN(blocks, bs, iters, sink, pat)                          \
    do {                                                                          \
        if (mma_acc == 2)       LAUNCH_KERNEL(golden_mma_kernel<1>, blocks, bs, iters, mma_groups, mma_reuse, sink, pat); \
        else if (mma_acc == 8)  LAUNCH_KERNEL(golden_mma_kernel<4>, blocks, bs, iters, mma_groups, mma_reuse, sink, pat); \
        else if (mma_acc == 16) LAUNCH_KERNEL(golden_mma_kernel<8>, blocks, bs, iters, mma_groups, mma_reuse, sink, pat); \
        else                    LAUNCH_KERNEL(golden_mma_kernel<2>, blocks, bs, iters, mma_groups, mma_reuse, sink, pat); \
    } while (0)

#define MMA_LAUNCH_STRESS(blocks, bs, iters, sink, inj, pat)                     \
    do {                                                                          \
        if (mma_acc == 2)       LAUNCH_KERNEL(mma_virus_kernel<1>, blocks, bs, iters, mma_groups, mma_reuse, sink, inj, pat); \
        else if (mma_acc == 8)  LAUNCH_KERNEL(mma_virus_kernel<4>, blocks, bs, iters, mma_groups, mma_reuse, sink, inj, pat); \
        else if (mma_acc == 16) LAUNCH_KERNEL(mma_virus_kernel<8>, blocks, bs, iters, mma_groups, mma_reuse, sink, inj, pat); \
        else                    LAUNCH_KERNEL(mma_virus_kernel<2>, blocks, bs, iters, mma_groups, mma_reuse, sink, inj, pat); \
    } while (0)

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
    int mma_acc = 16;          // Accumulators per warp; 16 measured best on B200
    int mma_groups = 4;        // Operand reloads per iteration
    int mma_reuse = 1;         // MMA passes per operand reload
    PantheonGemmOptions gemm_opt;   // vendor GEMM instead of WMMA, see vendor_gemm.h

    bool verify_mode = false;
    int inject_error = 0;

    for (int i = 1; i < argc; i++) {
        if (gemm_opt.parse(argc, argv, &i)) continue;
        if (std::string(argv[i]) == "--verify") verify_mode = true;
        if (std::string(argv[i]) == "--inject_error") inject_error = 1;
        if (std::string(argv[i]) == "--block_size" && i+1 < argc) block_size = atoi(argv[++i]);
        if (std::string(argv[i]) == "--grid_size" && i+1 < argc) grid_size = atoi(argv[++i]);
        if (std::string(argv[i]) == "--kernel_loops" && i+1 < argc) kernel_loops = atoi(argv[++i]);
        if (std::string(argv[i]) == "--warmup_iters" && i+1 < argc) warmup_iters = atoi(argv[++i]);
        if (std::string(argv[i]) == "--sync_mode" && i+1 < argc) sync_mode = atoi(argv[++i]);
        if (std::string(argv[i]) == "--mma_acc" && i+1 < argc) mma_acc = atoi(argv[++i]);
        if (std::string(argv[i]) == "--mma_groups" && i+1 < argc) mma_groups = atoi(argv[++i]);
        if (std::string(argv[i]) == "--mma_reuse" && i+1 < argc) mma_reuse = atoi(argv[++i]);
        if (std::string(argv[i]) == "--init_pattern" && i+1 < argc) {
            if (!pantheon_parse_init_pattern(argv[++i], &init_pattern)) {
                std::cerr << "[PANTHEON] Unknown --init_pattern '" << argv[i] << "'." << std::endl;
                pantheon_print_init_patterns(std::cerr);
                return 1;
            }
        }
    }

    normalize_kernel_launch_config(block_size, grid_size, kernel_loops, warmup_iters, sync_mode);

    if (mma_acc != 2 && mma_acc != 4 && mma_acc != 8 && mma_acc != 16) {
        std::cerr << "[PANTHEON] Warning: mma_acc " << mma_acc
                  << " is not one of 2, 4, 8, 16; using 16." << std::endl;
        mma_acc = 16;
    }
    if (mma_groups < 1) {
        std::cerr << "[PANTHEON] Warning: invalid mma_groups " << mma_groups
                  << "; using 4." << std::endl;
        mma_groups = 4;
    }
    if (mma_reuse < 1) {
        std::cerr << "[PANTHEON] Warning: invalid mma_reuse " << mma_reuse
                  << "; using 1." << std::endl;
        mma_reuse = 1;
    }

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

    // The vendor GEMM replaces the WMMA kernel when the library loads: it
    // reaches the part's asynchronous matrix instructions, which mma_sync
    // cannot. Verification compares each output with the first one.
    bool use_gemm = false;
    int gemm_batch = 0;
    void* d_golden_gemm = nullptr;
#if PANTHEON_VENDOR_GEMM
    PantheonGemm gemm;
    {
        int status = pantheon_gemm_setup(gemm, gemm_opt, gpu_id);
        if (status < 0) return 1;
        use_gemm = (status == 1);
    }
#endif

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
    std::cout << "  -> MMA Accums:    " << mma_acc << " per warp" << std::endl;
    std::cout << "  -> MMA Staging:   " << mma_groups << " reloads x " << mma_reuse
              << " passes" << std::endl;
    if (use_gemm) {
#if PANTHEON_VENDOR_GEMM
        std::cout << "  -> Tensor Engine: vendor GEMM (" << PantheonGemm::type_name(gemm.type) << " "
                  << gemm.m << "x" << gemm.n << "x" << gemm.k << ", FP32 accumulate)" << std::endl;
#endif
    } else {
        std::cout << "  -> Tensor Engine: portable WMMA" << std::endl;
    }
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- GOLDEN PASS ---
    float* d_golden_sink = nullptr;
    
    if (verify_mode) {
        std::cout << "[PANTHEON] Generating expected Tensor Core baseline (Golden Pass)..." << std::endl;
        CHECK(hipMalloc(&d_golden_sink, total_threads * sizeof(float)));
#if PANTHEON_VENDOR_GEMM
        if (use_gemm) {
            // One matmul is the reference: beta is 0, so every later run
            // must reproduce these bits exactly.
            CHECK(hipMalloc(&d_golden_gemm, gemm.out_size_bytes()));
            if (!gemm.run(0)) { std::cerr << "[PANTHEON] Vendor GEMM matmul failed." << std::endl; return 1; }
            CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(d_golden_gemm, gemm.c, gemm.out_size_bytes(), hipMemcpyDeviceToDevice));
        } else
#endif
        {
            MMA_LAUNCH_GOLDEN(num_blocks, block_size, kernel_loops, d_golden_sink, init_pattern);
            CHECK(hipDeviceSynchronize());
        }
    }

    // --- 4. WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup iterations..." << std::endl;
        for(int i = 0; i < warmup_iters; i++) {
#if PANTHEON_VENDOR_GEMM
            if (use_gemm) { gemm.run(0); continue; }
#endif
            MMA_LAUNCH_STRESS(num_blocks, block_size, kernel_loops, d_sink, inject_error, init_pattern);
        }
        CHECK(hipDeviceSynchronize());
    }

#if PANTHEON_VENDOR_GEMM
    if (use_gemm) gemm_batch = pantheon_gemm_batch(gemm, 0, 200);
#endif

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t ops_performed = 0;
    
    size_t flops_per_wmma = 8192;
    int warps_per_block = block_size / prop.warpSize;

    // --- 5. ACTIVE LOOP ---
    while(true) {
#if PANTHEON_VENDOR_GEMM
        if (use_gemm) {
            for (int r = 0; r < gemm_batch; ++r) {
                if (!gemm.run(0)) { std::cerr << "[PANTHEON] Vendor GEMM matmul failed." << std::endl; return 1; }
            }
            CHECK(hipDeviceSynchronize());
            ops_performed += (size_t)(gemm.flops_per_run() * (double)gemm_batch);
        } else
#endif
        {
            MMA_LAUNCH_STRESS(num_blocks, block_size, kernel_loops, d_sink, inject_error, init_pattern);
            CHECK(hipDeviceSynchronize());
            ops_performed += (size_t)num_blocks * warps_per_block * kernel_loops * (size_t)mma_groups * (size_t)mma_reuse * (size_t)mma_acc * flops_per_wmma;
        }
        
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
        float* verify_actual = d_sink;
        float* verify_expected = d_golden_sink;
        size_t verify_count = total_threads;
#if PANTHEON_VENDOR_GEMM
        if (use_gemm) {
            if (inject_error) LAUNCH_KERNEL(pantheon_gemm_flip_bit, 1, 1, (unsigned int*)gemm.c);
            verify_actual = (float*)gemm.c;
            verify_expected = (float*)d_golden_gemm;
            verify_count = gemm.out_words();
            verify_blocks = (int)((verify_count + 255) / 256);
        }
#endif
        LAUNCH_KERNEL(verify_mma_kernel, verify_blocks, 256, verify_actual, verify_expected, verify_count, d_err_count);
        CHECK(hipDeviceSynchronize());
        
        unsigned int h_err_count = 0;
        CHECK(hipMemcpy(&h_err_count, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));
        
        CHECK(hipFree(d_err_count));
        CHECK(hipFree(d_golden_sink));
        if (d_golden_gemm) CHECK(hipFree(d_golden_gemm));

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

#if PANTHEON_VENDOR_GEMM
    gemm.release();
#endif
    CHECK(hipFree(d_sink));
    return 0;
#endif
}
