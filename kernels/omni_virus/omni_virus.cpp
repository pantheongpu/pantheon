#include "../common/common.h"
#include "../common/fp16_shim.h"
#include "../common/toggle_chaos.h"
#include "../common/vendor_gemm.h"
#include <chrono>
#include <string>
#include <iostream>

// --- CROSS-PLATFORM WMMA SHIM ---
// The tensor pipes are where a modern datacenter part spends its power budget:
// on Hopper and Blackwell the matrix cores are worth more than an order of
// magnitude more FLOPS than the vector pipes, and they draw accordingly. A
// mixed-workload furnace that never issues an MMA leaves the largest power
// consumer on the die idle, which caps the whole test around half of TDP.
#ifdef __CUDACC__
    #include <mma.h>
    using namespace nvcuda;
    #define OMNI_WMMA 1
#elif defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    #if __has_include(<rocwmma/rocwmma.hpp>)
        #include <rocwmma/rocwmma.hpp>
        namespace wmma = rocwmma;
        #define OMNI_WMMA 1
    #else
        #define OMNI_WMMA 0
        #pragma message("rocWMMA header not found. omni_virus builds without its tensor stream.")
    #endif
#else
    #define OMNI_WMMA 0
#endif

// --- INDEPENDENT CHAIN WIDTH ---
// One FMA chain per thread cannot saturate an FMA pipe: every instruction
// consumes the previous result, so a thread issues at most one FMA per FMA
// latency (four cycles on every current part). Eight independent chains per
// thread keep the pipe issuing every cycle at the occupancy this test runs at.
#define OMNI_ILP      8
#define OMNI_SFU_ILP  4

// Inner unroll counts. These only set the work per outer iteration; the outer
// counts themselves are calibrated at runtime (see omni_calibrate).
#define OMNI_FP_INNER  32
#define OMNI_SFU_INNER  8

// The bounded-chaos state and the SFU chain live in common/toggle_chaos.h,
// which documents why a saturating or converging recurrence keeps the pipes
// issuing while the datapath stops toggling.

// --- STREAM 1: MEMORY FURNACE ---
__global__ void mem_stream(uint4* data, size_t n, size_t base, int loops, int chunk, int inject_error) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    uint4 pA = make_uint4(0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA);
    uint4 pB = make_uint4(0x55555555, 0x55555555, 0x55555555, 0x55555555);

    // --- DYNAMIC FAULT INJECTION ---
    // Both patterns, because which one this thread stores depends on the
    // parity of its addresses. Corrupting only pA left the fault unwritten
    // for every odd-indexed thread.
    if (inject_error && idx == 1337) {
        pA.x ^= 0xBADBEEF;
        pB.x ^= 0xBADBEEF;
    }

    // A dense, coalesced walk over a window that moves with each launch,
    // rather than a sweep of the whole buffer.
    //
    // Sweeping everything ties this launch's duration to how much VRAM was
    // allocated and to how few blocks this stream holds -- and the two
    // multiply. One block asked to cover 160 GB runs for seconds while the
    // other four streams sit behind the device sync waiting for it, which is
    // exactly the idle-pipe problem the per-stream calibration exists to
    // prevent. A window of `loops * gridThreads * chunk` words depends on
    // neither, so the calibration can actually hit its target.
    //
    // The window advances by its own length every launch, so the buffer is
    // still covered over a run; verification accepts any word that is one of
    // the two patterns or still at its initialised value.
    if (n == 0) return;
    size_t span = stride * (size_t)chunk;   // words touched per loop
    if (span == 0 || span > n) span = n;

    for(int l = 0; l < loops; ++l) {
        // block_size is forced to a multiple of the warp size, so the grid
        // stride is even and every address a thread touches shares one
        // parity: pick the pattern once per loop instead of selecting it per
        // store. Flipping it with the loop index also drives every address
        // rail to rail between passes, on top of the spatial alternation
        // between neighbouring threads.
        uint4 pat = ((idx ^ (size_t)l) & 1) ? pB : pA;
        size_t start = (base + (size_t)l * span) % n;

        #pragma unroll 4
        for (size_t k = idx; k < span; k += stride) {
            // start < n and k < span <= n, so one conditional subtract wraps
            // the index. A modulo here would cost more than the store.
            size_t off = start + k;
            if (off >= n) off -= n;
            store_nt(&data[off], pat);
        }
    }
}

// --- STREAM 2: TENSOR FURNACE (WMMA) ---
#if OMNI_WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Operand tiles rotated through the fragments, and NA x NB accumulators per
// warp. The accumulator count is what keeps the tensor pipe issuing:
// mma_sync into a single accumulator serialises on its own result, exactly
// like a single FMA chain does on the vector pipe. How many it takes to fill
// the pipe is a property of the part, and more accumulators cost registers
// and therefore occupancy, so the count is a knob (--mma_acc) rather than a
// guess baked into the source.
#define OMNI_MMA_TILES   4
#define OMNI_TILE_ELEMS  (WMMA_M * WMMA_K)

template <int NA>
__global__ void mma_stream(int iters, int groups, int reuse, float* sink, int inject_error, int init_pattern) {
    const int NB = 2;
    const int ACC = NA * NB;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    // Uncompressible operand tiles held in shared memory. Constant fragments
    // -- what fill_fragment alone gives you -- let the multiplier array
    // recompute identical products every cycle, and a multiplier that never
    // toggles draws a fraction of what a real GEMM draws. Reloading the
    // fragments also puts the LSU and the shared-memory banks under the
    // traffic a GEMM's operand staging generates.
    __shared__ __half tiles[OMNI_MMA_TILES * OMNI_TILE_ELEMS];
    for (int i = threadIdx.x; i < OMNI_MMA_TILES * OMNI_TILE_ELEMS; i += blockDim.x) {
        unsigned int h = pantheon_operand_hash(
            (unsigned int)i + (init_pattern == 1 ? 0x9e3779b9u : 0u));
        tiles[i] = __float2half(PANTHEON_OPERAND_VALUE(h));
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a[NA];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b[NB];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c[ACC];

    #pragma unroll
    for (int k = 0; k < ACC; ++k) wmma::fill_fragment(c[k], 0.0f);

    int phase = 0;
    for (int i = 0; i < iters; ++i) {
        // Operand staging, and how much MMA work each staging feeds.
        //
        // Raising `reuse` issues more MMAs per shared-memory load, which is
        // what a throughput-tuned GEMM wants. It is not obviously what a
        // furnace wants: the loads themselves burn power in the LSU and the
        // shared-memory banks, so trading them away can cost more watts than
        // the freed issue slots gain. Measured on a B200, reuse=1 with
        // groups=4 drew more than the higher-reuse shapes, hence the
        // defaults -- but the balance is a property of the part, so both
        // directions stay reachable.
        #pragma unroll
        for (int m = 1; m < NA; ++m) {
            wmma::load_matrix_sync(
                a[m], tiles + ((phase + m + 1) & (OMNI_MMA_TILES - 1)) * OMNI_TILE_ELEMS, WMMA_K);
        }
        wmma::load_matrix_sync(
            b[0], tiles + ((phase + NA + 1) & (OMNI_MMA_TILES - 1)) * OMNI_TILE_ELEMS, WMMA_N);

        #pragma unroll 4
        for (int g = 0; g < groups; ++g) {
            wmma::load_matrix_sync(
                a[0], tiles + ((phase + g + 0) & (OMNI_MMA_TILES - 1)) * OMNI_TILE_ELEMS, WMMA_K);
            wmma::load_matrix_sync(
                b[1], tiles + ((phase + g + 1) & (OMNI_MMA_TILES - 1)) * OMNI_TILE_ELEMS, WMMA_N);

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
        phase = (phase + 1) & (OMNI_MMA_TILES - 1);

        // --- DYNAMIC FAULT INJECTION ---
        // On the final iteration: an injected value has to survive to the
        // sink, and further MMAs would bury it under the accumulation.
        if (inject_error && tid == 1337 && i == iters - 1) {
            c[0].x[0] += 1000.0f;
        }
    }

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < ACC; ++k) acc += c[k].x[0];
    sink[tid] = acc;
}

// --mma_acc selects the instantiation. Kept as one macro so the calibration,
// golden, warmup and stress launches cannot drift apart.
#define OMNI_LAUNCH_MMA(blocks, bs, stream, iters, sink, inj, pat)                                          \
    do {                                                                                                     \
        if (mma_acc == 2) {                                                                                  \
            LAUNCH_KERNEL_ASYNC(mma_stream<1>, blocks, bs, 0, stream, iters, mma_groups, mma_reuse, sink, inj, pat);    \
        } else if (mma_acc == 8) {                                                                           \
            LAUNCH_KERNEL_ASYNC(mma_stream<4>, blocks, bs, 0, stream, iters, mma_groups, mma_reuse, sink, inj, pat);    \
        } else if (mma_acc == 16) {                                                                          \
            LAUNCH_KERNEL_ASYNC(mma_stream<8>, blocks, bs, 0, stream, iters, mma_groups, mma_reuse, sink, inj, pat);    \
        } else {                                                                                             \
            LAUNCH_KERNEL_ASYNC(mma_stream<2>, blocks, bs, 0, stream, iters, mma_groups, mma_reuse, sink, inj, pat);    \
        }                                                                                                    \
    } while (0)
#endif

// --- STREAM 3: TENSOR-ADJACENT VECTOR FURNACE (FP16) ---
__global__ void compute_fp16(int iters, float* sink, int inject_error) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    __half2 x[OMNI_ILP], c[OMNI_ILP];
    #pragma unroll
    for (int k = 0; k < OMNI_ILP; ++k) {
        x[k] = make_half2_universal(PANTHEON_CHAOS_SEED(k));
        c[k] = make_half2_universal(PANTHEON_CHAOS_CONST(k));
    }

    for(int i = 0; i < iters; ++i) {
        #pragma unroll 32
        for(int j = 0; j < OMNI_FP_INNER; ++j) {
            #pragma unroll
            for (int k = 0; k < OMNI_ILP; ++k) {
                x[k] = __hfma2(x[k], x[k], c[k]);
            }
        }

        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && tid == 1337 && i == iters - 1) {
            x[0] = make_half2_universal(9999.0f);
        }
    }

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < OMNI_ILP; ++k) acc += __half22float2(x[k]).x;
    sink[tid] = acc;
}

// --- STREAM 4: VECTOR FURNACE (FP32 PURE) ---
__global__ void compute_fp32(int iters, float* sink, int inject_error) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    float x[OMNI_ILP], c[OMNI_ILP];
    #pragma unroll
    for (int k = 0; k < OMNI_ILP; ++k) {
        x[k] = PANTHEON_CHAOS_SEED(k);
        c[k] = PANTHEON_CHAOS_CONST(k);
    }

    for(int i = 0; i < iters; ++i) {
        #pragma unroll 32
        for(int j = 0; j < OMNI_FP_INNER; ++j) {
            #pragma unroll
            for (int k = 0; k < OMNI_ILP; ++k) {
            #ifdef __HIP_PLATFORM_AMD__
                // Explicitly force 32-bit FMA to keep heat in registers
                x[k] = __builtin_fmaf(x[k], x[k], c[k]);
            #else
                x[k] = fmaf(x[k], x[k], c[k]);
            #endif
            }
        }

        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && tid == 1337 && i == iters - 1) {
            x[0] += 9999.0f;
        }
    }

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < OMNI_ILP; ++k) acc += x[k];
    sink[tid] = acc;
}

// --- STREAM 5: SFU FURNACE (TRANSCENDENTAL PURE) ---
__global__ void compute_sfu(int iters, float* sink, int inject_error) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    float a[OMNI_SFU_ILP], b[OMNI_SFU_ILP];
    #pragma unroll
    for (int k = 0; k < OMNI_SFU_ILP; ++k) {
        a[k] = PANTHEON_CHAOS_SFU_SEED_A(k);
        b[k] = PANTHEON_CHAOS_SFU_SEED_B(k);
    }

    for(int i = 0; i < iters; ++i) {
        #pragma unroll 8
        for(int j = 0; j < OMNI_SFU_INNER; ++j) {
            #pragma unroll
            for (int k = 0; k < OMNI_SFU_ILP; ++k) {
                PANTHEON_CHAOS_SFU_STEP(a[k], b[k]);
            }
        }

        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && tid == 1337 && i == iters - 1) {
            b[0] += 9999.0f;
        }
    }

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < OMNI_SFU_ILP; ++k) acc += b[k];
    sink[tid] = acc;
}

// --- VERIFICATION KERNELS ---
__global__ void verify_mem_stream(uint4* data, size_t n, unsigned int* err_count, int init_pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    uint32_t expected_unwritten = (init_pattern == 1) ? 0xFFFFFFFF : 0x00000000;

    for(size_t i = idx; i < n; i += stride) {
        uint4 v = load_nt(&data[i]);

        // Ensure memory is either unwritten, purely Pattern A, or purely Pattern B
        if (!(v.x == 0xAAAAAAAA && v.y == 0xAAAAAAAA && v.z == 0xAAAAAAAA && v.w == 0xAAAAAAAA) &&
            !(v.x == 0x55555555 && v.y == 0x55555555 && v.z == 0x55555555 && v.w == 0x55555555) &&
            !(v.x == expected_unwritten && v.y == expected_unwritten && v.z == expected_unwritten && v.w == expected_unwritten)) {

            printf("[SDC FAULT][OMNI_VIRUS] Memory Stream Error! Index: %llu | Act: {%x, %x, %x, %x}\n",
                   (unsigned long long)i, v.x, v.y, v.z, v.w);
            atomicAdd(err_count, 1);
        }
    }
}

__global__ void verify_compute_stream(int stream_id, float* sink, float* golden, size_t n, unsigned int* err_count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        unsigned int act = pantheon_bit_cast<unsigned int>(sink[tid]);
        unsigned int exp = pantheon_bit_cast<unsigned int>(golden[tid]);

        if (act != exp) {
            unsigned int xor_bits = act ^ exp;
            if (stream_id == 1) printf("[SDC FAULT][OMNI_VIRUS] FP16 Stream Error! TID: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n", (unsigned long long)tid, exp, act, xor_bits);
            else if (stream_id == 2) printf("[SDC FAULT][OMNI_VIRUS] FP32 Stream Error! TID: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n", (unsigned long long)tid, exp, act, xor_bits);
            else if (stream_id == 3) printf("[SDC FAULT][OMNI_VIRUS] SFU Stream Error! TID: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n", (unsigned long long)tid, exp, act, xor_bits);
            else if (stream_id == 5) printf("[SDC FAULT][OMNI_VIRUS] GEMM Stream Error! Word: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n", (unsigned long long)tid, exp, act, xor_bits);
            else if (stream_id == 4) printf("[SDC FAULT][OMNI_VIRUS] Tensor Stream Error! TID: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n", (unsigned long long)tid, exp, act, xor_bits);
            atomicAdd(err_count, 1);
        }
    }
}

// Scale a stream's outer loop count so that one launch lands near target_s.
//
// Fixed loop counts do not survive contact with real hardware: on a large
// card the memory sweep finishes in milliseconds where the tensor stream
// takes seconds, and each stream's grid is a fixed slice of the machine. A
// stream that finishes early therefore parks its share of the SMs until the
// next device sync, so most of every launch ran with most of the pipes idle.
// Matching the launch times keeps all five furnaces lit for the whole run.
template <typename Fn>
static int omni_calibrate(const char* name, double target_s, int probe, Fn launch) {
    double dt = 0.0;
    for (int attempt = 0; attempt < 5; ++attempt) {
        auto t0 = std::chrono::high_resolution_clock::now();
        launch(probe);
        CHECK(hipDeviceSynchronize());
        dt = std::chrono::duration<double>(
                 std::chrono::high_resolution_clock::now() - t0).count();
        // Below a couple of milliseconds the measurement is mostly launch
        // overhead, and scaling from it lands nowhere near the target.
        if (dt >= 0.002 || probe >= PANTHEON_MAX_KERNEL_LOOPS) break;
        long long next = (long long)probe * 16;
        probe = (int)(next > PANTHEON_MAX_KERNEL_LOOPS ? PANTHEON_MAX_KERNEL_LOOPS : next);
    }

    if (dt <= 0.0) dt = 1e-6;
    double scaled = (double)probe * (target_s / dt);
    if (scaled < 1.0) scaled = 1.0;
    if (scaled > (double)PANTHEON_MAX_KERNEL_LOOPS) scaled = (double)PANTHEON_MAX_KERNEL_LOOPS;

    int loops = (int)scaled;
    std::cout << "  -> " << name << " loops: " << loops
              << " (probe " << probe << " took " << (dt * 1e3) << " ms)" << std::endl;
    return loops;
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);
    int mem_pct = atoi(argv[3]);

    // --- PANTHEON CONFIG KNOBS ---
    int block_size = 256;
    int grid_size = 0;         // 0 = auto-calculate
    int kernel_loops = 5000;   // Fallback loop scale when calibration is off
    int warmup_iters = 5;
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 0;      // 0=Zeroes, 1=Ones
    int mma_pct = 90;          // Share of the grid given to the tensor stream
    int mma_acc = 16;          // Accumulators per warp in the tensor stream
    int mma_groups = 4;        // Operand reloads per iteration
    int mma_reuse = 1;         // MMA passes per operand reload
    int launch_ms = 200;       // Per-launch target; 0 = use kernel_loops
    PantheonGemmOptions gemm_opt;   // vendor GEMM tensor stream, see vendor_gemm.h

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
        if (std::string(argv[i]) == "--mma_pct" && i+1 < argc) mma_pct = atoi(argv[++i]);
        if (std::string(argv[i]) == "--mma_acc" && i+1 < argc) mma_acc = atoi(argv[++i]);
        if (std::string(argv[i]) == "--mma_groups" && i+1 < argc) mma_groups = atoi(argv[++i]);
        if (std::string(argv[i]) == "--mma_reuse" && i+1 < argc) mma_reuse = atoi(argv[++i]);
        if (std::string(argv[i]) == "--launch_ms" && i+1 < argc) launch_ms = atoi(argv[++i]);
        if (gemm_opt.parse(argc, argv, &i)) continue;
        if (std::string(argv[i]) == "--init_pattern" && i+1 < argc) {
            if (!pantheon_parse_init_pattern(argv[++i], &init_pattern)) {
                std::cerr << "[PANTHEON] Unknown --init_pattern '" << argv[i] << "'." << std::endl;
                pantheon_print_init_patterns(std::cerr);
                return 1;
            }
        }
    }

    normalize_kernel_launch_config(block_size, grid_size, kernel_loops, warmup_iters, sync_mode);

    if (mma_pct < 0 || mma_pct > 100) {
        std::cerr << "[PANTHEON] Warning: mma_pct " << mma_pct
                  << " out of range; clamping to [0, 100]." << std::endl;
        mma_pct = mma_pct < 0 ? 0 : 100;
    }
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
    if (launch_ms < 0) {
        std::cerr << "[PANTHEON] Warning: invalid launch_ms " << launch_ms
                  << "; using 200." << std::endl;
        launch_ms = 200;
    }
#if !OMNI_WMMA
    if (mma_pct > 0) {
        std::cout << "[PANTHEON] Matrix cores or WMMA headers unavailable; "
                  << "running without the tensor stream." << std::endl;
        mma_pct = 0;
    }
#endif

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

    // WMMA issues at warp granularity -- a wavefront of 64 on CDNA -- so a
    // block that is not a whole number of warps leaves a ragged tail that
    // never issues an MMA. The memory stream also relies on an even grid
    // stride to keep each thread's addresses at one parity.
    if (block_size % prop.warpSize != 0) {
        int rounded = ((block_size + prop.warpSize - 1) / prop.warpSize) * prop.warpSize;
        if (rounded > 1024) rounded = (1024 / prop.warpSize) * prop.warpSize;
        std::cerr << "[PANTHEON] Warning: block_size " << block_size
                  << " is not a multiple of the warp size (" << prop.warpSize
                  << "); using " << rounded << "." << std::endl;
        block_size = rounded;
    }

    // The vendor GEMM takes its operands first, so the memory stream's share
    // is computed from what is left instead of colliding with them.
    bool use_gemm = false;
    int gemm_loops = 0;
    void* d_gold_gemm = nullptr;
#if PANTHEON_VENDOR_GEMM
    PantheonGemm gemm;
    if (mma_pct > 0) {
        int status = pantheon_gemm_setup(gemm, gemm_opt, gpu_id);
        if (status < 0) return 1;
        use_gemm = (status == 1);
    }
#else
    if (gemm_opt.enabled && mma_pct > 0) {
        std::cout << "[PANTHEON] Vendor GEMM not built for this platform; "
                  << "using the portable WMMA tensor stream." << std::endl;
    }
#endif

    size_t free, total; CHECK(hipMemGetInfo(&free, &total));
    if (mem_pct > 99) mem_pct = 99;
    size_t alloc_size = (free * mem_pct) / 100;
    size_t num_elements = alloc_size / 16;

    uint4* d_data;
    CHECK(hipMalloc(&d_data, alloc_size));

    // Initialize Memory Pattern
    if (init_pattern == 1) {
        CHECK(hipMemset(d_data, 0xFF, alloc_size));
    } else {
        CHECK(hipMemset(d_data, 0x00, alloc_size));
    }

    hipStream_t stream_mem, stream_mma, stream_fp16, stream_fp32, stream_sfu;
    CHECK(hipStreamCreate(&stream_mem));
    CHECK(hipStreamCreate(&stream_mma));
    CHECK(hipStreamCreate(&stream_fp16));
    CHECK(hipStreamCreate(&stream_fp32));
    CHECK(hipStreamCreate(&stream_sfu));

    int sms = prop.multiProcessorCount;

    // --- 2. EXPLICIT OCCUPANCY ---
    int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
    if (max_blocks_per_sm < 1) max_blocks_per_sm = 1;
    int total_blocks = grid_size;
    bool auto_grid = false;

    if (total_blocks == 0) {
        total_blocks = sms * max_blocks_per_sm;
        auto_grid = true;
    }

    // Weighted, not equal. An even five-way split spends most of the machine
    // on pipes that cost a fraction of the power the matrix cores do; the
    // tensor stream is what closes the gap to TDP, so it gets the largest
    // share and the remaining pipes ride along to add the datapaths a pure
    // GEMM leaves cold.
    int blocks_mma = 0;
    if (mma_pct > 0) {
        blocks_mma = (int)(((long long)total_blocks * mma_pct) / 100);
        if (blocks_mma < 1) blocks_mma = 1;
    }

    // mma_pct 100 is the shape gpu-fryer runs: effectively nothing competes
    // with the matrix cores for an SM slot or a register. Worth having as a
    // reference point, because mixing pipes only wins while the part still
    // has power headroom. The four companion streams keep one block each --
    // 0.1% of the grid, thermally nothing -- so that every code path,
    // verification included, stays alive instead of needing a second set of
    // launch, allocation and verify branches that nothing would exercise.
    bool pure_tensor = (mma_pct == 100 && total_blocks > 4);
    int blocks_mem, blocks_fp16, blocks_fp32, blocks_sfu;

    if (pure_tensor) {
        blocks_mma = total_blocks - 4;
        blocks_mem = blocks_fp16 = blocks_fp32 = blocks_sfu = 1;
    } else {
        int rest = total_blocks - blocks_mma;
        if (rest < 4) {
            // The four vector/memory streams need a block each before the
            // tensor stream takes the rest.
            blocks_mma = total_blocks > 4 ? total_blocks - 4 : 0;
            rest = total_blocks - blocks_mma;
        }
        blocks_mem  = rest / 4 < 1 ? 1 : rest / 4;
        blocks_fp16 = rest / 4 < 1 ? 1 : rest / 4;
        blocks_sfu  = rest / 4 < 1 ? 1 : rest / 4;
        blocks_fp32 = rest - blocks_mem - blocks_fp16 - blocks_sfu;
        if (blocks_fp32 < 1) blocks_fp32 = 1;
    }

    // The vendor GEMM fills the machine by itself and cannot be confined to a
    // block count, so it replaces the WMMA stream instead of sharing the
    // tensor share with it. mma_pct then only decides how much of the grid
    // the four companion streams get to fill around it.
    if (use_gemm) blocks_mma = 0;

    // Words each memory-stream thread writes per loop. Bounded so that one
    // loop cannot exceed the buffer, which keeps the wrap in the kernel to a
    // single conditional subtract.
    size_t mem_threads = (size_t)blocks_mem * block_size;
    int mem_chunk = 64;
    if (mem_threads > 0 && (size_t)mem_chunk * mem_threads > num_elements) {
        size_t fit = num_elements / mem_threads;
        mem_chunk = (int)(fit < 1 ? 1 : fit);
    }
    size_t mem_span = mem_threads * (size_t)mem_chunk;
    if (mem_span == 0 || mem_span > num_elements) mem_span = num_elements;

    size_t mma_threads  = (size_t)blocks_mma * block_size;
    size_t fp16_threads = (size_t)blocks_fp16 * block_size;
    size_t fp32_threads = (size_t)blocks_fp32 * block_size;
    size_t sfu_threads  = (size_t)blocks_sfu * block_size;

    // Allocate isolated sinks to prevent data races across streams
    float *d_sink_mma = nullptr, *d_sink_fp16, *d_sink_fp32, *d_sink_sfu;
    if (mma_threads) CHECK(hipMalloc(&d_sink_mma, mma_threads * sizeof(float)));
    CHECK(hipMalloc(&d_sink_fp16, fp16_threads * sizeof(float)));
    CHECK(hipMalloc(&d_sink_fp32, fp32_threads * sizeof(float)));
    CHECK(hipMalloc(&d_sink_sfu,  sfu_threads * sizeof(float)));

    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running OMNI VIRUS (Multi-Stream Pipeline Saturator)..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << total_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Grid Split:    tensor " << blocks_mma << " | mem " << blocks_mem
              << " | fp16 " << blocks_fp16 << " | fp32 " << blocks_fp32
              << " | sfu " << blocks_sfu << std::endl;
    if (use_gemm) {
#if PANTHEON_VENDOR_GEMM
        std::cout << "  -> Tensor Engine: vendor GEMM (" << PantheonGemm::type_name(gemm.type) << " "
                  << gemm.m << "x" << gemm.n << "x" << gemm.k << ", FP32 accumulate)" << std::endl;
#endif
    } else {
        std::cout << "  -> Tensor Engine: portable WMMA" << std::endl;
    }
    std::cout << "  -> MMA Share:     " << mma_pct << "%"
              << (pure_tensor ? " (pure tensor)" : "") << std::endl;
    std::cout << "  -> MMA Accums:    " << mma_acc << " per warp" << std::endl;
    {
        long long mma_per_iter = (long long)mma_groups * mma_reuse * mma_acc;
        long long loads_per_iter = (mma_acc / 2) + (long long)mma_groups * 2;
        std::cout << "  -> MMA Staging:   " << mma_groups << " reloads x " << mma_reuse
                  << " passes (" << (mma_per_iter / loads_per_iter)
                  << " MMA per load)" << std::endl;
    }
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- 3. LOOP COUNTS ---
    int mma_loops = 0, fp16_loops = 0, fp32_loops = 0, sfu_loops = 0, mem_loops = 0;

    if (launch_ms > 0) {
        double target_s = (double)launch_ms / 1000.0;
        std::cout << "[PANTHEON] Calibrating stream loop counts for ~"
                  << launch_ms << " ms per launch..." << std::endl;

        mem_loops = omni_calibrate("mem   ", target_s, 1, [&](int n) {
            LAUNCH_KERNEL_ASYNC(mem_stream, blocks_mem, block_size, 0, stream_mem,
                                d_data, num_elements, (size_t)0, n, mem_chunk, inject_error);
        });
        fp16_loops = omni_calibrate("fp16  ", target_s, 64, [&](int n) {
            LAUNCH_KERNEL_ASYNC(compute_fp16, blocks_fp16, block_size, 0, stream_fp16,
                                n, d_sink_fp16, 0);
        });
        fp32_loops = omni_calibrate("fp32  ", target_s, 64, [&](int n) {
            LAUNCH_KERNEL_ASYNC(compute_fp32, blocks_fp32, block_size, 0, stream_fp32,
                                n, d_sink_fp32, 0);
        });
        sfu_loops = omni_calibrate("sfu   ", target_s, 64, [&](int n) {
            LAUNCH_KERNEL_ASYNC(compute_sfu, blocks_sfu, block_size, 0, stream_sfu,
                                n, d_sink_sfu, 0);
        });
#if PANTHEON_VENDOR_GEMM
        if (use_gemm) {
            gemm_loops = pantheon_gemm_batch(gemm, stream_mma, launch_ms);
            std::cout << "  -> gemm   loops: " << gemm_loops << std::endl;
        }
#endif
#if OMNI_WMMA
        if (blocks_mma > 0) {
            mma_loops = omni_calibrate("tensor", target_s, 16, [&](int n) {
                OMNI_LAUNCH_MMA(blocks_mma, block_size, stream_mma,
                                n, d_sink_mma, 0, init_pattern);
            });
        }
#endif
    } else {
        // Manual scaling, kept so that --kernel_loops still means something
        // for deep tuning. The divisors are rough per-pipe cost ratios and
        // will not balance the launches on every part.
        fp16_loops = kernel_loops;
        fp32_loops = kernel_loops;
        sfu_loops  = (kernel_loops / 4 > 0) ? kernel_loops / 4 : 1;
        mem_loops  = (kernel_loops / 1000 > 0) ? kernel_loops / 1000 : 1;
        mma_loops  = (kernel_loops / 16 > 0) ? kernel_loops / 16 : 1;
        gemm_loops = (kernel_loops / 500 > 0) ? kernel_loops / 500 : 1;
        std::cout << "  -> Kernel Loops:  " << kernel_loops
                  << " (calibration off; mem " << mem_loops << ", sfu " << sfu_loops
                  << ", tensor " << mma_loops << ")" << std::endl;
    }

    // Bound what a launch actually touches. The footprint is the window, not
    // the allocation: measured against the whole buffer this clamp would cut
    // the loop count by the ratio between the two and starve the stream
    // instead of protecting it.
    scale_kernel_loops_for_large_alloc(mem_loops, mem_span * 16,
                                       launch_ms > 0 ? mem_loops : 1);

    // --- GOLDEN PASS ---
    float *d_gold_mma = nullptr, *d_gold_fp16 = nullptr, *d_gold_fp32 = nullptr, *d_gold_sfu = nullptr;
    if (verify_mode) {
        std::cout << "[PANTHEON] Generating expected stream baselines (Golden Pass)..." << std::endl;
        CHECK(hipMalloc(&d_gold_fp16, fp16_threads * sizeof(float)));
        CHECK(hipMalloc(&d_gold_fp32, fp32_threads * sizeof(float)));
        CHECK(hipMalloc(&d_gold_sfu,  sfu_threads * sizeof(float)));

        // Run compute cleanly to establish baseline
        LAUNCH_KERNEL_ASYNC(compute_fp16, blocks_fp16, block_size, 0, stream_fp16, fp16_loops, d_gold_fp16, 0);
        LAUNCH_KERNEL_ASYNC(compute_fp32, blocks_fp32, block_size, 0, stream_fp32, fp32_loops, d_gold_fp32, 0);
        LAUNCH_KERNEL_ASYNC(compute_sfu,  blocks_sfu,  block_size, 0, stream_sfu,  sfu_loops,  d_gold_sfu,  0);
#if PANTHEON_VENDOR_GEMM
        if (use_gemm) {
            // One matmul is the whole reference: beta is 0, so every later
            // run must reproduce these bits exactly.
            CHECK(hipMalloc(&d_gold_gemm, gemm.out_size_bytes()));
            if (!gemm.run(stream_mma)) { std::cerr << "[PANTHEON] Vendor GEMM matmul failed." << std::endl; return 1; }
            CHECK(hipStreamSynchronize(stream_mma));
            CHECK(hipMemcpy(d_gold_gemm, gemm.c, gemm.out_size_bytes(), hipMemcpyDeviceToDevice));
        }
#endif
#if OMNI_WMMA
        if (blocks_mma > 0) {
            CHECK(hipMalloc(&d_gold_mma, mma_threads * sizeof(float)));
            OMNI_LAUNCH_MMA(blocks_mma, block_size, stream_mma, mma_loops, d_gold_mma, 0, init_pattern);
        }
#endif
        CHECK(hipDeviceSynchronize());
    }

    // Pre-calculate operations. FMA counts two flops; one 16x16x16 MMA counts
    // 2 * 16^3.
    size_t fp16_ops_per_launch = fp16_threads * (size_t)fp16_loops * OMNI_FP_INNER * OMNI_ILP * 4;
    size_t fp32_ops_per_launch = fp32_threads * (size_t)fp32_loops * OMNI_FP_INNER * OMNI_ILP * 2;
    // Twelve ops per SFU step: five transcendentals counted as one each,
    // plus the seven arithmetic operations that compose them.
    size_t sfu_ops_per_launch  = sfu_threads  * (size_t)sfu_loops  * OMNI_SFU_INNER * OMNI_SFU_ILP * 12;
    size_t mma_ops_per_launch  = 0;
#if OMNI_WMMA
    // One MMA is issued per warp (wavefront on AMD), not per thread.
    mma_ops_per_launch = (mma_threads / (size_t)prop.warpSize) * (size_t)mma_loops
                       * (size_t)mma_groups * (size_t)mma_reuse * (size_t)mma_acc * 8192;
#endif
#if PANTHEON_VENDOR_GEMM
    if (use_gemm) mma_ops_per_launch += (size_t)(gemm.flops_per_run() * (double)gemm_loops);
#endif
    size_t total_ops_per_launch = mma_ops_per_launch + fp16_ops_per_launch
                                + fp32_ops_per_launch + sfu_ops_per_launch;
    size_t mem_bytes_per_launch = mem_span * 16 * (size_t)mem_loops;

    // The window walks the buffer across launches, so a run still covers the
    // whole allocation even though no single launch does.
    size_t mem_base = 0;
    size_t mem_advance = (size_t)mem_loops * mem_span;

    // --- WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup iterations..." << std::endl;
        for (int i = 0; i < warmup_iters; i++) {
            LAUNCH_KERNEL_ASYNC(mem_stream,   blocks_mem,  block_size, 0, stream_mem,  d_data, num_elements, mem_base, mem_loops, mem_chunk, inject_error);
            LAUNCH_KERNEL_ASYNC(compute_fp16, blocks_fp16, block_size, 0, stream_fp16, fp16_loops, d_sink_fp16, inject_error);
            LAUNCH_KERNEL_ASYNC(compute_fp32, blocks_fp32, block_size, 0, stream_fp32, fp32_loops, d_sink_fp32, inject_error);
            LAUNCH_KERNEL_ASYNC(compute_sfu,  blocks_sfu,  block_size, 0, stream_sfu,  sfu_loops,  d_sink_sfu,  inject_error);
#if PANTHEON_VENDOR_GEMM
            if (use_gemm) {
                for (int r = 0; r < gemm_loops; ++r) gemm.run(stream_mma);
            }
#endif
#if OMNI_WMMA
            if (blocks_mma > 0) {
                OMNI_LAUNCH_MMA(blocks_mma, block_size, stream_mma, mma_loops, d_sink_mma, inject_error, init_pattern);
            }
#endif
        }
        CHECK(hipDeviceSynchronize());
    }

    // Re-initialize the memory buffer to a clean state. Calibration and
    // warmup both wrote patterns into it.
    if (init_pattern == 1) {
        CHECK(hipMemset(d_data, 0xFF, alloc_size));
    } else {
        CHECK(hipMemset(d_data, 0x00, alloc_size));
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t ops_performed = 0;
    size_t bytes_written = 0;

    // --- MAIN STRESS LOOP ---
    while(true) {
        LAUNCH_KERNEL_ASYNC(mem_stream,   blocks_mem,  block_size, 0, stream_mem,  d_data, num_elements, mem_base, mem_loops, mem_chunk, inject_error);
        LAUNCH_KERNEL_ASYNC(compute_fp16, blocks_fp16, block_size, 0, stream_fp16, fp16_loops, d_sink_fp16, inject_error);
        LAUNCH_KERNEL_ASYNC(compute_fp32, blocks_fp32, block_size, 0, stream_fp32, fp32_loops, d_sink_fp32, inject_error);
        LAUNCH_KERNEL_ASYNC(compute_sfu,  blocks_sfu,  block_size, 0, stream_sfu,  sfu_loops,  d_sink_sfu,  inject_error);
#if PANTHEON_VENDOR_GEMM
        if (use_gemm) {
            for (int r = 0; r < gemm_loops; ++r) {
                if (!gemm.run(stream_mma)) { std::cerr << "[PANTHEON] Vendor GEMM matmul failed." << std::endl; return 1; }
            }
        }
#endif
#if OMNI_WMMA
        if (blocks_mma > 0) {
            OMNI_LAUNCH_MMA(blocks_mma, block_size, stream_mma, mma_loops, d_sink_mma, inject_error, init_pattern);
        }
#endif

        CHECK(hipDeviceSynchronize());
        ops_performed += total_ops_per_launch;
        bytes_written += mem_bytes_per_launch;
        if (num_elements) mem_base = (mem_base + mem_advance) % num_elements;

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (ops_performed / 1e12) / seconds << " TFLOPS" << std::endl;
    std::cout << "  -> Memory Stream: " << (bytes_written / 1e9) / seconds << " GB/s (writes)" << std::endl;

    // --- VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running Multi-Stream Cross-Verification Pass..." << std::endl;

        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));

        // Verify Memory
        int mem_verify_grid = init_launch_grid_size(prop, num_elements, 256);
        LAUNCH_KERNEL_ASYNC(verify_mem_stream, mem_verify_grid, 256, 0, stream_mem, d_data, num_elements, d_err_count, init_pattern);

        // Verify FP16
        LAUNCH_KERNEL_ASYNC(verify_compute_stream, blocks_fp16, block_size, 0, stream_fp16, 1, d_sink_fp16, d_gold_fp16, fp16_threads, d_err_count);

        // Verify FP32
        LAUNCH_KERNEL_ASYNC(verify_compute_stream, blocks_fp32, block_size, 0, stream_fp32, 2, d_sink_fp32, d_gold_fp32, fp32_threads, d_err_count);

        // Verify SFU
        LAUNCH_KERNEL_ASYNC(verify_compute_stream, blocks_sfu, block_size, 0, stream_sfu, 3, d_sink_sfu, d_gold_sfu, sfu_threads, d_err_count);

#if PANTHEON_VENDOR_GEMM
        // Verify vendor GEMM: the last stress output against the reference.
        if (d_gold_gemm) {
            if (inject_error) LAUNCH_KERNEL_ASYNC(pantheon_gemm_flip_bit, 1, 1, 0, stream_mma, (unsigned int*)gemm.c);
            int gemm_grid = (int)((gemm.out_words() + 255) / 256);
            LAUNCH_KERNEL_ASYNC(verify_compute_stream, gemm_grid, 256, 0, stream_mma, 5,
                                (float*)gemm.c, (float*)d_gold_gemm, gemm.out_words(), d_err_count);
        }
#endif

        // Verify Tensor
        if (d_gold_mma) {
            LAUNCH_KERNEL_ASYNC(verify_compute_stream, blocks_mma, block_size, 0, stream_mma, 4, d_sink_mma, d_gold_mma, mma_threads, d_err_count);
        }

        CHECK(hipDeviceSynchronize());

        unsigned int h_err_count = 0;
        CHECK(hipMemcpy(&h_err_count, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));

        CHECK(hipFree(d_err_count));
        if (d_gold_mma) CHECK(hipFree(d_gold_mma));
        if (d_gold_gemm) CHECK(hipFree(d_gold_gemm));
        CHECK(hipFree(d_gold_fp16));
        CHECK(hipFree(d_gold_fp32));
        CHECK(hipFree(d_gold_sfu));

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
    CHECK(hipStreamDestroy(stream_mem));
    CHECK(hipStreamDestroy(stream_mma));
    CHECK(hipStreamDestroy(stream_fp16));
    CHECK(hipStreamDestroy(stream_fp32));
    CHECK(hipStreamDestroy(stream_sfu));
    CHECK(hipFree(d_data));
    if (d_sink_mma) CHECK(hipFree(d_sink_mma));
    CHECK(hipFree(d_sink_fp16));
    CHECK(hipFree(d_sink_fp32));
    CHECK(hipFree(d_sink_sfu));

    return 0;
}
