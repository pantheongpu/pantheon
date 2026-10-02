#ifndef PANTHEON_VENDOR_GEMM_H
#define PANTHEON_VENDOR_GEMM_H

// Optional vendor-library GEMM for the tests that want the matrix cores at
// the rate the hardware's own instructions reach.
//
// Portable WMMA lowers to the synchronous MMA instructions, which measured
// about a third of the peak tensor rate on a B200. The architecture's
// asynchronous matrix path (WGMMA, the Blackwell tensor-core instructions) is
// reachable through the vendor BLAS without writing any architecture-specific
// code here, and a BLAS picks the right one for whatever part it finds. That
// is the same reason gpu-fryer reaches the power limit with plain matmuls.
//
// Two backends sit behind one code path: cuBLASLt on NVIDIA and hipBLASLt on
// AMD. The two APIs have the same shape and differ in names and a few enums,
// which the PG() macros and pg_* aliases below absorb.
//
// The library is loaded at run time rather than linked. Every binary in this
// tree is built with one link line, and a hard dependency on the BLAS would
// keep every other test from starting on a machine whose loader cannot find
// it. With dlopen, a missing library is a notice and a fallback to WMMA, not
// a failed launch.

#include "common.h"
#include "toggle_chaos.h"
#include <string>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Options. Defined for every build so the flags parse everywhere, even where
// no backend exists and the tests keep their portable path.
// ---------------------------------------------------------------------------
struct PantheonGemmOptions {
    int enabled = 1;                  // --gemm 0 forces the portable path
    std::string type = "auto";        // auto | bf16 | fp16 | tf32 | fp32 | fp8
    int probe_ms = 1500;              // per-format power probe for auto; 0 = skip
    int margin_pct = 15;              // auto keeps the default format unless another beats it by this much
    int m = 8192, n = 8192, k = 8192; // C is m x n, contracted over k

    // Consumes argv[*i] (and its value) if it is one of the GEMM flags.
    bool parse(int argc, char** argv, int* i) {
        std::string a = argv[*i];
        if (*i + 1 >= argc) return false;
        if (a == "--gemm")          { enabled = atoi(argv[++*i]); return true; }
        if (a == "--gemm_type")     { type = argv[++*i]; return true; }
        if (a == "--gemm_probe_ms") { probe_ms = atoi(argv[++*i]); return true; }
        if (a == "--gemm_margin")   { margin_pct = atoi(argv[++*i]); return true; }
        if (a == "--gemm_size")     { m = n = k = atoi(argv[++*i]); return true; }
        if (a == "--gemm_m")        { m = atoi(argv[++*i]); return true; }
        if (a == "--gemm_n")        { n = atoi(argv[++*i]); return true; }
        if (a == "--gemm_k")        { k = atoi(argv[++*i]); return true; }
        return false;
    }
};

// Operand formats, the way gpu-fryer offers them. Each one exercises a
// different datapath at a different rate, so the choice is a workload choice:
// bf16/fp16 are the usual training shapes, tf32 and fp32 keep the operands
// wide, and fp8 is the densest tensor-core mode on parts that have it.
enum PantheonGemmType { GEMM_BF16 = 0, GEMM_FP16, GEMM_TF32, GEMM_FP32, GEMM_FP8 };

// ---------------------------------------------------------------------------
// Backend selection.
// ---------------------------------------------------------------------------
#if !defined(PANTHEON_MOCK) && defined(__has_include)
  #if defined(__CUDACC__)
    #if __has_include(<cublasLt.h>) && __has_include(<cuda_bf16.h>) && __has_include(<cuda_fp16.h>)
      #define PANTHEON_VENDOR_GEMM 1
      #define PANTHEON_GEMM_CUDA 1
    #endif
  #elif defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    #if __has_include(<hipblaslt/hipblaslt.h>) && __has_include(<hip/hip_bf16.h>) && __has_include(<hip/hip_fp16.h>)
      #define PANTHEON_VENDOR_GEMM 1
      #define PANTHEON_GEMM_HIP 1
    #endif
  #endif
#endif
#ifndef PANTHEON_VENDOR_GEMM
  #define PANTHEON_VENDOR_GEMM 0
#endif

#if PANTHEON_VENDOR_GEMM

#include <dlfcn.h>
#include <chrono>
#include <iostream>
#include <strings.h>

#if PANTHEON_GEMM_CUDA
  #include <cublasLt.h>
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
  #if __has_include(<cuda_fp8.h>)
    #include <cuda_fp8.h>
    #define PANTHEON_GEMM_HAS_FP8 1
  #endif

  #define PG(x) cublasLt##x
  typedef __nv_bfloat16        pg_bf16;
  typedef cublasStatus_t       pg_status_t;
  typedef cublasComputeType_t  pg_compute_t;
  typedef cudaDataType         pg_data_t;
  typedef cublasOperation_t    pg_op_t;
  #define PG_OK            CUBLAS_STATUS_SUCCESS
  #define PG_COMPUTE_32F   CUBLAS_COMPUTE_32F
  #define PG_COMPUTE_TF32  CUBLAS_COMPUTE_32F_FAST_TF32
  #define PG_R_16BF        CUDA_R_16BF
  #define PG_R_16F         CUDA_R_16F
  #define PG_R_32F         CUDA_R_32F
  #define PG_R_8F          CUDA_R_8F_E4M3
  #define PG_OP_T          CUBLAS_OP_T
  #define PG_ATTR_TRANSA   CUBLASLT_MATMUL_DESC_TRANSA
  #define PG_PREF_WS       CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
  #define PANTHEON_GEMM_LIBS { "libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so.11", "libcublasLt.so" }
#else
  #include <hipblaslt/hipblaslt.h>
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>

  #define PG(x) hipblasLt##x
  typedef __hip_bfloat16       pg_bf16;
  typedef hipblasStatus_t      pg_status_t;
  typedef hipblasComputeType_t pg_compute_t;
  typedef hipDataType          pg_data_t;
  typedef hipblasOperation_t   pg_op_t;
  #define PG_OK            HIPBLAS_STATUS_SUCCESS
  #define PG_COMPUTE_32F   HIPBLAS_COMPUTE_32F
  #define PG_R_16BF        HIP_R_16BF
  #define PG_R_16F         HIP_R_16F
  #define PG_R_32F         HIP_R_32F
  #define PG_OP_T          HIPBLAS_OP_T
  #define PG_ATTR_TRANSA   HIPBLASLT_MATMUL_DESC_TRANSA
  #define PG_PREF_WS       HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
  #define PANTHEON_GEMM_LIBS { "libhipblaslt.so.1", "libhipblaslt.so.0", "libhipblaslt.so" }
  // FP8 is left out on AMD: MI300 uses the FNUZ encodings and later parts the
  // OCP ones, so one enum cannot serve both. TF32 is left out because the
  // compute-type constant is not present in every ROCm release. Both report
  // "not supported" and "auto" skips them.
#endif
#ifndef PANTHEON_GEMM_HAS_FP8
  #define PANTHEON_GEMM_HAS_FP8 0
#endif

#define PG_STR2(x) #x
#define PG_STR(x) PG_STR2(x)

#define PANTHEON_GEMM_SYMBOLS(X)  \
    X(Create)                     \
    X(Destroy)                    \
    X(MatmulDescCreate)           \
    X(MatmulDescDestroy)          \
    X(MatmulDescSetAttribute)     \
    X(MatrixLayoutCreate)         \
    X(MatrixLayoutDestroy)        \
    X(MatmulPreferenceCreate)     \
    X(MatmulPreferenceDestroy)    \
    X(MatmulPreferenceSetAttribute) \
    X(MatmulAlgoGetHeuristic)     \
    X(Matmul)

// Conversions from the shared float payload to each operand format.
__device__ __forceinline__ void pantheon_gemm_cvt(pg_bf16& o, float v) { o = __float2bfloat16(v); }
__device__ __forceinline__ void pantheon_gemm_cvt(__half& o, float v)  { o = __float2half(v); }
__device__ __forceinline__ void pantheon_gemm_cvt(float& o, float v)   { o = v; }
#if PANTHEON_GEMM_HAS_FP8
__device__ __forceinline__ void pantheon_gemm_cvt(__nv_fp8_e4m3& o, float v) { o = __nv_fp8_e4m3(v); }
#endif

// Random operands, so the multiplier arrays toggle for the same reason the
// WMMA tiles are hashed: a GEMM of constants draws a fraction of the power.
template <typename T>
__global__ void pantheon_gemm_fill(T* out, size_t count, unsigned int seed) {
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        unsigned int h = pantheon_operand_hash((unsigned int)i * 2654435761u + seed);
        pantheon_gemm_cvt(out[i], PANTHEON_OPERAND_VALUE(h));
    }
}

__global__ void pantheon_gemm_flip_bit(unsigned int* word) {
    word[0] ^= 0x00000080u;
}

struct PantheonGemm {
    void* lib = nullptr;
#define PANTHEON_GEMM_MEMBER(name) decltype(&PG(name)) p_##name = nullptr;
    PANTHEON_GEMM_SYMBOLS(PANTHEON_GEMM_MEMBER)
#undef PANTHEON_GEMM_MEMBER

    PG(Handle_t) handle = nullptr;
    PG(MatmulDesc_t) desc = nullptr;
    PG(MatrixLayout_t) la = nullptr, lb = nullptr, lc = nullptr;
    PG(MatmulPreference_t) pref = nullptr;
    PG(MatmulAlgo_t) algo;
    void *a = nullptr, *b = nullptr, *c = nullptr, *ws = nullptr;
    size_t ws_bytes = (size_t)32 << 20;
    PantheonGemmType type = GEMM_BF16;
    int m = 0, n = 0, k = 0;
    std::string why;

    static bool parse_type(const char* name, PantheonGemmType* out) {
        if (!strcasecmp(name, "bf16")) { *out = GEMM_BF16; return true; }
        if (!strcasecmp(name, "fp16")) { *out = GEMM_FP16; return true; }
        if (!strcasecmp(name, "tf32")) { *out = GEMM_TF32; return true; }
        if (!strcasecmp(name, "fp32")) { *out = GEMM_FP32; return true; }
        if (!strcasecmp(name, "fp8"))  { *out = GEMM_FP8;  return true; }
        return false;
    }
    static const char* type_name(PantheonGemmType t) {
        switch (t) {
            case GEMM_BF16: return "BF16";
            case GEMM_FP16: return "FP16";
            case GEMM_TF32: return "TF32";
            case GEMM_FP32: return "FP32";
            case GEMM_FP8:  return "FP8 (E4M3)";
        }
        return "?";
    }
    // Input element size, and output element size. FP8 inputs write BF16.
    static size_t in_bytes(PantheonGemmType t) {
        switch (t) {
            case GEMM_BF16: case GEMM_FP16: return 2;
            case GEMM_TF32: case GEMM_FP32: return 4;
            case GEMM_FP8: return 1;
        }
        return 2;
    }
    static size_t out_bytes(PantheonGemmType t) { return t == GEMM_FP8 ? 2 : in_bytes(t); }

    size_t out_size_bytes() const { return (size_t)m * (size_t)n * out_bytes(type); }
    size_t out_words() const { return out_size_bytes() / 4; }   // 32-bit words of output
    double flops_per_run() const { return 2.0 * (double)m * (double)n * (double)k; }

    bool fail(const char* what) { why = what; release(); return false; }

    bool load() {
        static const char* names[] = PANTHEON_GEMM_LIBS;
        for (const char* nm : names) {
            lib = dlopen(nm, RTLD_NOW | RTLD_LOCAL);
            if (lib) break;
        }
        if (!lib) { why = "the BLAS library was not found by the dynamic loader"; return false; }
#define PANTHEON_GEMM_LOAD(name)                                          \
        p_##name = (decltype(&PG(name)))dlsym(lib, PG_STR(PG(name)));     \
        if (!p_##name) { why = "missing symbol " PG_STR(PG(name)); return false; }
        PANTHEON_GEMM_SYMBOLS(PANTHEON_GEMM_LOAD)
#undef PANTHEON_GEMM_LOAD
        return true;
    }

    // Bytes of device memory a shape needs, so the caller can pick one that
    // leaves the memory stream its allocation.
    static size_t footprint(PantheonGemmType t, int dm, int dn, int dk) {
        return (size_t)dm * dk * in_bytes(t) + (size_t)dk * dn * in_bytes(t)
             + (size_t)dm * dn * out_bytes(t) + ((size_t)32 << 20);
    }

    template <typename T>
    void fill(void* dst, size_t count, unsigned int seed) {
        pantheon_gemm_fill<T><<<1024, 256>>>((T*)dst, count, seed);
    }

    // C[m,n] = A[m,k] * B[k,n], both operands "TN" (A transposed, so both are
    // read down their k dimension), FP32 accumulate. TN is the layout every
    // format supports, FP8 included, so one code path serves them all.
    bool init(PantheonGemmType t, int dm, int dn, int dk) {
        type = t;
        if (t == GEMM_FP8 && !PANTHEON_GEMM_HAS_FP8) return fail("FP8 is not available through this backend");
#if PANTHEON_GEMM_HIP
        if (t == GEMM_TF32) return fail("TF32 is not available through this backend");
#endif
        if (t == GEMM_FP8 && (dm % 16 || dn % 16 || dk % 16)) return fail("FP8 needs every dimension to be a multiple of 16");
        if (!load()) { release(); return false; }
        if (p_Create(&handle) != PG_OK) return fail("handle creation failed");

        size_t na = (size_t)dm * dk, nb = (size_t)dk * dn, nc = (size_t)dm * dn;
        if (hipMalloc(&a, na * in_bytes(t)) != hipSuccess ||
            hipMalloc(&b, nb * in_bytes(t)) != hipSuccess ||
            hipMalloc(&c, nc * out_bytes(t)) != hipSuccess ||
            hipMalloc(&ws, ws_bytes) != hipSuccess) {
            hipGetLastError();
            return fail("could not allocate the GEMM operands");
        }
        switch (t) {
            case GEMM_BF16: fill<pg_bf16>(a, na, 0x1234abcdu); fill<pg_bf16>(b, nb, 0x9e3779b9u); break;
            case GEMM_FP16: fill<__half>(a, na, 0x1234abcdu);  fill<__half>(b, nb, 0x9e3779b9u);  break;
            case GEMM_TF32:
            case GEMM_FP32: fill<float>(a, na, 0x1234abcdu);   fill<float>(b, nb, 0x9e3779b9u);   break;
            case GEMM_FP8:
#if PANTHEON_GEMM_HAS_FP8
                fill<__nv_fp8_e4m3>(a, na, 0x1234abcdu); fill<__nv_fp8_e4m3>(b, nb, 0x9e3779b9u);
#endif
                break;
        }
        hipMemset(c, 0, nc * out_bytes(t));
        if (hipDeviceSynchronize() != hipSuccess) return fail("operand fill failed");

        pg_data_t in_t = PG_R_16BF, out_t = PG_R_16BF;
        pg_compute_t compute = PG_COMPUTE_32F;
        switch (t) {
            case GEMM_BF16: in_t = out_t = PG_R_16BF; break;
            case GEMM_FP16: in_t = out_t = PG_R_16F;  break;
            case GEMM_FP32: in_t = out_t = PG_R_32F;  break;
            case GEMM_TF32:
#ifdef PG_COMPUTE_TF32
                in_t = out_t = PG_R_32F; compute = PG_COMPUTE_TF32;
#endif
                break;
            case GEMM_FP8:
#ifdef PG_R_8F
                in_t = PG_R_8F; out_t = PG_R_16BF;
#endif
                break;
        }

        if (p_MatmulDescCreate(&desc, compute, PG_R_32F) != PG_OK)
            return fail("descriptor creation failed");
        pg_op_t op_t = PG_OP_T;
        if (p_MatmulDescSetAttribute(desc, PG_ATTR_TRANSA, &op_t, sizeof(op_t)) != PG_OK)
            return fail("could not select the transposed layout");
        // A is stored k x m (read down k), B is k x n, C is m x n.
        if (p_MatrixLayoutCreate(&la, in_t,  dk, dm, dk) != PG_OK ||
            p_MatrixLayoutCreate(&lb, in_t,  dk, dn, dk) != PG_OK ||
            p_MatrixLayoutCreate(&lc, out_t, dm, dn, dm) != PG_OK)
            return fail("matrix layout creation failed");
        if (p_MatmulPreferenceCreate(&pref) != PG_OK)
            return fail("preference creation failed");
        if (p_MatmulPreferenceSetAttribute(pref, PG_PREF_WS, &ws_bytes, sizeof(ws_bytes)) != PG_OK)
            return fail("could not set the workspace limit");

        PG(MatmulHeuristicResult_t) result;
        int found = 0;
        if (p_MatmulAlgoGetHeuristic(handle, desc, la, lb, lc, lc, pref, 1, &result, &found)
                != PG_OK || found < 1)
            return fail("the library has no algorithm for this format and shape on this part");
        algo = result.algo;
        m = dm; n = dn; k = dk;
        return true;
    }

    // One matmul, asynchronous. beta is 0, so every run overwrites the same
    // output and a corrupted run is visible against a snapshot of the first.
    bool run(hipStream_t stream) {
        float alpha = 1.0f, beta = 0.0f;
        return p_Matmul(handle, desc, &alpha, a, la, b, lb, &beta, c, lc, c, lc,
                        &algo, ws, ws_bytes, stream) == PG_OK;
    }

    void release() {
        if (pref)   p_MatmulPreferenceDestroy(pref);
        if (la)     p_MatrixLayoutDestroy(la);
        if (lb)     p_MatrixLayoutDestroy(lb);
        if (lc)     p_MatrixLayoutDestroy(lc);
        if (desc)   p_MatmulDescDestroy(desc);
        if (handle) p_Destroy(handle);
        pref = nullptr; la = lb = lc = nullptr; desc = nullptr; handle = nullptr;
        if (a)  hipFree(a);
        if (b)  hipFree(b);
        if (c)  hipFree(c);
        if (ws) hipFree(ws);
        a = b = c = ws = nullptr;
        m = n = k = 0;
    }
};

// Board energy meter, so "auto" can choose by measured watts. Only NVML is
// wired up: it is loaded the same way the BLAS is, and the device is matched
// by PCI bus ID because NVML and CUDA do not promise the same ordinals
// (CUDA_VISIBLE_DEVICES and FASTEST_FIRST both reorder). On AMD open() fails,
// and "auto" falls back to the first format the library accepts.
struct PantheonEnergyMeter {
    void* lib = nullptr;
    void* dev = nullptr;
    int (*p_init)(void) = nullptr;
    int (*p_handle)(const char*, void**) = nullptr;
    int (*p_energy)(void*, unsigned long long*) = nullptr;

    bool open(int device) {
#if PANTHEON_GEMM_CUDA
        lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!lib) return false;
        p_init   = (int (*)(void))dlsym(lib, "nvmlInit_v2");
        p_handle = (int (*)(const char*, void**))dlsym(lib, "nvmlDeviceGetHandleByPciBusId_v2");
        p_energy = (int (*)(void*, unsigned long long*))dlsym(lib, "nvmlDeviceGetTotalEnergyConsumption");
        if (!p_init || !p_handle || !p_energy || p_init() != 0) return false;
        char bus[32];
        if (cudaDeviceGetPCIBusId(bus, sizeof(bus), device) != cudaSuccess) return false;
        if (p_handle(bus, &dev) != 0) return false;
        unsigned long long probe;
        return p_energy(dev, &probe) == 0;
#else
        (void)device;
        return false;
#endif
    }
    bool millijoules(unsigned long long* out) { return dev && p_energy(dev, out) == 0; }
};

// Shrink a shape, all three dimensions together, until the operands fit in an
// eighth of free memory, keeping every dimension a multiple of 16.
static inline void pantheon_gemm_fit(PantheonGemmType t, size_t free_bytes, int* m, int* n, int* k) {
    while ((*m > 256 || *n > 256 || *k > 256)
           && PantheonGemm::footprint(t, *m, *n, *k) > free_bytes / 8) {
        *m = (*m / 2) & ~15; *n = (*n / 2) & ~15; *k = (*k / 2) & ~15;
        if (*m < 16) *m = 16;
        if (*n < 16) *n = 16;
        if (*k < 16) *k = 16;
    }
}

// Choose the operand format for this part.
//
// The formats are tried in a fixed order, BF16, FP16, TF32, FP8, FP32, and the
// first one the part and library accept is the default. BF16 is not universal
// (it needs Ampere or later, FP8 Ada/Hopper or later), so compatibility comes
// first. Every accepted format is then run alone for probe_ms and its board
// power measured, and the default is replaced only by a format that draws at
// least margin_pct more.
//
// The margin is what makes the choice repeatable. On a B200 the accepted
// formats measured 956 to 1041 W, the same format varied by up to 9% between
// probes, and the gaps between formats were 1 to 4%. Taking the maximum of
// those readings gave a different format on different GPUs of one node, and
// an 8% margin still did. 15% is above the largest spread seen, so the same
// part resolves to the same format every time unless one format is clearly
// ahead, and then every GPU sees that lead. The cost is that formats closer
// than the margin are not told apart; on a B200 that is all of them.
//
// The probe runs the GEMM without any companion streams, so it ranks formats
// by what the tensor path alone draws, not by the final mix.
//
// Without a meter there is nothing to measure, and the default is used.
static inline bool pantheon_gemm_pick(int device, int probe_ms, int margin_pct, int m, int n, int k,
                                      PantheonGemmType* chosen) {
    static const PantheonGemmType order[] = { GEMM_BF16, GEMM_FP16, GEMM_TF32, GEMM_FP8, GEMM_FP32 };
    PantheonEnergyMeter meter;
    bool metered = probe_ms > 0 && meter.open(device);
    size_t gfree, gtotal;
    hipMemGetInfo(&gfree, &gtotal);

    bool have_default = false;
    PantheonGemmType def = GEMM_BF16;
    double def_w = 0.0, best_w = 0.0;
    PantheonGemmType best = GEMM_BF16;
    bool have_best = false;

    for (PantheonGemmType t : order) {
        int tm = m, tn = n, tk = k;
        pantheon_gemm_fit(t, gfree, &tm, &tn, &tk);
        PantheonGemm g;
        if (!g.init(t, tm, tn, tk)) {
            std::cout << "  -> Probe " << PantheonGemm::type_name(t) << ": skipped ("
                      << g.why << ")" << std::endl;
            continue;
        }
        if (!have_default) { have_default = true; def = t; }
        if (!metered) {
            g.release();
            break;                          // nothing to measure: first accepted wins
        }
        for (int r = 0; r < 4; ++r) g.run(0);
        hipStreamSynchronize(0);
        unsigned long long e0 = 0, e1 = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        meter.millijoules(&e0);
        double dt = 0.0;
        while (dt < probe_ms / 1000.0) {
            for (int r = 0; r < 8; ++r) g.run(0);
            hipStreamSynchronize(0);
            dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        }
        bool ok = meter.millijoules(&e1) && e1 > e0 && dt > 0.0;
        double watts = ok ? ((double)(e1 - e0) / 1000.0) / dt : 0.0;
        g.release();
        std::cout << "  -> Probe " << PantheonGemm::type_name(t) << ": "
                  << (ok ? std::to_string((int)watts) + " W" : std::string("no energy reading")) << std::endl;
        if (!ok) continue;
        if (t == def) def_w = watts;
        if (!have_best || watts > best_w) { best_w = watts; best = t; have_best = true; }
    }
    if (!have_default) return false;
    *chosen = def;
    if (metered && def_w > 0.0 && have_best && best != def
        && best_w > def_w * (1.0 + margin_pct / 100.0)) {
        *chosen = best;
    }
    std::cout << "  -> Selected " << PantheonGemm::type_name(*chosen)
              << (*chosen == def ? " (default order; no format ahead by " : " (ahead of the default by more than ")
              << margin_pct << "%)" << std::endl;
    return true;
}

// Validate the options, pick the format, and bring the GEMM up on the current
// device. Returns 1 when the GEMM is ready, 0 when the test should keep its
// portable path (a notice is printed), and -1 for invalid options (an error
// is printed and the test should exit).
static inline int pantheon_gemm_setup(PantheonGemm& g, PantheonGemmOptions& o, int device) {
    if (!o.enabled) return 0;
    PantheonGemmType type = GEMM_BF16;
    bool auto_type = (o.type == "auto");
    if (!auto_type && !PantheonGemm::parse_type(o.type.c_str(), &type)) {
        std::cerr << "[PANTHEON] Unknown --gemm_type '" << o.type
                  << "'. Use auto, bf16, fp16, tf32, fp32 or fp8." << std::endl;
        return -1;
    }
    if (o.m < 16 || o.n < 16 || o.k < 16) {
        std::cerr << "[PANTHEON] --gemm_m/n/k must each be at least 16." << std::endl;
        return -1;
    }
    // Round to 16: every format's tensor path wants it, FP8 requires it.
    o.m &= ~15; o.n &= ~15; o.k &= ~15;
    if (auto_type) {
        std::cout << "[PANTHEON] Choosing the GEMM format by measured power ("
                  << o.probe_ms << " ms each)..." << std::endl;
        if (!pantheon_gemm_pick(device, o.probe_ms, o.margin_pct, o.m, o.n, o.k, &type)) type = GEMM_BF16;
    }
    size_t gfree, gtotal;
    hipMemGetInfo(&gfree, &gtotal);
    pantheon_gemm_fit(type, gfree, &o.m, &o.n, &o.k);
    if (!g.init(type, o.m, o.n, o.k)) {
        std::cout << "[PANTHEON] Vendor GEMM unavailable (" << g.why
                  << "); using the portable WMMA tensor stream." << std::endl;
        return 0;
    }
    return 1;
}

// Matmuls per launch so that one launch takes about target_ms. The first call
// pays for library and kernel initialisation, so it is warmed up before it is
// timed; timed cold, the estimate comes out several times too long and the
// launches several times too short.
static inline int pantheon_gemm_batch(PantheonGemm& g, hipStream_t stream, int target_ms) {
    for (int r = 0; r < 4; ++r) g.run(stream);
    hipStreamSynchronize(stream);
    const int probe = 8;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < probe; ++r) g.run(stream);
    hipStreamSynchronize(stream);
    double per = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() / probe;
    if (per <= 0.0) per = 1e-6;
    int batch = (int)((target_ms / 1000.0) / per);
    return batch < 1 ? 1 : batch;
}

#endif // PANTHEON_VENDOR_GEMM
#endif // PANTHEON_VENDOR_GEMM_H
