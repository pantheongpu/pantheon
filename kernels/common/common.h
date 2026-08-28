#ifndef PANTHEON_COMMON_H
#define PANTHEON_COMMON_H

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <climits>
#include <string>

// ==========================================
// PATH 1: CPU MOCK SIMULATION (CI/CD)
// ==========================================
#ifdef PANTHEON_MOCK
    #include "mock_gpu.h"

    // In Mock mode, "Launch" just calls the function directly on the CPU.
    #define LAUNCH_KERNEL(kernel_name, grid, block, ...) kernel_name(__VA_ARGS__) 
    
    // NEW: Async Mock Macro
    #define LAUNCH_KERNEL_ASYNC(kernel_name, grid, block, shmem, stream, ...) kernel_name(__VA_ARGS__)

// ==========================================
// PATH 2: REAL GPU (CUDA / ROCm)
// ==========================================
#else
    // --- CROSS-PLATFORM SHIM (HIP -> CUDA) ---
    #ifdef __CUDACC__
        // NVIDIA / CUDA MODE
        #include <cuda_runtime.h>
        #include <device_launch_parameters.h>

        #define hipError_t cudaError_t
        #define hipSuccess cudaSuccess
        #define hipDeviceProp_t cudaDeviceProp
        #define hipGetErrorString cudaGetErrorString
        #define hipStream_t cudaStream_t
        #define hipSetDevice cudaSetDevice
        #define hipGetDeviceProperties cudaGetDeviceProperties
        #define hipMalloc cudaMalloc
        #define hipFree cudaFree
        #define hipMemcpy cudaMemcpy
        #define hipMemGetInfo cudaMemGetInfo
        #define hipDeviceSynchronize cudaDeviceSynchronize
        #define hipStreamCreate cudaStreamCreate
        #define hipStreamDestroy cudaStreamDestroy
        #define hipStreamSynchronize cudaStreamSynchronize
        #define hipMemset cudaMemset 
       
	#define hipGetLastError cudaGetLastError

        // --- NEW: PCIe / Async Mappings ---
        #define hipHostMalloc(ptr, size) cudaMallocHost(ptr, size)
        #define hipHostFree cudaFreeHost
        #define hipMemcpyAsync cudaMemcpyAsync
        #define hipMemcpyHostToDevice cudaMemcpyHostToDevice
        #define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
        #define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice

        #define hipGetDeviceCount cudaGetDeviceCount
        #define hipDeviceCanAccessPeer cudaDeviceCanAccessPeer
        #define hipDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
        #define hipMemcpyPeerAsync cudaMemcpyPeerAsync

        // Graph APIs are exposed through HIP names so graph-based workloads
        // build against CUDA and ROCm from the same source.
        #define hipGraph_t cudaGraph_t
        #define hipGraphExec_t cudaGraphExec_t
        #define hipGraphNode_t cudaGraphNode_t
        #define hipStreamCaptureModeGlobal cudaStreamCaptureModeGlobal
        #define hipStreamBeginCapture cudaStreamBeginCapture
        #define hipStreamEndCapture cudaStreamEndCapture
        #define hipGraphInstantiate cudaGraphInstantiate
        #define hipGraphLaunch cudaGraphLaunch
        #define hipGraphDestroy cudaGraphDestroy
        #define hipGraphExecDestroy cudaGraphExecDestroy

    #else
        // AMD / ROCm MODE
        #include <hip/hip_runtime.h>
    #endif

    // Real Kernel Launch Syntax. The do/while wrapper keeps the launch and
    // its error check together as one statement, so an unbraced `if` or
    // `for` guards both instead of only the launch.
    #define LAUNCH_KERNEL(kernel_name, grid, block, ...) \
        do { \
            kernel_name<<<grid, block>>>(__VA_ARGS__); \
            CHECK(hipGetLastError()); \
        } while (0)

    // Real Async Macro
    #define LAUNCH_KERNEL_ASYNC(kernel_name, grid, block, shmem, stream, ...) \
        do { \
            kernel_name<<<grid, block, shmem, stream>>>(__VA_ARGS__); \
            CHECK(hipGetLastError()); \
        } while (0)

    // 128-bit Vector Types (GCC/Clang Vector Extensions for GPU)
    typedef float float4_ __attribute__((vector_size(16)));
    typedef unsigned int uint4_ __attribute__((vector_size(16)));
#endif


// ==========================================
// SHARED UTILITIES
// ==========================================

#define BLOCK_SIZE 256

// Data backgrounds shared by the memory workloads.
//
//   0 = solid zeros        3 = rail-to-rail (0x00000000 / 0xFFFFFFFF) + entropy
//   1 = solid ones         4 = checkerboard (0xAAAAAAAA / 0x55555555), no entropy
//   2 = crosstalk (0xAAAAAAAA / 0x55555555) + entropy
//   5 = walking ones       6 = walking zeros
//
// Patterns 2 and 3 XOR a per-address base with entropy so the payload stays
// uncompressible while adjacent words toggle the bus differently. Patterns 4-6
// are deliberately *not* mixed with entropy: a checkerboard only exposes
// coupling if the neighbouring values are actually complementary, and a walking
// pattern only walks if exactly one bit moves per address. Entropy would
// destroy both properties.
//
// Keeping the selection here stops workloads from drifting apart, which is how
// the memory read and write tests ended up as separate binaries.


// ---------------------------------------------------------------------------
// Named data backgrounds
//
// The device-side selector stays numeric because it runs once per element, but
// the command line takes names: --init_pattern crosstalk says what it does,
// where --init_pattern 2 has to be looked up in a table. Numbers still parse,
// so existing scripts and recorded fault maps keep working unchanged.
// ---------------------------------------------------------------------------
struct PantheonPatternName { const char* name; int value; };

static const PantheonPatternName PANTHEON_PATTERN_NAMES[] = {
    {"zeros",         0},
    {"ones",          1},
    {"crosstalk",     2},
    {"rail_to_rail",  3},
    {"checkerboard",  4},
    {"walking_ones",  5},
    {"walking_zeros", 6},
};
static const int PANTHEON_PATTERN_COUNT =
    (int)(sizeof(PANTHEON_PATTERN_NAMES) / sizeof(PANTHEON_PATTERN_NAMES[0]));

// Case-insensitive, and '-' and '_' are skipped, so rail_to_rail, rail-to-rail
// and railtorail all name the same background.
inline bool pantheon_pattern_name_matches(const char* a, const char* b) {
    for (;;) {
        while (*a == '-' || *a == '_') ++a;
        while (*b == '-' || *b == '_') ++b;
        if (!*a || !*b) return *a == '\0' && *b == '\0';
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return false;
        ++a; ++b;
    }
}

inline const char* pantheon_init_pattern_name(int value) {
    for (int i = 0; i < PANTHEON_PATTERN_COUNT; ++i) {
        if (PANTHEON_PATTERN_NAMES[i].value == value) return PANTHEON_PATTERN_NAMES[i].name;
    }
    return "custom";
}

inline void pantheon_print_init_patterns(std::ostream& os) {
    os << "[PANTHEON] --init_pattern accepts:";
    for (int i = 0; i < PANTHEON_PATTERN_COUNT; ++i) {
        os << (i ? ", " : " ") << PANTHEON_PATTERN_NAMES[i].name
           << " (" << PANTHEON_PATTERN_NAMES[i].value << ")";
    }
    os << std::endl;
}

// Accepts a name or a bare number. Returns false and leaves *out untouched on
// anything else -- a typo silently becoming pattern 0 would quietly test the
// wrong background, which is worse than refusing to start.
inline bool pantheon_parse_init_pattern(const char* arg, int* out) {
    if (!arg || !*arg) return false;
    for (int i = 0; i < PANTHEON_PATTERN_COUNT; ++i) {
        if (pantheon_pattern_name_matches(arg, PANTHEON_PATTERN_NAMES[i].name)) {
            *out = PANTHEON_PATTERN_NAMES[i].value;
            return true;
        }
    }
    char* end = nullptr;
    long v = strtol(arg, &end, 10);
    if (end && *end == '\0' && v >= 0 && v <= INT_MAX) { *out = (int)v; return true; }
    return false;
}

__device__ __host__ __forceinline__ unsigned int pantheon_pattern_base(
    unsigned long long index, int init_pattern
) {
    if (init_pattern == 3) {
        return (index % 2 == 0) ? 0x00000000u : 0xFFFFFFFFu;
    }
    return (index % 2 == 0) ? 0xAAAAAAAAu : 0x55555555u;
}

// The value a workload should store at `index` for the selected background.
// `entropy` is only consumed by the patterns that mix it in.
__device__ __host__ __forceinline__ unsigned int pantheon_pattern_value(
    unsigned long long index, int init_pattern, unsigned int entropy
) {
    switch (init_pattern) {
        case 0: return 0x00000000u;
        case 1: return 0xFFFFFFFFu;
        case 4: return (index % 2 == 0) ? 0xAAAAAAAAu : 0x55555555u;
        case 5: return 1u << (unsigned int)(index % 32);          // walking ones
        case 6: return ~(1u << (unsigned int)(index % 32));       // walking zeros
        default: return pantheon_pattern_base(index, init_pattern) ^ entropy;
    }
}

// Reinterpret value bits without the undefined behavior of pointer punning
// (`*(unsigned int*)&f`), which g++ flags under strict aliasing. The memcpy
// compiles to a plain register move on CUDA, HIP, and host compilers.
template <typename To, typename From>
__device__ __host__ __forceinline__ To pantheon_bit_cast(const From& from) {
    static_assert(sizeof(To) == sizeof(From), "bit_cast requires equal sizes");
    To to;
    __builtin_memcpy(&to, &from, sizeof(To));
    return to;
}

#define CHECK(cmd) \
{ \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
        std::cerr << "[PANTHEON ERROR] Code: " << error \
                  << " (" << hipGetErrorString(error) << ")" \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// Normalize user-facing stress knobs before they reach launch geometry or
// inner loops.  Direct kernel binaries are intentionally tunable, but invalid
// values such as --block_size 0 or --kernel_loops -1 should never turn a stress
// test into a divide-by-zero, invalid launch, or no-op.
// Blocks per launch beyond this buy nothing and start to cost: the launch stops
// returning in reasonable time, which presents as a hang.
#define PANTHEON_MAX_GRID_BLOCKS 262144

// A single kernel launch has to stay bounded. Past this the launch stops
// returning in any reasonable time, which presents as a hang and can trip the
// display driver's timeout detection.
#define PANTHEON_MAX_KERNEL_LOOPS 1000000

inline void normalize_kernel_launch_config(
    int& block_size,
    int& grid_size,
    int& kernel_loops,
    int& warmup_iters,
    int& sync_mode
) {
    if (block_size <= 0) {
        std::cerr << "[PANTHEON] Warning: invalid block_size " << block_size
                  << "; using 256." << std::endl;
        block_size = 256;
    } else if (block_size > 1024) {
        std::cerr << "[PANTHEON] Warning: block_size " << block_size
                  << " exceeds common GPU launch limits; clamping to 1024." << std::endl;
        block_size = 1024;
    }

    if (grid_size < 0) {
        std::cerr << "[PANTHEON] Warning: invalid grid_size " << grid_size
                  << "; using auto-grid." << std::endl;
        grid_size = 0;
    } else if (grid_size > PANTHEON_MAX_GRID_BLOCKS) {
        // A user-supplied grid went straight to the launch unchecked, while the
        // internal fill grids were already capped at the same number.
        std::cerr << "[PANTHEON] Warning: grid_size " << grid_size
                  << " exceeds practical launch limits; clamping to "
                  << PANTHEON_MAX_GRID_BLOCKS << "." << std::endl;
        grid_size = PANTHEON_MAX_GRID_BLOCKS;
    }

    if (kernel_loops < 1) {
        std::cerr << "[PANTHEON] Warning: invalid kernel_loops " << kernel_loops
                  << "; using 1." << std::endl;
        kernel_loops = 1;
    } else if (kernel_loops > PANTHEON_MAX_KERNEL_LOOPS) {
        // Kernel runtime scales linearly with this, so an unbounded value is a
        // GPU hang: the launch never returns, and on a display GPU the driver
        // resets the device out from under the run. Deep tuning stays possible
        // well inside this bound.
        std::cerr << "[PANTHEON] Warning: kernel_loops " << kernel_loops
                  << " would make a single launch effectively unbounded; clamping to "
                  << PANTHEON_MAX_KERNEL_LOOPS << "." << std::endl;
        kernel_loops = PANTHEON_MAX_KERNEL_LOOPS;
    }

    if (warmup_iters < 0) {
        std::cerr << "[PANTHEON] Warning: invalid warmup_iters " << warmup_iters
                  << "; using 0." << std::endl;
        warmup_iters = 0;
    }

    if (sync_mode < 0 || sync_mode > 2) {
        std::cerr << "[PANTHEON] Warning: invalid sync_mode " << sync_mode
                  << "; using blocking sync mode 2." << std::endl;
        sync_mode = 2;
    }
}

// Large VRAM fills must use a wide grid-stride launch without exceeding driver limits.
inline int init_launch_grid_size(const hipDeviceProp_t& prop, size_t num_elements, int block_size) {
    if (block_size <= 0) block_size = BLOCK_SIZE;
    const unsigned int practical_max = PANTHEON_MAX_GRID_BLOCKS;
#ifdef PANTHEON_MOCK
    (void)prop;
    unsigned int max_grid = practical_max;
#else
    unsigned int max_grid = prop.maxGridSize[0];
    if (max_grid == 0) max_grid = 65535;
    if (max_grid > practical_max) max_grid = practical_max;
#endif
    size_t needed = (num_elements + (size_t)block_size - 1) / (size_t)block_size;
    if (needed == 0) needed = 1;
    if (needed > max_grid) return (int)max_grid;
    return (int)needed;
}

inline void scale_warmup_for_large_alloc(int& warmup_iters, size_t alloc_bytes) {
    const size_t gb64 = 64ULL * 1024 * 1024 * 1024;
    const size_t gb200 = 200ULL * 1024 * 1024 * 1024;
    if (alloc_bytes > gb200 && warmup_iters > 0) {
        std::cerr << "[PANTHEON] Large VRAM allocation (" << (alloc_bytes / (1024 * 1024))
                  << " MB): skipping warmup." << std::endl;
        warmup_iters = 0;
    } else if (alloc_bytes > gb64 && warmup_iters > 1) {
        std::cerr << "[PANTHEON] Large VRAM allocation (" << (alloc_bytes / (1024 * 1024))
                  << " MB): reducing warmup to 1 iteration." << std::endl;
        warmup_iters = 1;
    }
}

// A launch that walks the buffer per loop costs loops * alloc_bytes. Bound that
// product rather than the loop count alone: the per-loop cost differs by orders
// of magnitude between a bandwidth workload and an ALU one, so no single loop
// limit suits both. Chosen high enough that real defaults are never touched --
// the largest cards run about a quarter of this at their default settings.
#define PANTHEON_MAX_BYTES_PER_LAUNCH (4ULL * 1024 * 1024 * 1024 * 1024)

inline void scale_kernel_loops_for_large_alloc(int& kernel_loops, size_t alloc_bytes, int cap) {
    const size_t gb200 = 200ULL * 1024 * 1024 * 1024;
    if (cap < 1) cap = 1;
    if (alloc_bytes > gb200 && kernel_loops > cap) {
        std::cerr << "[PANTHEON] Large VRAM allocation (" << (alloc_bytes / (1024 * 1024))
                  << " MB): reducing kernel_loops from " << kernel_loops << " to " << cap << "."
                  << std::endl;
        kernel_loops = cap;
    }

    // Guard against a launch that never returns. Without this, a large
    // --kernel_loops on a bandwidth workload presents as a hang and can trip
    // the display driver's timeout detection.
    if (alloc_bytes > 0) {
        size_t max_loops = PANTHEON_MAX_BYTES_PER_LAUNCH / alloc_bytes;
        if (max_loops < 1) max_loops = 1;
        if ((size_t)kernel_loops > max_loops) {
            std::cerr << "[PANTHEON] Warning: kernel_loops " << kernel_loops << " over a "
                      << (alloc_bytes / (1024 * 1024)) << " MB buffer would make a single "
                      << "launch effectively unbounded; clamping to " << max_loops << "."
                      << std::endl;
            kernel_loops = (int)max_loops;
        }
    }
}

// --- NON-TEMPORAL STORE (Bypass Cache -> Write Memory) ---
// 1. CPU Mock: Standard Store
// 2. AMD: Builtin or Cast
// 3. CUDA: PTX ASM
__device__ __host__ __forceinline__ void store_nt(void* addr, uint4 val) {
#ifdef PANTHEON_MOCK
    // CPU Mock: Just write to memory
    *(uint4*)addr = val;
#elif defined(__HIP_PLATFORM_AMD__)
    // AMD: Use builtin for NT store
    typedef unsigned int __attribute__((vector_size(16))) vec_uint4;
    __builtin_nontemporal_store(*(vec_uint4*)&val, (vec_uint4*)addr);
#elif defined(__CUDACC__)
    // NVIDIA: PTX ASM
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" 
                 :: "l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#else
    *(uint4*)addr = val;
#endif
}

// --- NON-TEMPORAL LOAD (CRASH-PROOF VERSION) ---
// 1. CPU Mock: Standard Load
// 2. AMD: Decomposed Loads (RDNA Fix)
// 3. CUDA: PTX ASM
__device__ __host__ __forceinline__ uint4 load_nt(void* addr) {
    uint4 ret;
#ifdef PANTHEON_MOCK
    // CPU Mock: Just read memory
    ret = *(uint4*)addr;
#elif defined(__HIP_PLATFORM_AMD__)
    // RDNA CRASH FIX: Decompose into 4x 32-bit loads.
    // Attempting a single 128-bit vector load (*(uint4*)) segfaults on RDNA
    // if the buffer isn't perfectly 128-bit aligned.
    unsigned int* p = (unsigned int*)addr;
    ret.x = p[0];
    ret.y = p[1];
    ret.z = p[2];
    ret.w = p[3];
#elif defined(__CUDACC__)
    // NVIDIA: PTX Streaming Load
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];" 
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(addr));
#else
    ret = *(uint4*)addr;
#endif
    return ret;
}

#endif // PANTHEON_COMMON_H
