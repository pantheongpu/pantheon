#ifndef MOCK_GPU_H
#define MOCK_GPU_H

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <map>

// --- MOCK STATE ---
static std::map<void*, size_t> g_mock_allocations;
static size_t g_mock_total_bytes = 0;

// --- MOCK TYPES ---
typedef int hipError_t;
typedef int hipStream_t;
#define hipSuccess 0
struct hipDeviceProp_t { 
	int multiProcessorCount; 
	
	int maxThreadsPerMultiProcessor;
	int maxGridSize[3];
	int memoryClockRate;
        int memoryBusWidth;
        int clockRate;
        int l2CacheSize;
};

// --- MOCK VECTOR TYPES ---
struct float4_ { float x, y, z, w; };
struct uint4_ { unsigned int x, y, z, w; };
struct float2_ { float x, y; };
typedef float4_ float4;
typedef uint4_ uint4;
typedef float2_ float2;

// --- MOCK QUALIFIERS ---
#define __global__
#define __device__
#define __host__
#define __shared__
#define __forceinline__ inline
#define __launch_bounds__(...)

// --- MOCK THREADING ---
struct uint3 { unsigned int x, y, z; };
static uint3 threadIdx = {0,0,0};
static uint3 blockIdx = {0,0,0};
static uint3 blockDim = {1,1,1};
static uint3 gridDim = {1,1,1};

inline void __syncthreads() {} 

// --- MOCK API ---
inline hipError_t hipSetDevice(int dev) { return hipSuccess; }
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* p, int d) { 
	p->multiProcessorCount = 1; 

	p->maxThreadsPerMultiProcessor = 1024;
	p->maxGridSize[0] = 65535;
	p->maxGridSize[1] = 65535;
	p->maxGridSize[2] = 65535;
        p->memoryClockRate = 1000000; // 1 GHz
        p->memoryBusWidth = 256;      // 256-bit bus
        p->clockRate = 1500000;       // 1.5 GHz
        p->l2CacheSize = 4194304;     // 4MB fake L2 Cache
	return hipSuccess; 
}
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }
inline hipError_t hipMemGetInfo(size_t* f, size_t* t) { *f=1e9; *t=2e9; return hipSuccess; }
inline const char* hipGetErrorString(hipError_t error) { return error == hipSuccess ? "Mock Success" : "Mock Error"; }

// Tell the mock environment it only has 1 fake GPU to safely skip the test
inline hipError_t hipGetDeviceCount(int* count) { 
    *count = 1; 
    return hipSuccess; 
}
inline hipError_t hipDeviceCanAccessPeer(int* canAccess, int dev, int peer) { 
    *canAccess = 0; 
    return hipSuccess; 
}
inline hipError_t hipDeviceEnablePeerAccess(int peer, unsigned int flags) { 
    return hipSuccess; 
}
inline hipError_t hipMemcpyPeerAsync(void* dst, int dstDev, const void* src, int srcDev, size_t count, hipStream_t stream = 0) { 
    memcpy(dst, src, count); 
    return hipSuccess; 
}

// --- TRACKED MEMORY MANAGEMENT ---

// Device Malloc. A failed allocation must report failure so CHECK() catches
// it, instead of handing the kernel a null pointer with a success code.
template <typename T>
inline hipError_t hipMalloc(T** p, size_t s) {
    *p = (T*)malloc(s);
    if (!*p) return 2; // mirrors hipErrorOutOfMemory
    g_mock_allocations[*p] = s;
    g_mock_total_bytes += s;
    return hipSuccess;
}

// Host Malloc (Pinned) - Simulates standard malloc for CPU
inline hipError_t hipHostMalloc(void** p, size_t s) {
    *p = malloc(s);
    if (!*p) return 2; // mirrors hipErrorOutOfMemory
    g_mock_allocations[*p] = s;
    g_mock_total_bytes += s;
    return hipSuccess;
}

// Universal Free (Handles both Device and Host pointers in Mock)
inline hipError_t hipFree(void* p) {
    if (p) {
        if (g_mock_allocations.find(p) != g_mock_allocations.end()) {
            g_mock_total_bytes -= g_mock_allocations[p];
            g_mock_allocations.erase(p);
        }
        free(p);
    }
    return hipSuccess;
}

inline hipError_t hipHostFree(void* p) { return hipFree(p); }
inline hipError_t hipMemset(void* p, int v, size_t s) { memset(p, v, s); return hipSuccess; }

// --- MOCK STREAMS ---
inline hipError_t hipStreamCreate(hipStream_t* s) { *s = 0; return hipSuccess; }
inline hipError_t hipStreamDestroy(hipStream_t s) { return hipSuccess; }
inline hipError_t hipStreamSynchronize(hipStream_t s) { return hipSuccess; }

// --- MOCK MEMCPY ---
enum hipMemcpyKind {
    hipMemcpyHostToDevice = 0,
    hipMemcpyDeviceToHost = 1,
    hipMemcpyDeviceToDevice = 2
};

// Async Copy (Just does a sync memcpy in mock)
inline hipError_t hipMemcpyAsync(void* dst, const void* src, size_t count, hipMemcpyKind kind, hipStream_t stream = 0) {
    memcpy(dst, src, count);
    return hipSuccess;
}

// NEW: Sync Copy (Required for RAS and Retention Bake verification)
inline hipError_t hipMemcpy(void* dst, const void* src, size_t count, hipMemcpyKind kind) {
    memcpy(dst, src, count);
    return hipSuccess;
}

// --- LEAK CHECKER ---
inline void mock_check_leaks() {
    if (g_mock_total_bytes > 0 || !g_mock_allocations.empty()) {
        std::cerr << "[MOCK ERROR] Memory Leak Detected! Leaked Bytes: " << g_mock_total_bytes << std::endl;
        exit(1);
    }
}

// --- MOCK INTRINSICS ---
typedef float __half2;
inline float __float2half2_rn(float f) { return f; }
inline float __hfma2(float a, float b, float c) { return a*b+c; }
inline float __hneg2(float a) { return -a; }
inline float2 __half22float2(float a) { return {a, a}; }
inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { return {x,y,z,w}; }
// Returns the previous value, matching CUDA/HIP semantics. Without this the
// mock cannot express the atomic-append idiom (claim a slot, then write it),
// which is why callers previously had to avoid the return value entirely.
inline unsigned int atomicAdd(unsigned int* address, unsigned int val) {
    unsigned int old = *address;
    *address += val;
    return old;
}
inline unsigned int atomicAdd(unsigned int* address, int val) {
    return atomicAdd(address, (unsigned int)val);
}

// Math Intrinsics
inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
inline float __sinf(float x) { return sinf(x); }
inline float __cosf(float x) { return cosf(x); }
inline float __expf(float x) { return expf(x); }
inline void __sincosf(float x, float* s, float* c) { *s = sinf(x); *c = cosf(x); }

// Bit reinterpretation, via memcpy so this stays defined behaviour rather than
// relying on type punning through a union.
inline unsigned int __float_as_uint(float x) {
    unsigned int out;
    std::memcpy(&out, &x, sizeof(out));
    return out;
}
inline float __uint_as_float(unsigned int x) {
    float out;
    std::memcpy(&out, &x, sizeof(out));
    return out;
}

// The mock executes a single logical thread, so a warp is one lane wide: every
// lane a shuffle could read from is this lane. Returning the value unchanged is
// the faithful single-thread reduction, not a stub.
inline float __shfl_xor_sync(unsigned int, float value, int, int = 32) { return value; }
inline unsigned int __shfl_xor_sync(unsigned int, unsigned int value, int, int = 32) { return value; }
inline int __shfl_xor_sync(unsigned int, int value, int, int = 32) { return value; }
inline float __logf(float x) { return logf(x); }

// --- AUTO-MAGIC LEAK CHECKER ---
struct MockLeakDetector {
    ~MockLeakDetector() { mock_check_leaks(); }
};
static MockLeakDetector g_leak_detector;

#endif
