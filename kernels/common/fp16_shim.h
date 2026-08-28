#ifndef PANTHEON_FP16_SHIM_H
#define PANTHEON_FP16_SHIM_H

// --- FIX: GUARD AGAINST GPU HEADERS IN MOCK MODE ---
#ifdef PANTHEON_MOCK
    // No headers needed. Types (like __half2) come from mock_gpu.h
#elif defined(__CUDACC__)
    #include <cuda_fp16.h>
#else
    #include <hip/hip_fp16.h>
#endif

// Universal FP16 Constructors & Math
// This hides the architecture differences for __half2 initialization
__device__ __forceinline__ __half2 make_half2_universal(float f) {
    // Works for Mock (function), CUDA (intrinsic), and HIP (intrinsic)
    return __float2half2_rn(f);
}

#endif // PANTHEON_FP16_SHIM_H
