#include "../common/common.h"
#include <chrono>
#include <iostream>
#include <string>

// Decode is intentionally a latency-oriented kernel.  Each thread models a
// token stream: it performs a small projection-like FMA chain while walking a
// strided KV-cache window.  It is not a model implementation or a claim of
// end-to-end LLM accuracy; it isolates the compute and cache traffic that
// dominates autoregressive token generation.
__device__ __forceinline__ unsigned int decode_hash(unsigned int value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    return value;
}

__global__ void initialize_decode_kv(uint4* kv, size_t count, int inject_error) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < count; idx += stride) {
        unsigned int value = decode_hash(static_cast<unsigned int>(idx));
        if (inject_error && idx == 1337) value = 0;
        kv[idx] = make_uint4(value | 1u, value ^ 0xA5A5A5A5u, value + 17u, value ^ 0x5A5A5A5Au);
    }
}

__global__ void llm_decode_kernel(const uint4* kv, size_t count, unsigned int* sink, int loops) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t window = count < 4096 ? count : 4096;
    unsigned int state = decode_hash(static_cast<unsigned int>(tid) + 1u);
    float projection = 0.125f + static_cast<float>(state & 31u) * 0.001f;

    for (int step = 0; step < loops; ++step) {
        // A dependent gather approximates reading one token's KV history.
        size_t base = (static_cast<size_t>(state) + static_cast<size_t>(step) * 131u) % (count - window + 1u);
        #pragma unroll 8
        for (size_t lane = 0; lane < window; lane += 512) {
            uint4 value = kv[base + lane];
            state ^= value.x + value.y + value.z + value.w;
            projection = fmaf(projection, 1.00031f, static_cast<float>(state & 255u) * 0.00001f);
        }
        state = decode_hash(state ^ static_cast<unsigned int>(projection * 65536.0f));
    }
    sink[tid] = state;
}

__global__ void verify_decode_kv(const uint4* kv, size_t count, unsigned int* errors) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < count; idx += stride) {
        unsigned int value = decode_hash(static_cast<unsigned int>(idx));
        if (kv[idx].x != (value | 1u)) atomicAdd(errors, 1);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);
    int mem_pct = atoi(argv[3]);
    int block_size = 256, grid_size = 0, kernel_loops = 32, warmup_iters = 3, sync_mode = 2;
    bool verify_mode = false;
    int inject_error = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--verify") verify_mode = true;
        else if (arg == "--inject_error") inject_error = 1;
        else if (arg == "--block_size" && i + 1 < argc) block_size = atoi(argv[++i]);
        else if (arg == "--grid_size" && i + 1 < argc) grid_size = atoi(argv[++i]);
        else if (arg == "--kernel_loops" && i + 1 < argc) kernel_loops = atoi(argv[++i]);
        else if (arg == "--warmup_iters" && i + 1 < argc) warmup_iters = atoi(argv[++i]);
        else if (arg == "--sync_mode" && i + 1 < argc) sync_mode = atoi(argv[++i]);
    }
    normalize_kernel_launch_config(block_size, grid_size, kernel_loops, warmup_iters, sync_mode);
    CHECK(hipSetDevice(gpu_id));
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    int blocks = grid_size ? grid_size : prop.multiProcessorCount * ((prop.maxThreadsPerMultiProcessor / block_size) ? (prop.maxThreadsPerMultiProcessor / block_size) : 8);
    size_t free_bytes, total_bytes; CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    mem_pct = mem_pct < 1 ? 1 : (mem_pct > 99 ? 99 : mem_pct);
    size_t bytes = (free_bytes * static_cast<size_t>(mem_pct)) / 100;
    if (bytes < 4096) bytes = 4096;
    size_t count = bytes / sizeof(uint4);
    scale_warmup_for_large_alloc(warmup_iters, bytes);
    uint4* kv = nullptr; unsigned int* sink = nullptr;
    CHECK(hipMalloc(&kv, count * sizeof(uint4)));
    CHECK(hipMalloc(&sink, static_cast<size_t>(blocks) * block_size * sizeof(unsigned int)));
    std::cout << "[PANTHEON] GPU " << gpu_id << ": LLM Decode (KV-cache gather and projection)" << std::endl;
    std::cout << "  -> KV Cache:      " << (count * sizeof(uint4) / (1024 * 1024)) << " MiB" << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << std::endl;
    int init_grid = init_launch_grid_size(prop, count, 256);
    LAUNCH_KERNEL(initialize_decode_kv, init_grid, 256, kv, count, inject_error); CHECK(hipDeviceSynchronize());
    for (int i = 0; i < warmup_iters; ++i) LAUNCH_KERNEL(llm_decode_kernel, blocks, block_size, kv, count, sink, kernel_loops);
    CHECK(hipDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now(); size_t tokens = 0;
    do {
        LAUNCH_KERNEL(llm_decode_kernel, blocks, block_size, kv, count, sink, kernel_loops); CHECK(hipDeviceSynchronize());
        tokens += static_cast<size_t>(blocks) * block_size * kernel_loops;
    } while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < duration);
    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "Throughput: " << tokens / seconds << " tokens/s" << std::endl;
    int exit_code = 0;
    if (verify_mode) {
        unsigned int* errors = nullptr; unsigned int host_errors = 0;
        CHECK(hipMalloc(&errors, sizeof(unsigned int))); CHECK(hipMemset(errors, 0, sizeof(unsigned int)));
        int verify_grid = init_launch_grid_size(prop, count, 256);
        LAUNCH_KERNEL(verify_decode_kv, verify_grid, 256, kv, count, errors); CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(&host_errors, errors, sizeof(unsigned int), hipMemcpyDeviceToHost)); CHECK(hipFree(errors));
        std::cout << "Verification: " << (host_errors ? "FAIL" : "PASS") << " (" << host_errors << " KV-cache errors)" << std::endl;
        exit_code = host_errors ? 1 : 0;
    }
    CHECK(hipFree(sink)); CHECK(hipFree(kv)); return exit_code;
}
