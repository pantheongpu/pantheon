#include "../common/common.h"
#include <chrono>
#include <iostream>
#include <string>

// Prefill is throughput-oriented. It combines projection-like math with a
// causal attention scan over a bounded context window. The workload avoids
// shipping model weights while retaining the memory and arithmetic shape that
// makes long-prompt inference different from decode.
__device__ __forceinline__ unsigned int prefill_hash(unsigned int value) {
    value ^= value >> 16; value *= 0x45d9f3bu; value ^= value >> 16; return value;
}
__global__ void initialize_prefill(uint4* data, size_t count, int inject_error) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    for (; idx < count; idx += stride) { unsigned int x = prefill_hash(static_cast<unsigned int>(idx)); if (inject_error && idx == 1337) x = 0; data[idx] = make_uint4(x | 1u, x + 3u, x ^ 0x9E3779B9u, x + 11u); }
}
__global__ void llm_prefill_kernel(const uint4* data, size_t count, unsigned int* sink, int loops) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t context = count < 8192 ? count : 8192;
    unsigned int state = prefill_hash(static_cast<unsigned int>(tid)); float q = 0.25f;
    for (int pass = 0; pass < loops; ++pass) {
        size_t start = (static_cast<size_t>(state) + static_cast<size_t>(pass) * 257u) % (count - context + 1u);
        // Causal scan: later elements consume all earlier tiles in the selected window.
        #pragma unroll 8
        for (size_t key = 0; key < context; key += 256) {
            uint4 kv = data[start + key];
            float score = static_cast<float>((kv.x ^ state) & 1023u) * 0.0009765625f;
            q = fmaf(q, 0.9991f, score);
            state = prefill_hash(state ^ kv.y ^ kv.z ^ static_cast<unsigned int>(q * 4096.0f));
        }
    }
    sink[tid] = state;
}
__global__ void verify_prefill(const uint4* data, size_t count, unsigned int* errors) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    for (; idx < count; idx += stride) { unsigned int x = prefill_hash(static_cast<unsigned int>(idx)); if (data[idx].x != (x | 1u)) atomicAdd(errors, 1); }
}
int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]), duration = atoi(argv[2]), mem_pct = atoi(argv[3]);
    int block_size = 256, grid_size = 0, kernel_loops = 16, warmup_iters = 3, sync_mode = 2; bool verify_mode = false; int inject_error = 0;
    for (int i = 1; i < argc; ++i) { std::string arg(argv[i]); if (arg == "--verify") verify_mode = true; else if (arg == "--inject_error") inject_error = 1; else if (arg == "--block_size" && i + 1 < argc) block_size = atoi(argv[++i]); else if (arg == "--grid_size" && i + 1 < argc) grid_size = atoi(argv[++i]); else if (arg == "--kernel_loops" && i + 1 < argc) kernel_loops = atoi(argv[++i]); else if (arg == "--warmup_iters" && i + 1 < argc) warmup_iters = atoi(argv[++i]); else if (arg == "--sync_mode" && i + 1 < argc) sync_mode = atoi(argv[++i]); }
    normalize_kernel_launch_config(block_size, grid_size, kernel_loops, warmup_iters, sync_mode); CHECK(hipSetDevice(gpu_id)); hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    int blocks = grid_size ? grid_size : prop.multiProcessorCount * ((prop.maxThreadsPerMultiProcessor / block_size) ? (prop.maxThreadsPerMultiProcessor / block_size) : 8);
    size_t free_bytes, total_bytes; CHECK(hipMemGetInfo(&free_bytes, &total_bytes)); mem_pct = mem_pct < 1 ? 1 : (mem_pct > 99 ? 99 : mem_pct); size_t bytes = (free_bytes * static_cast<size_t>(mem_pct)) / 100; if (bytes < 4096) bytes = 4096; size_t count = bytes / sizeof(uint4);
    scale_warmup_for_large_alloc(warmup_iters, bytes);
    uint4* data = nullptr; unsigned int* sink = nullptr; CHECK(hipMalloc(&data, count * sizeof(uint4))); CHECK(hipMalloc(&sink, static_cast<size_t>(blocks) * block_size * sizeof(unsigned int)));
    std::cout << "[PANTHEON] GPU " << gpu_id << ": LLM Prefill (causal attention and projections)" << std::endl; std::cout << "  -> Context Store: " << (count * sizeof(uint4) / (1024 * 1024)) << " MiB" << std::endl;
    int init_grid = init_launch_grid_size(prop, count, 256);
    LAUNCH_KERNEL(initialize_prefill, init_grid, 256, data, count, inject_error); CHECK(hipDeviceSynchronize()); for (int i = 0; i < warmup_iters; ++i) LAUNCH_KERNEL(llm_prefill_kernel, blocks, block_size, data, count, sink, kernel_loops); CHECK(hipDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now(); size_t tokens = 0; do { LAUNCH_KERNEL(llm_prefill_kernel, blocks, block_size, data, count, sink, kernel_loops); CHECK(hipDeviceSynchronize()); tokens += static_cast<size_t>(blocks) * block_size * kernel_loops; } while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < duration);
    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count(); std::cout << "Throughput: " << tokens / seconds << " prompt-tokens/s" << std::endl;
    int exit_code = 0; if (verify_mode) { unsigned int* errors = nullptr, host_errors = 0; CHECK(hipMalloc(&errors, sizeof(unsigned int))); CHECK(hipMemset(errors, 0, sizeof(unsigned int))); int verify_grid = init_launch_grid_size(prop, count, 256); LAUNCH_KERNEL(verify_prefill, verify_grid, 256, data, count, errors); CHECK(hipDeviceSynchronize()); CHECK(hipMemcpy(&host_errors, errors, sizeof(unsigned int), hipMemcpyDeviceToHost)); CHECK(hipFree(errors)); std::cout << "Verification: " << (host_errors ? "FAIL" : "PASS") << " (" << host_errors << " context errors)" << std::endl; exit_code = host_errors ? 1 : 0; }
    CHECK(hipFree(sink)); CHECK(hipFree(data)); return exit_code;
}
