#include "../common/common.h"
#include <chrono>
#include <iostream>
#include <string>

// Models paged/ragged KV-cache maintenance: sparse reads, per-token updates,
// and non-sequential page selection. This deliberately stresses allocation-like
// cache churn rather than a sequential bandwidth path.
__device__ __forceinline__ unsigned int churn_hash(unsigned int value) { value ^= value >> 17; value *= 0xed5ad4bbu; value ^= value >> 11; value *= 0xac4c1b51u; return value ^ (value >> 15); }
__global__ void initialize_churn(uint4* cache, size_t count, int inject_error) { size_t idx = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x; for (; idx < count; idx += stride) { unsigned int x = churn_hash(static_cast<unsigned int>(idx)); if (inject_error && idx == 1337) cache[idx] = make_uint4(0, 0, 0, 0); else cache[idx] = make_uint4(x | 1u, x + 7u, x ^ 0xC2B2AE35u, x + 19u); } }
__global__ void kv_cache_churn_kernel(uint4* cache, size_t count, unsigned int* sink, int loops) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x; unsigned int state = churn_hash(static_cast<unsigned int>(tid) + 17u);
    for (int iteration = 0; iteration < loops; ++iteration) {
        // 256 entries acts as a page; the two offsets imitate a ragged cache lookup and append.
        size_t page = (static_cast<size_t>(state) * 256u) % (count - 256u); size_t read_idx = page + ((state >> 8) & 255u); size_t write_idx = page + ((state >> 16) & 255u);
        uint4 prior = cache[read_idx]; state = churn_hash(state ^ prior.x ^ prior.z);
        cache[write_idx] = make_uint4((state | 1u), prior.y ^ state, prior.z + 1u, prior.w ^ 0x9E3779B9u);
    }
    sink[tid] = state;
}
__global__ void inject_churn_error(uint4* cache, size_t count) { if (count > 1337) cache[1337] = make_uint4(0, 0, 0, 0); }
__global__ void verify_churn(const uint4* cache, size_t count, unsigned int* errors) { size_t idx = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x; for (; idx < count; idx += stride) if (cache[idx].x == 0) atomicAdd(errors, 1); }
int main(int argc, char* argv[]) {
    if (argc < 4) return 1; int gpu_id = atoi(argv[1]), duration = atoi(argv[2]), mem_pct = atoi(argv[3]); int block_size = 256, grid_size = 0, kernel_loops = 128, warmup_iters = 3, sync_mode = 2; bool verify_mode = false; int inject_error = 0;
    for (int i = 1; i < argc; ++i) { std::string arg(argv[i]); if (arg == "--verify") verify_mode = true; else if (arg == "--inject_error") inject_error = 1; else if (arg == "--block_size" && i + 1 < argc) block_size = atoi(argv[++i]); else if (arg == "--grid_size" && i + 1 < argc) grid_size = atoi(argv[++i]); else if (arg == "--kernel_loops" && i + 1 < argc) kernel_loops = atoi(argv[++i]); else if (arg == "--warmup_iters" && i + 1 < argc) warmup_iters = atoi(argv[++i]); else if (arg == "--sync_mode" && i + 1 < argc) sync_mode = atoi(argv[++i]); }
    normalize_kernel_launch_config(block_size, grid_size, kernel_loops, warmup_iters, sync_mode); CHECK(hipSetDevice(gpu_id)); hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id)); int blocks = grid_size ? grid_size : prop.multiProcessorCount * ((prop.maxThreadsPerMultiProcessor / block_size) ? (prop.maxThreadsPerMultiProcessor / block_size) : 8);
    size_t free_bytes, total_bytes; CHECK(hipMemGetInfo(&free_bytes, &total_bytes)); mem_pct = mem_pct < 1 ? 1 : (mem_pct > 99 ? 99 : mem_pct); size_t bytes = (free_bytes * static_cast<size_t>(mem_pct)) / 100; if (bytes < 8192) bytes = 8192; size_t count = bytes / sizeof(uint4);
    scale_warmup_for_large_alloc(warmup_iters, bytes);
    uint4* cache = nullptr; unsigned int* sink = nullptr; CHECK(hipMalloc(&cache, count * sizeof(uint4))); CHECK(hipMalloc(&sink, static_cast<size_t>(blocks) * block_size * sizeof(unsigned int)));
    std::cout << "[PANTHEON] GPU " << gpu_id << ": KV Cache Churn (paged/ragged cache updates)" << std::endl; std::cout << "  -> Cache Size:    " << (count * sizeof(uint4) / (1024 * 1024)) << " MiB" << std::endl;
    int init_grid = init_launch_grid_size(prop, count, 256);
    LAUNCH_KERNEL(initialize_churn, init_grid, 256, cache, count, inject_error); CHECK(hipDeviceSynchronize()); for (int i = 0; i < warmup_iters; ++i) LAUNCH_KERNEL(kv_cache_churn_kernel, blocks, block_size, cache, count, sink, kernel_loops); CHECK(hipDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now(); size_t updates = 0; do { LAUNCH_KERNEL(kv_cache_churn_kernel, blocks, block_size, cache, count, sink, kernel_loops); CHECK(hipDeviceSynchronize()); updates += static_cast<size_t>(blocks) * block_size * kernel_loops; } while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < duration);
    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count(); std::cout << "Throughput: " << updates / seconds << " cache-updates/s" << std::endl;
    int exit_code = 0; if (verify_mode) { if (inject_error) { LAUNCH_KERNEL(inject_churn_error, 1, 1, cache, count); CHECK(hipDeviceSynchronize()); } unsigned int* errors = nullptr, host_errors = 0; CHECK(hipMalloc(&errors, sizeof(unsigned int))); CHECK(hipMemset(errors, 0, sizeof(unsigned int))); int verify_grid = init_launch_grid_size(prop, count, 256); LAUNCH_KERNEL(verify_churn, verify_grid, 256, cache, count, errors); CHECK(hipDeviceSynchronize()); CHECK(hipMemcpy(&host_errors, errors, sizeof(unsigned int), hipMemcpyDeviceToHost)); CHECK(hipFree(errors)); std::cout << "Verification: " << (host_errors ? "FAIL" : "PASS") << " (" << host_errors << " invalid cache entries)" << std::endl; exit_code = host_errors ? 1 : 0; }
    CHECK(hipFree(sink)); CHECK(hipFree(cache)); return exit_code;
}
