#include "common.h"

#include <chrono>
#include <iostream>
#include <string>

// This workload measures an actual runtime graph capture and replay path.
// CUDA builds map the HIP calls below to CUDA Graphs; ROCm builds use HIP Graphs.

__device__ __forceinline__ unsigned int graph_hash(unsigned int value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

__global__ void graph_initialize(uint4* data, size_t count, int inject_error) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < count; idx += stride) {
        unsigned int value = graph_hash(static_cast<unsigned int>(idx));
        data[idx] = (inject_error && idx == 1337)
            ? make_uint4(0, 0, 0, 0)
            : make_uint4(value | 1u, value ^ 0x9E3779B9u, value + 17u, value ^ 0xC2B2AE35u);
    }
}

__global__ void graph_replay_kernel(const uint4* data, size_t count, unsigned int* sink, int loops) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t window = count < 4096 ? count : 4096;
    unsigned int state = graph_hash(static_cast<unsigned int>(tid));
    float acc = 0.125f + static_cast<float>(state & 63u) * 0.001f;
    for (int iteration = 0; iteration < loops; ++iteration) {
        size_t base = (static_cast<size_t>(state) + static_cast<size_t>(iteration) * 257u) % (count - window + 1u);
        #pragma unroll 8
        for (size_t offset = 0; offset < window; offset += 256) {
            uint4 value = data[base + offset];
            unsigned int mix = value.x ^ value.y ^ value.z ^ value.w ^ state;
            acc = fmaf(acc, 1.00031f, static_cast<float>(mix & 1023u) * 0.00001f);
            state = graph_hash(mix ^ static_cast<unsigned int>(acc * 65536.0f));
        }
    }
    sink[tid] = state;
}

__global__ void graph_verify_input(const uint4* data, size_t count, unsigned int* errors) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < count; idx += stride) {
        unsigned int value = graph_hash(static_cast<unsigned int>(idx));
        if (data[idx].x != (value | 1u)) atomicAdd(errors, 1);
    }
}

__global__ void graph_verify_output(const unsigned int* actual, const unsigned int* expected, size_t count, unsigned int* errors) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < count; idx += stride) if (actual[idx] != expected[idx]) atomicAdd(errors, 1);
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]), duration = atoi(argv[2]), mem_pct = atoi(argv[3]);
    int block_size = 256, grid_size = 0, kernel_loops = 64, warmup_iters = 3, sync_mode = 2;
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
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, gpu_id));
    int blocks = grid_size ? grid_size : prop.multiProcessorCount * ((prop.maxThreadsPerMultiProcessor / block_size) ? (prop.maxThreadsPerMultiProcessor / block_size) : 8);
    size_t free_bytes, total_bytes;
    CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    mem_pct = mem_pct < 1 ? 1 : (mem_pct > 99 ? 99 : mem_pct);
    size_t bytes = (free_bytes * static_cast<size_t>(mem_pct)) / 100;
    if (bytes < 4096) bytes = 4096;
    size_t count = bytes / sizeof(uint4);
    size_t sink_count = static_cast<size_t>(blocks) * block_size;

    uint4* data = nullptr;
    unsigned int* sink = nullptr;
    unsigned int* expected = nullptr;
    CHECK(hipMalloc(&data, count * sizeof(uint4)));
    CHECK(hipMalloc(&sink, sink_count * sizeof(unsigned int)));
    if (verify_mode) CHECK(hipMalloc(&expected, sink_count * sizeof(unsigned int)));
    // Zero both result buffers: lanes a backend does not write (the CPU mock
    // executes a single logical thread) must compare as zero-vs-zero rather
    // than as whatever the allocator happened to return.
    CHECK(hipMemset(sink, 0, sink_count * sizeof(unsigned int)));
    if (verify_mode) CHECK(hipMemset(expected, 0, sink_count * sizeof(unsigned int)));
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Graph Replay" << std::endl;
    scale_warmup_for_large_alloc(warmup_iters, bytes);
    int init_grid = init_launch_grid_size(prop, count, 256);
    LAUNCH_KERNEL(graph_initialize, init_grid, 256, data, count, inject_error);
    CHECK(hipDeviceSynchronize());

#ifdef PANTHEON_MOCK
    std::cout << "[PANTHEON] Mock graph fallback: replaying the captured sequence directly." << std::endl;
    for (int i = 0; i < warmup_iters; ++i) LAUNCH_KERNEL(graph_replay_kernel, blocks, block_size, data, count, sink, kernel_loops);
    CHECK(hipDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    size_t work = 0;
    do {
        LAUNCH_KERNEL(graph_replay_kernel, blocks, block_size, data, count, sink, kernel_loops);
        CHECK(hipDeviceSynchronize());
        work += static_cast<size_t>(blocks) * block_size * kernel_loops;
    } while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < duration);
#else
    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));
    hipGraph_t graph;
    hipGraphExec_t graph_exec;
    CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    LAUNCH_KERNEL_ASYNC(graph_replay_kernel, blocks, block_size, 0, stream, data, count, sink, kernel_loops);
    CHECK(hipStreamEndCapture(stream, &graph));
    CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    for (int i = 0; i < warmup_iters; ++i) CHECK(hipGraphLaunch(graph_exec, stream));
    CHECK(hipStreamSynchronize(stream));
    auto start = std::chrono::high_resolution_clock::now();
    size_t work = 0;
    do {
        CHECK(hipGraphLaunch(graph_exec, stream));
        CHECK(hipStreamSynchronize(stream));
        work += static_cast<size_t>(blocks) * block_size * kernel_loops;
    } while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < duration);
#endif
    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "Throughput: " << work / seconds << " graph-steps/s" << std::endl;

    int exit_code = 0;
    if (verify_mode) {
        unsigned int* errors = nullptr;
        unsigned int host_errors = 0;
        LAUNCH_KERNEL(graph_replay_kernel, blocks, block_size, data, count, expected, kernel_loops);
#ifdef PANTHEON_MOCK
        LAUNCH_KERNEL(graph_replay_kernel, blocks, block_size, data, count, sink, kernel_loops);
#else
        CHECK(hipGraphLaunch(graph_exec, stream));
        CHECK(hipStreamSynchronize(stream));
#endif
        CHECK(hipMalloc(&errors, sizeof(unsigned int)));
        CHECK(hipMemset(errors, 0, sizeof(unsigned int)));
        int verify_grid = init_launch_grid_size(prop, count, 256);
        int verify_out_grid = init_launch_grid_size(prop, sink_count, 256);
        LAUNCH_KERNEL(graph_verify_input, verify_grid, 256, data, count, errors);
        LAUNCH_KERNEL(graph_verify_output, verify_out_grid, 256, sink, expected, sink_count, errors);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(&host_errors, errors, sizeof(unsigned int), hipMemcpyDeviceToHost));
        CHECK(hipFree(errors));
        std::cout << "Verification: " << (host_errors ? "FAIL" : "PASS") << " (" << host_errors << " input/output errors)" << std::endl;
        exit_code = host_errors ? 1 : 0;
    }
#ifndef PANTHEON_MOCK
    CHECK(hipGraphExecDestroy(graph_exec));
    CHECK(hipGraphDestroy(graph));
    CHECK(hipStreamDestroy(stream));
#endif
    if (expected) CHECK(hipFree(expected));
    CHECK(hipFree(sink));
    CHECK(hipFree(data));
    return exit_code;
}
