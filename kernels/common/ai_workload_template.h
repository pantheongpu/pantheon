#ifndef PANTHEON_AI_WORKLOAD_TEMPLATE_H
#define PANTHEON_AI_WORKLOAD_TEMPLATE_H

#include "common.h"
#include <chrono>
#include <iostream>
#include <string>

// Shared implementation for the AI-path workloads.  Each small translation
// unit selects a workload shape with PANTHEON_AI_WORKLOAD_KIND, while keeping
// the build system's one-source/one-binary convention intact.
#ifndef PANTHEON_AI_WORKLOAD_KIND
#error "PANTHEON_AI_WORKLOAD_KIND must be defined"
#endif

// The unit below is the same for every kind on purpose. This harness measures
// thread-iterations per second, which is a property of the launch geometry and
// the machine, not of the AI operation each kind is named after -- all ten
// converge to within about 1% of each other on the same GPU. Kinds 2, 3, 6 and
// 7 differ in their inner FMA expression; the rest run identical arithmetic and
// differ only in name. Naming the metric after a semantic work unit would
// invite readers to compare these numbers against real serving or training
// rates, which they do not measure.

#if PANTHEON_AI_WORKLOAD_KIND == 1
#define AI_NAME "Fused Attention"
#define AI_UNIT "ai-ops/s"
#define AI_LOOPS 32
#elif PANTHEON_AI_WORKLOAD_KIND == 2
#define AI_NAME "RoPE Stress"
#define AI_UNIT "ai-ops/s"
#define AI_LOOPS 64
#elif PANTHEON_AI_WORKLOAD_KIND == 3
#define AI_NAME "Quantized GEMM"
#define AI_UNIT "ai-ops/s"
#define AI_LOOPS 64
#elif PANTHEON_AI_WORKLOAD_KIND == 4
#define AI_NAME "Serving Mix"
#define AI_UNIT "ai-ops/s"
#define AI_LOOPS 32
#elif PANTHEON_AI_WORKLOAD_KIND == 5
#define AI_NAME "Speculative Decode"
#define AI_UNIT "ai-ops/s"
#define AI_LOOPS 48
#elif PANTHEON_AI_WORKLOAD_KIND == 6
#define AI_NAME "MoE Router"
#define AI_UNIT "ai-ops/s"
#define AI_LOOPS 64
#elif PANTHEON_AI_WORKLOAD_KIND == 7
#define AI_NAME "Transformer Train Step"
#define AI_UNIT "ai-ops/s"
#define AI_LOOPS 24
#elif PANTHEON_AI_WORKLOAD_KIND == 8
#define AI_NAME "Allocation Fragmentation"
#define AI_UNIT "ai-ops/s"
#define AI_LOOPS 96
#elif PANTHEON_AI_WORKLOAD_KIND == 10
#define AI_NAME "RAG Embedding"
#define AI_UNIT "ai-ops/s"
#define AI_LOOPS 48
#elif PANTHEON_AI_WORKLOAD_KIND == 11
#define AI_NAME "Vision Encoder"
#define AI_UNIT "ai-ops/s"
#define AI_LOOPS 48
#endif

__device__ __forceinline__ unsigned int ai_hash(unsigned int value) {
    value ^= value >> 16; value *= 0x7feb352du; value ^= value >> 15;
    value *= 0x846ca68bu; return value ^ (value >> 16);
}

__global__ void ai_initialize(uint4* data, size_t count, int inject_error) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < count; idx += stride) {
        unsigned int x = ai_hash(static_cast<unsigned int>(idx));
        data[idx] = (inject_error && idx == 1337)
            ? make_uint4(0, 0, 0, 0)
            : make_uint4(x | 1u, x ^ 0x9E3779B9u, x + 17u, x ^ 0xC2B2AE35u);
    }
}

// Each kind must stress a DIFFERENT bottleneck, not just a different
// arithmetic expression. Ten kinds previously shared one loop whose cost was
// dominated by window reads and the hash chain, so bolting a few FMAs inside it
// left every workload measuring the same thing -- all ten landed within 0.9% of
// each other on the same GPU. What follows gives each kind its own access
// pattern or divergence behaviour, which is what actually moves the number.
//
// Every variant stays deterministic in tid: verification runs the kernel twice
// and compares the sink, so data-dependent branching is fine but anything
// clock- or scheduler-dependent is not.
__global__ void ai_workload_kernel(const uint4* data, size_t count, unsigned int* sink, int loops) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t window = count < 4096 ? count : 4096;
    unsigned int state = ai_hash(static_cast<unsigned int>(tid) + PANTHEON_AI_WORKLOAD_KIND);
    float acc = 0.125f + static_cast<float>(state & 63u) * 0.001f;

#if PANTHEON_AI_WORKLOAD_KIND == 1
    // Fused Attention -- SFU bound. Softmax needs a transcendental per element
    // and a cross-lane reduction, which the plain FMA loop never exercised.
    for (int iteration = 0; iteration < loops; ++iteration) {
        size_t base = (static_cast<size_t>(state) + static_cast<size_t>(iteration) * 257u) % (count - window + 1u);
        float running_max = -1.0e30f, denom = 0.0f;
        // Few loads, heavy transcendental work: a softmax tile is bound by the
        // SFU and the reduction, not by fetching the tile.
        #pragma unroll 2
        for (size_t offset = 0; offset < window; offset += 2048) {
            uint4 v = data[base + offset];
            unsigned int mix = v.x ^ v.y ^ v.z ^ v.w ^ state;
            float score = static_cast<float>(mix & 1023u) * 0.001953125f - 1.0f;
            #pragma unroll 8
            for (int e = 0; e < 24; ++e) { score = __expf(score * 0.5f) - 0.5f; }
            running_max = fmaxf(running_max, score);
            denom += __expf(score - running_max);
            state = ai_hash(mix ^ __float_as_uint(denom));
        }
        // Reduce across the warp, as an attention tile does.
        for (int shift = 16; shift > 0; shift >>= 1) {
            denom += __shfl_xor_sync(0xffffffffu, denom, shift);
        }
        acc = fmaf(acc, 0.999f, denom * 1.0e-6f);
    }

#elif PANTHEON_AI_WORKLOAD_KIND == 2
    // RoPE -- SFU bound on sine/cosine rather than exp.
    for (int iteration = 0; iteration < loops; ++iteration) {
        size_t base = (static_cast<size_t>(state) + static_cast<size_t>(iteration) * 257u) % (count - window + 1u);
        #pragma unroll 2
        for (size_t offset = 0; offset < window; offset += 2048) {
            uint4 v = data[base + offset];
            unsigned int mix = v.x ^ v.y ^ v.z ^ v.w ^ state;
            float theta = static_cast<float>(mix & 4095u) * 0.001533981f;
            float s = 0.0f, c = 0.0f;
            // Rotate every position in the head dimension, not just one.
            #pragma unroll 8
            for (int d = 0; d < 32; ++d) { __sincosf(theta + static_cast<float>(d), &s, &c); theta = fmaf(s, 0.01f, theta); }
            float x = static_cast<float>((mix >> 12) & 1023u) * 0.0009765625f;
            acc = fmaf(x, c, acc * 0.999f) + s * 0.001f;
            state = ai_hash(mix ^ __float_as_uint(acc));
        }
    }

#elif PANTHEON_AI_WORKLOAD_KIND == 3
    // Quantized GEMM -- integer pipe. No float in the inner loop at all, so it
    // loads the INT units rather than the FP ones.
    unsigned int iacc = state;
    for (int iteration = 0; iteration < loops; ++iteration) {
        size_t base = (static_cast<size_t>(state) + static_cast<size_t>(iteration) * 257u) % (count - window + 1u);
        #pragma unroll 4
        for (size_t offset = 0; offset < window; offset += 1024) {
            uint4 v = data[base + offset];
            unsigned int mix = v.x ^ v.y ^ v.z ^ v.w ^ state;
            #pragma unroll 8
            for (int q = 0; q < 32; ++q) { iacc += (mix >> (q & 15)) & 15u; iacc *= 3u; }
            // Four packed 4-bit lanes, multiplied and accumulated as integers.
            unsigned int p0 = (mix >> 0) & 15u, p1 = (mix >> 4) & 15u;
            unsigned int p2 = (mix >> 8) & 15u, p3 = (mix >> 12) & 15u;
            iacc += p0 * p1 + p2 * p3;
            iacc = (iacc << 1) | (iacc >> 31);
            state = ai_hash(mix ^ iacc);
        }
    }
    acc = static_cast<float>(iacc & 1023u) * 0.001f;

#elif PANTHEON_AI_WORKLOAD_KIND == 4
    // Serving Mix -- warp divergence. Requests in a real server have wildly
    // different lengths, so lanes in a warp do different amounts of work and
    // the warp waits for its slowest lane.
    {
        int lane_work = 1 + static_cast<int>((state >> 3) & 31u);
        for (int iteration = 0; iteration < loops; ++iteration) {
            size_t base = (static_cast<size_t>(state) + static_cast<size_t>(iteration) * 257u) % (count - window + 1u);
            // Each lane fetches a different number of tiles, so the warp runs
            // at the pace of its longest request.
            for (int r = 0; r < lane_work; ++r) {
                size_t offset = (static_cast<size_t>(r) * 256u) % window;
                uint4 v = data[base + offset];
                unsigned int mix = v.x ^ v.y ^ v.z ^ v.w ^ state;
                acc = fmaf(acc, 1.00031f, static_cast<float>(mix & 255u) * 1.0e-5f);
                state = ai_hash(mix ^ __float_as_uint(acc));
            }
        }
    }

#elif PANTHEON_AI_WORKLOAD_KIND == 5
    // Speculative Decode -- branch divergence with early exit. Drafted tokens
    // are verified and rejected part way through, so trip counts vary per lane
    // and the loop leaves early.
    for (int iteration = 0; iteration < loops; ++iteration) {
        size_t base = (static_cast<size_t>(state) + static_cast<size_t>(iteration) * 257u) % (count - window + 1u);
        // Verify drafted tokens one fetch at a time and stop at the first
        // reject, so the number of loads issued is data dependent.
        int accepted = 0;
        #pragma unroll 1
        for (size_t offset = 0; offset < window; offset += 256) {
            uint4 v = data[base + offset];
            unsigned int mix = v.x ^ v.y ^ v.z ^ v.w ^ state;
            state = ai_hash(mix ^ static_cast<unsigned int>(accepted));
            acc = fmaf(acc, 1.0002f, static_cast<float>(mix & 255u) * 1.0e-5f);
            if ((mix & 7u) >= 5u) break;
            ++accepted;
        }
    }

#elif PANTHEON_AI_WORKLOAD_KIND == 6
    // MoE Router -- shared-memory atomics. Routing tokens to experts is a
    // scatter with contention, not a dense FMA stream.
    {
        __shared__ unsigned int expert_load[32];
        if (threadIdx.x < 32) expert_load[threadIdx.x] = 0u;
        __syncthreads();
        for (int iteration = 0; iteration < loops; ++iteration) {
            size_t base = (static_cast<size_t>(state) + static_cast<size_t>(iteration) * 257u) % (count - window + 1u);
            #pragma unroll 2
            for (size_t offset = 0; offset < window; offset += 1024) {
                uint4 v = data[base + offset];
                unsigned int mix = v.x ^ v.y ^ v.z ^ v.w ^ state;
                // Real routing is skewed: most tokens land on a few experts,
                // so the atomic serialises rather than spreading over banks.
                // Same-address atomics are warp-aggregated in hardware, so
                // contention alone changes nothing measurable. What actually
                // costs is the routed gather: each token reads the weights of
                // whichever expert it was assigned, so lanes in a warp fetch
                // from unrelated places.
                unsigned int expert = (mix >> 8) & 31u;
                // The atomic is here for the routing contention, and its return
                // value is deliberately unused: atomicAdd returns whatever the
                // counter happened to hold, which depends on execution order.
                // Feeding that into the result made the kernel nondeterministic,
                // so verification compared two differently-ordered runs and
                // failed on healthy hardware every time.
                atomicAdd(&expert_load[expert], 1u);
                size_t expert_row = (static_cast<size_t>(expert) * 1048573u
                                     + static_cast<size_t>(mix & 1023u)) % count;
                uint4 w = data[expert_row];
                mix ^= w.x ^ w.y ^ w.z ^ w.w;
                acc = fmaf(acc, 1.0001f, static_cast<float>(mix & 255u) * 1.0e-5f);
                state = ai_hash(mix ^ expert);
            }
        }
        __syncthreads();
        state ^= expert_load[threadIdx.x & 31u];
    }

#elif PANTHEON_AI_WORKLOAD_KIND == 7
    // Training Step -- arithmetic intensity. Forward, backward and optimizer
    // state means many FLOPs per byte loaded, unlike the memory-fed variants.
    for (int iteration = 0; iteration < loops; ++iteration) {
        size_t base = (static_cast<size_t>(state) + static_cast<size_t>(iteration) * 257u) % (count - window + 1u);
        // One load per iteration, not sixteen: the arithmetic has to dominate
        // for this to be compute bound rather than latency bound.
        {
            uint4 v = data[base];
            unsigned int mix = v.x ^ v.y ^ v.z ^ v.w ^ state;
            float g = static_cast<float>(mix & 511u) * 1.0e-4f;
            float m = acc * 0.9f, vv = acc * 0.999f;
            #pragma unroll 8
            for (int step = 0; step < 128; ++step) {
                m = fmaf(m, 0.9f, g);
                vv = fmaf(vv, 0.999f, g * g);
                acc = fmaf(m, vv * 1.0e-3f, acc * 0.9999f);  // no sqrt: FP pipe, not SFU
            }
            state = ai_hash(mix ^ __float_as_uint(acc));
        }
    }

#elif PANTHEON_AI_WORKLOAD_KIND == 10
    // RAG Embedding -- random gather, latency bound. Embedding lookups hit
    // scattered rows of a large table; nothing about that is sequential.
    for (int iteration = 0; iteration < loops; ++iteration) {
        // Four times the gathers of the other kinds, each to an unrelated row.
        #pragma unroll 4
        for (size_t step = 0; step < window; step += 64) {
            // Index the whole allocation pseudo-randomly rather than walking a
            // contiguous window.
            size_t idx = (static_cast<size_t>(state) * 2654435761u + step) % count;
            uint4 v = data[idx];
            unsigned int mix = v.x ^ v.y ^ v.z ^ v.w ^ state;
            acc = fmaf(acc, 1.00017f, static_cast<float>(mix & 1023u) * 1.0e-5f);
            state = ai_hash(mix);
        }
    }

#elif PANTHEON_AI_WORKLOAD_KIND == 11
    // Vision Encoder -- strided access. Patch extraction walks a row stride, so
    // consecutive lanes touch addresses far apart and coalescing is poor.
    {
        const size_t stride = 262144;  // ~4 MB apart: a new page every patch
        for (int iteration = 0; iteration < loops; ++iteration) {
            size_t base = (static_cast<size_t>(state) + static_cast<size_t>(iteration) * 257u) % count;
            #pragma unroll 4
            for (size_t patch = 0; patch < 32; ++patch) {
                size_t idx = (base + patch * stride) % count;
                uint4 v = data[idx];
                unsigned int mix = v.x ^ v.y ^ v.z ^ v.w ^ state;
                acc = fmaf(acc, 1.00021f, static_cast<float>(mix & 1023u) * 1.0e-5f);
                state = ai_hash(mix);
            }
        }
    }

#else
    // Remaining kinds keep the original shared loop.
    for (int iteration = 0; iteration < loops; ++iteration) {
        size_t base = (static_cast<size_t>(state) + static_cast<size_t>(iteration) * 257u) % (count - window + 1u);
        #pragma unroll 8
        for (size_t offset = 0; offset < window; offset += 256) {
            uint4 value = data[base + offset];
            unsigned int mix = value.x ^ value.y ^ value.z ^ value.w ^ state;
            acc = fmaf(acc, 1.00031f, static_cast<float>(mix & 1023u) * 0.00001f);
            state = ai_hash(mix ^ static_cast<unsigned int>(acc * 65536.0f));
        }
    }
#endif

    sink[tid] = state ^ __float_as_uint(acc);
}

__global__ void ai_verify(const uint4* data, size_t count, unsigned int* errors) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    for (; idx < count; idx += stride) {
        unsigned int x = ai_hash(static_cast<unsigned int>(idx));
        if (data[idx].x != (x | 1u)) atomicAdd(errors, 1);
    }
}

__global__ void ai_verify_output(const unsigned int* actual, const unsigned int* expected, size_t count, unsigned int* errors) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < count; idx += stride) {
        if (actual[idx] != expected[idx]) atomicAdd(errors, 1);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]), duration = atoi(argv[2]), mem_pct = atoi(argv[3]);
    int block_size = 256, grid_size = 0, kernel_loops = AI_LOOPS, warmup_iters = 3, sync_mode = 2;
    bool verify_mode = false; int inject_error = 0;
    for (int i = 1; i < argc; ++i) { std::string arg(argv[i]); if (arg == "--verify") verify_mode = true; else if (arg == "--inject_error") inject_error = 1; else if (arg == "--block_size" && i + 1 < argc) block_size = atoi(argv[++i]); else if (arg == "--grid_size" && i + 1 < argc) grid_size = atoi(argv[++i]); else if (arg == "--kernel_loops" && i + 1 < argc) kernel_loops = atoi(argv[++i]); else if (arg == "--warmup_iters" && i + 1 < argc) warmup_iters = atoi(argv[++i]); else if (arg == "--sync_mode" && i + 1 < argc) sync_mode = atoi(argv[++i]); }
    normalize_kernel_launch_config(block_size, grid_size, kernel_loops, warmup_iters, sync_mode); CHECK(hipSetDevice(gpu_id)); hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    int blocks = grid_size ? grid_size : prop.multiProcessorCount * ((prop.maxThreadsPerMultiProcessor / block_size) ? (prop.maxThreadsPerMultiProcessor / block_size) : 8);
    size_t free_bytes, total_bytes; CHECK(hipMemGetInfo(&free_bytes, &total_bytes)); mem_pct = mem_pct < 1 ? 1 : (mem_pct > 99 ? 99 : mem_pct);     size_t bytes = (free_bytes * static_cast<size_t>(mem_pct)) / 100; if (bytes < 4096) bytes = 4096; size_t count = bytes / sizeof(uint4);
    scale_warmup_for_large_alloc(warmup_iters, bytes);
    uint4* data = nullptr; unsigned int* sink = nullptr; unsigned int* expected = nullptr; size_t sink_count = static_cast<size_t>(blocks) * block_size; CHECK(hipMalloc(&data, count * sizeof(uint4))); CHECK(hipMalloc(&sink, sink_count * sizeof(unsigned int))); if (verify_mode) CHECK(hipMalloc(&expected, sink_count * sizeof(unsigned int)));
    // Zero both result buffers: lanes a backend does not write (the CPU mock
    // executes a single logical thread) must compare as zero-vs-zero rather
    // than as whatever the allocator happened to return.
    CHECK(hipMemset(sink, 0, sink_count * sizeof(unsigned int)));
    if (verify_mode) CHECK(hipMemset(expected, 0, sink_count * sizeof(unsigned int)));
    int init_grid = init_launch_grid_size(prop, count, 256);
    std::cout << "[PANTHEON] GPU " << gpu_id << ": " << AI_NAME << std::endl; LAUNCH_KERNEL(ai_initialize, init_grid, 256, data, count, inject_error); CHECK(hipDeviceSynchronize());
    for (int i = 0; i < warmup_iters; ++i) LAUNCH_KERNEL(ai_workload_kernel, blocks, block_size, data, count, sink, kernel_loops); CHECK(hipDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now(); size_t work = 0; do { LAUNCH_KERNEL(ai_workload_kernel, blocks, block_size, data, count, sink, kernel_loops); CHECK(hipDeviceSynchronize()); work += static_cast<size_t>(blocks) * block_size * kernel_loops; } while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < duration);
    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    // work counts thread-iterations, which is what this harness actually
    // measures. It is deliberately NOT labelled with a semantic unit: these
    // workloads share one loop, so "requests/s" or "train-steps/s" would be
    // the same number wearing a different name, and a reader would reasonably
    // read 2e8 requests/s as an inference rate rather than a synthetic op rate.
    std::cout << "Throughput: " << work / seconds << " " << AI_UNIT << std::endl;
    int exit_code = 0; if (verify_mode) { unsigned int* errors = nullptr, host_errors = 0; LAUNCH_KERNEL(ai_workload_kernel, blocks, block_size, data, count, expected, kernel_loops); LAUNCH_KERNEL(ai_workload_kernel, blocks, block_size, data, count, sink, kernel_loops); CHECK(hipMalloc(&errors, sizeof(unsigned int))); CHECK(hipMemset(errors, 0, sizeof(unsigned int))); int verify_grid = init_launch_grid_size(prop, count, 256); int verify_out_grid = init_launch_grid_size(prop, sink_count, 256); LAUNCH_KERNEL(ai_verify, verify_grid, 256, data, count, errors); LAUNCH_KERNEL(ai_verify_output, verify_out_grid, 256, sink, expected, sink_count, errors); CHECK(hipDeviceSynchronize()); CHECK(hipMemcpy(&host_errors, errors, sizeof(unsigned int), hipMemcpyDeviceToHost)); CHECK(hipFree(errors)); std::cout << "Verification: " << (host_errors ? "FAIL" : "PASS") << " (" << host_errors << " input/output errors)" << std::endl; exit_code = host_errors ? 1 : 0; }
    if (expected) CHECK(hipFree(expected)); CHECK(hipFree(sink)); CHECK(hipFree(data)); return exit_code;
}
#endif
