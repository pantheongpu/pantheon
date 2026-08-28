#include "../common/common.h"

#include <chrono>
#include <iostream>
#include <string>

// A real two-rank all-reduce. GPU 1's input is copied to GPU 0, reduced
// there, then the completed value is broadcast back to GPU 1.
__global__ void init_collective(unsigned int* data, size_t count, unsigned int value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < count; idx += stride) data[idx] = value;
}

__global__ void add_collective(unsigned int* dst, const unsigned int* src, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < count; idx += stride) dst[idx] += src[idx];
}

__global__ void corrupt_collective(unsigned int* data) {
    if (blockIdx.x == 0 && threadIdx.x == 0) data[0] ^= 1u;
}

__global__ void verify_collective(const unsigned int* data, size_t count, unsigned int expected, unsigned int* errors) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < count; idx += stride) if (data[idx] != expected) atomicAdd(errors, 1);
}

int main(int argc, char** argv) {
    if (argc < 4) return 1;
    int primary = atoi(argv[1]);
    int duration = atoi(argv[2]);
    bool verify = false;
    bool inject_error = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--verify") verify = true;
        else if (arg == "--inject_error") inject_error = true;
    }

    int devices = 0;
    CHECK(hipGetDeviceCount(&devices));
    if (devices < 2) {
        std::cout << "[PANTHEON] Skipping ALL_REDUCE: fewer than two GPUs." << std::endl;
        std::cout << "Throughput: 0.0 GB/s" << std::endl;
        return 0;
    }

    int peer = (primary + 1) % devices;
    int forward_peer_access = 0;
    int reverse_peer_access = 0;
    CHECK(hipDeviceCanAccessPeer(&forward_peer_access, primary, peer));
    CHECK(hipDeviceCanAccessPeer(&reverse_peer_access, peer, primary));
    bool p2p = forward_peer_access && reverse_peer_access;
    if (p2p) {
        CHECK(hipSetDevice(primary));
        CHECK(hipDeviceEnablePeerAccess(peer, 0));
        CHECK(hipSetDevice(peer));
        CHECK(hipDeviceEnablePeerAccess(primary, 0));
    }

    const size_t bytes = 64ull * 1024 * 1024;
    const size_t count = bytes / sizeof(unsigned int);
    const int blocks = 4096;
    unsigned int* primary_data = nullptr;
    unsigned int* peer_data = nullptr;
    unsigned int* scratch = nullptr;
    void* host_stage = nullptr;

    CHECK(hipSetDevice(primary));
    CHECK(hipMalloc(&primary_data, bytes));
    CHECK(hipMalloc(&scratch, bytes));
    if (!p2p) CHECK(hipHostMalloc(&host_stage, bytes));
    CHECK(hipSetDevice(peer));
    CHECK(hipMalloc(&peer_data, bytes));
    CHECK(hipSetDevice(primary));
    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));

    LAUNCH_KERNEL(init_collective, blocks, 256, primary_data, count, 1u);
    CHECK(hipSetDevice(peer));
    LAUNCH_KERNEL(init_collective, blocks, 256, peer_data, count, 2u);
    CHECK(hipDeviceSynchronize());
    CHECK(hipSetDevice(primary));

    auto reduce_and_broadcast = [&]() {
        if (p2p) {
            CHECK(hipMemcpyPeerAsync(scratch, primary, peer_data, peer, bytes, stream));
        } else {
            CHECK(hipSetDevice(peer));
            CHECK(hipMemcpy(host_stage, peer_data, bytes, hipMemcpyDeviceToHost));
            CHECK(hipSetDevice(primary));
            CHECK(hipMemcpy(scratch, host_stage, bytes, hipMemcpyHostToDevice));
        }
        LAUNCH_KERNEL(add_collective, blocks, 256, primary_data, scratch, count);
        if (p2p) {
            CHECK(hipMemcpyPeerAsync(peer_data, peer, primary_data, primary, bytes, stream));
            CHECK(hipStreamSynchronize(stream));
        } else {
            CHECK(hipMemcpy(host_stage, primary_data, bytes, hipMemcpyDeviceToHost));
            CHECK(hipSetDevice(peer));
            CHECK(hipMemcpy(peer_data, host_stage, bytes, hipMemcpyHostToDevice));
            CHECK(hipDeviceSynchronize());
            CHECK(hipSetDevice(primary));
        }
    };

    std::cout << "[PANTHEON] GPU " << primary << ": ALL_REDUCE with GPU " << peer
              << (p2p ? " (peer DMA)" : " (host-staged fallback)") << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    // Time only the collective itself: the between-iteration re-initialization
    // is bookkeeping, not interconnect traffic, and must not dilute GB/s.
    double reduce_seconds = 0.0;
    size_t transferred = 0;
    do {
        auto step_start = std::chrono::high_resolution_clock::now();
        reduce_and_broadcast();
        reduce_seconds += std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - step_start).count();
        LAUNCH_KERNEL(init_collective, blocks, 256, primary_data, count, 1u);
        CHECK(hipSetDevice(peer));
        LAUNCH_KERNEL(init_collective, blocks, 256, peer_data, count, 2u);
        CHECK(hipDeviceSynchronize());
        CHECK(hipSetDevice(primary));
        // One gather crossing plus one broadcast crossing per collective.
        transferred += bytes * 2;
    } while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < duration);
    double seconds = reduce_seconds > 0.0 ? reduce_seconds : 1e-6;
    std::cout << "Throughput: " << (transferred / 1e9) / seconds << " GB/s" << std::endl;

    int exit_code = 0;
    if (verify) {
        reduce_and_broadcast();
        if (inject_error) {
            CHECK(hipSetDevice(peer));
            LAUNCH_KERNEL(corrupt_collective, 1, 1, peer_data);
            CHECK(hipDeviceSynchronize());
        }

        unsigned int* primary_errors = nullptr;
        unsigned int* peer_errors = nullptr;
        unsigned int primary_error_count = 0;
        unsigned int peer_error_count = 0;
        CHECK(hipSetDevice(primary));
        CHECK(hipMalloc(&primary_errors, sizeof(unsigned int)));
        CHECK(hipMemset(primary_errors, 0, sizeof(unsigned int)));
        LAUNCH_KERNEL(verify_collective, blocks, 256, primary_data, count, 3u, primary_errors);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(&primary_error_count, primary_errors, sizeof(primary_error_count), hipMemcpyDeviceToHost));
        CHECK(hipFree(primary_errors));

        CHECK(hipSetDevice(peer));
        CHECK(hipMalloc(&peer_errors, sizeof(unsigned int)));
        CHECK(hipMemset(peer_errors, 0, sizeof(unsigned int)));
        LAUNCH_KERNEL(verify_collective, blocks, 256, peer_data, count, 3u, peer_errors);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(&peer_error_count, peer_errors, sizeof(peer_error_count), hipMemcpyDeviceToHost));
        CHECK(hipFree(peer_errors));
        unsigned int errors = primary_error_count + peer_error_count;
        if (errors) {
            std::cout << "Verification: FAIL (" << errors << " errors)" << std::endl;
            exit_code = 1;
        } else {
            std::cout << "Verification: PASS" << std::endl;
        }
    }

    CHECK(hipSetDevice(primary));
    CHECK(hipStreamDestroy(stream));
    CHECK(hipFree(primary_data));
    CHECK(hipFree(scratch));
    if (host_stage) CHECK(hipHostFree(host_stage));
    CHECK(hipSetDevice(peer));
    CHECK(hipFree(peer_data));
    return exit_code;
}
