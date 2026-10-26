#include "../common/common.h"
#include "../common/p2p_matrix_stats.h"
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

// P2P Matrix: how fast GPU g can send to every other GPU, one peer at a time.
//
// Pantheon starts one copy of this workload per selected GPU. Each copy measures
// its own row of the matrix (the transfers that start at its GPU), so together
// the copies cover every ordered pair exactly once. They take turns, through a
// lock file shared by the copies of one run, because two GPUs sending at once
// would share a link and read slow for it.

// --- PAYLOAD ---
// The value depends on the sending GPU, so data that arrives from the wrong card
// does not look right by accident.
__device__ __forceinline__ uint4 matrix_payload(size_t i, unsigned int src_gpu) {
    uint32_t salt = 0x9E3779B9u * (src_gpu + 1);
    uint32_t val = (uint32_t)i ^ salt;
    return make_uint4(val, val ^ 0x55555555u, ~val, val ^ 0xAAAAAAAAu);
}

__global__ void init_matrix_kernel(uint4* data, size_t n, unsigned int src_gpu) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) data[i] = matrix_payload(i, src_gpu);
}

__global__ void inject_matrix_error_kernel(uint4* data, size_t target_idx) {
    if (blockIdx.x == 0 && threadIdx.x == 0) data[target_idx].x ^= 0xBADBEEF;
}

__global__ void verify_matrix_kernel(const uint4* data, size_t n, unsigned int src_gpu, unsigned int* err_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        uint4 exp = matrix_payload(i, src_gpu);
        uint4 act = data[i];
        if (act.x != exp.x || act.y != exp.y || act.z != exp.z || act.w != exp.w) {
            if (*err_count < 5) {
                printf("[SDC FAULT][P2P_MATRIX] Peer copy from GPU %u arrived wrong! Index: %llu | Exp: {%x, %x, %x, %x} | Act: {%x, %x, %x, %x}\n",
                       src_gpu, (unsigned long long)i, exp.x, exp.y, exp.z, exp.w, act.x, act.y, act.z, act.w);
            }
            atomicAdd(err_count, 1);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int primary_gpu = atoi(argv[1]);
    int duration = atoi(argv[2]);
    // argv[3], the share of free memory to use, does not apply: each peer is sent
    // one fixed-size buffer.

    int block_size = 256;
    int grid_size = 0;          // 0 = auto-calculate
    int batch_copies = 8;       // copies between two timings
    int buffer_mb = 256;
    bool verify_mode = false;
    bool inject_error = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--verify") verify_mode = true;
        if (arg == "--inject_error") inject_error = true;
        if (arg == "--block_size" && i + 1 < argc) block_size = atoi(argv[++i]);
        if (arg == "--grid_size" && i + 1 < argc) grid_size = atoi(argv[++i]);
        if (arg == "--batch_copies" && i + 1 < argc) batch_copies = atoi(argv[++i]);
        if (arg == "--buffer_mb" && i + 1 < argc) buffer_mb = atoi(argv[++i]);
    }
    if (block_size < 32) block_size = 256;
    if (batch_copies < 1) batch_copies = 1;
    if (buffer_mb < 1) buffer_mb = 1;

    int device_count = 0;
    CHECK(hipGetDeviceCount(&device_count));
    if (device_count < 2) {
        if (verify_mode) std::cout << "Verification: SKIPPED (only 1 GPU detected)" << std::endl;
        return pantheon_skip(primary_gpu, "P2P_MATRIX", "only 1 GPU detected");
    }

    // The peers this GPU can send to directly, and the ones it cannot.
    std::vector<int> peers, unreachable;
    for (int peer = 0; peer < device_count; peer++) {
        if (peer == primary_gpu) continue;
        int forward = 0, backward = 0;
        CHECK(hipDeviceCanAccessPeer(&forward, primary_gpu, peer));
        CHECK(hipDeviceCanAccessPeer(&backward, peer, primary_gpu));
        (forward && backward ? peers : unreachable).push_back(peer);
    }
    if (peers.empty()) {
        if (verify_mode) std::cout << "Verification: SKIPPED (no bidirectional P2P routing)" << std::endl;
        return pantheon_skip(primary_gpu, "P2P_MATRIX", "no bidirectional P2P routing from GPU " + std::to_string(primary_gpu) + " to any other GPU");
    }

    size_t buffer_size = (size_t)buffer_mb * 1024 * 1024;
    size_t num_elements = buffer_size / sizeof(uint4);

    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, primary_gpu));
    int num_blocks = grid_size;
    if (num_blocks == 0) {
        int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
        if (max_blocks_per_sm < 1) max_blocks_per_sm = 16;
        num_blocks = prop.multiProcessorCount * max_blocks_per_sm;
    }

    double pair_seconds = pantheon_pair_seconds(duration, device_count);

    std::cout << "[PANTHEON] GPU " << primary_gpu << ": Running P2P MATRIX (row of " << peers.size() << " peers)..." << std::endl;
    std::cout << "  -> Duration (s):   " << duration << std::endl;
    std::cout << "  -> Per peer (s):   " << pair_seconds << std::endl;
    std::cout << "  -> Buffer (MiB):   " << buffer_mb << std::endl;
    std::cout << "  -> Batch copies:   " << batch_copies << std::endl;
    std::cout << "  -> Verify Mode:    " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- SENDER BUFFER ---
    uint4* d_src = nullptr;
    hipStream_t stream;
    CHECK(hipSetDevice(primary_gpu));
    CHECK(hipMalloc(&d_src, buffer_size));
    CHECK(hipStreamCreate(&stream));
    LAUNCH_KERNEL_ASYNC(init_matrix_kernel, num_blocks, block_size, 0, stream, d_src, num_elements, (unsigned int)primary_gpu);
    CHECK(hipStreamSynchronize(stream));
    for (int peer : peers) CHECK(hipDeviceEnablePeerAccess(peer, 0));

    // --- RECEIVE BUFFERS ---
    // One on each peer, made before the turn: creating a context on a GPU and
    // allocating on it take far longer than a copy, and must not be timed or
    // held against the other GPUs that are waiting.
    std::vector<uint4*> d_dsts(peers.size(), nullptr);
    for (size_t k = 0; k < peers.size(); k++) {
        CHECK(hipSetDevice(peers[k]));
        CHECK(hipMalloc(&d_dsts[k], buffer_size));
    }
    CHECK(hipSetDevice(primary_gpu));

    // --- WAIT FOR THIS GPU'S TURN ---
    PantheonMatrixTurn turn;
    if (!turn.acquire(duration + 60.0)) {
        std::cerr << "[PANTHEON ERROR] P2P_MATRIX: another GPU kept the interconnect for more than " << (duration + 60) << " s" << std::endl;
        return 1;
    }

    std::vector<double> bandwidth(peers.size(), 0.0);
    unsigned int total_errors = 0;
    bool injected = false;

    for (size_t k = 0; k < peers.size(); k++) {
        int peer = peers[k];
        uint4* d_dst = d_dsts[k];

        // Warm the path: the first copy pays for setting it up.
        for (int i = 0; i < 3; i++) CHECK(hipMemcpyPeerAsync(d_dst, peer, d_src, primary_gpu, buffer_size, stream));
        CHECK(hipStreamSynchronize(stream));

        std::vector<double> batches;
        auto row_start = std::chrono::steady_clock::now();
        do {
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < batch_copies; i++) CHECK(hipMemcpyPeerAsync(d_dst, peer, d_src, primary_gpu, buffer_size, stream));
            CHECK(hipStreamSynchronize(stream));
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (seconds > 0) batches.push_back((double)buffer_size * batch_copies / 1e9 / seconds);
        } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - row_start).count() < pair_seconds || batches.size() < 3);
        bandwidth[k] = pantheon_pair_bandwidth(batches);

        unsigned int errors = 0;
        if (verify_mode) {
            CHECK(hipSetDevice(peer));
            hipStream_t peer_stream;
            CHECK(hipStreamCreate(&peer_stream));
            if (inject_error && !injected) {
                // A link that delivers a corrupted payload: flip a bit in what arrived.
                LAUNCH_KERNEL_ASYNC(inject_matrix_error_kernel, 1, 1, 0, peer_stream, d_dst, 1337);
                CHECK(hipStreamSynchronize(peer_stream));
                injected = true;
            }
            unsigned int* d_err_count = nullptr;
            CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
            CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
            LAUNCH_KERNEL_ASYNC(verify_matrix_kernel, num_blocks, block_size, 0, peer_stream, d_dst, num_elements, (unsigned int)primary_gpu, d_err_count);
            CHECK(hipStreamSynchronize(peer_stream));
            CHECK(hipMemcpy(&errors, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));
            CHECK(hipFree(d_err_count));
            CHECK(hipStreamDestroy(peer_stream));
            total_errors += errors;
            CHECK(hipSetDevice(primary_gpu));
        }

        std::cout << "  -> GPU " << primary_gpu << " to GPU " << peer << ": " << bandwidth[k] << " GB/s"
                  << (verify_mode ? (errors ? "  (data wrong: " + std::to_string(errors) + " errors)" : "  (data ok)") : "")
                  << std::endl;

    }

    for (int peer : unreachable) {
        std::cout << "  -> GPU " << primary_gpu << " to GPU " << peer << ": no direct peer access, not measured" << std::endl;
    }

    PantheonPairStats stats = pantheon_pair_stats(bandwidth);
    std::cout << "[PANTHEON] GPU " << primary_gpu << ": " << peers.size() << " peers measured, slowest GPU "
              << primary_gpu << " to GPU " << peers[stats.slowest] << " at " << stats.min
              << " GB/s, median " << stats.median << " GB/s, fastest " << stats.max << " GB/s" << std::endl;
    // The score of this GPU is its slowest direction.
    std::cout << "Throughput: " << stats.min << " GB/s" << std::endl;

    if (verify_mode) {
        std::cout << "Verification: " << (total_errors ? "FAIL" : "PASS")
                  << " (" << total_errors << " errors)" << std::endl;
    }

    for (size_t k = 0; k < peers.size(); k++) {
        CHECK(hipSetDevice(peers[k]));
        CHECK(hipFree(d_dsts[k]));
    }
    CHECK(hipSetDevice(primary_gpu));
    CHECK(hipStreamDestroy(stream));
    CHECK(hipFree(d_src));
    return total_errors ? 1 : 0;
}
