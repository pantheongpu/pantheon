#include "../common/common.h"
#include "../common/p2p_matrix_stats.h"
#include "../common/nccl_collective.h"
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

// NCCL All-Reduce: one all-reduce across every selected GPU at once, done by the
// vendor's own collective library (NCCL on NVIDIA, RCCL on AMD).
//
// all_reduce moves data between two GPUs with plain copies; this is the call that
// training jobs make, with the library choosing the ring or tree, the transport (NVLink,
// NVSwitch, PCIe, shared memory) and the number of channels. It is the closest a test
// gets to the traffic a multi-GPU job puts on the interconnect.
//
// Pantheon starts one copy per selected GPU and tells each which devices take part
// (PANTHEON_PEER_GPUS). The copies share the library's unique id through a file named
// for the run, join one communicator, and all call the collective together. The score
// is the "bus bandwidth" nccl-tests reports, so it compares with the speed of a link.

// Every rank sends its own number plus one, so the sum is known and exact in a float.
__global__ void init_allreduce_kernel(float* data, size_t n, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) data[i] = value;
}

__global__ void inject_allreduce_error_kernel(float* data, size_t target_idx) {
    if (blockIdx.x == 0 && threadIdx.x == 0) data[target_idx] += 1.0f;
}

__global__ void verify_allreduce_kernel(const float* data, size_t n, float expected, unsigned int rank, unsigned int* err_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        if (data[i] != expected) {
            if (*err_count < 5) {
                printf("[SDC FAULT][NCCL_ALL_REDUCE] Rank %u: all-reduce result wrong! Index: %llu | Exp: %f | Act: %f\n",
                       rank, (unsigned long long)i, expected, data[i]);
            }
            atomicAdd(err_count, 1);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int device = atoi(argv[1]);
    int duration = atoi(argv[2]);
    // argv[3], the share of free memory to use, does not apply: the collective works on
    // one fixed-size buffer.

    int buffer_mb = 256;
    int batch_iters = 10;       // collectives between two timings
    bool verify_mode = false;
    bool inject_error = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--verify") verify_mode = true;
        if (arg == "--inject_error") inject_error = true;
        if (arg == "--buffer_mb" && i + 1 < argc) buffer_mb = atoi(argv[++i]);
        if (arg == "--batch_iters" && i + 1 < argc) batch_iters = atoi(argv[++i]);
    }
    if (buffer_mb < 1) buffer_mb = 1;
    if (batch_iters < 1) batch_iters = 1;

    PantheonCollectiveGroup group = pantheon_collective_group(device, getenv("PANTHEON_PEER_GPUS"), getenv("PANTHEON_RUN_TOKEN"));
    if (!group.error.empty()) {
        if (verify_mode) std::cout << "Verification: SKIPPED (" << group.error << ")" << std::endl;
        return pantheon_skip(device, "NCCL_ALL_REDUCE", group.error);
    }

#ifdef PANTHEON_MOCK
    if (verify_mode) std::cout << "Verification: SKIPPED (mock backend)" << std::endl;
    return pantheon_skip(device, "NCCL_ALL_REDUCE", "the mock backend has no collective library");
#else
    // A copy that never arrives would hold the others in the library for ever.
    const std::string id_path = pantheon_collective_id_path(group.token);
    pantheon_collective_deadline((unsigned int)duration + 180, group.rank == 0 ? id_path : std::string());

    PantheonNccl nccl;
    if (!nccl.load(pantheon_nccl_candidates(getenv("PANTHEON_NCCL_LIB")))) {
        if (verify_mode) std::cout << "Verification: SKIPPED (" << nccl.error << ")" << std::endl;
        return pantheon_skip(device, "NCCL_ALL_REDUCE", nccl.error);
    }

#define NCCL_CHECK(call)                                                                                  \
    do {                                                                                                  \
        int nccl_result_ = (call);                                                                        \
        if (nccl_result_ != 0) {                                                                          \
            std::cerr << "[PANTHEON ERROR] NCCL_ALL_REDUCE: " #call " failed: " << nccl.describe(nccl_result_) << std::endl; \
            return 1;                                                                                     \
        }                                                                                                 \
    } while (0)

    const int ranks = (int)group.devices.size();
    const int rank = group.rank;
    const size_t bytes = (size_t)buffer_mb * 1024 * 1024;
    const size_t count = bytes / sizeof(float);
    const int blocks = 4096;
    const int threads = 256;

    int version = 0;
    NCCL_CHECK(nccl.GetVersion(&version));

    std::string members;
    for (int d : group.devices) members += (members.empty() ? "" : ",") + std::to_string(d);
    std::cout << "[PANTHEON] GPU " << device << ": NCCL_ALL_REDUCE rank " << rank << " of " << ranks << " (devices " << members
              << ", NCCL " << pantheon_nccl_version_string(version) << ")" << std::endl;
    std::cout << "  -> Duration (s):   " << duration << std::endl;
    std::cout << "  -> Buffer (MiB):   " << buffer_mb << std::endl;
    std::cout << "  -> Batch size:     " << batch_iters << std::endl;
    std::cout << "  -> Library:        " << nccl.path << std::endl;
    std::cout << "  -> Verify Mode:    " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- BUFFERS ---
    // Made before joining the group: allocating takes far longer than a collective, and
    // the other copies are waiting to be joined by this one.
    float* d_send = nullptr;
    float* d_recv = nullptr;
    float* d_flag = nullptr;
    hipStream_t stream;
    CHECK(hipSetDevice(device));
    CHECK(hipMalloc(&d_send, bytes));
    CHECK(hipMalloc(&d_recv, bytes));
    CHECK(hipMalloc(&d_flag, sizeof(float)));
    CHECK(hipStreamCreate(&stream));
    LAUNCH_KERNEL_ASYNC(init_allreduce_kernel, blocks, threads, 0, stream, d_send, count, pantheon_allreduce_input(rank));
    CHECK(hipMemset(d_recv, 0, bytes));
    CHECK(hipStreamSynchronize(stream));

    // --- JOIN THE GROUP ---
    PantheonNcclUniqueId id;
    std::memset(&id, 0, sizeof(id));
    PantheonCollectiveIdFile id_file;       // rank 0's file goes however rank 0 leaves
    if (rank == 0) {
        NCCL_CHECK(nccl.GetUniqueId(&id));
        if (!pantheon_collective_publish(id_path, &id, sizeof(id))) {
            std::cerr << "[PANTHEON ERROR] NCCL_ALL_REDUCE: could not write " << id_path << std::endl;
            return 1;
        }
        id_file.path = id_path;
    } else if (!pantheon_collective_wait(id_path, &id, sizeof(id), 120.0)) {
        std::cerr << "[PANTHEON ERROR] NCCL_ALL_REDUCE: rank 0 did not publish the unique id within 120 s" << std::endl;
        return 1;
    }
    void* comm = nullptr;
    NCCL_CHECK(nccl.CommInitRank(&comm, ranks, id, rank));
    // Every copy has read the id by now: the call returns once all of them have joined.
    if (rank == 0) pantheon_collective_remove(id_path);

    auto run_batch = [&](int iterations) -> int {
        for (int i = 0; i < iterations; i++) {
            int result = nccl.AllReduce(d_send, d_recv, count, PANTHEON_NCCL_FLOAT32, PANTHEON_NCCL_SUM, comm, (void*)stream);
            if (result != 0) return result;
        }
        return 0;
    };

    // Warm up: the first collectives pay for the library's setup.
    NCCL_CHECK(run_batch(5));
    CHECK(hipStreamSynchronize(stream));

    // --- TIMED RUN ---
    // Every copy has to call the collective the same number of times, or the ones still
    // calling wait for ever. So the copies do not each watch their own clock: after each
    // batch they all-reduce a flag, and stop together as soon as any of them has run out
    // of time.
    std::vector<double> batches;
    auto run_start = std::chrono::steady_clock::now();
    bool stop = false;
    while (!stop) {
        auto start = std::chrono::steady_clock::now();
        NCCL_CHECK(run_batch(batch_iters));
        CHECK(hipStreamSynchronize(stream));
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (seconds > 0) batches.push_back(pantheon_allreduce_bus_bandwidth((double)bytes * batch_iters, seconds, ranks));

        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start).count();
        float out_of_time = (elapsed >= duration && batches.size() >= 3) ? 1.0f : 0.0f;
        CHECK(hipMemcpy(d_flag, &out_of_time, sizeof(float), hipMemcpyHostToDevice));
        NCCL_CHECK(nccl.AllReduce(d_flag, d_flag, 1, PANTHEON_NCCL_FLOAT32, PANTHEON_NCCL_SUM, comm, (void*)stream));
        CHECK(hipStreamSynchronize(stream));
        CHECK(hipMemcpy(&out_of_time, d_flag, sizeof(float), hipMemcpyDeviceToHost));
        stop = out_of_time > 0.0f;
    }
    double bus_bandwidth = pantheon_median(batches);
    double algorithm_bandwidth = pantheon_allreduce_bus_factor(ranks) > 0 ? bus_bandwidth / pantheon_allreduce_bus_factor(ranks) : 0.0;

    std::cout << "  -> Collectives:    " << batches.size() * (size_t)batch_iters << std::endl;
    std::cout << "  -> Algorithm bandwidth: " << algorithm_bandwidth << " GB/s" << std::endl;
    std::cout << "Throughput: " << bus_bandwidth << " GB/s" << std::endl;

    // --- VERIFY ---
    // One more collective into a buffer that was cleared first, so a result left over from
    // the timed run cannot stand for it.
    unsigned int errors = 0;
    if (verify_mode) {
        CHECK(hipMemset(d_recv, 0, bytes));
        NCCL_CHECK(nccl.AllReduce(d_send, d_recv, count, PANTHEON_NCCL_FLOAT32, PANTHEON_NCCL_SUM, comm, (void*)stream));
        CHECK(hipStreamSynchronize(stream));
        if (inject_error) {
            // A fabric that delivers a corrupted sum: change one element of what arrived.
            LAUNCH_KERNEL_ASYNC(inject_allreduce_error_kernel, 1, 1, 0, stream, d_recv, 1337);
            CHECK(hipStreamSynchronize(stream));
        }
        unsigned int* d_err_count = nullptr;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        LAUNCH_KERNEL_ASYNC(verify_allreduce_kernel, blocks, threads, 0, stream, d_recv, count,
                            pantheon_allreduce_expected(ranks), (unsigned int)rank, d_err_count);
        CHECK(hipStreamSynchronize(stream));
        CHECK(hipMemcpy(&errors, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));
        CHECK(hipFree(d_err_count));
        std::cout << "Verification: " << (errors ? "FAIL" : "PASS") << " (" << errors << " errors)" << std::endl;
    }

    NCCL_CHECK(nccl.CommDestroy(comm));
    CHECK(hipStreamDestroy(stream));
    CHECK(hipFree(d_send));
    CHECK(hipFree(d_recv));
    CHECK(hipFree(d_flag));
    return errors ? 1 : 0;
#endif
}
