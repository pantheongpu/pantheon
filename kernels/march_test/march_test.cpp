#include "../common/common.h"
#include "../common/fault_log.h"
#include <chrono>
#include <iostream>
#include <string>

// --- MARCH C- ---
// The standard linear memory test. Six march elements, 10N operations:
//
//   1. |(w0)        write background
//   2. ^(r0, w1)    ascending:  expect 0, write 1
//   3. ^(r1, w0)    ascending:  expect 1, write 0
//   4. v(r0, w1)    descending: expect 0, write 1
//   5. v(r1, w0)    descending: expect 1, write 0
//   6. |(r0)        final read
//
// Together these detect stuck-at, transition, address-decoder and coupling
// faults with defined coverage, which is what separates a march test from
// simply writing a pattern and reading it back.
//
// GPU adaptation and its limit: a march element is order-sensitive, and
// thousands of concurrent threads have no global ordering. Each thread is
// therefore given a private contiguous chunk and marches it alone, so ordering
// holds *within* a chunk. Coupling is consequently detected between addresses
// that share a chunk, not across the whole allocation -- which is the useful
// case anyway, since physically adjacent cells land near each other.
//
// Access is non-temporal so each march element reaches DRAM instead of being
// answered by the cache that the previous element just filled.

__device__ __forceinline__ uint4 march_splat(unsigned int value) {
    return make_uint4(value, value, value, value);
}

__device__ __forceinline__ bool march_differs(const uint4& v, unsigned int expected) {
    return v.x != expected || v.y != expected || v.z != expected || v.w != expected;
}

__global__ void march_c_minus(uint4* data, size_t n, unsigned int* err_count,
                              PantheonFaultLog fault_log, int inject_error) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t threads = (size_t)blockDim.x * gridDim.x;
    if (threads == 0) return;

    // Private contiguous chunk: [begin, end)
    size_t chunk = (n + threads - 1) / threads;   // ceil, so no thread absorbs the remainder
    size_t begin = tid * chunk;
    if (begin >= n) return;
    size_t end = begin + chunk;
    if (end > n) end = n;

    const unsigned int ZERO = 0x00000000u;
    const unsigned int ONES = 0xFFFFFFFFu;

    // 1. |(w0)
    for (size_t i = begin; i < end; ++i) store_nt(&data[i], march_splat(ZERO));

    // Fault injection has to happen *inside* the march. The first element
    // writes the background across the whole array, so anything corrupted
    // beforehand is overwritten before a single read element runs. Corrupting
    // one cell here, in this thread's own chunk, is what the next element sees.
    if (inject_error && tid == 0 && begin < end) {
        uint4 poisoned = load_nt(&data[begin]);
        poisoned.x ^= 0xBADBEEF;
        store_nt(&data[begin], poisoned);
    }

    // 2. ^(r0, w1)
    for (size_t i = begin; i < end; ++i) {
        uint4 v = load_nt(&data[i]);
        if (march_differs(v, ZERO)) {
            pantheon_fault_log_append(fault_log, (unsigned long long)i, ZERO, v.x);
            atomicAdd(err_count, 1u);
        }
        store_nt(&data[i], march_splat(ONES));
    }

    // 3. ^(r1, w0)
    for (size_t i = begin; i < end; ++i) {
        uint4 v = load_nt(&data[i]);
        if (march_differs(v, ONES)) {
            pantheon_fault_log_append(fault_log, (unsigned long long)i, ONES, v.x);
            atomicAdd(err_count, 1u);
        }
        store_nt(&data[i], march_splat(ZERO));
    }

    // 4. v(r0, w1)  -- descending
    for (size_t i = end; i-- > begin; ) {
        uint4 v = load_nt(&data[i]);
        if (march_differs(v, ZERO)) {
            pantheon_fault_log_append(fault_log, (unsigned long long)i, ZERO, v.x);
            atomicAdd(err_count, 1u);
        }
        store_nt(&data[i], march_splat(ONES));
    }

    // 5. v(r1, w0)  -- descending
    for (size_t i = end; i-- > begin; ) {
        uint4 v = load_nt(&data[i]);
        if (march_differs(v, ONES)) {
            pantheon_fault_log_append(fault_log, (unsigned long long)i, ONES, v.x);
            atomicAdd(err_count, 1u);
        }
        store_nt(&data[i], march_splat(ZERO));
    }

    // 6. |(r0)
    for (size_t i = begin; i < end; ++i) {
        uint4 v = load_nt(&data[i]);
        if (march_differs(v, ZERO)) {
            pantheon_fault_log_append(fault_log, (unsigned long long)i, ZERO, v.x);
            atomicAdd(err_count, 1u);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);
    int mem_pct = atoi(argv[3]);

    int block_size = 256;
    int grid_size = 0;
    int kernel_loops = 1;
    int warmup_iters = 0;
    int sync_mode = 2;
    int init_pattern = 0;   // March C- defines its own backgrounds; kept for CLI symmetry.
    std::string fault_map_path;

    bool verify_mode = false;
    int inject_error = 0;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--verify") verify_mode = true;
        if (std::string(argv[i]) == "--inject_error") inject_error = 1;
        if (std::string(argv[i]) == "--block_size" && i+1 < argc) block_size = atoi(argv[++i]);
        if (std::string(argv[i]) == "--grid_size" && i+1 < argc) grid_size = atoi(argv[++i]);
        if (std::string(argv[i]) == "--kernel_loops" && i+1 < argc) kernel_loops = atoi(argv[++i]);
        if (std::string(argv[i]) == "--warmup_iters" && i+1 < argc) warmup_iters = atoi(argv[++i]);
        if (std::string(argv[i]) == "--sync_mode" && i+1 < argc) sync_mode = atoi(argv[++i]);
        if (std::string(argv[i]) == "--init_pattern" && i+1 < argc) {
            if (!pantheon_parse_init_pattern(argv[++i], &init_pattern)) {
                std::cerr << "[PANTHEON] Unknown --init_pattern '" << argv[i] << "'." << std::endl;
                pantheon_print_init_patterns(std::cerr);
                return 1;
            }
        }
        if (std::string(argv[i]) == "--fault_map" && i+1 < argc) fault_map_path = argv[++i];
    }

    normalize_kernel_launch_config(block_size, grid_size, kernel_loops, warmup_iters, sync_mode);
    CHECK(hipSetDevice(gpu_id));

    size_t free_bytes, total_bytes;
    CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    if (mem_pct > 99) mem_pct = 99;
    if (mem_pct < 1) mem_pct = 1;
    size_t alloc_size = (free_bytes * mem_pct) / 100;
    size_t num_elements = alloc_size / sizeof(uint4);
    if (num_elements == 0) { std::cerr << "[PANTHEON] Allocation too small." << std::endl; return 1; }

    uint4* d_data = nullptr;
    CHECK(hipMalloc(&d_data, alloc_size));
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));

    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) {
        num_blocks = init_launch_grid_size(prop, num_elements, block_size);
        auto_grid = true;
    }

    // A march element is only meaningful over an ordered run of addresses. With
    // one element per thread there is no ordering left and the test degenerates
    // into writing a pattern and reading it back, losing exactly the coupling
    // and decoder coverage that justifies a march. Cap the thread count so each
    // one owns a run of at least this many elements.
    const size_t MIN_CHUNK_ELEMENTS = 4096;
    size_t threads = (size_t)num_blocks * block_size;
    size_t max_useful_threads = num_elements / MIN_CHUNK_ELEMENTS;
    if (max_useful_threads < 1) max_useful_threads = 1;
    if (threads > max_useful_threads) {
        num_blocks = (int)((max_useful_threads + block_size - 1) / block_size);
        if (num_blocks < 1) num_blocks = 1;
        threads = (size_t)num_blocks * block_size;
    }
    size_t chunk_elements = (num_elements + threads - 1) / threads;

    unsigned int* d_err_count = nullptr;
    CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
    CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));

    PantheonFaultLog fault_log = fault_map_path.empty()
        ? pantheon_fault_log_none()
        : pantheon_fault_log_create(PANTHEON_FAULT_LOG_DEFAULT_CAPACITY);

    std::cout << "[PANTHEON] GPU " << gpu_id << ": MARCH C- (Linear Memory Test)" << std::endl;
    std::cout << "  -> Coverage:      " << (alloc_size / (1024 * 1024)) << " MiB" << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Chunk/thread:  " << chunk_elements << " elements (ordered run)" << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << " (March C- defines its own backgrounds)" << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t passes = 0;
    do {
        LAUNCH_KERNEL(march_c_minus, num_blocks, block_size, d_data, num_elements, d_err_count, fault_log, inject_error);
        CHECK(hipDeviceSynchronize());
        passes++;
    } while (std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::high_resolution_clock::now() - start_time).count() < duration);

    double seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start_time).count();

    // March C- performs 10 operations per element per pass.
    double march_ops = (double)passes * (double)num_elements * 10.0;
    std::cout << "Throughput: " << march_ops / seconds << " march-ops/s" << std::endl;
    std::cout << "  -> Passes:        " << passes << std::endl;

    unsigned int h_err_count = 0;
    CHECK(hipMemcpy(&h_err_count, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));

    if (!fault_map_path.empty()) {
        pantheon_fault_log_write(fault_log, fault_map_path.c_str(), "march_test", gpu_id, init_pattern);
    }
    pantheon_fault_log_destroy(fault_log);
    CHECK(hipFree(d_err_count));
    CHECK(hipFree(d_data));

    int exit_code = 0;
    if (verify_mode) {
        // The march elements are the verification: every read is a comparison.
        std::cout << "Verification: " << (h_err_count ? "FAIL" : "PASS")
                  << " (" << h_err_count << " march errors over " << passes << " passes)" << std::endl;
        exit_code = h_err_count ? 1 : 0;
    }
    return exit_code;
}
