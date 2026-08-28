#include "../common/common.h"
#include "../common/fault_log.h"
#include <chrono>
#include <iostream>
#include <string>

// --- MEMORY HAMMER (DISTURB TEST) ---
// Repeatedly activates a pair of aggressor addresses and then checks whether
// anything *else* changed. Nothing here ever writes to the payload during the
// hammer phase: every reported error is a value that changed without being
// written, which is what a disturb fault is.
//
// The classic double-sided arrangement is used -- two aggressors sandwiching a
// victim -- because a victim between two hammered rows is disturbed far more
// than one beside a single aggressor.
//
// Honest limitation: a GPU exposes no virtual-to-physical mapping, let alone
// the bank/row/column geometry, and the memory controller scrambles addresses
// anyway. The aggressor spacing is therefore a *statistical* attempt to land on
// different rows, not a targeted one. That makes this a probabilistic test:
// finding a disturb error is meaningful, finding none proves less than it would
// on a platform where rows can be addressed directly. Sweeping --hammer_stride
// is how the useful spacings get found.
//
// Access is non-temporal throughout: a cached read never reaches the DRAM row
// and therefore never activates anything.

// A hammer read must reach DRAM. The shared load_nt uses a streaming load that
// bypasses L1 but still hits L2, and the aggressor working set is small enough
// to live entirely in L2 -- so every activation after the first was answered by
// cache and disturbed nothing. This load bypasses L1 *and* L2 so each hit
// actually opens a row.
__device__ __forceinline__ uint4 hammer_load(const uint4* addr) {
    uint4 ret;
#ifdef PANTHEON_MOCK
    ret = *(const uint4*)addr;
#elif defined(__CUDACC__)
    asm volatile("ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
                 : "l"(addr) : "memory");
#elif defined(__HIP_PLATFORM_AMD__)
    const unsigned int* p = (const unsigned int*)addr;
    ret.x = __builtin_nontemporal_load(&p[0]);
    ret.y = __builtin_nontemporal_load(&p[1]);
    ret.z = __builtin_nontemporal_load(&p[2]);
    ret.w = __builtin_nontemporal_load(&p[3]);
#else
    ret = *(const uint4*)addr;
#endif
    return ret;
}

inline __device__ uint32_t hammer_hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

__global__ void init_hammer_payload(uint4* data, size_t n, int init_pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        unsigned int value = pantheon_pattern_value(i, init_pattern, hammer_hash((uint32_t)i));
        store_nt(&data[i], make_uint4(value, value, value, value));
    }
}

// Hammers aggressor pairs. Read-only by construction.
__global__ void hammer_kernel(uint4* data, size_t n, size_t hammer_stride,
                              int activations, int pairs, size_t pair_span,
                              unsigned int* sink) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t threads = (size_t)blockDim.x * gridDim.x;
    if (hammer_stride == 0 || n <= 2 * hammer_stride || pairs < 1) return;

    unsigned int accumulator = 0;
    for (size_t base = tid; base + 2 * hammer_stride < n; base += threads * (2 * hammer_stride + 1)) {
        for (int hit = 0; hit < activations; ++hit) {
            // Every pair is touched once per round, and the pairs are spread
            // across the whole allocation. A thread that hammered a single pair
            // in a tight loop would be answered by L2 forever -- the aggressor
            // footprint has to exceed L2 before a re-read can reach DRAM and
            // actually open a row. See --hammer_pairs.
            for (int p = 0; p < pairs; ++p) {
                size_t aggressor_low = base + (size_t)p * pair_span;
                size_t aggressor_high = aggressor_low + 2 * hammer_stride;
                if (aggressor_high >= n) break;
                accumulator ^= hammer_load(&data[aggressor_low]).x
                             ^ hammer_load(&data[aggressor_high]).x;
            }
        }
    }

    // Consume the reads so the compiler cannot delete the hammer loop.
    if (accumulator == 0xDEADBEEF) sink[tid % 1024] = accumulator;
}

__global__ void verify_hammer_payload(uint4* data, size_t n, unsigned int* err_count,
                                      int init_pattern, PantheonFaultLog fault_log) {
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        unsigned int expected = pantheon_pattern_value(i, init_pattern, hammer_hash((uint32_t)i));
        uint4 actual = load_nt(&data[i]);
        if (actual.x != expected || actual.y != expected || actual.z != expected || actual.w != expected) {
            if (*err_count < 5) {
                printf("[SDC FAULT][MEMORY_HAMMER] Disturb Error! Index: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n",
                       (unsigned long long)i, expected, actual.x, expected ^ actual.x);
            }
            pantheon_fault_log_append(fault_log, (unsigned long long)i, expected, actual.x);
            atomicAdd(err_count, 1u);
        }
    }
}

__global__ void inject_hammer_error(uint4* data, size_t n, size_t hammer_stride) {
    // Corrupt a victim address: the cell between an aggressor pair, which is
    // exactly where a real disturb fault would appear.
    if (blockIdx.x == 0 && threadIdx.x == 0 && n > hammer_stride) {
        data[hammer_stride].x ^= 0xBADBEEF;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);
    int mem_pct = atoi(argv[3]);

    int block_size = 256;
    int grid_size = 0;
    int kernel_loops = 4096;   // activations per aggressor pair per pass
    int warmup_iters = 0;
    int sync_mode = 2;
    int init_pattern = 2;
    // 32768 uint4 elements = 512 KiB apart, chosen to clear typical row and
    // channel interleaving so the pair is likely to land on distinct rows.
    long long hammer_stride = 32768;
    int hammer_pairs = 8;
    const int MAX_HAMMER_PAIRS = 256;
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
        if (std::string(argv[i]) == "--hammer_stride" && i+1 < argc) hammer_stride = atoll(argv[++i]);
        if (std::string(argv[i]) == "--hammer_pairs" && i+1 < argc) hammer_pairs = atoi(argv[++i]);
        if (std::string(argv[i]) == "--fault_map" && i+1 < argc) fault_map_path = argv[++i];
    }

    normalize_kernel_launch_config(block_size, grid_size, kernel_loops, warmup_iters, sync_mode);
    if (hammer_pairs < 1) {
        std::cerr << "[PANTHEON] Warning: invalid hammer_pairs; using 8." << std::endl;
        hammer_pairs = 8;
    } else if (hammer_pairs > MAX_HAMMER_PAIRS) {
        // Each pair is walked on every activation round, so this multiplies
        // kernel runtime directly -- a large value simply hangs. There is also
        // nothing to gain: the point of extra pairs is to push the aggressor
        // set past L2, and a handful already clears it by an order of
        // magnitude.
        std::cerr << "[PANTHEON] Warning: hammer_pairs " << hammer_pairs
                  << " would make each launch unbounded; clamping to "
                  << MAX_HAMMER_PAIRS << "." << std::endl;
        hammer_pairs = MAX_HAMMER_PAIRS;
    }
    if (hammer_stride < 1) {
        std::cerr << "[PANTHEON] Warning: invalid hammer_stride; using 32768." << std::endl;
        hammer_stride = 32768;
    }

    CHECK(hipSetDevice(gpu_id));
    size_t free_bytes, total_bytes;
    CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    if (mem_pct > 99) mem_pct = 99;
    if (mem_pct < 1) mem_pct = 1;
    size_t alloc_size = (free_bytes * mem_pct) / 100;
    size_t num_elements = alloc_size / sizeof(uint4);
    if (num_elements <= (size_t)(2 * hammer_stride)) {
        std::cerr << "[PANTHEON] Allocation too small for hammer_stride "
                  << hammer_stride << "; increase --mem or lower --hammer_stride." << std::endl;
        return 1;
    }

    uint4* d_data = nullptr;
    unsigned int* d_sink = nullptr;
    CHECK(hipMalloc(&d_data, alloc_size));
    CHECK(hipMalloc(&d_sink, 1024 * sizeof(unsigned int)));
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));

    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) {
        num_blocks = prop.multiProcessorCount * 8;
        if (num_blocks < 1) num_blocks = 1;
        auto_grid = true;
    }

    std::cout << "[PANTHEON] GPU " << gpu_id << ": MEMORY HAMMER (Row Disturb)" << std::endl;
    std::cout << "  -> Payload Size:  " << (alloc_size / (1024 * 1024)) << " MiB" << std::endl;
    std::cout << "  -> Hammer Stride: " << hammer_stride << " elements ("
              << (hammer_stride * sizeof(uint4) / 1024) << " KiB)" << std::endl;
    std::cout << "  -> Activations:   " << kernel_loops << " per aggressor pair per pass" << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    size_t pair_span = num_elements / (size_t)hammer_pairs;
    if (pair_span == 0) pair_span = 1;
    // Aggressor lines actually in flight. If this fits in L2 the hammer never
    // reaches DRAM, so surface it next to the cache size rather than hiding it.
    // Consecutive threads read consecutive elements, so they coalesce into
    // shared lines; counting whole lines per thread would overstate this ~8x.
    double footprint_mb = (double)num_blocks * block_size * hammer_pairs * 2.0
                        * sizeof(uint4) / (1024.0 * 1024.0);
    std::cout << "  -> Hammer Pairs:  " << hammer_pairs
              << " per thread (span " << pair_span << " elements)" << std::endl;
    std::cout << "  -> Aggressor Set: ~" << (long long)footprint_mb << " MiB vs L2 "
              << (prop.l2CacheSize / (1024 * 1024)) << " MiB"
              << (footprint_mb * 1024.0 * 1024.0 > prop.l2CacheSize
                    ? " (exceeds L2)" : " (FITS IN L2 - reads will be cache hits)")
              << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    int init_grid = init_launch_grid_size(prop, num_elements, block_size);
    LAUNCH_KERNEL(init_hammer_payload, init_grid, block_size, d_data, num_elements, init_pattern);
    CHECK(hipDeviceSynchronize());

    std::cout << "[PANTHEON] Hammering..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t passes = 0;
    do {
        LAUNCH_KERNEL(hammer_kernel, num_blocks, block_size, d_data, num_elements,
                      (size_t)hammer_stride, kernel_loops, hammer_pairs, pair_span, d_sink);
        CHECK(hipDeviceSynchronize());
        passes++;
    } while (std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::high_resolution_clock::now() - start_time).count() < duration);

    double seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start_time).count();
    // Two aggressors are read per pair per round. This counts issued reads, not
    // confirmed DRAM row activations -- without the physical address mapping we
    // cannot prove a read opened a row, so the metric is deliberately named for
    // what it actually measures.
    double reads = (double)passes * (double)num_blocks * block_size
                 * kernel_loops * hammer_pairs * 2.0;
    std::cout << "Throughput: " << reads / seconds << " aggressor-reads/s" << std::endl;
    std::cout << "  -> Passes:        " << passes << std::endl;

    int exit_code = 0;
    if (verify_mode) {
        if (inject_error) {
            LAUNCH_KERNEL(inject_hammer_error, 1, 1, d_data, num_elements, (size_t)hammer_stride);
            CHECK(hipDeviceSynchronize());
        }
        std::cout << "[PANTHEON] Checking every cell for disturb..." << std::endl;
        unsigned int* d_err_count = nullptr;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));

        PantheonFaultLog fault_log = fault_map_path.empty()
            ? pantheon_fault_log_none()
            : pantheon_fault_log_create(PANTHEON_FAULT_LOG_DEFAULT_CAPACITY);

        int verify_grid = init_launch_grid_size(prop, num_elements, 256);
        LAUNCH_KERNEL(verify_hammer_payload, verify_grid, 256, d_data, num_elements, d_err_count, init_pattern, fault_log);
        CHECK(hipDeviceSynchronize());

        unsigned int h_err_count = 0;
        CHECK(hipMemcpy(&h_err_count, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));
        CHECK(hipFree(d_err_count));

        if (!fault_map_path.empty()) {
            pantheon_fault_log_write(fault_log, fault_map_path.c_str(), "memory_hammer", gpu_id, init_pattern);
        }
        pantheon_fault_log_destroy(fault_log);

        std::cout << "Verification: " << (h_err_count ? "FAIL" : "PASS")
                  << " (" << h_err_count << " disturbed cells)" << std::endl;
        exit_code = h_err_count ? 1 : 0;
    }

    CHECK(hipFree(d_sink));
    CHECK(hipFree(d_data));
    return exit_code;
}
