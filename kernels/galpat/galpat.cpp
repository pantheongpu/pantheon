#include "../common/common.h"
#include "../common/fault_log.h"
#include <chrono>
#include <iostream>
#include <string>

// --- GALPAT (GALLOPING PATTERN) ---
// The most thorough classical coupling and address-decoder test: one cell is
// inverted and then "galloped" against every other cell, reading the pair back
// and forth. If writing cell b disturbs cell i, or if the decoder aliases the
// two addresses, the alternating reads catch it.
//
// The cost is O(n^2), which is why this is region-bounded rather than a
// full-VRAM sweep. It is the confirmation step: a linear test such as
// march_test narrows a suspect address, then GALPAT is pointed at a small
// region around it with --region_offset and --region_size.
//
// GPU adaptation: the gallop is order-sensitive, so each thread takes a private
// sub-region and gallops within it. Coupling is therefore detected between
// addresses sharing a sub-region. Widening --region_chunk deepens the coupling
// coverage at quadratic cost; narrowing it trades coverage for speed.

inline __device__ uint32_t galpat_hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

__global__ void init_galpat_region(uint4* data, size_t begin, size_t count, int init_pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < count; i += stride) {
        size_t at = begin + i;
        unsigned int value = pantheon_pattern_value(at, init_pattern, galpat_hash((uint32_t)at));
        store_nt(&data[at], make_uint4(value, value, value, value));
    }
}

__device__ __forceinline__ unsigned int galpat_expected(size_t at, int init_pattern) {
    return pantheon_pattern_value(at, init_pattern, galpat_hash((uint32_t)at));
}

__global__ void galpat_kernel(uint4* data, size_t region_begin, size_t region_count,
                              size_t chunk, int init_pattern,
                              unsigned int* err_count, PantheonFaultLog fault_log) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk == 0) return;

    size_t sub_begin = region_begin + tid * chunk;
    size_t region_end = region_begin + region_count;
    if (sub_begin >= region_end) return;
    size_t sub_end = sub_begin + chunk;
    if (sub_end > region_end) sub_end = region_end;

    for (size_t b = sub_begin; b < sub_end; ++b) {
        unsigned int base = galpat_expected(b, init_pattern);
        unsigned int inverted = ~base;

        // Invert the galloping cell.
        store_nt(&data[b], make_uint4(inverted, inverted, inverted, inverted));

        for (size_t i = sub_begin; i < sub_end; ++i) {
            if (i == b) continue;

            // Read the other cell: it must still hold its background. A change
            // here means writing b disturbed i, or the decoder aliased them.
            uint4 other = load_nt(&data[i]);
            unsigned int other_expected = galpat_expected(i, init_pattern);
            if (other.x != other_expected) {
                pantheon_fault_log_append(fault_log, (unsigned long long)i, other_expected, other.x);
                atomicAdd(err_count, 1u);
            }

            // Read the galloping cell back: it must still hold the inversion.
            uint4 gallop = load_nt(&data[b]);
            if (gallop.x != inverted) {
                pantheon_fault_log_append(fault_log, (unsigned long long)b, inverted, gallop.x);
                atomicAdd(err_count, 1u);
            }
        }

        // Restore the background before moving on.
        store_nt(&data[b], make_uint4(base, base, base, base));
    }
}

__global__ void inject_galpat_error(uint4* data, size_t region_begin, size_t region_count) {
    // Corrupt a cell that the gallop will read as a neighbour before it ever
    // becomes the galloping cell itself.
    if (blockIdx.x == 0 && threadIdx.x == 0 && region_count > 2) {
        data[region_begin + 1].x ^= 0xBADBEEF;
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
    int init_pattern = 2;
    long long region_offset = 0;
    long long region_size = 1 << 20;   // 1M elements = 16 MiB; O(n^2) forbids more by default
    long long region_chunk = 512;      // per-thread gallop window
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
        if (std::string(argv[i]) == "--region_offset" && i+1 < argc) region_offset = atoll(argv[++i]);
        if (std::string(argv[i]) == "--region_size" && i+1 < argc) region_size = atoll(argv[++i]);
        if (std::string(argv[i]) == "--region_chunk" && i+1 < argc) region_chunk = atoll(argv[++i]);
        if (std::string(argv[i]) == "--fault_map" && i+1 < argc) fault_map_path = argv[++i];
    }

    normalize_kernel_launch_config(block_size, grid_size, kernel_loops, warmup_iters, sync_mode);
    if (region_offset < 0) region_offset = 0;
    if (region_size < 2) { std::cerr << "[PANTHEON] Warning: region_size too small; using 1024." << std::endl; region_size = 1024; }
    if (region_chunk < 2) { std::cerr << "[PANTHEON] Warning: region_chunk too small; using 512." << std::endl; region_chunk = 512; }

    CHECK(hipSetDevice(gpu_id));
    size_t free_bytes, total_bytes;
    CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    if (mem_pct > 99) mem_pct = 99;
    if (mem_pct < 1) mem_pct = 1;
    size_t alloc_size = (free_bytes * mem_pct) / 100;
    size_t num_elements = alloc_size / sizeof(uint4);
    if (num_elements == 0) { std::cerr << "[PANTHEON] Allocation too small." << std::endl; return 1; }

    // Clamp the region into the allocation instead of failing: a search narrowing
    // toward a suspect address should not have to know the allocation size.
    if ((size_t)region_offset >= num_elements) region_offset = 0;
    size_t region_count = (size_t)region_size;
    if (region_offset + region_count > num_elements) region_count = num_elements - region_offset;
    if ((size_t)region_chunk > region_count) region_chunk = region_count;

    uint4* d_data = nullptr;
    CHECK(hipMalloc(&d_data, alloc_size));
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));

    size_t needed_threads = (region_count + region_chunk - 1) / region_chunk;
    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) {
        num_blocks = (int)((needed_threads + block_size - 1) / block_size);
        if (num_blocks < 1) num_blocks = 1;
        auto_grid = true;
    }

    std::cout << "[PANTHEON] GPU " << gpu_id << ": GALPAT (Galloping Pattern)" << std::endl;
    std::cout << "  -> Allocation:    " << (alloc_size / (1024 * 1024)) << " MiB" << std::endl;
    std::cout << "  -> Region:        [" << region_offset << ", " << (region_offset + region_count)
              << ") = " << region_count << " elements" << std::endl;
    std::cout << "  -> Gallop Window: " << region_chunk << " elements per thread" << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    int init_grid = init_launch_grid_size(prop, region_count, block_size);
    LAUNCH_KERNEL(init_galpat_region, init_grid, block_size, d_data, (size_t)region_offset, region_count, init_pattern);
    CHECK(hipDeviceSynchronize());

    if (inject_error) {
        LAUNCH_KERNEL(inject_galpat_error, 1, 1, d_data, (size_t)region_offset, region_count);
        CHECK(hipDeviceSynchronize());
    }

    unsigned int* d_err_count = nullptr;
    CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
    CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));

    PantheonFaultLog fault_log = fault_map_path.empty()
        ? pantheon_fault_log_none()
        : pantheon_fault_log_create(PANTHEON_FAULT_LOG_DEFAULT_CAPACITY);

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t passes = 0;
    do {
        LAUNCH_KERNEL(galpat_kernel, num_blocks, block_size, d_data, (size_t)region_offset,
                      region_count, (size_t)region_chunk, init_pattern, d_err_count, fault_log);
        CHECK(hipDeviceSynchronize());
        passes++;
    } while (std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::high_resolution_clock::now() - start_time).count() < duration);

    double seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start_time).count();
    // Each gallop step performs two reads for every (base, other) pair.
    double gallops = (double)passes * (double)needed_threads
                   * (double)region_chunk * (double)(region_chunk - 1) * 2.0;
    std::cout << "Throughput: " << gallops / seconds << " gallop-reads/s" << std::endl;
    std::cout << "  -> Passes:        " << passes << std::endl;

    unsigned int h_err_count = 0;
    CHECK(hipMemcpy(&h_err_count, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));

    if (!fault_map_path.empty()) {
        pantheon_fault_log_write(fault_log, fault_map_path.c_str(), "galpat", gpu_id, init_pattern);
    }
    pantheon_fault_log_destroy(fault_log);
    CHECK(hipFree(d_err_count));
    CHECK(hipFree(d_data));

    int exit_code = 0;
    if (verify_mode) {
        // Every gallop step is itself a comparison, so the walk is the verification.
        std::cout << "Verification: " << (h_err_count ? "FAIL" : "PASS")
                  << " (" << h_err_count << " gallop errors over " << passes << " passes)" << std::endl;
        exit_code = h_err_count ? 1 : 0;
    }
    return exit_code;
}
