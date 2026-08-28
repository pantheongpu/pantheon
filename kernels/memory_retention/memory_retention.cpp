#include "../common/common.h"
#include "../common/fault_log.h"
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

// --- MEMORY RETENTION (CHARGE LEAK OVER TIME) ---
// Writes a payload, leaves it completely untouched for a chosen delay, then
// verifies it.
//
// This is distinct from memory_retention_bake, which keeps the GPU busy with
// ALU work so the payload sits in a *hot* device. That finds cells that fail at
// temperature. This finds cells that fail with *time*: a weak cell leaks charge
// and loses its value between refreshes regardless of temperature, and a marginal
// one only fails past some retention interval. Sweeping --retention_delay is how
// that interval is found.
//
// The idle period has to live inside this binary. Device memory is released
// when the process exits, so an external tool cannot write, wait, and read
// back across separate invocations -- the payload would not survive.

__device__ __forceinline__ uint32_t retention_hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Device-only: the expected value is recomputed inside the kernels rather
// than stored, so a full-VRAM payload needs no reference copy.
__device__ __forceinline__ uint32_t retention_expected(uint32_t index, int init_pattern) {
    if (init_pattern == 0) return 0x00000000u;
    if (init_pattern == 1) return 0xFFFFFFFFu;
    return (index % 2 == 0 ? 0xAAAAAAAAu : 0x55555555u) ^ retention_hash(index);
}

__global__ void write_retention_payload(uint4* data, size_t n, int inject_error, int init_pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        uint32_t value = retention_expected((uint32_t)i, init_pattern);
        if (inject_error && i == 1337) value ^= 0xBADBEEF;
        // Non-temporal so the payload reaches DRAM rather than lingering in
        // cache, where it would never be exposed to a retention failure.
        store_nt(&data[i], make_uint4(value, value, value, value));
    }
}

__global__ void verify_retention_payload(uint4* data, size_t n, unsigned int* err_count,
                                         int init_pattern, PantheonFaultLog fault_log) {
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint32_t expected = retention_expected((uint32_t)i, init_pattern);
        // Read past the caches: a cached hit would return the value we wrote
        // instead of what the DRAM cell still holds.
        uint4 actual = load_nt(&data[i]);
        if (actual.x != expected || actual.y != expected || actual.z != expected || actual.w != expected) {
            if (*err_count < 5) {
                printf("[SDC FAULT][MEMORY_RETENTION] Charge Loss! Index: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n",
                       (unsigned long long)i, expected, actual.x, expected ^ actual.x);
            }
            pantheon_fault_log_append(fault_log, (unsigned long long)i, expected, actual.x);
            atomicAdd(err_count, 1);
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
    int init_pattern = 2;      // 0=Zeroes, 1=Ones, 2=Entropy
    int retention_delay = -1;  // seconds idle; defaults to duration
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
        if (std::string(argv[i]) == "--retention_delay" && i+1 < argc) retention_delay = atoi(argv[++i]);
        if (std::string(argv[i]) == "--fault_map" && i+1 < argc) fault_map_path = argv[++i];
    }

    normalize_kernel_launch_config(block_size, grid_size, kernel_loops, warmup_iters, sync_mode);
    if (retention_delay < 0) retention_delay = duration;
    if (retention_delay < 0) retention_delay = 0;

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

    std::cout << "[PANTHEON] GPU " << gpu_id << ": MEMORY RETENTION (Charge Leak Over Time)" << std::endl;
    std::cout << "  -> Payload Size:  " << (alloc_size / (1024 * 1024)) << " MiB" << std::endl;
    std::cout << "  -> Retention (s): " << retention_delay << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    std::cout << "[PANTHEON] Writing retention payload..." << std::endl;
    LAUNCH_KERNEL(write_retention_payload, num_blocks, block_size, d_data, num_elements, inject_error, init_pattern);
    CHECK(hipDeviceSynchronize());

    // The idle period. Nothing may touch the payload here: any access would
    // refresh the cells being tested and invalidate the measurement.
    std::cout << "[PANTHEON] Holding payload untouched for " << retention_delay << "s..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    if (retention_delay > 0) {
        std::this_thread::sleep_for(std::chrono::seconds(retention_delay));
    }
    double held_seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start_time).count();

    // The useful result is how much memory held its contents for how long, so
    // report the retained payload rather than a bandwidth figure: nothing was
    // transferred during the hold.
    std::cout << "Throughput: " << (double)(alloc_size / (1024 * 1024)) << " retained-MiB" << std::endl;
    std::cout << "  -> Held for:      " << held_seconds << "s" << std::endl;

    int exit_code = 0;
    if (verify_mode) {
        std::cout << "[PANTHEON] Verifying retained payload..." << std::endl;
        unsigned int* d_err_count = nullptr;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));

        PantheonFaultLog fault_log = fault_map_path.empty()
            ? pantheon_fault_log_none()
            : pantheon_fault_log_create(PANTHEON_FAULT_LOG_DEFAULT_CAPACITY);

        int verify_grid = init_launch_grid_size(prop, num_elements, 256);
        LAUNCH_KERNEL(verify_retention_payload, verify_grid, 256, d_data, num_elements, d_err_count, init_pattern, fault_log);
        CHECK(hipDeviceSynchronize());

        unsigned int h_err_count = 0;
        CHECK(hipMemcpy(&h_err_count, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));
        CHECK(hipFree(d_err_count));

        if (!fault_map_path.empty()) {
            pantheon_fault_log_write(fault_log, fault_map_path.c_str(), "memory_retention", gpu_id, init_pattern);
        }
        pantheon_fault_log_destroy(fault_log);

        std::cout << "Verification: " << (h_err_count ? "FAIL" : "PASS")
                  << " (" << h_err_count << " retention errors after " << retention_delay << "s)" << std::endl;
        exit_code = h_err_count ? 1 : 0;
    }

    CHECK(hipFree(d_data));
    return exit_code;
}
