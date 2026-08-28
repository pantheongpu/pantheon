#include "../common/common.h"
#include "../common/fault_log.h"
#include <vector>
#include <chrono>
#include <string>
#include <iostream>

// --- ENTROPY GENERATOR ---
// Generates uncompressible noise to blind the memory compressor (DCC)
// This forces the physical VRAM bus to transmit every single bit.
inline __device__ uint32_t fast_hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// --- WRITE KERNEL (Rail-to-Rail + Entropy) ---
__global__ void memory_write_kernel(uint4* data, size_t n, int loops, int inject_error, int init_pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (int l = 0; l < loops; ++l) {
        // Proper Coalesced Grid-Stride Loop
        for (size_t i = idx; i < n; i += stride * 16) {
            #pragma unroll
            for (int u = 0; u < 16; ++u) {
                size_t offset = i + stride * u;
                if (offset < n) {
                    uint32_t final_val;

                    if (init_pattern == 0) {
                        final_val = 0x00000000;
                    } else if (init_pattern == 1) {
                        final_val = 0xFFFFFFFF;
                    } else {
                        // The base selects toggle behaviour (2 = crosstalk,
                        // 3 = rail-to-rail); the hash keeps the cache line
                        // uncompressible either way.
                        uint32_t entropy = fast_hash((uint32_t)offset);
                        final_val = pantheon_pattern_value(offset, init_pattern, entropy);
                    }

                    // --- DYNAMIC FAULT INJECTION ---
                    if (inject_error && offset == 1337) {
                        // Intentionally corrupt the payload before writing to physical memory
                        final_val ^= 0xBADBEEF;
                    }

                    uint4 my_pattern = make_uint4(final_val, final_val, final_val, final_val);

                    // Non-Temporal Store: Bypasses the L2 cache directly to VRAM
                    store_nt(&data[offset], my_pattern);
                }
            }
        }
    }
}

// --- VERIFICATION KERNEL ---
__global__ void verify_memory_write_kernel(uint4* data, size_t n, unsigned int* err_count, int init_pattern, PantheonFaultLog fault_log) {
    // Grid-stride like the init kernel: the launch grid is capped, so a plain
    // bounds check would silently verify only the first ~67M elements of a
    // large-VRAM allocation.
    size_t stride = blockDim.x * gridDim.x;
    for (size_t offset = blockIdx.x * blockDim.x + threadIdx.x; offset < n; offset += stride) {
        uint4 actual = data[offset];

        uint32_t expected_val;
        if (init_pattern == 0) {
            expected_val = 0x00000000;
        } else if (init_pattern == 1) {
            expected_val = 0xFFFFFFFF;
        } else {
            // Recalculate the exact entropy and base for this memory address
            uint32_t entropy = fast_hash((uint32_t)offset);
            expected_val = pantheon_pattern_value(offset, init_pattern, entropy);
        }

        // Verify all 4 vector components
        if (actual.x != expected_val || actual.y != expected_val || actual.z != expected_val || actual.w != expected_val) {
            uint32_t xor_val = actual.x ^ expected_val;
            // Cap prints so a widescale corruption cannot flood the log.
            if (*err_count < 5) {
                printf("[SDC FAULT][Memory_WRITE] Write/Retention Error! VA: %p | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n",
                       (void*)&data[offset], expected_val, actual.x, xor_val);
            }
            
            // Record the full location for the fault map; console output is capped.
            pantheon_fault_log_append(fault_log, (unsigned long long)offset, expected_val, actual.x);

            // Report the error back to the host
            atomicAdd(err_count, 1);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);
    int mem_pct = atoi(argv[3]);

    // --- PANTHEON CONFIG KNOBS ---
    int block_size = 256;      
    int grid_size = 0;         // 0 = auto-calculate
    int kernel_loops = 10;     // Passes over the entire VRAM buffer per launch
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    // Default 3 preserves this workload's historical rail-to-rail background.
    int init_pattern = 3;      // 0=Zeroes, 1=Ones, 2=Crosstalk+entropy, 3=Rail-to-rail+entropy

    std::string fault_map_path; // --fault_map <file>: every failing address

    bool verify_mode = false;
    int inject_error = 0;

    // Parse arguments
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

    // --- 1. SET SYNC MODE ---
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__) || defined(__HIP_PLATFORM_HCC__)
    unsigned int sync_flag = hipDeviceScheduleBlockingSync;
    if (sync_mode == 0) sync_flag = hipDeviceScheduleSpin;
    else if (sync_mode == 1) sync_flag = hipDeviceScheduleYield;
    hipSetDeviceFlags(sync_flag);
#elif defined(__CUDACC__)
    unsigned int sync_flag = cudaDeviceScheduleBlockingSync;
    if (sync_mode == 0) sync_flag = cudaDeviceScheduleSpin;
    else if (sync_mode == 1) sync_flag = cudaDeviceScheduleYield;
    cudaSetDeviceFlags(sync_flag);
#else
    // MOCK PLATFORM: CPU execution doesn't require hardware scheduling flags
#endif

    CHECK(hipSetDevice(gpu_id));

    size_t free, total;
    CHECK(hipMemGetInfo(&free, &total));
    if (mem_pct > 99) mem_pct = 99;
    size_t alloc_size = (free * mem_pct) / 100;
    size_t num_elements = alloc_size / 16;

    scale_warmup_for_large_alloc(warmup_iters, alloc_size);
    scale_kernel_loops_for_large_alloc(kernel_loops, alloc_size, 1);

    uint4* d_data;
    CHECK(hipMalloc(&d_data, alloc_size));

    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));

    // --- 2. EXPLICIT OCCUPANCY ---
    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) {
        int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
        if (max_blocks_per_sm < 1) max_blocks_per_sm = 20;
        num_blocks = prop.multiProcessorCount * max_blocks_per_sm;
        auto_grid = true;
    }

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Memory WRITE (NT Store | Rail-to-Rail) | " 
              << mem_pct << "% VRAM..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- 4. WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup iterations..." << std::endl;
        for(int i = 0; i < warmup_iters; i++) {
            LAUNCH_KERNEL(memory_write_kernel, num_blocks, block_size, d_data, num_elements, kernel_loops, inject_error, init_pattern);
        }
        CHECK(hipDeviceSynchronize());
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t bytes_transferred = 0;

    // --- 5. ACTIVE LOOP ---
    while (true) {
        LAUNCH_KERNEL(memory_write_kernel, num_blocks, block_size, d_data, num_elements, kernel_loops, inject_error, init_pattern);
        CHECK(hipDeviceSynchronize());

        bytes_transferred += alloc_size * kernel_loops;
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (bytes_transferred / 1e9) / seconds << " GB/s" << std::endl;

    // --- 6. VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running Memory Write Verification Pass..." << std::endl;
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        PantheonFaultLog fault_log = fault_map_path.empty()
        
            ? pantheon_fault_log_none()
        
            : pantheon_fault_log_create(PANTHEON_FAULT_LOG_DEFAULT_CAPACITY);

        
        int verify_grid = init_launch_grid_size(prop, num_elements, 256);
        LAUNCH_KERNEL(verify_memory_write_kernel, verify_grid, 256, d_data, num_elements, d_err_count, init_pattern, fault_log);
        CHECK(hipDeviceSynchronize());
        
        unsigned int h_err_count = 0;
        CHECK(hipMemcpy(&h_err_count, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));
        CHECK(hipFree(d_err_count));

        if (!fault_map_path.empty()) {
            pantheon_fault_log_write(fault_log, fault_map_path.c_str(), "memory_write", gpu_id, init_pattern);
            pantheon_fault_log_destroy(fault_log);
        }

        // Say so on success too. Silence is indistinguishable from a
        // verification that never ran, which is how a self-test that
        // could never fail went unnoticed in memory_pc_pingpong.
        std::cout << "Verification: " << (h_err_count ? "FAIL" : "PASS")
                  << " (" << h_err_count << " errors)" << std::endl;
        if (h_err_count > 0) {
            // CRITICAL: Exit with non-zero code to fail the CI step
            return 1;
        }
    }

    CHECK(hipFree(d_data));
    return 0;
}
