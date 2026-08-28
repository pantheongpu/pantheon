#include "../common/common.h"
#include "../common/fault_log.h"
#include <vector>
#include <chrono>
#include <string>
#include <iostream>

// --- RAS VALIDATOR (ECC SCRUBBER STRESS) ---
// Writes a pristine payload, then reads it continuously.
// When a physical bit is flipped via AMD RAS injection, this kernel will
// detect Uncorrectable Errors (UEs) or expose the latency jitter of 
// Correctable Errors (CEs) being actively scrubbed by the UMC.

__global__ void init_pristine_memory(uint4* data, size_t n, int inject_error, int init_pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    uint32_t exp;
    if (init_pattern == 0) {
        exp = 0x00000000;
    } else if (init_pattern == 1) {
        exp = 0xFFFFFFFF;
    } else {
        // Default: A highly distinct alternating pattern (1010... and 0101...)
        exp = 0xAAAA5555;
    }
    
    uint4 pristine = make_uint4(exp, exp, exp, exp);
    
    for (size_t i = idx; i < n; i += stride) {
        uint4 local_val = pristine;
        
        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && i == 1337) {
            // Simulate a physical bit-flip that bypassed hardware ECC
            local_val.x ^= 0xBADBEEF; 
        }

        data[i] = local_val;
    }
}

__global__ void ecc_scrub_reader(uint4* data, size_t n, unsigned int* err_count, uint4* sink, int loops, int init_pattern, PantheonFaultLog fault_log) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    unsigned int local_err = 0;
    unsigned int acc = 0;
    
    uint32_t exp;
    if (init_pattern == 0) {
        exp = 0x00000000;
    } else if (init_pattern == 1) {
        exp = 0xFFFFFFFF;
    } else {
        exp = 0xAAAA5555;
    }
    
    // 4x Unroll using proper Coalesced Grid-Stride
    // The faster we read, the harder the UMC has to work if it hits a poisoned block.
    size_t step = stride * 4;
    
    for (int l = 0; l < loops; ++l) {
        for (size_t i = idx; i + stride * 3 < n; i += step) {
           
            // Use Non-Temporal loads to bypass L1/L2 and force physical DRAM fetches
            uint4 r0 = load_nt(&data[i]);
            uint4 r1 = load_nt(&data[i + stride]);
            uint4 r2 = load_nt(&data[i + stride * 2]);
            uint4 r3 = load_nt(&data[i + stride * 3]);

            // Validate all 4 vector components for corruption
            // Record the failing address, not just a tally: a bare count
            // cannot locate a defective cell.
            if (r0.x != exp || r0.y != exp || r0.z != exp || r0.w != exp) {
                local_err++;
                pantheon_fault_log_append(fault_log, (unsigned long long)i, exp, r0.x);
            }
            if (r1.x != exp || r1.y != exp || r1.z != exp || r1.w != exp) {
                local_err++;
                pantheon_fault_log_append(fault_log, (unsigned long long)(i + stride), exp, r1.x);
            }
            if (r2.x != exp || r2.y != exp || r2.z != exp || r2.w != exp) {
                local_err++;
                pantheon_fault_log_append(fault_log, (unsigned long long)(i + stride * 2), exp, r2.x);
            }
            if (r3.x != exp || r3.y != exp || r3.z != exp || r3.w != exp) {
                local_err++;
                pantheon_fault_log_append(fault_log, (unsigned long long)(i + stride * 3), exp, r3.x);
            }
            
            // Consume data to satisfy compiler
            acc += r0.y + r1.z;
        }
    }

    if (local_err > 0) {
        // Log uncorrectable/silent data corruption to host
        atomicAdd(err_count, local_err);
    }
    
    if (acc == 0xDEADBEEF) sink[idx] = make_uint4(acc, 0, 0, 0);
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);
    int mem_pct = atoi(argv[3]);

    // --- PANTHEON CONFIG KNOBS ---
    int block_size = 256;      
    int grid_size = 0;         // 0 = auto-calculate
    int kernel_loops = 10;     // Memory sweeps per launch
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 2;      // 0=Zeroes, 1=Ones, 2=0xAAAA5555

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

    size_t free, total; CHECK(hipMemGetInfo(&free, &total));
    if (mem_pct > 99) mem_pct = 99;
    
    size_t alloc_size = (free * mem_pct) / 100;
    size_t num_elements = alloc_size / 16; 

    scale_warmup_for_large_alloc(warmup_iters, alloc_size);
    scale_kernel_loops_for_large_alloc(kernel_loops, alloc_size, 1);

    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    
    // --- 2. EXPLICIT OCCUPANCY ---
    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) {
        int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
        if (max_blocks_per_sm < 1) max_blocks_per_sm = 16; 
        num_blocks = prop.multiProcessorCount * max_blocks_per_sm;
        auto_grid = true;
    }

    uint4* d_data; CHECK(hipMalloc(&d_data, alloc_size));
    uint4* d_sink; CHECK(hipMalloc(&d_sink, num_blocks * block_size * sizeof(uint4)));
    unsigned int* d_err_count; CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
    CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": RAS VALIDATOR (Active Scrubbing) | " 
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
              
    // --- 3. INITIALIZE PRISTINE MEMORY ---
    std::cout << "[PANTHEON] Init Pristine Pattern on " << alloc_size / (1024*1024) << " MB..." << std::endl;
    int init_grid = init_launch_grid_size(prop, num_elements, block_size);
    LAUNCH_KERNEL(init_pristine_memory, init_grid, block_size, d_data, num_elements, inject_error, init_pattern);
    CHECK(hipDeviceSynchronize());

    PantheonFaultLog fault_log = fault_map_path.empty()
        ? pantheon_fault_log_none()
        : pantheon_fault_log_create(PANTHEON_FAULT_LOG_DEFAULT_CAPACITY);

    // --- 4. WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup iterations..." << std::endl;
        for(int i = 0; i < warmup_iters; i++) {
            // Warmup faults are deliberately not mapped: the counter is reset
            // below, so recording them would leave the map inconsistent with it.
            LAUNCH_KERNEL(ecc_scrub_reader, num_blocks, block_size, d_data, num_elements, d_err_count, d_sink, kernel_loops, init_pattern, pantheon_fault_log_none());
        }
        CHECK(hipDeviceSynchronize());
        
        // CRITICAL: Clear the error counter after warmup so telemetry starts fresh
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t bytes_transferred = 0;

    // --- 5. EXECUTION / VERIFICATION LOOP ---
    // Note: The scrub reader actively verifies the data in real-time, 
    // so we don't need a separate Phase 6 verify pass.
    while (true) {
        LAUNCH_KERNEL(ecc_scrub_reader, num_blocks, block_size, d_data, num_elements, d_err_count, d_sink, kernel_loops, init_pattern, fault_log);
        CHECK(hipDeviceSynchronize());

        bytes_transferred += alloc_size * kernel_loops;
        
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (bytes_transferred / 1e9) / seconds << " GB/s" << std::endl;

    // Verify if any silent corruption leaked past the hardware ECC
    unsigned int h_err_count = 0;
    CHECK(hipMemcpy(&h_err_count, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));

    if (!fault_map_path.empty()) {
        pantheon_fault_log_write(fault_log, fault_map_path.c_str(), "ras_validator", gpu_id, init_pattern);
    }
    pantheon_fault_log_destroy(fault_log);

    
    CHECK(hipFree(d_data)); 
    CHECK(hipFree(d_sink)); 
    CHECK(hipFree(d_err_count));
    
    // Say so on success too. Silence is indistinguishable from a
    // verification that never ran, which is how a self-test that
    // could never fail went unnoticed in memory_pc_pingpong.
    std::cout << "Verification: " << (h_err_count ? "FAIL" : "PASS")
              << " (" << h_err_count << " errors)" << std::endl;
    if (h_err_count > 0) {
        std::cout << "[SDC FAULT][RAS_VALIDATOR] Uncorrectable Data Corruption Detected: " << h_err_count << " bad vector components!" << std::endl;
        return 1; // Always fail if data is physically corrupted
    }

    return 0;
}
