#include "../common/common.h"
#include "../common/fault_log.h"
#include <vector>
#include <chrono>
#include <string>
#include <iostream>

// --- ATOMIC VIRUS ---
// Sweeps VRAM with atomic Read-Modify-Write operations. Each thread updates
// distinct addresses, so this measures uncontended atomic RMW throughput
// through the L2/memory fabric rather than same-address arbiter contention.
__global__ void atomic_virus_kernel(uint4* data, size_t n, int inject_error, int kernel_loops) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Cast to uint for atomic operations
    unsigned int* atom_ptr = (unsigned int*)data;
    size_t num_ints = n * 4; 

    // Stride loop
    for (size_t i = tid; i < num_ints; i += stride) {
        // Internal loop constraint to dictate duty cycle
        for (int j = 0; j < kernel_loops; j++) {
            // Atomic Add: Forces hardware to lock the cache line, update, and unlock.
            // Extremely taxing on the internal fabric.
            atomicAdd(&atom_ptr[i], 1);
        }
        
        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && i == 1337) {
            // Intentionally corrupt this specific cache line
            atomicAdd(&atom_ptr[i], 100); 
        }
    }
}

// --- VERIFICATION KERNEL ---
__global__ void verify_atomic_kernel(unsigned int* data, size_t num_ints, unsigned int expected_val, unsigned int* err_count, PantheonFaultLog fault_log) {
    // Grid-stride like the stress kernel: the launch grid is capped, so a
    // plain bounds check would silently verify only the first ~67M of a
    // large-VRAM allocation's integers.
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_ints; i += stride) {
        unsigned int actual_val = data[i];

        if (actual_val != expected_val) {
            unsigned int xor_bits = expected_val ^ actual_val;

            // Cap prints so a widescale corruption cannot flood the log.
            if (*err_count < 5) {
                printf("[SDC FAULT][ATOMIC_VIRUS] Atomic RMW Error! Index: %llu | Exp: %u (0x%08x) | Act: %u (0x%08x) | XOR: 0x%08x\n",
                       (unsigned long long)i, expected_val, expected_val, actual_val, actual_val, xor_bits);
            }

            // Record the full location for the fault map; console output is capped.
            pantheon_fault_log_append(fault_log, (unsigned long long)i, expected_val, actual_val);

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
    int kernel_loops = 100;    
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 0;      // 0=Zeroes, 1=Ones

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
    
    // Use 60% VRAM. If too large, latency hides the stress. 
    // If too small, it stays in L1. 60% ensures L2 thrashing.
    size_t alloc_size = (free * mem_pct) / 100;

    // Each kernel launch runs a full grid sweep over the whole buffer; work scales as
    // ~(alloc_bytes/4) * kernel_loops atomics before hipDeviceSynchronize returns.
    // Multi‑hundred‑GB buffers imply trillions of atomics per launch (hours), so external
    // watchdogs (e.g. duration+30s in pantheon.py) always kill the process. Clamp to keep
    // per-launch work in a sane range while still stressing cache arbiters.
    constexpr size_t kAtomicVirusMaxAlloc = 128ULL * 1024 * 1024; // 128 MiB
    if (alloc_size > kAtomicVirusMaxAlloc) {
        std::cout << "[PANTHEON] atomic_virus: clamping GPU buffer to "
                  << (kAtomicVirusMaxAlloc / (1024 * 1024))
                  << " MiB (requested ~"
                  << ((free * mem_pct / 100) / (1024 * 1024))
                  << " MiB would stall one kernel launch far beyond any reasonable timeout). "
                     "Lower --mem or use another test for full-VRAM allocation checks.\n";
        alloc_size = kAtomicVirusMaxAlloc;
    }

    size_t num_elements = alloc_size / 16; 

    uint4* d_data; CHECK(hipMalloc(&d_data, alloc_size));
    
    // --- 2. SET MEMORY INIT PATTERN ---
    if (init_pattern == 1) {
        CHECK(hipMemset(d_data, 0xFF, alloc_size)); 
    } else {
        CHECK(hipMemset(d_data, 0x00, alloc_size)); 
    }

    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    
    // --- 3. EXPLICIT OCCUPANCY ---
    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) { // Fallback to calculation if not provided by AI
        int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
        if (max_blocks_per_sm < 1) max_blocks_per_sm = 16;
        num_blocks = prop.multiProcessorCount * max_blocks_per_sm;
        auto_grid = true;
    }

    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running ATOMIC VIRUS (L2/ROP Stress)..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Mem Alloc (%): " << mem_pct << std::endl;
    std::cout << "  -> Buffer size:   " << (alloc_size / (1024 * 1024)) << " MiB (actual GPU allocation for this test)" << std::endl;
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
            LAUNCH_KERNEL(atomic_virus_kernel, num_blocks, block_size, d_data, num_elements, inject_error, kernel_loops);
        }
        CHECK(hipDeviceSynchronize());

        // --- THE FIX: RESET MEMORY AFTER WARMUP ---
        if (init_pattern == 1) {
            CHECK(hipMemset(d_data, 0xFF, alloc_size));
        } else {
            CHECK(hipMemset(d_data, 0x00, alloc_size));
        }
        std::cout << "[PANTHEON] Warmup complete. Memory reset to baseline." << std::endl;
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t ops_performed = 0;
    unsigned int kernel_launches = 0;

    while (true) {
        // --- 5. ACTIVE LOOP ---
        LAUNCH_KERNEL(atomic_virus_kernel, num_blocks, block_size, d_data, num_elements, inject_error, kernel_loops);
        CHECK(hipDeviceSynchronize());
        
        kernel_launches++;
        
        // 4 ints per uint4 element * kernel loops
        ops_performed += num_elements * 4 * kernel_loops;

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    // Metric: MAPS (Million Atomic Operations Per Second)
    std::cout << "Throughput: " << (ops_performed / 1e6) / seconds << " MAPS" << std::endl;

    // --- VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running L2 Arbiter Verification Pass..." << std::endl;
        
        // Allocate error counter
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        size_t total_ints = num_elements * 4;
        PantheonFaultLog fault_log = fault_map_path.empty()
            ? pantheon_fault_log_none()
            : pantheon_fault_log_create(PANTHEON_FAULT_LOG_DEFAULT_CAPACITY);

        int verify_grid = init_launch_grid_size(prop, total_ints, 256);
        
        // --- 6. VERIFICATION UPDATE ---
        unsigned int expected_val = kernel_launches * kernel_loops;
        if (init_pattern == 1) expected_val += 0xFFFFFFFF; // Account for the initial 0xFF pattern

        LAUNCH_KERNEL(verify_atomic_kernel, verify_grid, 256, (unsigned int*)d_data, total_ints, expected_val, d_err_count, fault_log);
        CHECK(hipDeviceSynchronize());
        
        // Check for errors
        unsigned int h_err_count = 0;
        CHECK(hipMemcpy(&h_err_count, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));
        CHECK(hipFree(d_err_count));

        if (!fault_map_path.empty()) {
            pantheon_fault_log_write(fault_log, fault_map_path.c_str(), "atomic_virus", gpu_id, init_pattern);
            pantheon_fault_log_destroy(fault_log);
        }

        // Say so on success too. Silence is indistinguishable from a
        // verification that never ran, which is how a self-test that
        // could never fail went unnoticed in memory_pc_pingpong.
        std::cout << "Verification: " << (h_err_count ? "FAIL" : "PASS")
                  << " (" << h_err_count << " errors)" << std::endl;
        if (h_err_count > 0) {
            // CRITICAL: Exit with a non-zero code so Python knows the hardware failed
            return 1;
        }
    }

    CHECK(hipFree(d_data));
    return 0;
}
