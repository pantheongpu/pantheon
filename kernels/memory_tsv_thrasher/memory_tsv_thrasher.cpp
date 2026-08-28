#include "../common/common.h"
#include <vector>
#include <chrono>
#include <string>
#include <iostream>

// --- TSV TOGGLE-RATE THRASHER ---
// Alternates between driving the bus completely HIGH (0xFF) and completely LOW (0x00).
// This maximizes Data Bus Inversion (DBI) stress, inducing massive di/dt (current spikes) 
// and electrical crosstalk on the physical interposer TSVs.

__global__ 
void memory_tsv_thrasher_kernel(uint4* data, size_t n, int loops, int inject_error) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // The electrical payload: 
    // p1 forces all 128 bits of the vector to HIGH voltage
    // p0 forces all 128 bits of the vector to LOW voltage
    uint4 p1 = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    uint4 p0 = make_uint4(0x00000000, 0x00000000, 0x00000000, 0x00000000);

    // 16x Unroll to keep the Unified Memory Controllers (UMCs) fully saturated
    size_t step = stride * 16;
    
    for (int l = 0; l < loops; ++l) {
        for (size_t i = idx; i + step < n; i += step) {
            
            uint4 local_p1 = p1;
            // --- DYNAMIC FAULT INJECTION ---
            if (inject_error && idx == 1337) {
                // Intentionally corrupt the HIGH voltage vector to simulate a dropped TSV trace
                local_p1.z ^= 0xBADBEEF; 
            }

            // We write in alternating waves to force the maximum number of physical 
            // trace state-changes per clock cycle.
            store_nt(&data[i],              local_p1);
            store_nt(&data[i + stride],     p0);
            store_nt(&data[i + stride*2],   local_p1);
            store_nt(&data[i + stride*3],   p0);
            
            store_nt(&data[i + stride*4],   local_p1);
            store_nt(&data[i + stride*5],   p0);
            store_nt(&data[i + stride*6],   local_p1);
            store_nt(&data[i + stride*7],   p0);
            
            store_nt(&data[i + stride*8],   local_p1);
            store_nt(&data[i + stride*9],   p0);
            store_nt(&data[i + stride*10],  local_p1);
            store_nt(&data[i + stride*11],  p0);
            
            store_nt(&data[i + stride*12],  local_p1);
            store_nt(&data[i + stride*13],  p0);
            store_nt(&data[i + stride*14],  local_p1);
            store_nt(&data[i + stride*15],  p0);
        }
    }
}

// --- VERIFICATION KERNEL ---
__global__ void verify_tsv_thrasher_kernel(uint4* data, size_t n, unsigned int* err_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t step = stride * 16;

    uint32_t e1 = 0xFFFFFFFF;
    uint32_t e0 = 0x00000000;

    // Mirror the exact stride logic of the write kernel to verify safely
    for (size_t i = idx; i + step < n; i += step) {
        for (int k = 0; k < 16; ++k) {
            size_t offset = i + stride * k;
            
            // Use non-temporal load to ensure we read physical Memory, not L2 cache
            uint4 val = load_nt(&data[offset]);
            
            // Even unrolls are p1 (HIGH), Odd unrolls are p0 (LOW)
            uint32_t expected = (k % 2 == 0) ? e1 : e0;
            
            if (val.x != expected || val.y != expected || val.z != expected || val.w != expected) {
                printf("[SDC FAULT][TSV_THRASHER] Interposer/DBI Error! Index: %llu | Exp: 0x%08x | Act: {0x%08x, 0x%08x, 0x%08x, 0x%08x}\n",
                       (unsigned long long)offset, expected, val.x, val.y, val.z, val.w);
                atomicAdd(err_count, 1);
            }
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
    int kernel_loops = 10;     // Sweeps the TSV trace lines per launch
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 0;      // 0=Zeroes, 1=Ones

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
    
    // Size the allocation
    size_t alloc_size = (free * mem_pct) / 100;
    size_t num_elements = alloc_size / 16; // 16 bytes per uint4

    uint4* d_data; 
    CHECK(hipMalloc(&d_data, alloc_size));
    
    // --- 2. SET MEMORY INIT PATTERN ---
    // Initialize to prevent false positives at the unwritten tail end
    if (init_pattern == 1) {
        CHECK(hipMemset(d_data, 0xFF, alloc_size)); 
    } else {
        CHECK(hipMemset(d_data, 0x00, alloc_size)); 
    }

    // --- 3. EXPLICIT OCCUPANCY ---
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) {
        int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
        if (max_blocks_per_sm < 1) max_blocks_per_sm = 20; // Oversubscribe for memory workloads
        num_blocks = prop.multiProcessorCount * max_blocks_per_sm;
        auto_grid = true;
    }

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": TSV TOGGLE-RATE THRASHER (Interposer Stress) | " 
              << mem_pct << "% VRAM..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << " (Overridden: Forced Toggle Pattern)" << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- 4. WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup iterations..." << std::endl;
        for(int i = 0; i < warmup_iters; i++) {
            LAUNCH_KERNEL(memory_tsv_thrasher_kernel, num_blocks, block_size, d_data, num_elements, kernel_loops, inject_error);
        }
        CHECK(hipDeviceSynchronize());
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t bytes_transferred = 0;

    // --- 5. ACTIVE LOOP ---
    while (true) {
        LAUNCH_KERNEL(memory_tsv_thrasher_kernel, num_blocks, block_size, d_data, num_elements, kernel_loops, inject_error);
        CHECK(hipDeviceSynchronize());
        
        bytes_transferred += alloc_size * kernel_loops;
        
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (bytes_transferred / 1e9) / seconds << " GB/s" << std::endl;

    // --- 6. VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running TSV Toggle-Rate Verification Pass..." << std::endl;
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        // Launch using the exact same grid dimensions to perfectly align the stride offsets
        LAUNCH_KERNEL(verify_tsv_thrasher_kernel, num_blocks, block_size, d_data, num_elements, d_err_count);
        CHECK(hipDeviceSynchronize());
        
        unsigned int h_err_count = 0;
        CHECK(hipMemcpy(&h_err_count, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));
        CHECK(hipFree(d_err_count));

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
