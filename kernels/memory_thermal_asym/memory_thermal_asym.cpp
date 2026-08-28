#include "../common/common.h"
#include <chrono>
#include <string>
#include <iostream>

// --- ASYMMETRIC THERMAL GRADIENT ---
// Hammers a small, isolated memory region while drawing maximum compute power.
__global__ void thermal_asym_kernel(uint4* data, size_t n, int loops, int inject_error, int init_pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float a = 1.0f, b = -1.0f, c = 0.5f;
    
    // Determine the physical bits to write based on fuzzer config
    uint32_t val = (init_pattern == 1) ? 0xFFFFFFFF : 0x00000000;
    uint4 pattern = make_uint4(val, val, val, val);

    for (int i = 0; i < loops; ++i) {
        // 1. Extreme Compute Burn (Heat Generation)
        #pragma unroll 32
        for (int j = 0; j < 32; ++j) {
            #ifdef __HIP_PLATFORM_AMD__
                a = __builtin_fmaf(a, b, c);
                b = __builtin_fmaf(b, c, a);
            #else
                a = fmaf(a, b, c);
                b = fmaf(b, c, a);
            #endif
        }

        // --- DYNAMIC FAULT INJECTION ---
        uint4 write_val = pattern;
        if (inject_error && idx == 1337 && i == 500) {
            // Intentionally corrupt the payload before writing
            write_val.x ^= 0xBADBEEF; 
        }

        // 2. Localized Memory Write (Confined to specific stack)
        // We use modulo 'n' to trap the writes inside the isolated buffer
        size_t write_idx = (idx + i * stride) % n;
        store_nt(&data[write_idx], write_val);
        
        if ((i & 0xF) == 0) a = -a;
    }

    // Dependency sink to prevent DCE
    if (a == 12345.0f) data[0].x = 1;
}

// --- VERIFICATION KERNEL ---
__global__ void verify_thermal_asym_kernel(uint4* data, size_t n, unsigned int* err_count, int init_pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    uint32_t expected_val = (init_pattern == 1) ? 0xFFFFFFFF : 0x00000000;

    for (size_t i = idx; i < n; i += stride) {
        // Non-temporal load to bypass caches
        uint4 val = load_nt(&data[i]);
        
        if (val.x != expected_val || val.y != expected_val || val.z != expected_val || val.w != expected_val) {
            printf("[SDC FAULT][THERMAL_ASYM] Asymmetric Heat Corruption! Index: %llu | Exp: 0x%08x | Act: {0x%08x, 0x%08x, 0x%08x, 0x%08x}\n",
                   (unsigned long long)i, expected_val, val.x, val.y, val.z, val.w);
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
    int kernel_loops = 1000;   // Balances ALU burn with physical memory writes
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 1;      // 0=Zeroes, 1=Ones (Default to 1 to maximize power delivery stress)

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

    // Force allocation to exactly 16GB (or less depending on mem_pct)
    // This isolates the traffic to typically 1 or 2 physical Memory stacks.
    size_t free, total; CHECK(hipMemGetInfo(&free, &total));
    if (mem_pct > 99) mem_pct = 99;
    
    size_t target_size = 16ULL * 1024 * 1024 * 1024; 
    size_t alloc_size = (free > target_size) ? target_size : (free * mem_pct) / 100;
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
    if (num_blocks == 0) {
        int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
        if (max_blocks_per_sm < 1) max_blocks_per_sm = 16; 
        num_blocks = prop.multiProcessorCount * max_blocks_per_sm;
        auto_grid = true;
    }

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": ASYMMETRIC THERMAL STRESS | " 
              << (alloc_size / 1e9) << " GB Target..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Mem Alloc (%): " << mem_pct << std::endl;
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
            LAUNCH_KERNEL(thermal_asym_kernel, num_blocks, block_size, d_data, num_elements, kernel_loops, inject_error, init_pattern);
        }
        CHECK(hipDeviceSynchronize());
        
        // Reset the memory after warmup to ensure verification runs on a clean buffer
        if (init_pattern == 1) {
            CHECK(hipMemset(d_data, 0xFF, alloc_size)); 
        } else {
            CHECK(hipMemset(d_data, 0x00, alloc_size)); 
        }
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t ops_performed = 0;

    // --- 5. ACTIVE LOOP ---
    while (true) {
        LAUNCH_KERNEL(thermal_asym_kernel, num_blocks, block_size, d_data, num_elements, kernel_loops, inject_error, init_pattern);
        CHECK(hipDeviceSynchronize());
        
        ops_performed += (size_t)num_blocks * block_size * kernel_loops * 64; // Approx FLOPs
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (ops_performed / 1e12) / seconds << " TFLOPS" << std::endl;

    // --- 6. VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running Asymmetric Thermal Verification Pass..." << std::endl;
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        int verify_blocks = (num_elements + 255) / 256;
        // Launch a 1D grid over the entire memory buffer
        LAUNCH_KERNEL(verify_thermal_asym_kernel, verify_blocks, 256, d_data, num_elements, d_err_count, init_pattern);
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
            // Exit with non-zero code to fail the CI step
            return 1;
        }
    }

    CHECK(hipFree(d_data));
    return 0;
}
