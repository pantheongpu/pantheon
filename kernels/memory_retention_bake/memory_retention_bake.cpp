#include "../common/common.h"
#include <chrono>
#include <iostream>
#include <string>

// --- DATA RETENTION BAKE ---
__global__ void write_payload_kernel(uint4* data, size_t n, int inject_error, int init_pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Determine the baseline retention payload
    uint32_t v1, v2, v3, v4;
    if (init_pattern == 0) {
        v1 = v2 = v3 = v4 = 0x00000000;
    } else if (init_pattern == 1) {
        v1 = v2 = v3 = v4 = 0xFFFFFFFF;
    } else {
        // Default: Complex alternating bit-pattern to test distinct cell states
        v1 = 0x12345678; 
        v2 = 0x9ABCDEF0; 
        v3 = 0x0FEDCBA9; 
        v4 = 0x87654321;
    }
    uint4 pattern = make_uint4(v1, v2, v3, v4);
    
    for (size_t i = idx; i < n; i += stride) {
        uint4 val = pattern;
        
        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && i == 1337) {
            // Intentionally corrupt the payload before baking
            val.x ^= 0xBADBEEF; 
        }
        
        data[i] = val;
    }
}

// Pure ALU virus to generate heat without touching memory
__global__ void bake_kernel(int iters, float* sink) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = 1.0f, b = -1.0f, c = 0.5f;
    
    for (int i = 0; i < iters; ++i) {
        #pragma unroll 32
        for (int j = 0; j < 32; ++j) {
            #ifdef __HIP_PLATFORM_AMD__
                // Use fmaf for explicitly 32-bit float math (fma defaults to double)
                a = __builtin_fmaf(a, b, c); 
                b = __builtin_fmaf(b, c, a);
            #else
                a = fmaf(a, b, c); 
                b = fmaf(b, c, a);
            #endif
        }
        if ((i & 0xF) == 0) a = -a;
    }
    
    // Sink accumulator to prevent Dead Code Elimination
    if (a == 12345.0f) sink[tid] = 1.0f;
}

__global__ void verify_payload_kernel(uint4* data, size_t n, unsigned int* err_count, int init_pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    uint32_t e1, e2, e3, e4;
    if (init_pattern == 0) {
        e1 = e2 = e3 = e4 = 0x00000000;
    } else if (init_pattern == 1) {
        e1 = e2 = e3 = e4 = 0xFFFFFFFF;
    } else {
        e1 = 0x12345678; 
        e2 = 0x9ABCDEF0; 
        e3 = 0x0FEDCBA9; 
        e4 = 0x87654321;
    }
    
    for (size_t i = idx; i < n; i += stride) {
        // Use non-temporal load to bypass L1/L2 and read physical memory
        uint4 val = load_nt(&data[i]);
        
        if (val.x != e1 || val.y != e2 || val.z != e3 || val.w != e4) {
            
            printf("[SDC FAULT][Memory_RETENTION_BAKE] Thermal Charge Leakage! Index: %llu | Exp: {0x%08x, 0x%08x, 0x%08x, 0x%08x} | Act: {0x%08x, 0x%08x, 0x%08x, 0x%08x}\n",
                   (unsigned long long)i, e1, e2, e3, e4, val.x, val.y, val.z, val.w);
            
            atomicAdd(err_count, 1);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]); // Duration acts as the "Bake Time"
    int mem_pct = atoi(argv[3]);

    // --- PANTHEON CONFIG KNOBS ---
    int block_size = 256;      
    int grid_size = 0;         // 0 = auto-calculate
    int kernel_loops = 50000;  // Maps directly to the ALU 'iters' for the thermal bake
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 2;      // 0=Zeroes, 1=Ones, 2=Standard Hex Pattern

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
    size_t alloc_size = (free * mem_pct) / 100;
    size_t num_elements = alloc_size / 16; 

    uint4* d_data; CHECK(hipMalloc(&d_data, alloc_size));

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
    
    size_t total_threads = (size_t)num_blocks * block_size;
    float* d_sink; CHECK(hipMalloc(&d_sink, total_threads * sizeof(float)));

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": BAKING (Thermal Leakage Test)..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << " (ALU only)" << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;
    
    // --- PHASE 1: WRITE PAYLOAD ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Writing Payload to VRAM..." << std::endl;
    LAUNCH_KERNEL(write_payload_kernel, num_blocks, block_size, d_data, num_elements, inject_error, init_pattern);
    CHECK(hipDeviceSynchronize());

    // --- PHASE 2: WARMUP ALU ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup iterations..." << std::endl;
        for(int i = 0; i < warmup_iters; i++) {
            LAUNCH_KERNEL(bake_kernel, num_blocks, block_size, kernel_loops, d_sink);
        }
        CHECK(hipDeviceSynchronize());
    }

    // --- PHASE 3: BAKE ---
    std::cout << "[PANTHEON] Starting active thermal bake telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t ops_performed = 0;
    
    while (true) {
        // Run ALU virus to spike die temp
        LAUNCH_KERNEL(bake_kernel, num_blocks, block_size, kernel_loops, d_sink);
        CHECK(hipDeviceSynchronize());
        
        // 2 FMA operations per float = 4 ops per j-loop. Unrolled 32 times = 128 ops per i-loop.
        ops_performed += total_threads * kernel_loops * 128;
        
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (ops_performed / 1e12) / seconds << " TFLOPS" << std::endl;

    // --- PHASE 4: VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] GPU " << gpu_id << ": Verifying Data Retention..." << std::endl;
        
        unsigned int* d_err; CHECK(hipMalloc(&d_err, sizeof(unsigned int)));
        CHECK(hipMemset(d_err, 0, sizeof(unsigned int)));
        
        LAUNCH_KERNEL(verify_payload_kernel, num_blocks, block_size, d_data, num_elements, d_err, init_pattern);
        CHECK(hipDeviceSynchronize());

        unsigned int h_err = 0;
        CHECK(hipMemcpy(&h_err, d_err, sizeof(unsigned int), hipMemcpyDeviceToHost));
        CHECK(hipFree(d_err));
        
        // Say so on success too. Silence is indistinguishable from a
        // verification that never ran.
        std::cout << "Verification: " << (h_err ? "FAIL" : "PASS")
                  << " (" << h_err << " errors)" << std::endl;
        if (h_err > 0) {
            std::cout << "\n[FAIL] Thermal Charge Leakage Detected: " << h_err << " corrupt blocks!" << std::endl;
            // CRITICAL: Exit with non-zero code to fail the CI step
            return 1;
        } else {
            std::cout << "\n[PASS] Memory capacitors retained charge flawlessly." << std::endl;
        }
    }

    CHECK(hipFree(d_data)); 
    CHECK(hipFree(d_sink)); 
    return 0;
}
