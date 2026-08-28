#include "../common/common.h"
#include <chrono>
#include <thread> // For sleep
#include <string>
#include <iostream>

// --- GOLDEN PASS KERNEL ---
__global__ void golden_pulse_kernel(int iters, unsigned int* golden_sink) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = 0.999f, b = -0.999f, c = 0.5f;

    for(int i = 0; i < iters; ++i) {
        #pragma unroll 32
        for(int j = 0; j < 32; ++j) {
            #ifdef __HIP_PLATFORM_AMD__
                a = __builtin_fmaf(a, b, c);
                b = __builtin_fmaf(b, c, a);
                c = __builtin_fmaf(c, a, b);
            #else
                a = fmaf(a, b, c);
                b = fmaf(b, c, a);
                c = fmaf(c, a, b);
            #endif
        }
        // Clamp variables to prevent infinity/NaN
        if ((i & 0xFF) == 0) { a = -a; b = -b; }
    }
    
    // Cast the float to bits so we can reliably accumulate it
    golden_sink[tid] = pantheon_bit_cast<unsigned int>(a);
}

// --- PULSE VIRUS ---
// Induces maximum dI/dt (Change in Current over Time).
// 1. Spikes load to 100% instantly (Heavy FMA).
// 2. Drops load to 0% instantly (Sleep).
__global__ void pulse_load_kernel(int iters, unsigned int* sink, int inject_error) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a = 0.999f, b = -0.999f, c = 0.5f;

    for(int i = 0; i < iters; ++i) {
        #pragma unroll 32
        for(int j = 0; j < 32; ++j) {
            #ifdef __HIP_PLATFORM_AMD__
                a = __builtin_fmaf(a, b, c);
                b = __builtin_fmaf(b, c, a);
                c = __builtin_fmaf(c, a, b);
            #else
                a = fmaf(a, b, c);
                b = fmaf(b, c, a);
                c = fmaf(c, a, b);
            #endif
        }
        if ((i & 0xFF) == 0) { a = -a; b = -b; }
    }
    
    unsigned int final_bits = pantheon_bit_cast<unsigned int>(a);
    
    // --- DYNAMIC FAULT INJECTION ---
    // Inject directly into the raw bits AFTER the math loop completes.
    // This guarantees the error survives even if the math saturated to Infinity or NaN.
    if (inject_error && tid == 1337) {
        final_bits ^= 0xBADBEEF;
    }
    
    // Accumulate the poisoned bits
    sink[tid] += final_bits;
}

// --- VERIFICATION KERNEL ---
__global__ void verify_pulse_kernel(unsigned int* sink, unsigned int* golden_sink, unsigned int launches, size_t total_threads, unsigned int* err_count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Safely boundary-check the threads for MOCK platform testing
    if (tid < total_threads) {
        // Calculate expected accumulated value
        unsigned int exp = golden_sink[tid] * launches;
        unsigned int act = sink[tid];
        
        if (exp != act) {
            // Decoupled atomicAdd return value to prevent MOCK compiler void-assignment errors
            if (*err_count < 5) {
                printf("[SDC FAULT][PULSE_VIRUS] Transient ALU Error! TID: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n",
                       (unsigned long long)tid, exp, act, exp ^ act);
            }
            atomicAdd(err_count, 1);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);

    // --- PANTHEON CONFIG KNOBS ---
    int block_size = 256;      
    int grid_size = 0;         // 0 = auto-calculate
    int kernel_loops = 20000;  // ALU math iterations during the spike
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 50;     // Hijacked! Acts as Sleep Duration in ms (Duty Cycle)

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

    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, gpu_id));
    
    // --- 2. EXPLICIT OCCUPANCY ---
    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) {
        int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
        if (max_blocks_per_sm < 1) max_blocks_per_sm = 16; // Fallback
        num_blocks = prop.multiProcessorCount * max_blocks_per_sm; 
        auto_grid = true;
    }
    
    // Allocate sink big enough for ALL threads
    size_t total_threads = (size_t)num_blocks * block_size;
    size_t sink_size = total_threads * sizeof(unsigned int);
    
    unsigned int* d_sink; 
    CHECK(hipMalloc(&d_sink, sink_size));
    CHECK(hipMemset(d_sink, 0, sink_size));

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running PULSE VIRUS (Transient Load)..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << " (Overridden: Used as Sleep ms)" << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- GOLDEN PASS ---
    unsigned int* d_golden_sink = nullptr;

    if (verify_mode) {
        std::cout << "[PANTHEON] Generating expected ALU baselines (Golden Pass)..." << std::endl;
        CHECK(hipMalloc(&d_golden_sink, sink_size));
        LAUNCH_KERNEL(golden_pulse_kernel, num_blocks, block_size, kernel_loops, d_golden_sink);
        CHECK(hipDeviceSynchronize());
    }

    // --- 4. WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup iterations..." << std::endl;
        for(int i = 0; i < warmup_iters; i++) {
            LAUNCH_KERNEL(pulse_load_kernel, num_blocks, block_size, kernel_loops, d_sink, inject_error);
        }
        CHECK(hipDeviceSynchronize());
        
        // Clear accumulator so telemetry phase math aligns correctly
        CHECK(hipMemset(d_sink, 0, sink_size));
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    unsigned int kernel_launches = 0;
    size_t ops_performed = 0;
    
    // --- 5. ACTIVE LOOP ---
    while(true) {
        // PHASE 1: SPIKE (Load ON)
        LAUNCH_KERNEL(pulse_load_kernel, num_blocks, block_size, kernel_loops, d_sink, inject_error);
        CHECK(hipDeviceSynchronize());
        
        kernel_launches++;
        // 3 FMAs * 2 ops/FMA * 32 unrolls
        ops_performed += total_threads * kernel_loops * 32 * 6;

        // PHASE 2: DROOP (Load OFF)
        // Uses the hijacked 'init_pattern' parameter as the sleep duration
        std::this_thread::sleep_for(std::chrono::milliseconds(init_pattern));
        
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (ops_performed / 1e12) / seconds << " TFLOPS" << std::endl;

    // --- 6. VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running Transient Load Verification Pass..." << std::endl;
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        int verify_blocks = (total_threads + 255) / 256;
        LAUNCH_KERNEL(verify_pulse_kernel, verify_blocks, 256, d_sink, d_golden_sink, kernel_launches, total_threads, d_err_count);
        CHECK(hipDeviceSynchronize());
        
        unsigned int h_err_count = 0;
        CHECK(hipMemcpy(&h_err_count, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));
        
        CHECK(hipFree(d_err_count));
        CHECK(hipFree(d_golden_sink));

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
    
    CHECK(hipFree(d_sink));
    return 0;
}
