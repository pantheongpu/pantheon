#include "../common/common.h"
#include <chrono>
#include <string>
#include <iostream>
#include "../common/fp16_shim.h"
#include "../common/toggle_chaos.h"

// Independent packed-FP16 chains per thread. One chain issues at a quarter of
// the pipe's rate because every __hfma2 consumes the previous result.
#define TENSOR_ILP 8

// --- GOLDEN PASS KERNEL ---
__global__ void golden_tensor_kernel(int iters, unsigned int* golden_sink, int init_pattern) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // init_pattern modulates the starting polarity of the FP16 chains.
    float sign = (init_pattern == 1) ? -1.0f : 1.0f;
    __half2 x[TENSOR_ILP], c[TENSOR_ILP];
    #pragma unroll
    for (int k = 0; k < TENSOR_ILP; ++k) {
        x[k] = make_half2_universal(PANTHEON_CHAOS_SEED(k) * sign);
        c[k] = make_half2_universal(PANTHEON_CHAOS_CONST(k));
    }

    for(int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int k = 0; k < TENSOR_ILP; ++k) {
            x[k] = __hfma2(x[k], x[k], c[k]);
        }
    }

    __half2 a = x[0];
    #pragma unroll
    for (int k = 1; k < TENSOR_ILP; ++k) a = __hfma2(a, make_half2_universal(1.0f), x[k]);

    // Convert to float2, extract the raw bits of both 16-bit calculations, 
    // and XOR them into a single deterministic 32-bit state hash.
    float2 res = __half22float2(a);
    unsigned int bits_x = pantheon_bit_cast<unsigned int>(res.x);
    unsigned int bits_y = pantheon_bit_cast<unsigned int>(res.y);
    golden_sink[tid] = bits_x + bits_y;
}

// --- TENSOR VIRUS (FP16 HAMMER) ---
// Saturates the packed half-precision vector pipe (__hfma2). This is the
// FP16 ALU path, not the matrix cores -- mma_virus covers those.
__global__ void tensor_virus_kernel(int iters, unsigned int* sink, int inject_error, int init_pattern) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    float sign = (init_pattern == 1) ? -1.0f : 1.0f;
    __half2 x[TENSOR_ILP], c[TENSOR_ILP];
    #pragma unroll
    for (int k = 0; k < TENSOR_ILP; ++k) {
        x[k] = make_half2_universal(PANTHEON_CHAOS_SEED(k) * sign);
        c[k] = make_half2_universal(PANTHEON_CHAOS_CONST(k));
    }

    for(int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int k = 0; k < TENSOR_ILP; ++k) {
            x[k] = __hfma2(x[k], x[k], c[k]);
        }

        // --- DYNAMIC FAULT INJECTION ---
        // Inject on the absolute final iteration. Mid-run the chaotic map
        // amplifies a real flip rather than healing it, but an injected value
        // this far out of range overflows the chain to infinity, and infinity
        // is what the old saturating recurrence produced on its own.
        if (inject_error && tid == 1337 && i == iters - 1) {
            x[0] = make_half2_universal(9999.0f);
        }
    }

    __half2 a = x[0];
    #pragma unroll
    for (int k = 1; k < TENSOR_ILP; ++k) a = __hfma2(a, make_half2_universal(1.0f), x[k]);

    float2 res = __half22float2(a);
    unsigned int bits_x = pantheon_bit_cast<unsigned int>(res.x);
    unsigned int bits_y = pantheon_bit_cast<unsigned int>(res.y);
    
    // Accumulate the state hash. Any dropped FMA instruction in ANY launch 
    // permanently corrupts this integer pool via wrap-around addition.
    sink[tid] += (bits_x + bits_y);
}

// --- VERIFICATION KERNEL ---
__global__ void verify_tensor_kernel(unsigned int* sink, unsigned int* golden_sink, unsigned int launches, size_t n, unsigned int* err_count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        unsigned int exp = golden_sink[tid] * launches;
        unsigned int act = sink[tid];
        
        if (exp != act) {
            // Decoupled atomicAdd return value for MOCK platform compatibility
            if (*err_count < 5) {
                printf("[SDC FAULT][TENSOR_VIRUS] FP16 ALU Error! TID: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n",
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
    int kernel_loops = 20000;  // Maps directly to FP16 'iters'
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 0;      // 0=Standard Polarity, 1=Inverted Polarity

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

    hipDeviceProp_t prop; 
    CHECK(hipGetDeviceProperties(&prop, gpu_id));
    
    // --- 2. EXPLICIT OCCUPANCY ---
    // Max Occupancy: Tensor tests need MASSIVE parallelism to fill the huge pipes.
    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) {
        int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
        if (max_blocks_per_sm < 1) max_blocks_per_sm = 16; 
        num_blocks = prop.multiProcessorCount * max_blocks_per_sm;
        auto_grid = true;
    }

    // Dynamic Sink Allocation: Prevent Buffer Overflows
    size_t total_threads = (size_t)num_blocks * block_size;
    size_t sink_size = total_threads * sizeof(unsigned int);
    
    unsigned int* d_sink;
    CHECK(hipMalloc(&d_sink, sink_size));
    CHECK(hipMemset(d_sink, 0, sink_size));

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running TENSOR VIRUS (FP16 Stress)..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << " (0=Standard Polarity, 1=Inverted Polarity)" << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- GOLDEN PASS ---
    unsigned int* d_golden_sink = nullptr;

    if (verify_mode) {
        std::cout << "[PANTHEON] Generating expected FP16 baseline (Golden Pass)..." << std::endl;
        CHECK(hipMalloc(&d_golden_sink, sink_size));
        LAUNCH_KERNEL(golden_tensor_kernel, num_blocks, block_size, kernel_loops, d_golden_sink, init_pattern);
        CHECK(hipDeviceSynchronize());
    }

    // --- 4. WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup iterations..." << std::endl;
        for(int i = 0; i < warmup_iters; i++) {
            LAUNCH_KERNEL(tensor_virus_kernel, num_blocks, block_size, kernel_loops, d_sink, inject_error, init_pattern);
        }
        CHECK(hipDeviceSynchronize());
        
        // Reset accumulation sink so verification passes accurately
        CHECK(hipMemset(d_sink, 0, sink_size));
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t ops_performed = 0;
    unsigned int kernel_launches = 0;

    // --- 5. MAIN STRESS LOOP ---
    while(true) {
        LAUNCH_KERNEL(tensor_virus_kernel, num_blocks, block_size, kernel_loops, d_sink, inject_error, init_pattern);
        CHECK(hipDeviceSynchronize());
        
        kernel_launches++;

        // 12 FLOPs per loop * iterations per thread
        ops_performed += (size_t)num_blocks * block_size * kernel_loops * TENSOR_ILP * 4;

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (ops_performed / 1e12) / seconds << " TFLOPS" << std::endl;

    // --- 6. VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running Packed FP16 State Verification Pass..." << std::endl;
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        int verify_blocks = (total_threads + 255) / 256;
        LAUNCH_KERNEL(verify_tensor_kernel, verify_blocks, 256, d_sink, d_golden_sink, kernel_launches, total_threads, d_err_count);
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
