#include "../common/common.h"
#include "../common/toggle_chaos.h"

// Independent transcendental chains per thread. SFU latency is long and its
// throughput is a fraction of the FMA pipe's, so one chain per thread leaves
// the unit idle between results.
#define SFU_ILP 4
#include <chrono>
#include <string>
#include <iostream>

// --- GOLDEN PASS KERNEL ---
__global__ void golden_sfu_kernel(int iters, unsigned int* golden_sink, int init_pattern) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use init_pattern to modulate the starting scale and sign of the chains.
    float a[SFU_ILP], b[SFU_ILP];
    #pragma unroll
    for (int k = 0; k < SFU_ILP; ++k) {
        a[k] = PANTHEON_CHAOS_SFU_SEED_A(k)
             + (float)tid * 0.0001f * (float)(init_pattern + 1);
        b[k] = (init_pattern % 2 == 1) ? -PANTHEON_CHAOS_SFU_SEED_B(k)
                                      :  PANTHEON_CHAOS_SFU_SEED_B(k);
    }

    for(int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int k = 0; k < SFU_ILP; ++k) {
            PANTHEON_CHAOS_SFU_STEP(a[k], b[k]);
        }
    }

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < SFU_ILP; ++k) acc += a[k];

    // Cast the final float to bits for exact hardware verification
    golden_sink[tid] = pantheon_bit_cast<unsigned int>(acc);
}

// --- SFU (SPECIAL FUNCTION UNIT) VIRUS ---
// Hammers the transcendental math pipelines (SIN, COS, EXP, LOG, RSQRT).
// These units often share power rails with Texture units or Tensor cores.
__global__ void sfu_stress_kernel(int iters, unsigned int* sink, int inject_error, int init_pattern) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Seed with thread ID to prevent caching
    float a[SFU_ILP], b[SFU_ILP];
    #pragma unroll
    for (int k = 0; k < SFU_ILP; ++k) {
        a[k] = PANTHEON_CHAOS_SFU_SEED_A(k)
             + (float)tid * 0.0001f * (float)(init_pattern + 1);
        b[k] = (init_pattern % 2 == 1) ? -PANTHEON_CHAOS_SFU_SEED_B(k)
                                      :  PANTHEON_CHAOS_SFU_SEED_B(k);
    }

    // The "Transcendental Torture" Chain: high-latency, low-throughput
    // instructions, amplified before the sine so the orbit stays chaotic.
    // The old chain needed a periodic nudge to "avoid convergence to 0/INF"
    // and still settled onto a short orbit toggling 1.2 bits of 32; this one
    // is bounded by construction and needs no nudge.
    for(int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int k = 0; k < SFU_ILP; ++k) {
            PANTHEON_CHAOS_SFU_STEP(a[k], b[k]);
        }

        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && tid == 1337 && i == iters - 1) {
            a[0] += 9999.0f; 
        }
    }

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < SFU_ILP; ++k) acc += a[k];

    // Accumulate the bits. If an SDC occurs during ANY transient spike,
    // it permanently poisons this thread's accumulator via integer addition.
    sink[tid] += pantheon_bit_cast<unsigned int>(acc);
}

// --- VERIFICATION KERNEL ---
__global__ void verify_sfu_kernel(unsigned int* sink, unsigned int* golden_sink, unsigned int launches, size_t n, unsigned int* err_count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Unsigned integer overflow safely mirrors the wrap-around additions happening on the GPU
        unsigned int exp = golden_sink[tid] * launches;
        unsigned int act = sink[tid];
        
        if (exp != act) {
            // Decoupled atomicAdd return value for MOCK platform compatibility
            if (*err_count < 5) {
                printf("[SDC FAULT][SFU_VIRUS] Transcendental ALU Error! TID: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n",
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
    int kernel_loops = 5000;   // Transcendental math chain iterations per launch
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 0;      // Used to modulate starting seed vectors

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

    // --- 2. EXPLICIT OCCUPANCY ---
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, gpu_id));
    
    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) {
        int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
        if (max_blocks_per_sm > 4) max_blocks_per_sm -= 1; // Slight safety margin
        if (max_blocks_per_sm < 1) max_blocks_per_sm = 32; 
        num_blocks = prop.multiProcessorCount * max_blocks_per_sm; 
        auto_grid = true;
    }
    
    // Dynamically allocate sink buffer based on total threads
    size_t total_threads = (size_t)num_blocks * block_size;
    size_t sink_size = total_threads * sizeof(unsigned int);
    
    unsigned int* d_sink; 
    CHECK(hipMalloc(&d_sink, sink_size));
    CHECK(hipMemset(d_sink, 0, sink_size)); // Crucial for clean accumulation

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running SFU VIRUS (Transcendental Math)..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << " (Transcendental Math Seed)" << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- GOLDEN PASS ---
    unsigned int* d_golden_sink = nullptr;

    if (verify_mode) {
        std::cout << "[PANTHEON] Generating expected SFU baseline (Golden Pass)..." << std::endl;
        CHECK(hipMalloc(&d_golden_sink, sink_size));
        LAUNCH_KERNEL(golden_sfu_kernel, num_blocks, block_size, kernel_loops, d_golden_sink, init_pattern);
        CHECK(hipDeviceSynchronize());
    }

    // --- 4. WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup iterations..." << std::endl;
        for(int i = 0; i < warmup_iters; i++) {
            LAUNCH_KERNEL(sfu_stress_kernel, num_blocks, block_size, kernel_loops, d_sink, inject_error, init_pattern);
        }
        CHECK(hipDeviceSynchronize());
        
        // Reset accumulation sink so verification passes accurately
        CHECK(hipMemset(d_sink, 0, sink_size));
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t ops_performed = 0;
    unsigned int kernel_launches = 0;

    // --- 5. ACTIVE LOOP ---
    while(true) {
        LAUNCH_KERNEL(sfu_stress_kernel, num_blocks, block_size, kernel_loops, d_sink, inject_error, init_pattern);
        CHECK(hipDeviceSynchronize());
        
        kernel_launches++;

        // 13 FLOPs per loop * iterations per thread
        // Twelve ops per chain step: five transcendentals counted as one each,
        // plus the seven arithmetic operations that compose them.
        ops_performed += (size_t)num_blocks * block_size * kernel_loops * SFU_ILP * 12;

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (ops_performed / 1e12) / seconds << " TFLOPS" << std::endl;

    // --- 6. VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running Transcendental ALU Verification Pass..." << std::endl;
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        int verify_blocks = (total_threads + 255) / 256;
        LAUNCH_KERNEL(verify_sfu_kernel, verify_blocks, 256, d_sink, d_golden_sink, kernel_launches, total_threads, d_err_count);
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
