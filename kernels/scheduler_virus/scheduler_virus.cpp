#include "../common/common.h"
#include <chrono>
#include <vector>
#include <string>
#include <iostream>

// --- GOLDEN MICRO KERNEL ---
__global__ void golden_micro_kernel(int loops, unsigned int* golden_sink, int stream_idx) {
    int tid = threadIdx.x;
    unsigned int val = tid ^ 0xCAFEBABE;
    
    // Tiny loop just to keep the kernel alive for a few nanoseconds
    for(int i = 0; i < loops; ++i) {
        val = (val << 3) | (val >> 29);
        val ^= 0xDEADBEEF;
        val += i;
    }
    
    int sink_idx = stream_idx * blockDim.x + tid;
    golden_sink[sink_idx] = val;
}

// --- MICRO KERNEL ---
// A deliberately tiny workload designed to finish almost instantly.
// The goal is not to stress the ALUs, but to force the hardware 
// scheduler to constantly load and unload thread contexts.
__global__ void micro_kernel(int loops, unsigned int* sink, int stream_idx, int inject_now) {
    int tid = threadIdx.x;
    unsigned int val = tid ^ 0xCAFEBABE;
    
    for(int i = 0; i < loops; ++i) {
        val = (val << 3) | (val >> 29);
        val ^= 0xDEADBEEF;
        val += i;
    }

    // --- DYNAMIC FAULT INJECTION ---
    // Simulates the hardware scheduler corrupting a register during a context load/unload
    if (inject_now && stream_idx == 13 && tid == 32) {
        val ^= 0xBADBEEF;
    }
    
    int sink_idx = stream_idx * blockDim.x + tid;
    
    // Accumulate. If a kernel launch is dropped by the queue or mis-scheduled, 
    // it will fail to add its value, permanently poisoning the verification.
    sink[sink_idx] += val;
}

// --- VERIFICATION KERNEL ---
__global__ void verify_scheduler_kernel(unsigned int* sink, unsigned int* golden_sink, unsigned int launches, size_t n, unsigned int* err_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        unsigned int exp = golden_sink[i] * launches;
        unsigned int act = sink[i];

        if (exp != act) {
            // Decoupled atomicAdd return value for MOCK platform compatibility
            if (*err_count < 5) {
                printf("[SDC FAULT][SCHEDULER_VIRUS] Context Switch / State Error! Index: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n",
                       (unsigned long long)i, exp, act, exp ^ act);
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
    int block_size = 64;       // Defines kernel fragment size (smaller = more context switching overhead)
    int grid_size = 64;        // Hijacked: Defines the Number of Concurrent Streams (Queue Depth)
    int kernel_loops = 10;     // Number of micro-kernel launches per stream before sync barrier
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 100;    // Hijacked: Defines inner loop count for the micro-kernel (duration)

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
    if (init_pattern < 1) {
        std::cerr << "[PANTHEON] Warning: invalid scheduler inner-loop count "
                  << init_pattern << "; using 1." << std::endl;
        init_pattern = 1;
    }

    // Safety check for hijacked stream count
    if (grid_size < 1) grid_size = 64;
    int NUM_STREAMS = grid_size;

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

    // Create a massive pool of independent streams
    // This will force hardware queues (NVIDIA Hyper-Q / AMD ACEs) into multiplexing mode
    std::vector<hipStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK(hipStreamCreate(&streams[i]));
    }

    // Allocate an isolated sink for every thread in every stream to prevent race conditions
    size_t total_elements = NUM_STREAMS * block_size;
    size_t sink_size = total_elements * sizeof(unsigned int);
    
    unsigned int* d_sink;
    CHECK(hipMalloc(&d_sink, sink_size));
    CHECK(hipMemset(d_sink, 0, sink_size));

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running SCHEDULER VIRUS (Context Switching)..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << " (Fragment Size)" << std::endl;
    std::cout << "  -> Grid Size:     " << NUM_STREAMS << " (Overridden: Queue/Stream Count)" << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << " (Launches per Sync Batch)" << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << " (Overridden: Micro-Kernel Inner Loops)" << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- GOLDEN PASS ---
    unsigned int* d_golden_sink = nullptr;
    
    if (verify_mode) {
        std::cout << "[PANTHEON] Generating expected Context State baseline (Golden Pass)..." << std::endl;
        CHECK(hipMalloc(&d_golden_sink, sink_size));
        // Zero every lane: the verification pass reads all of them, and lanes
        // a backend does not write must compare as zero-vs-zero, not as
        // whatever the allocator left behind.
        CHECK(hipMemset(d_golden_sink, 0, sink_size));

        for (int i = 0; i < NUM_STREAMS; ++i) {
            LAUNCH_KERNEL_ASYNC(golden_micro_kernel, 1, block_size, 0, streams[i], init_pattern, d_golden_sink, i);
        }
        CHECK(hipDeviceSynchronize());
    }

    // --- 4. WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup scheduling batches..." << std::endl;
        for (int w = 0; w < warmup_iters; w++) {
            for (int i = 0; i < NUM_STREAMS; ++i) {
                LAUNCH_KERNEL_ASYNC(micro_kernel, 1, block_size, 0, streams[i], init_pattern, d_sink, i, 0);
            }
        }
        CHECK(hipDeviceSynchronize());
        
        // Reset accumulation sink so verification passes
        CHECK(hipMemset(d_sink, 0, sink_size));
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t kernels_launched = 0;
    unsigned int host_launches = 0;

    // --- 5. MAIN STRESS LOOP ---
    while (true) {
        int inject_now = (inject_error && host_launches == 50) ? 1 : 0;

        // Batch execution across all streams to minimize PCI-e stall time
        for (int k = 0; k < kernel_loops; k++) {
            for (int i = 0; i < NUM_STREAMS; ++i) {
                // We launch with exactly 1 Block per stream.
                // This forces maximum fragmentation on the SMs.
                LAUNCH_KERNEL_ASYNC(micro_kernel, 1, block_size, 0, streams[i], init_pattern, d_sink, i, inject_now);
            }
        }
        
        // Wait for the GPU to clear the massive traffic jam
        CHECK(hipDeviceSynchronize());
        
        kernels_launched += (NUM_STREAMS * kernel_loops);
        host_launches += kernel_loops;

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    
    // Output in KIPS (Thousand Kernel Issues Per Second)
    std::cout << "Throughput: " << (kernels_launched / 1000.0) / seconds << " KIPS" << std::endl;

    // --- 6. VERIFICATION PASS ---
    int exit_code = 0;
    if (verify_mode) {
        std::cout << "[PANTHEON] Running Hardware Scheduler Context Verification Pass..." << std::endl;
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        int verify_grid = init_launch_grid_size(prop, total_elements, 256);
        LAUNCH_KERNEL(verify_scheduler_kernel, verify_grid, 256, d_sink, d_golden_sink, host_launches, total_elements, d_err_count);
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
            // Fail the run, but only after every allocation is released below.
            exit_code = 1;
        }
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK(hipStreamDestroy(streams[i]));
    }
    CHECK(hipFree(d_sink));

    return exit_code;
}
