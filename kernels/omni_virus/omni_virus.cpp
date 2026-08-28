#include "../common/common.h"
#include "../common/fp16_shim.h"
#include <chrono>
#include <string>
#include <iostream>

// --- STREAM 1: MEMORY FURNACE ---
__global__ void mem_stream(uint4* data, size_t n, int loops, int l2_stride, int inject_error) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (blockDim.x * gridDim.x) + l2_stride;
    uint4 pA = make_uint4(0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA);
    uint4 pB = make_uint4(0x55555555, 0x55555555, 0x55555555, 0x55555555);

    // --- DYNAMIC FAULT INJECTION ---
    if (inject_error && idx == 1337) {
        // Induce a continuous memory crosstalk error for this specific thread
        pA.x ^= 0xBADBEEF; 
    }

    for(int l = 0; l < loops; ++l) {
        #pragma unroll 4
        for (size_t i = idx; i + stride < n; i += stride * 2) {
            store_nt(&data[i], pA);
            store_nt(&data[i+stride], pB);
        }
    }
}

// --- STREAM 2: TENSOR FURNACE (FP16) ---
__global__ void compute_fp16(int iters, float* sink, int inject_error) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    __half2 a = make_half2_universal(0.999f), b = make_half2_universal(-0.999f), c = make_half2_universal(0.5f);

    for(int i = 0; i < iters; ++i) {
        #pragma unroll 128
        for(int j = 0; j < 128; ++j) {
            a = __hfma2(a, b, c);
            b = __hfma2(b, c, a);
            c = __hfma2(c, a, b);
        }
        a = __hneg2(a);
        
        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && tid == 1337 && i == 50) {
            a = make_half2_universal(9999.0f);
        }
    }
    sink[tid] = __half22float2(a).x;
}

// --- STREAM 3: VECTOR FURNACE (FP32 PURE) ---
__global__ void compute_fp32(int iters, float* sink, int inject_error) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.999f, b = -0.999f, c = 0.5f;

    for(int i = 0; i < iters; ++i) {
        #pragma unroll 128
        for(int j = 0; j < 128; ++j) {
            #ifdef __HIP_PLATFORM_AMD__
                // Explicitly force 32-bit FMA to keep heat in registers
                a = __builtin_fmaf(a, b, c);
                b = __builtin_fmaf(b, c, a);
                c = __builtin_fmaf(c, a, b);
            #else
                a = fmaf(a, b, c);
                b = fmaf(b, c, a);
                c = fmaf(c, a, b);
            #endif
        }
        a = -a;

        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && tid == 1337 && i == 50) {
            a += 9999.0f;
        }
    }
    sink[tid] = a;
}

// --- STREAM 4: SFU FURNACE (TRANSCENDENTAL PURE) ---
__global__ void compute_sfu(int iters, float* sink, int inject_error) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = (float)(tid % 100) * 0.001f;
    float b = 1.0f;

    for(int i = 0; i < iters; ++i) {
        #pragma unroll 16
        for(int j = 0; j < 16; ++j) {
            a = sinf(a) * cosf(b);
            b = rsqrtf(fabsf(a) + 1.0f);
        }
        a += 0.01f;

        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && tid == 1337 && i == 50) {
            b += 9999.0f;
        }
    }
    sink[tid] = b;
}

// --- VERIFICATION KERNELS ---
__global__ void verify_mem_stream(uint4* data, size_t n, unsigned int* err_count, int init_pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    uint32_t expected_unwritten = (init_pattern == 1) ? 0xFFFFFFFF : 0x00000000;

    for(size_t i = idx; i < n; i += stride) {
        uint4 v = load_nt(&data[i]);
        
        // Ensure memory is either unwritten, purely Pattern A, or purely Pattern B
        if (!(v.x == 0xAAAAAAAA && v.y == 0xAAAAAAAA && v.z == 0xAAAAAAAA && v.w == 0xAAAAAAAA) &&
            !(v.x == 0x55555555 && v.y == 0x55555555 && v.z == 0x55555555 && v.w == 0x55555555) &&
            !(v.x == expected_unwritten && v.y == expected_unwritten && v.z == expected_unwritten && v.w == expected_unwritten)) {
            
            printf("[SDC FAULT][OMNI_VIRUS] Memory Stream Error! Index: %llu | Act: {%x, %x, %x, %x}\n",
                   (unsigned long long)i, v.x, v.y, v.z, v.w);
            atomicAdd(err_count, 1);
        }
    }
}

__global__ void verify_compute_stream(int stream_id, float* sink, float* golden, size_t n, unsigned int* err_count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        unsigned int act = pantheon_bit_cast<unsigned int>(sink[tid]);
        unsigned int exp = pantheon_bit_cast<unsigned int>(golden[tid]);
        
        if (act != exp) {
            unsigned int xor_bits = act ^ exp;
            if (stream_id == 1) printf("[SDC FAULT][OMNI_VIRUS] FP16 Stream Error! TID: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n", (unsigned long long)tid, exp, act, xor_bits);
            else if (stream_id == 2) printf("[SDC FAULT][OMNI_VIRUS] FP32 Stream Error! TID: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n", (unsigned long long)tid, exp, act, xor_bits);
            else if (stream_id == 3) printf("[SDC FAULT][OMNI_VIRUS] SFU Stream Error! TID: %llu | Exp: 0x%08x | Act: 0x%08x | XOR: 0x%08x\n", (unsigned long long)tid, exp, act, xor_bits);
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
    int kernel_loops = 5000;   // Master control loop for the dominant FP compute pipelines
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 0;      // 0=Zeroes, 1=Ones

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
    
    size_t free, total; CHECK(hipMemGetInfo(&free, &total));
    if (mem_pct > 99) mem_pct = 99;
    size_t alloc_size = (free * mem_pct) / 100; 
    size_t num_elements = alloc_size / 16; 

    uint4* d_data; 
    CHECK(hipMalloc(&d_data, alloc_size));
    
    // Initialize Memory Pattern
    if (init_pattern == 1) {
        CHECK(hipMemset(d_data, 0xFF, alloc_size));
    } else {
        CHECK(hipMemset(d_data, 0x00, alloc_size)); 
    }

    hipStream_t stream_mem, stream_fp16, stream_fp32, stream_sfu;
    CHECK(hipStreamCreate(&stream_mem));
    CHECK(hipStreamCreate(&stream_fp16));
    CHECK(hipStreamCreate(&stream_fp32));
    CHECK(hipStreamCreate(&stream_sfu));

    int sms = prop.multiProcessorCount;
    
    // --- 2. EXPLICIT OCCUPANCY ---
    int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
    int total_blocks = grid_size;
    bool auto_grid = false;
    
    if (total_blocks == 0) {
        total_blocks = sms * max_blocks_per_sm;
        auto_grid = true;
    }

    // Divide occupancy perfectly into 4 parallel chunks
    int blocks_mem  = total_blocks / 4;  if (blocks_mem < 1) blocks_mem = 1;
    int blocks_fp16 = total_blocks / 4;  if (blocks_fp16 < 1) blocks_fp16 = 1;
    int blocks_fp32 = total_blocks / 4;  if (blocks_fp32 < 1) blocks_fp32 = 1;
    int blocks_sfu  = total_blocks / 4;  if (blocks_sfu < 1)  blocks_sfu = 1;

    size_t fp16_threads = (size_t)blocks_fp16 * block_size;
    size_t fp32_threads = (size_t)blocks_fp32 * block_size;
    size_t sfu_threads  = (size_t)blocks_sfu * block_size;

    // Allocate isolated sinks to prevent data races across streams
    float *d_sink_fp16, *d_sink_fp32, *d_sink_sfu; 
    CHECK(hipMalloc(&d_sink_fp16, fp16_threads * sizeof(float)));
    CHECK(hipMalloc(&d_sink_fp32, fp32_threads * sizeof(float)));
    CHECK(hipMalloc(&d_sink_sfu,  sfu_threads * sizeof(float)));

    // Scale peripheral loops relative to the master kernel_loops compute scale
    int comp_loops = kernel_loops;
    int mem_loops = (kernel_loops / 1000 > 0) ? (kernel_loops / 1000) : 1; 
    int sfu_loops = (kernel_loops / 10 > 0) ? (kernel_loops / 10) : 1; 

    int l2_stride_offset = prop.l2CacheSize / sizeof(uint4);
    if (l2_stride_offset < 1024) l2_stride_offset = 1024;

    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running OMNI VIRUS (Multi-Stream Pipeline Saturator)..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << total_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << " (Auto-Scaled for Memory/SFU pipelines)" << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- GOLDEN PASS ---
    float *d_gold_fp16 = nullptr, *d_gold_fp32 = nullptr, *d_gold_sfu = nullptr;
    if (verify_mode) {
        std::cout << "[PANTHEON] Generating expected stream baselines (Golden Pass)..." << std::endl;
        CHECK(hipMalloc(&d_gold_fp16, fp16_threads * sizeof(float)));
        CHECK(hipMalloc(&d_gold_fp32, fp32_threads * sizeof(float)));
        CHECK(hipMalloc(&d_gold_sfu,  sfu_threads * sizeof(float)));

        // Run compute cleanly to establish baseline
        LAUNCH_KERNEL_ASYNC(compute_fp16, blocks_fp16, block_size, 0, stream_fp16, comp_loops, d_gold_fp16, 0);
        LAUNCH_KERNEL_ASYNC(compute_fp32, blocks_fp32, block_size, 0, stream_fp32, comp_loops, d_gold_fp32, 0);
        LAUNCH_KERNEL_ASYNC(compute_sfu,  blocks_sfu,  block_size, 0, stream_sfu,  sfu_loops,  d_gold_sfu,  0);
        CHECK(hipDeviceSynchronize());
    }

    // Pre-calculate operations
    size_t fp16_ops_per_launch = (size_t)blocks_fp16 * block_size * comp_loops * 1536;
    size_t fp32_ops_per_launch = (size_t)blocks_fp32 * block_size * comp_loops * 768;
    size_t sfu_ops_per_launch  = (size_t)blocks_sfu * block_size * sfu_loops * 96;
    size_t total_ops_per_launch = fp16_ops_per_launch + fp32_ops_per_launch + sfu_ops_per_launch;

    // --- WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup iterations..." << std::endl;
        for (int i = 0; i < warmup_iters; i++) {
            LAUNCH_KERNEL_ASYNC(mem_stream,   blocks_mem,  block_size, 0, stream_mem,  d_data, num_elements, mem_loops, l2_stride_offset, inject_error);
            LAUNCH_KERNEL_ASYNC(compute_fp16, blocks_fp16, block_size, 0, stream_fp16, comp_loops, d_sink_fp16, inject_error);
            LAUNCH_KERNEL_ASYNC(compute_fp32, blocks_fp32, block_size, 0, stream_fp32, comp_loops, d_sink_fp32, inject_error);
            LAUNCH_KERNEL_ASYNC(compute_sfu,  blocks_sfu,  block_size, 0, stream_sfu,  sfu_loops,  d_sink_sfu,  inject_error);
        }
        CHECK(hipDeviceSynchronize());
        
        // Re-initialize memory buffer to clean state
        if (init_pattern == 1) {
            CHECK(hipMemset(d_data, 0xFF, alloc_size));
        } else {
            CHECK(hipMemset(d_data, 0x00, alloc_size)); 
        }
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t ops_performed = 0;

    // --- MAIN STRESS LOOP ---
    while(true) {
        LAUNCH_KERNEL_ASYNC(mem_stream,   blocks_mem,  block_size, 0, stream_mem,  d_data, num_elements, mem_loops, l2_stride_offset, inject_error);
        LAUNCH_KERNEL_ASYNC(compute_fp16, blocks_fp16, block_size, 0, stream_fp16, comp_loops, d_sink_fp16, inject_error);
        LAUNCH_KERNEL_ASYNC(compute_fp32, blocks_fp32, block_size, 0, stream_fp32, comp_loops, d_sink_fp32, inject_error);
        LAUNCH_KERNEL_ASYNC(compute_sfu,  blocks_sfu,  block_size, 0, stream_sfu,  sfu_loops,  d_sink_sfu,  inject_error);

        CHECK(hipDeviceSynchronize());
        ops_performed += total_ops_per_launch;

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (ops_performed / 1e12) / seconds << " TFLOPS" << std::endl;

    // --- VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running Multi-Stream Cross-Verification Pass..." << std::endl;
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        // Verify Memory
        int mem_verify_grid = init_launch_grid_size(prop, num_elements, 256);
        LAUNCH_KERNEL_ASYNC(verify_mem_stream, mem_verify_grid, 256, 0, stream_mem, d_data, num_elements, d_err_count, init_pattern);

        // Verify FP16
        LAUNCH_KERNEL_ASYNC(verify_compute_stream, blocks_fp16, block_size, 0, stream_fp16, 1, d_sink_fp16, d_gold_fp16, fp16_threads, d_err_count);
        
        // Verify FP32
        LAUNCH_KERNEL_ASYNC(verify_compute_stream, blocks_fp32, block_size, 0, stream_fp32, 2, d_sink_fp32, d_gold_fp32, fp32_threads, d_err_count);
        
        // Verify SFU
        LAUNCH_KERNEL_ASYNC(verify_compute_stream, blocks_sfu, block_size, 0, stream_sfu, 3, d_sink_sfu, d_gold_sfu, sfu_threads, d_err_count);

        CHECK(hipDeviceSynchronize());
        
        unsigned int h_err_count = 0;
        CHECK(hipMemcpy(&h_err_count, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));
        
        CHECK(hipFree(d_err_count));
        CHECK(hipFree(d_gold_fp16));
        CHECK(hipFree(d_gold_fp32));
        CHECK(hipFree(d_gold_sfu));

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

    CHECK(hipStreamDestroy(stream_mem));
    CHECK(hipStreamDestroy(stream_fp16));
    CHECK(hipStreamDestroy(stream_fp32));
    CHECK(hipStreamDestroy(stream_sfu));
    CHECK(hipFree(d_data));
    CHECK(hipFree(d_sink_fp16));
    CHECK(hipFree(d_sink_fp32));
    CHECK(hipFree(d_sink_sfu));

    return 0;
}
