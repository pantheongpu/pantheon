#include "../common/common.h"
#include <vector>
#include <chrono>
#include <string>
#include <iostream>

// --- DEVICE KERNELS ---

__global__ void init_pcie_device_kernel(uint4* data, size_t n, int init_pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        if (init_pattern == 0) {
            data[i] = make_uint4(0x00000000, 0x00000000, 0x00000000, 0x00000000);
        } else if (init_pattern == 1) {
            data[i] = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        } else {
            uint32_t val = (uint32_t)i ^ 0xDEADBEEF;
            data[i] = make_uint4(val, ~val, val ^ 0x55555555, val ^ 0xAAAAAAAA);
        }
    }
}

__global__ void inject_pcie_error_kernel(uint4* data, size_t target_idx) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Intentionally corrupt the outbound GPU buffer before DMA transfer
        data[target_idx].y ^= 0xBADBEEF;
    }
}

__global__ void verify_h2d_kernel(uint4* data, size_t n, unsigned int* err_count, int init_pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        uint4 exp;
        if (init_pattern == 0) {
            exp = make_uint4(0x00000000, 0x00000000, 0x00000000, 0x00000000);
        } else if (init_pattern == 1) {
            exp = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        } else {
            uint32_t val = (uint32_t)i ^ 0xCAFEBABE;
            exp = make_uint4(val, ~val, val ^ 0x55555555, val ^ 0xAAAAAAAA);
        }
        
        uint4 act = data[i];

    if (act.x != exp.x || act.y != exp.y || act.z != exp.z || act.w != exp.w) {
            if (*err_count < 5) {
                printf("[SDC FAULT][PCIE_H2D] Host-to-Device DMA Error! Index: %llu | Exp: {%x, %x, %x, %x} | Act: {%x, %x, %x, %x}\n",
                       (unsigned long long)i, exp.x, exp.y, exp.z, exp.w, act.x, act.y, act.z, act.w);
            }
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
    int kernel_loops = 10;     // Number of async DMA transfers per stream before sync
    int warmup_iters = 5;      
    int sync_mode = 2;         // 0=Spin, 1=Yield, 2=Block
    int init_pattern = 2;      // 0=Zeroes, 1=Ones, 2=Verifiable Entropy

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

    // 2. Allocate Pinned Host Memory (CPU side)
    // Separated into isolated source and destination buffers to prevent DMA tearing
    size_t buffer_size = 256 * 1024 * 1024; 
    size_t num_elements = buffer_size / sizeof(uint4);
    
    uint4 *h_src, *h_dst;
    CHECK(hipHostMalloc((void**)&h_src, buffer_size));
    CHECK(hipHostMalloc((void**)&h_dst, buffer_size));

    // 3. Allocate Device Memory (GPU side)
    uint4 *d_src, *d_dst;
    CHECK(hipMalloc(&d_src, buffer_size));
    CHECK(hipMalloc(&d_dst, buffer_size));

    // 4. Initialize Payloads
    // Initialize Host Outbound Buffer (H2D)
    for(size_t i = 0; i < num_elements; i++) {
        if (init_pattern == 0) {
            h_src[i].x = h_src[i].y = h_src[i].z = h_src[i].w = 0x00000000;
        } else if (init_pattern == 1) {
            h_src[i].x = h_src[i].y = h_src[i].z = h_src[i].w = 0xFFFFFFFF;
        } else {
            uint32_t val = (uint32_t)i ^ 0xCAFEBABE;
            h_src[i].x = val;
            h_src[i].y = ~val;
            h_src[i].z = val ^ 0x55555555;
            h_src[i].w = val ^ 0xAAAAAAAA;
        }
    }
    memset(h_dst, 0, buffer_size);

    hipDeviceProp_t prop; 
    CHECK(hipGetDeviceProperties(&prop, gpu_id));

    // --- 5. EXPLICIT OCCUPANCY ---
    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) {
        num_blocks = prop.multiProcessorCount * 16;
        auto_grid = true;
    }

    // Initialize Device Outbound Buffer (D2H)
    LAUNCH_KERNEL(init_pcie_device_kernel, num_blocks, block_size, d_src, num_elements, init_pattern);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemset(d_dst, 0, buffer_size));

    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running PCIE BANDWIDTH THRASHER (Duplex)..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << " (DMA Batches per Sync)" << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    hipStream_t stream1, stream2;
    CHECK(hipStreamCreate(&stream1));
    CHECK(hipStreamCreate(&stream2));

    // --- 6. WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup DMA iterations..." << std::endl;
        for (int i = 0; i < warmup_iters; i++) {
            CHECK(hipMemcpyAsync(d_dst, h_src, buffer_size, hipMemcpyHostToDevice, stream1));
            CHECK(hipMemcpyAsync(h_dst, d_src, buffer_size, hipMemcpyDeviceToHost, stream2));
        }
        CHECK(hipStreamSynchronize(stream1));
        CHECK(hipStreamSynchronize(stream2));
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t bytes_transferred = 0;
    unsigned int loops = 0;

    // --- 7. MAIN STRESS LOOP ---
    while (true) {
        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && loops == 1) {
            h_src[1337].x ^= 0xBADBEEF; // Corrupt CPU-side memory before H2D upload
            LAUNCH_KERNEL_ASYNC(inject_pcie_error_kernel, 1, 1, 0, stream2, d_src, 1337); // Corrupt GPU-side memory before D2H download
            CHECK(hipStreamSynchronize(stream2));
        }

        // Batch the DMA requests based on the kernel_loops parameter
        for (int k = 0; k < kernel_loops; k++) {
            // Stream 1: Upload (H2D)
            CHECK(hipMemcpyAsync(d_dst, h_src, buffer_size, hipMemcpyHostToDevice, stream1));
            
            // Stream 2: Download (D2H)
            CHECK(hipMemcpyAsync(h_dst, d_src, buffer_size, hipMemcpyDeviceToHost, stream2));
        }
        
        CHECK(hipStreamSynchronize(stream1));
        CHECK(hipStreamSynchronize(stream2));
        
        bytes_transferred += (buffer_size * 2 * kernel_loops);
        loops++;

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (bytes_transferred / 1e9) / seconds << " GB/s" << std::endl;

    // --- 8. VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running Bidirectional PCIe Verification Pass..." << std::endl;
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        // 1. Verify H2D (Did the GPU receive the CPU's data correctly?)
        int verify_blocks = (num_elements + 255) / 256;
        LAUNCH_KERNEL(verify_h2d_kernel, verify_blocks, 256, d_dst, num_elements, d_err_count, init_pattern);
        CHECK(hipDeviceSynchronize());
        
        unsigned int h_errs_h2d = 0;
        CHECK(hipMemcpy(&h_errs_h2d, d_err_count, sizeof(unsigned int), hipMemcpyDeviceToHost));
        
        // 2. Verify D2H (Did the CPU receive the GPU's data correctly?)
        unsigned int h_errs_d2h = 0;
        for (size_t i = 0; i < num_elements; i++) {
            uint32_t exp_x, exp_y, exp_z, exp_w;

            if (init_pattern == 0) {
                exp_x = exp_y = exp_z = exp_w = 0x00000000;
            } else if (init_pattern == 1) {
                exp_x = exp_y = exp_z = exp_w = 0xFFFFFFFF;
            } else {
                uint32_t val = (uint32_t)i ^ 0xDEADBEEF;
                exp_x = val;
                exp_y = ~val;
                exp_z = val ^ 0x55555555;
                exp_w = val ^ 0xAAAAAAAA;
            }

            if (h_dst[i].x != exp_x || h_dst[i].y != exp_y || h_dst[i].z != exp_z || h_dst[i].w != exp_w) {
                if (h_errs_d2h < 5) {
                    printf("[SDC FAULT][PCIE_D2H] Device-to-Host DMA Error! Index: %zu | Exp: {%x, %x, %x, %x} | Act: {%x, %x, %x, %x}\n",
                           i, exp_x, exp_y, exp_z, exp_w, h_dst[i].x, h_dst[i].y, h_dst[i].z, h_dst[i].w);
                }
                h_errs_d2h++;
            }
        }

        CHECK(hipFree(d_err_count));

        // Say so on success too. Silence is indistinguishable from a
        // verification that never ran. Both directions are reported
        // because either can fail on its own.
        std::cout << "Verification: " << ((h_errs_h2d || h_errs_d2h) ? "FAIL" : "PASS")
                  << " (" << h_errs_h2d << " h2d, " << h_errs_d2h << " d2h errors)"
                  << std::endl;
        if (h_errs_h2d > 0 || h_errs_d2h > 0) {
            // CRITICAL: Exit with non-zero code to fail the CI step
            return 1;
        }
    }

    CHECK(hipStreamDestroy(stream1));
    CHECK(hipStreamDestroy(stream2));
    CHECK(hipFree(d_src));
    CHECK(hipFree(d_dst));
    CHECK(hipHostFree(h_src));
    CHECK(hipHostFree(h_dst));
    return 0;
}
