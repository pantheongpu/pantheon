#include "../common/common.h"
#include <vector>
#include <chrono>
#include <string>
#include <iostream>

// --- INITIALIZATION KERNEL ---
__global__ void init_p2p_kernel(uint4* data, size_t n, int init_pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        if (init_pattern == 0) {
            data[i] = make_uint4(0x00000000, 0x00000000, 0x00000000, 0x00000000);
        } else if (init_pattern == 1) {
            data[i] = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        } else {
            // Generate a deterministic, verifiable pattern based on the memory address
            uint32_t val = (uint32_t)i;
            data[i] = make_uint4(val, val ^ 0x55555555, ~val, val ^ 0xAAAAAAAA);
        }
    }
}

// --- ERROR INJECTION KERNEL ---
__global__ void inject_p2p_error_kernel(uint4* data, size_t target_idx) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Intentionally corrupt the DMA payload before it crosses the bridge
        data[target_idx].x ^= 0xBADBEEF;
    }
}

// --- VERIFICATION KERNEL ---
__global__ void verify_p2p_kernel(uint4* data, size_t n, unsigned int* err_count, int init_pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        uint4 exp;
        if (init_pattern == 0) {
            exp = make_uint4(0x00000000, 0x00000000, 0x00000000, 0x00000000);
        } else if (init_pattern == 1) {
            exp = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
        } else {
            uint32_t val = (uint32_t)i;
            exp = make_uint4(val, val ^ 0x55555555, ~val, val ^ 0xAAAAAAAA);
        }
        
        uint4 act = data[i];
    
        if (act.x != exp.x || act.y != exp.y || act.z != exp.z || act.w != exp.w) {
            // Cap prints to prevent log spam if the bridge completely fails
            if (*err_count < 5) {
                printf("[SDC FAULT][P2P_THRASHER] NVLink/PCIe DMA Error! Index: %llu | Exp: {%x, %x, %x, %x} | Act: {%x, %x, %x, %x}\n",
                       (unsigned long long)i, exp.x, exp.y, exp.z, exp.w, act.x, act.y, act.z, act.w);
            }
            atomicAdd(err_count, 1);
        }    
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int primary_gpu = atoi(argv[1]);
    int duration = atoi(argv[2]);
    int mem_pct = atoi(argv[3]);

    // --- PANTHEON CONFIG KNOBS ---
    int block_size = 256;      
    int grid_size = 0;         // 0 = auto-calculate
    int kernel_loops = 10;     // Number of bidirectional async DMA batches before synchronization
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

    int device_count = 0;
    CHECK(hipGetDeviceCount(&device_count));

    if (device_count < 2) {
        std::cout << "[PANTHEON] GPU " << primary_gpu << ": Skipping P2P_THRASHER (Only 1 GPU detected)." << std::endl;
        std::cout << "Throughput: 0.0 GB/s" << std::endl;
        return 0;
    }

    // Target the "next" GPU in the topology
    int peer_gpu = (primary_gpu + 1) % device_count;
    
    // Attempt to enable Peer Access
    int can_access_fwd = 0;
    int can_access_bwd = 0;
    CHECK(hipDeviceCanAccessPeer(&can_access_fwd, primary_gpu, peer_gpu));
    CHECK(hipDeviceCanAccessPeer(&can_access_bwd, peer_gpu, primary_gpu));

    if (!can_access_fwd || !can_access_bwd) {
        std::cout << "[PANTHEON] GPU " << primary_gpu << ": Skipping P2P_THRASHER (Bidirectional P2P routing not supported to GPU " << peer_gpu << ")." << std::endl;
        std::cout << "Throughput: 0.0 GB/s" << std::endl;
        return 0;
    }
    
    // --- 1. SET SYNC MODE ACROSS BOTH DEVICES ---
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__) || defined(__HIP_PLATFORM_HCC__)
    unsigned int sync_flag = hipDeviceScheduleBlockingSync;
    if (sync_mode == 0) sync_flag = hipDeviceScheduleSpin;
    else if (sync_mode == 1) sync_flag = hipDeviceScheduleYield;
    
    CHECK(hipSetDevice(primary_gpu));
    hipSetDeviceFlags(sync_flag);
    CHECK(hipDeviceEnablePeerAccess(peer_gpu, 0));

    CHECK(hipSetDevice(peer_gpu));
    hipSetDeviceFlags(sync_flag);
    CHECK(hipDeviceEnablePeerAccess(primary_gpu, 0));

#elif defined(__CUDACC__)
    unsigned int sync_flag = cudaDeviceScheduleBlockingSync;
    if (sync_mode == 0) sync_flag = cudaDeviceScheduleSpin;
    else if (sync_mode == 1) sync_flag = cudaDeviceScheduleYield;
    
    CHECK(hipSetDevice(primary_gpu));
    cudaSetDeviceFlags(sync_flag);
    CHECK(hipDeviceEnablePeerAccess(peer_gpu, 0));

    CHECK(hipSetDevice(peer_gpu));
    cudaSetDeviceFlags(sync_flag);
    CHECK(hipDeviceEnablePeerAccess(primary_gpu, 0));

#else
    // MOCK PLATFORM: CPU execution doesn't require hardware scheduling flags
    CHECK(hipSetDevice(primary_gpu));
    CHECK(hipDeviceEnablePeerAccess(peer_gpu, 0));

    CHECK(hipSetDevice(peer_gpu));
    CHECK(hipDeviceEnablePeerAccess(primary_gpu, 0));
#endif

    // Allocate memory on BOTH GPUs
    // Bounding max burst to 256MB per transfer prevents stream buffer overflow
    size_t buffer_size = 256 * 1024 * 1024; 
    size_t num_elements = buffer_size / sizeof(uint4);
    
    uint4 *d_src, *d_dst;
    CHECK(hipSetDevice(primary_gpu));
    CHECK(hipMalloc(&d_src, buffer_size));
    CHECK(hipSetDevice(peer_gpu));
    CHECK(hipMalloc(&d_dst, buffer_size));

    hipStream_t stream;
    CHECK(hipSetDevice(primary_gpu));
    CHECK(hipStreamCreate(&stream));

    // --- 2. EXPLICIT OCCUPANCY ---
    hipDeviceProp_t prop; 
    CHECK(hipGetDeviceProperties(&prop, primary_gpu));
    
    int num_blocks = grid_size;
    bool auto_grid = false;
    if (num_blocks == 0) {
        int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
        if (max_blocks_per_sm < 1) max_blocks_per_sm = 16;
        num_blocks = prop.multiProcessorCount * max_blocks_per_sm;
        auto_grid = true;
    }

    // --- PRINT ARGUMENTS ---
    std::cout << "[PANTHEON] GPU " << primary_gpu << ": Running P2P THRASHER (Targeting GPU " << peer_gpu << ")..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << " (DMA Batches per Sync)" << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- INITIALIZE DMA PAYLOAD ---
    LAUNCH_KERNEL_ASYNC(init_p2p_kernel, num_blocks, block_size, 0, stream, d_src, num_elements, init_pattern);
    CHECK(hipStreamSynchronize(stream));

    // --- WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup interconnect iterations..." << std::endl;
        for (int i = 0; i < warmup_iters; i++) {
            CHECK(hipMemcpyPeerAsync(d_dst, peer_gpu, d_src, primary_gpu, buffer_size, stream));
            CHECK(hipMemcpyPeerAsync(d_src, primary_gpu, d_dst, peer_gpu, buffer_size, stream));
        }
        CHECK(hipStreamSynchronize(stream));
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t bytes_transferred = 0;
    unsigned int loops = 0;

    // --- EXECUTION LOOP ---
    while (true) {
        
        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && loops == 50) {
            LAUNCH_KERNEL_ASYNC(inject_p2p_error_kernel, 1, 1, 0, stream, d_src, 1337);
            CHECK(hipStreamSynchronize(stream));
        }

        // Direct GPU-to-GPU DMA over NVLink/PCIe (Batched via kernel_loops)
        for (int k = 0; k < kernel_loops; k++) {
            CHECK(hipMemcpyPeerAsync(d_dst, peer_gpu, d_src, primary_gpu, buffer_size, stream));
            CHECK(hipMemcpyPeerAsync(d_src, primary_gpu, d_dst, peer_gpu, buffer_size, stream));
        }
        CHECK(hipStreamSynchronize(stream));
        
        bytes_transferred += (buffer_size * 2 * kernel_loops);
        loops++;
        
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) break;
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (bytes_transferred / 1e9) / seconds << " GB/s" << std::endl;

    // --- VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running NVLink/PCIe Interconnect Verification Pass..." << std::endl;

        // Short runs can finish before the in-loop injection point (loop 50);
        // inject before verification so --inject_error stays deterministic.
        if (inject_error && loops <= 50) {
            LAUNCH_KERNEL_ASYNC(inject_p2p_error_kernel, 1, 1, 0, stream, d_src, 1337);
            CHECK(hipStreamSynchronize(stream));
        }
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        LAUNCH_KERNEL_ASYNC(verify_p2p_kernel, num_blocks, block_size, 0, stream, d_src, num_elements, d_err_count, init_pattern);
        CHECK(hipStreamSynchronize(stream));
        
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

    CHECK(hipStreamDestroy(stream));
    CHECK(hipSetDevice(primary_gpu)); CHECK(hipFree(d_src));
    CHECK(hipSetDevice(peer_gpu));    CHECK(hipFree(d_dst));
    return 0;
}
