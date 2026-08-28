#include "../common/common.h"
#include <chrono>
#include <string>
#include <iostream>

// --- PSEUDO-CHANNEL PING-PONG ---
// Forces data to cross the Infinity Fabric crossbar switches by reading from 
// one physical stack and writing to a completely separate one.
__global__ 
void pc_pingpong_kernel(uint4* data, size_t half_n, int direction, unsigned int launch_idx, int inject_error) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Ping-Pong Logic: Swap Source and Destination every launch
    size_t src_offset = direction ? half_n : 0;
    size_t dst_offset = direction ? 0 : half_n; 

    // Unroll 8x properly using Coalesced Grid-Stride
    size_t step = stride * 8;
    
    // The condition 'i + stride * 7 < half_n' ensures we don't read out of bounds.
    for (size_t i = idx; i + stride * 7 < half_n; i += step) {
        
        // Read from Source Stack
        uint4 r0 = load_nt(&data[i + src_offset]);
        uint4 r1 = load_nt(&data[i + stride + src_offset]);
        uint4 r2 = load_nt(&data[i + stride * 2 + src_offset]);
        uint4 r3 = load_nt(&data[i + stride * 3 + src_offset]);
        uint4 r4 = load_nt(&data[i + stride * 4 + src_offset]);
        uint4 r5 = load_nt(&data[i + stride * 5 + src_offset]);
        uint4 r6 = load_nt(&data[i + stride * 6 + src_offset]);
        uint4 r7 = load_nt(&data[i + stride * 7 + src_offset]);

        // Bitwise invert to simulate work
        r0.x = ~r0.x; r1.y = ~r1.y; r2.z = ~r2.z; r3.w = ~r3.w;
        r4.x = ~r4.x; r5.y = ~r5.y; r6.z = ~r6.z; r7.w = ~r7.w;

        // --- DYNAMIC FAULT INJECTION ---
        if (inject_error && launch_idx == 0 && idx == 1337 && i == idx) {
            // Inject a bit-flip. It will survive and bounce back and forth between stacks.
            r0.x ^= 0xBADBEEF; 
        }

        // Write to Destination Stack safely without collisions
        store_nt(&data[i + dst_offset], r0);
        store_nt(&data[i + stride + dst_offset], r1);
        store_nt(&data[i + stride * 2 + dst_offset], r2);
        store_nt(&data[i + stride * 3 + dst_offset], r3);
        store_nt(&data[i + stride * 4 + dst_offset], r4);
        store_nt(&data[i + stride * 5 + dst_offset], r5);
        store_nt(&data[i + stride * 6 + dst_offset], r6);
        store_nt(&data[i + stride * 7 + dst_offset], r7);
    }
}

// --- VERIFICATION KERNEL ---
__global__ void verify_pingpong_kernel(uint4* data, size_t half_n, unsigned int expected_val, unsigned int* err_count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Scan Stack A.
    for (size_t idx = i; idx < half_n; idx += stride) {
        // Use non-temporal load to bypass L2 cache and force physical reads
        uint4 val = load_nt(&data[idx]);
        
        if (val.x != expected_val || val.y != expected_val || val.z != expected_val || val.w != expected_val) {
            printf("[SDC FAULT][PC_PINGPONG] Crossbar Data Corruption! Index: %llu | Exp: 0x%08x | Act: {%x, %x, %x, %x}\n",
                   (unsigned long long)idx, expected_val, val.x, val.y, val.z, val.w);
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
    int kernel_loops = 1;      // Ignored for ping-pong, as it relies on external launch counter
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
    size_t alloc_size = (free * mem_pct) / 100;
    size_t num_elements = alloc_size / 16; 
    size_t half_elements = num_elements / 2;

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
    std::cout << "[PANTHEON] GPU " << gpu_id << ": PSEUDO-CHANNEL PING-PONG (Crossbar Stress) | " 
              << mem_pct << "% VRAM..." << std::endl;
    std::cout << "  -> Duration (s):  " << duration << std::endl;
    std::cout << "  -> Block Size:    " << block_size << std::endl;
    std::cout << "  -> Grid Size:     " << num_blocks << (auto_grid ? " (Auto-calculated)" : " (Explicit)") << std::endl;
    std::cout << "  -> Kernel Loops:  " << kernel_loops << " (Ignored for Ping-Pong)" << std::endl;
    std::cout << "  -> Warmup Iters:  " << warmup_iters << std::endl;
    std::cout << "  -> Sync Mode:     " << sync_mode << std::endl;
    std::cout << "  -> Init Pattern:  " << init_pattern << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    // --- 4. WARMUP PHASE ---
    if (warmup_iters > 0) {
        std::cout << "[PANTHEON] Running " << warmup_iters << " warmup iterations..." << std::endl;
        
        // Force an EVEN number of warmup iterations so the state is restored
        int safe_warmups = (warmup_iters % 2 == 0) ? warmup_iters : warmup_iters + 1;
        for(int i = 0; i < safe_warmups; i++) {
            int direction = i % 2;
            // Never inject during warmup. The injection fires on launch_idx 0,
            // which warmup and the active loop both have, so injecting here
            // flipped the same bit twice -- and two XORs of the same mask
            // cancel, so the fault silently disappeared before verification.
            LAUNCH_KERNEL(pc_pingpong_kernel, num_blocks, block_size, d_data, half_elements, direction, i, 0);
        }
        CHECK(hipDeviceSynchronize());
    }

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t bytes_transferred = 0;
    unsigned int kernel_launches = 0;

    // --- 5. ACTIVE LOOP ---
    while (true) {
        int direction = kernel_launches % 2;
        LAUNCH_KERNEL(pc_pingpong_kernel, num_blocks, block_size, d_data, half_elements, direction, kernel_launches, inject_error);
        CHECK(hipDeviceSynchronize());
        
        kernel_launches++;
        bytes_transferred += alloc_size; // Count both read and write
        
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= duration) {
            // Force an EVEN number of launches before exiting to restore the array state.
            if (kernel_launches % 2 != 0) {
                direction = kernel_launches % 2;
                LAUNCH_KERNEL(pc_pingpong_kernel, num_blocks, block_size, d_data, half_elements, direction, kernel_launches, inject_error);
                CHECK(hipDeviceSynchronize());
                kernel_launches++;
                bytes_transferred += alloc_size;
            }
            break;
        }
    }

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Throughput: " << (bytes_transferred / 1e9) / seconds << " GB/s" << std::endl;

    // --- 6. VERIFICATION PASS ---
    if (verify_mode) {
        std::cout << "[PANTHEON] Running Crossbar Ping-Pong Verification Pass..." << std::endl;
        
        unsigned int* d_err_count;
        CHECK(hipMalloc(&d_err_count, sizeof(unsigned int)));
        CHECK(hipMemset(d_err_count, 0, sizeof(unsigned int)));
        
        // Determine expected value based on initialization
        unsigned int expected_val = (init_pattern == 1) ? 0xFFFFFFFF : 0x00000000;
        
        int verify_grid = init_launch_grid_size(prop, half_elements, 256);
        LAUNCH_KERNEL(verify_pingpong_kernel, verify_grid, 256, d_data, half_elements, expected_val, d_err_count);
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
