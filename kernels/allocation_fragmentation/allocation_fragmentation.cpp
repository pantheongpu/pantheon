// Allocation Fragmentation
//
// This workload previously included the shared AI kernel template and performed
// no allocation at all -- it ran the same FMA loop as five other workloads while
// its name promised allocator behaviour. It now does what it says: repeatedly
// allocates and frees device memory in a pattern that fragments the heap, and
// measures how the allocator copes.
//
// The stress is on the driver's allocator, not on the GPU pipelines, so the
// device stays almost idle here by design.

#include "../common/common.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// Touch each block so the allocation is actually backed, and so a handed-back
// pointer that overlaps a live one shows up as corruption rather than passing
// silently.
__global__ void stamp_block(unsigned int* block, size_t words, unsigned int tag) {
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += stride) {
        block[i] = tag ^ static_cast<unsigned int>(i);
    }
}

__global__ void check_block(const unsigned int* block, size_t words, unsigned int tag,
                            unsigned int* errors) {
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += stride) {
        if (block[i] != (tag ^ static_cast<unsigned int>(i))) atomicAdd(errors, 1u);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);
    int mem_pct = atoi(argv[3]);

    bool verify_mode = false;
    int inject_error = 0;
    int block_size = 256, grid_size = 0, kernel_loops = 1, warmup_iters = 0, sync_mode = 2;
    for (int i = 4; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--verify") verify_mode = true;
        else if (a == "--inject_error") inject_error = 1;
        else if (a == "--block_size" && i + 1 < argc) block_size = atoi(argv[++i]);
        else if (a == "--grid_size" && i + 1 < argc) grid_size = atoi(argv[++i]);
        else if (a == "--warmup_iters" && i + 1 < argc) warmup_iters = atoi(argv[++i]);
        else if (a == "--sync_mode" && i + 1 < argc) sync_mode = atoi(argv[++i]);
    }
    normalize_kernel_launch_config(block_size, grid_size, kernel_loops, warmup_iters, sync_mode);

    CHECK(hipSetDevice(gpu_id));
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, gpu_id));

    size_t free_bytes = 0, total_bytes = 0;
    CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    mem_pct = mem_pct < 1 ? 1 : (mem_pct > 99 ? 99 : mem_pct);
    size_t budget = (free_bytes * static_cast<size_t>(mem_pct)) / 100;

    // A spread of sizes is what fragments a heap. Equal-sized blocks are always
    // reusable and would never expose the behaviour this test is named for.
    const size_t kSmall = 64u * 1024u;
    const size_t kLarge = 8u * 1024u * 1024u;
    const size_t kSlots = 512;

    std::cout << "[PANTHEON] GPU " << gpu_id << ": Allocation Fragmentation" << std::endl;
    std::cout << "  -> Budget:        " << (budget / (1024 * 1024)) << " MiB" << std::endl;
    std::cout << "  -> Block sizes:   " << (kSmall / 1024) << " KiB to "
              << (kLarge / (1024 * 1024)) << " MiB" << std::endl;
    std::cout << "  -> Slots:         " << kSlots << std::endl;
    std::cout << "  -> Verify Mode:   " << (verify_mode ? "ON" : "OFF") << std::endl;
    if (inject_error) std::cout << "[PANTHEON] Warning: SDC Fault Injection is ACTIVE!" << std::endl;

    std::vector<unsigned int*> slots(kSlots, nullptr);
    std::vector<size_t> sizes(kSlots, 0);
    std::vector<unsigned int> tags(kSlots, 0);
    size_t live_bytes = 0;
    unsigned long long allocations = 0, frees = 0, failures = 0;
    unsigned int* d_errors = nullptr;
    if (verify_mode) {
        CHECK(hipMalloc(&d_errors, sizeof(unsigned int)));
        CHECK(hipMemset(d_errors, 0, sizeof(unsigned int)));
    }

    unsigned int rng = 0x9E3779B9u;
    auto next = [&rng]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };

    std::cout << "[PANTHEON] Starting active telemetry phase..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    do {
        for (size_t round = 0; round < kSlots; ++round) {
            size_t slot = next() % kSlots;

            if (slots[slot]) {
                // Free roughly half the live blocks each pass, leaving holes of
                // mixed sizes behind rather than a clean sweep.
                CHECK(hipFree(slots[slot]));
                live_bytes -= sizes[slot];
                slots[slot] = nullptr;
                sizes[slot] = 0;
                ++frees;
                continue;
            }

            size_t want = kSmall + (static_cast<size_t>(next()) % (kLarge - kSmall));
            want &= ~static_cast<size_t>(255);
            if (live_bytes + want > budget) continue;

            unsigned int* block = nullptr;
            if (hipMalloc(&block, want) != hipSuccess || block == nullptr) {
                // A refusal while well under budget is the fragmentation signal.
                ++failures;
                continue;
            }
            size_t words = want / sizeof(unsigned int);
            unsigned int tag = next();
            int grid = init_launch_grid_size(prop, words, block_size);
            LAUNCH_KERNEL(stamp_block, grid, block_size, block, words, tag);
            slots[slot] = block;
            sizes[slot] = want;
            tags[slot] = tag;
            live_bytes += want;
            ++allocations;
        }
        CHECK(hipDeviceSynchronize());
    } while (std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::high_resolution_clock::now() - start).count() < duration);

    double seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start).count();

    std::cout << "Throughput: " << (static_cast<double>(allocations + frees) / seconds)
              << " alloc-events/s" << std::endl;
    std::cout << "  -> Allocations:   " << allocations << std::endl;
    std::cout << "  -> Frees:         " << frees << std::endl;
    std::cout << "  -> Refusals:      " << failures << " (allocator declined while under budget)" << std::endl;
    std::cout << "  -> Live at end:   " << (live_bytes / (1024 * 1024)) << " MiB" << std::endl;

    int exit_code = 0;
    if (verify_mode) {
        std::cout << "[PANTHEON] Verifying every live block still holds its tag..." << std::endl;
        if (inject_error) {
            for (size_t s = 0; s < kSlots; ++s) {
                if (slots[s]) {
                    unsigned int bad = 0x0BADBEEFu;
                    CHECK(hipMemcpy(slots[s], &bad, sizeof(bad), hipMemcpyHostToDevice));
                    break;
                }
            }
        }
        for (size_t s = 0; s < kSlots; ++s) {
            if (!slots[s]) continue;
            size_t words = sizes[s] / sizeof(unsigned int);
            int grid = init_launch_grid_size(prop, words, block_size);
            LAUNCH_KERNEL(check_block, grid, block_size, slots[s], words, tags[s], d_errors);
        }
        CHECK(hipDeviceSynchronize());
        unsigned int host_errors = 0;
        CHECK(hipMemcpy(&host_errors, d_errors, sizeof(unsigned int), hipMemcpyDeviceToHost));
        std::cout << "Verification: " << (host_errors ? "FAIL" : "PASS")
                  << " (" << host_errors << " corrupted words across live blocks)" << std::endl;
        exit_code = host_errors ? 1 : 0;
    }

    for (size_t s = 0; s < kSlots; ++s) if (slots[s]) CHECK(hipFree(slots[s]));
    if (d_errors) CHECK(hipFree(d_errors));
    return exit_code;
}
