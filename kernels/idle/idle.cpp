#include "../common/common.h"
#include <chrono>
#include <thread>

/**
 * IDLE KERNEL
 * This kernel does literally nothing. It exists to provide a 
 * valid execution target for Pantheon while the Hardware Monitor 
 * captures baseline metrics (temp, power, clocks).
 */
int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    int gpu_id = atoi(argv[1]);
    int duration = atoi(argv[2]);

    // Initialize GPU context
    CHECK(hipSetDevice(gpu_id));

    std::cout << "[PANTHEON] GPU " << gpu_id << ": Running BASELINE IDLE test..." << std::endl;

    // Just sleep for the requested duration
    // No kernels are launched, keeping the GPU in its resting state
    std::this_thread::sleep_for(std::chrono::seconds(duration));

    // Output dummy throughput to satisfy the parser
    std::cout << "Throughput: 0.0 GB/s" << std::endl;

    return 0;
}
