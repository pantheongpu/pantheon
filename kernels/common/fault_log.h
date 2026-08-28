#ifndef PANTHEON_FAULT_LOG_H
#define PANTHEON_FAULT_LOG_H

#include "common.h"
#include <cstdio>

// Locating a defective cell needs every failing address, not a sample of them.
// Console output is deliberately capped -- a widescale corruption would other-
// wise emit billions of lines -- so a run reports an accurate error *count*
// but only the first few error *locations*. That cap makes a fault map
// impossible to build from stdout.
//
// This records each failing location into a bounded device-side buffer that
// the host writes out after the run, so an external search can accumulate a
// complete map across many runs while the console stays readable.

#define PANTHEON_FAULT_LOG_DEFAULT_CAPACITY 4096

struct PantheonFault {
    unsigned long long index;   // element index within the workload's buffer
    unsigned int expected;
    unsigned int actual;
    unsigned int xor_bits;      // expected ^ actual: identifies the failing bits
};

// Header kept in device memory alongside the records so the kernel can append
// without a host round trip.
struct PantheonFaultLog {
    PantheonFault* records;
    unsigned int* count;        // total faults seen, may exceed capacity
    unsigned int capacity;
};

// Append one fault. Returns nothing: overflow is not an error, it is expected
// on a badly damaged device, and the count still reports the true total.
__device__ __forceinline__ void pantheon_fault_log_append(
    PantheonFaultLog log,
    unsigned long long index,
    unsigned int expected,
    unsigned int actual
) {
    if (log.records == nullptr || log.count == nullptr) return;
    unsigned int slot = atomicAdd(log.count, 1u);
    if (slot < log.capacity) {
        log.records[slot].index = index;
        log.records[slot].expected = expected;
        log.records[slot].actual = actual;
        log.records[slot].xor_bits = expected ^ actual;
    }
}

// A disabled log: appends become no-ops, so kernels take the same parameter
// whether or not the caller asked for a map.
inline PantheonFaultLog pantheon_fault_log_none() {
    PantheonFaultLog log;
    log.records = nullptr;
    log.count = nullptr;
    log.capacity = 0;
    return log;
}

// ---------------------------------------------------------------- host side

inline PantheonFaultLog pantheon_fault_log_create(unsigned int capacity) {
    PantheonFaultLog log;
    log.capacity = capacity;
    log.records = nullptr;
    log.count = nullptr;
    CHECK(hipMalloc(&log.records, (size_t)capacity * sizeof(PantheonFault)));
    CHECK(hipMalloc(&log.count, sizeof(unsigned int)));
    CHECK(hipMemset(log.count, 0, sizeof(unsigned int)));
    return log;
}

inline void pantheon_fault_log_destroy(PantheonFaultLog& log) {
    if (log.records) CHECK(hipFree(log.records));
    if (log.count) CHECK(hipFree(log.count));
    log.records = nullptr;
    log.count = nullptr;
}

// Write the map as CSV. Returns the number of records written, which is
// capped by capacity even when more faults occurred.
inline unsigned int pantheon_fault_log_write(
    const PantheonFaultLog& log,
    const char* path,
    const char* workload,
    int gpu_id,
    int init_pattern
) {
    if (!path || !path[0] || log.records == nullptr) return 0;

    unsigned int total = 0;
    CHECK(hipMemcpy(&total, log.count, sizeof(unsigned int), hipMemcpyDeviceToHost));
    unsigned int stored = total < log.capacity ? total : log.capacity;

    PantheonFault* host = (PantheonFault*)malloc((size_t)stored * sizeof(PantheonFault));
    if (!host) return 0;
    if (stored > 0) {
        CHECK(hipMemcpy(host, log.records, (size_t)stored * sizeof(PantheonFault),
                        hipMemcpyDeviceToHost));
    }

    FILE* out = fopen(path, "w");
    if (!out) {
        free(host);
        std::cerr << "[PANTHEON] Warning: could not write fault map to " << path << std::endl;
        return 0;
    }
    // A header keeps the file self-describing: consumers should not have to
    // know which workload or pattern produced it.
    // Record the background by name as well as number -- a fault map read back
    // months later should not require a lookup table to interpret.
    fprintf(out, "# workload=%s gpu=%d init_pattern=%d init_pattern_name=%s"
                 " total_faults=%u recorded=%u capacity=%u\n",
            workload, gpu_id, init_pattern, pantheon_init_pattern_name(init_pattern),
            total, stored, log.capacity);
    fprintf(out, "index,expected,actual,xor\n");
    for (unsigned int i = 0; i < stored; ++i) {
        fprintf(out, "%llu,0x%08x,0x%08x,0x%08x\n",
                (unsigned long long)host[i].index,
                host[i].expected, host[i].actual, host[i].xor_bits);
    }
    fclose(out);
    free(host);

    if (total > stored) {
        std::cerr << "[PANTHEON] Fault map truncated: " << total
                  << " faults occurred, " << stored << " recorded." << std::endl;
    }
    std::cout << "[PANTHEON] Fault map written to " << path
              << " (" << stored << " of " << total << " faults)" << std::endl;
    return stored;
}

#endif // PANTHEON_FAULT_LOG_H
