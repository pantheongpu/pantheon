#ifndef PANTHEON_P2P_MATRIX_STATS_H
#define PANTHEON_P2P_MATRIX_STATS_H

// The parts of the p2p_matrix workload that are not GPU calls (the arithmetic and
// the taking of turns), kept apart so they can be compiled and checked on a
// machine with no GPU.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <string>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#define PANTHEON_HAVE_FLOCK 1
#endif

// Median of a list; the mean of the two middle values for an even count, 0 for none.
inline double pantheon_median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2.0;
}

struct PantheonPairStats {
    double min = 0.0;
    double median = 0.0;
    double max = 0.0;
    int slowest = -1;   // index of the smallest value, the first of equals
};

inline PantheonPairStats pantheon_pair_stats(const std::vector<double>& bandwidths) {
    PantheonPairStats stats;
    if (bandwidths.empty()) return stats;
    stats.min = stats.max = bandwidths[0];
    stats.slowest = 0;
    for (size_t i = 1; i < bandwidths.size(); i++) {
        if (bandwidths[i] < stats.min) { stats.min = bandwidths[i]; stats.slowest = (int)i; }
        if (bandwidths[i] > stats.max) stats.max = bandwidths[i];
    }
    stats.median = pantheon_median(bandwidths);
    return stats;
}

// Seconds to spend on one transfer direction. Every GPU measures its own row,
// one GPU at a time, so the rows together take n * (n - 1) of these. Eight tenths
// of the requested duration are used, with a floor that still holds many copies
// of a 256 MiB buffer.
inline double pantheon_pair_seconds(int duration_s, int gpu_count) {
    if (gpu_count < 2) return 0.0;
    double each = 0.8 * duration_s / ((double)gpu_count * (gpu_count - 1));
    return std::max(0.1, each);
}

// The sub-batch bandwidths of one pair reduce to their median, so one stalled
// batch does not stand for the pair.
inline double pantheon_pair_bandwidth(const std::vector<double>& batch_bandwidths) {
    return pantheon_median(batch_bandwidths);
}

// Taking turns.
// An exclusive lock on a file named for the parent process, which is the runner
// that started every copy. Without a parent in common (a copy started by hand)
// there is nobody to wait for.
// Keep it. On 8 H100s behind NVSwitch, eight copies at once read 65 to 400 GB/s per
// row where taking turns read 396 GB/s for every pair: without the lock the numbers
// measure who else is copying. (On a V100 mesh it made no difference.)
struct PantheonMatrixTurn {
    int fd = -1;

    bool acquire(double give_up_after_s) {
#ifdef PANTHEON_HAVE_FLOCK
        std::string path = "/tmp/pantheon-p2p-matrix-" + std::to_string((long)getppid()) + ".lock";
        fd = open(path.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd < 0) return true;  // no lock available: measure anyway rather than fail the run
        auto start = std::chrono::steady_clock::now();
        while (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > give_up_after_s) return false;
            usleep(20000);
        }
#else
        (void)give_up_after_s;
#endif
        return true;
    }

    ~PantheonMatrixTurn() {
#ifdef PANTHEON_HAVE_FLOCK
        if (fd >= 0) { flock(fd, LOCK_UN); close(fd); }
#endif
    }
};

#endif  // PANTHEON_P2P_MATRIX_STATS_H
