#ifndef PANTHEON_NCCL_COLLECTIVE_H
#define PANTHEON_NCCL_COLLECTIVE_H

// The parts of the nccl_all_reduce workload that are not GPU calls: who takes part,
// the arithmetic, how the copies share the library's unique id, and the loading of
// NCCL (RCCL on ROCm). Kept apart so they can be compiled and checked on a machine
// with no GPU.
//
// The library is loaded at run time rather than linked, for the reason vendor_gemm.h
// gives: every binary in this tree is built with one link line, and a hard dependency
// would stop every other test from starting wherever the loader cannot find it. With
// dlopen a missing library is a skip with a reason.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#ifndef PANTHEON_MOCK
#include <dlfcn.h>
#endif

// ---------------------------------------------------------------------------
// Who takes part.
//
// Pantheon starts one copy per selected GPU. A collective only works if every copy
// calls it, so each copy has to know which devices are in the group and where it
// sits among them. The runner says so in PANTHEON_PEER_GPUS (the device numbers as
// the workload sees them, comma separated) and gives the run a name in
// PANTHEON_RUN_TOKEN, which the copies use to find each other.
// ---------------------------------------------------------------------------
struct PantheonCollectiveGroup {
    std::vector<int> devices;   // the devices that take part, in rank order
    int rank = -1;              // the place of this copy's device in devices
    std::string token;          // names this run's rendezvous
    std::string error;          // why there is no group; empty when there is one
};

inline PantheonCollectiveGroup pantheon_collective_group(int own_device, const char* peers, const char* token) {
    PantheonCollectiveGroup group;
    if (!peers || !*peers || !token || !*token) {
        group.error = "PANTHEON_PEER_GPUS and PANTHEON_RUN_TOKEN are not set: the copies of this workload find each other "
                      "through them, so start it with pantheon rather than by hand";
        return group;
    }
    for (const char* c = token; *c; ++c) {
        bool plain = (*c >= '0' && *c <= '9') || (*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') || *c == '-' || *c == '_';
        if (!plain) {
            group.error = "PANTHEON_RUN_TOKEN may only hold letters, digits, - and _";
            return group;
        }
    }
    std::string list(peers);
    size_t begin = 0;
    while (begin <= list.size()) {
        size_t end = list.find(',', begin);
        if (end == std::string::npos) end = list.size();
        std::string item = list.substr(begin, end - begin);
        char* stop = nullptr;
        long value = std::strtol(item.c_str(), &stop, 10);
        if (item.empty() || *stop != '\0' || value < 0) {
            group.error = "PANTHEON_PEER_GPUS is not a list of device numbers: " + list;
            group.devices.clear();
            return group;
        }
        if (std::find(group.devices.begin(), group.devices.end(), (int)value) != group.devices.end()) {
            group.error = "PANTHEON_PEER_GPUS names a device twice: " + list;
            group.devices.clear();
            return group;
        }
        group.devices.push_back((int)value);
        begin = end + 1;
    }
    if (group.devices.size() < 2) {
        group.error = "fewer than two GPUs selected";
        return group;
    }
    auto place = std::find(group.devices.begin(), group.devices.end(), own_device);
    if (place == group.devices.end()) {
        group.error = "device " + std::to_string(own_device) + " is not in PANTHEON_PEER_GPUS (" + list + ")";
        return group;
    }
    group.rank = (int)(place - group.devices.begin());
    group.token = token;
    return group;
}

// ---------------------------------------------------------------------------
// The arithmetic.
//
// An all-reduce of S bytes over n ranks makes each rank's links carry 2(n-1)/n of S.
// The "bus bandwidth" that nccl-tests reports multiplies the plain S/time by that
// factor, so the figure can be compared with the speed of a link whatever the size
// of the group.
//
// Every rank starts with its own rank number plus one in each element, so the sum is
// n(n+1)/2. These are whole numbers a float holds exactly, so the answer does not
// depend on the order the library adds in, and a single wrong bit shows.
// ---------------------------------------------------------------------------
inline double pantheon_allreduce_bus_factor(int ranks) { return ranks < 2 ? 0.0 : 2.0 * (ranks - 1) / ranks; }

inline double pantheon_allreduce_bus_bandwidth(double bytes, double seconds, int ranks) {
    return seconds > 0.0 ? bytes / 1e9 / seconds * pantheon_allreduce_bus_factor(ranks) : 0.0;
}

inline float pantheon_allreduce_input(int rank) { return (float)(rank + 1); }

inline float pantheon_allreduce_expected(int ranks) { return 0.5f * (float)ranks * (float)(ranks + 1); }

// NCCL reports its version as one number: major*10000 + minor*100 + patch from 2.9,
// major*1000 + minor*100 + patch before.
inline std::string pantheon_nccl_version_string(int code) {
    int major = code >= 10000 ? code / 10000 : code / 1000;
    int minor = code >= 10000 ? (code / 100) % 100 : (code / 100) % 10;
    int patch = code % 100;
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}

// ---------------------------------------------------------------------------
// Sharing the unique id.
//
// NCCL joins the ranks of a group through an id that one rank makes and all the
// others are given. The copies are separate processes, so rank 0 writes the id to a
// file named for the run and the others wait for it. It is written under another
// name and renamed, so a reader never sees half of it.
// ---------------------------------------------------------------------------
inline std::string pantheon_collective_id_path(const std::string& token) { return "/tmp/pantheon-nccl-" + token + ".id"; }

inline bool pantheon_collective_publish(const std::string& path, const void* bytes, size_t size) {
    std::string partial = path + "." + std::to_string((long)getpid());
    int fd = open(partial.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return false;
    const char* data = static_cast<const char*>(bytes);
    size_t written = 0;
    while (written < size) {
        ssize_t n = write(fd, data + written, size - written);
        if (n <= 0) { close(fd); unlink(partial.c_str()); return false; }
        written += (size_t)n;
    }
    close(fd);
    if (rename(partial.c_str(), path.c_str()) != 0) { unlink(partial.c_str()); return false; }
    return true;
}

// Wait for the id; false if it does not appear within give_up_after_s seconds.
inline bool pantheon_collective_wait(const std::string& path, void* bytes, size_t size, double give_up_after_s) {
    auto start = std::chrono::steady_clock::now();
    for (;;) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd >= 0) {
            ssize_t n = read(fd, bytes, size);
            close(fd);
            if (n == (ssize_t)size) return true;
        }
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > give_up_after_s) return false;
        usleep(20000);
    }
}

inline void pantheon_collective_remove(const std::string& path) { unlink(path.c_str()); }

// The id file of rank 0, removed however rank 0 leaves (including the deadline below).
struct PantheonCollectiveIdFile {
    std::string path;
    ~PantheonCollectiveIdFile() { if (!path.empty()) unlink(path.c_str()); }
};

// A copy that never arrives would hold the others in the library for ever. After the
// given time the process says so, removes the id file it is responsible for (if any),
// and ends with 142, the status a process killed by SIGALRM reports.
static char pantheon_collective_deadline_path[256];

inline void pantheon_collective_on_deadline(int) {
    static const char message[] = "[PANTHEON ERROR] NCCL_ALL_REDUCE: the group did not form or finish in time\n";
    if (pantheon_collective_deadline_path[0]) unlink(pantheon_collective_deadline_path);
    ssize_t ignored = write(2, message, sizeof message - 1);
    (void)ignored;
    _exit(142);
}

inline void pantheon_collective_deadline(unsigned int seconds, const std::string& id_path) {
    std::snprintf(pantheon_collective_deadline_path, sizeof pantheon_collective_deadline_path, "%s", id_path.c_str());
    std::signal(SIGALRM, pantheon_collective_on_deadline);
    alarm(seconds);
}

// ---------------------------------------------------------------------------
// The library.
// ---------------------------------------------------------------------------
#ifndef PANTHEON_MOCK

// NCCL's ncclUniqueId: an opaque 128 bytes, passed to ncclCommInitRank by value.
struct PantheonNcclUniqueId { char internal[128]; };

// The values of ncclDataType_t and ncclRedOp_t that are used, as nccl.h and rccl.h give them.
enum { PANTHEON_NCCL_SUM = 0, PANTHEON_NCCL_FLOAT32 = 7 };

// Where to look: the file named in PANTHEON_NCCL_LIB first (a pip install of NCCL keeps
// it out of the loader's path), then the usual names.
inline std::vector<std::string> pantheon_nccl_candidates(const char* override_path) {
    std::vector<std::string> names;
    if (override_path && *override_path) names.push_back(override_path);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    names.push_back("librccl.so.1");
    names.push_back("librccl.so");
#else
    names.push_back("libnccl.so.2");
    names.push_back("libnccl.so");
#endif
    return names;
}

struct PantheonNccl {
    void* handle = nullptr;
    std::string path;     // the file that was loaded
    std::string error;    // why it could not be, when it could not

    int (*GetVersion)(int*) = nullptr;
    int (*GetUniqueId)(PantheonNcclUniqueId*) = nullptr;
    int (*CommInitRank)(void**, int, PantheonNcclUniqueId, int) = nullptr;
    int (*AllReduce)(const void*, void*, size_t, int, int, void*, void*) = nullptr;
    int (*CommDestroy)(void*) = nullptr;
    const char* (*GetErrorString)(int) = nullptr;

    template <typename F>
    bool symbol(const char* name, F* out) {
        *out = reinterpret_cast<F>(dlsym(handle, name));
        if (!*out) error = path + " has no " + name;
        return *out != nullptr;
    }

    bool load(const std::vector<std::string>& candidates) {
        std::string tried, last;
        for (const std::string& candidate : candidates) {
            handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (handle) { path = candidate; break; }
            const char* why = dlerror();
            last = why ? why : "could not be loaded";
            tried += (tried.empty() ? "" : ", ") + candidate;
        }
        if (!handle) {
            error = "no NCCL library found (tried " + tried + "; last error: " + last +
                    "); install NCCL (RCCL on AMD) or set PANTHEON_NCCL_LIB to its path";
            return false;
        }
        return symbol("ncclGetVersion", &GetVersion) && symbol("ncclGetUniqueId", &GetUniqueId)
            && symbol("ncclCommInitRank", &CommInitRank) && symbol("ncclAllReduce", &AllReduce)
            && symbol("ncclCommDestroy", &CommDestroy) && symbol("ncclGetErrorString", &GetErrorString);
    }

    std::string describe(int result) const {
        const char* text = GetErrorString ? GetErrorString(result) : nullptr;
        return std::string(text ? text : "unknown error") + " (code " + std::to_string(result) + ")";
    }
};

#endif  // PANTHEON_MOCK

#endif  // PANTHEON_NCCL_COLLECTIVE_H
