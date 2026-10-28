"""nccl_all_reduce: who takes part, the arithmetic, the sharing of the unique id, the loading of
NCCL, how the runner starts the copies, and the place of a failure in the verdict.

The collective itself needs two or more GPUs and the real library and is not exercised here.
The header that holds everything else is compiled on its own, and the library is stood in for
by a small fake one that checks the calls are made the way NCCL's header says: above all that
the unique id is passed by value, which a wrong function type would get wrong without a
compiler noticing. The workload's deliberate skip on the mock backend is covered with the
other skips in test_kernel_contract.py.
"""
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import pantheon

ROOT = Path(__file__).resolve().parent.parent
HEADER_DIR = ROOT / "kernels" / "common"

needs_gpp = pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")

# A stand-in for libnccl.so.2 with the six calls the workload makes.
FAKE_NCCL = r'''
#include <cstring>
#include <cstddef>
extern "C" {
struct FakeId { char internal[128]; };
static FakeId made;
int ncclGetVersion(int* v) { *v = 22005; return 0; }
int ncclGetUniqueId(FakeId* id) { for (int i = 0; i < 128; i++) id->internal[i] = (char)(i * 7 + 3); made = *id; return 0; }
// The id arrives by value; anything but the bytes ncclGetUniqueId made is refused.
int ncclCommInitRank(void** comm, int nranks, FakeId id, int rank) {
    if (std::memcmp(&id, &made, sizeof id) != 0) return 2;
    *comm = (void*)(long)(1000 + rank * 10 + nranks);
    return 0;
}
int ncclAllReduce(const void* send, void* recv, size_t count, int type, int op, void* comm, void* stream) {
    if (type != 7 || op != 0 || comm == nullptr) return 3;
    const float* in = (const float*)send; float* out = (float*)recv;
    for (size_t i = 0; i < count; i++) out[i] = in[i] * 2.0f;
    return 0;
}
int ncclCommDestroy(void*) { return 0; }
const char* ncclGetErrorString(int r) { return r == 2 ? "invalid argument" : "fake error"; }
}
'''


def _build(tmp_path, source_text, name, extra=()):
    source = tmp_path / f"{name}.cpp"
    source.write_text(source_text)
    binary = tmp_path / name
    build = subprocess.run(["g++", "-std=c++14", "-O1", f"-I{HEADER_DIR}", str(source), "-o", str(binary), *extra],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr[-3000:]
    return binary


def _run_harness(tmp_path, body, timeout=60, preamble=""):
    binary = _build(tmp_path, '#include "nccl_collective.h"\n#include <cstdio>\n#include <sys/wait.h>\n' + preamble +
                    '\nint main() {\n' + body + '\n}\n', "harness", extra=["-ldl"])
    return subprocess.run([str(binary)], capture_output=True, text=True, timeout=timeout)


def _fake_library(tmp_path, name="libnccl.so.2", source=FAKE_NCCL):
    src = tmp_path / (name + ".cpp")
    src.write_text(source)
    lib = tmp_path / name
    build = subprocess.run(["g++", "-std=c++14", "-shared", "-fPIC", str(src), "-o", str(lib)], capture_output=True, text=True)
    assert build.returncode == 0, build.stderr[-3000:]
    return lib


# --- who takes part -----------------------------------------------------------------

@needs_gpp
def test_a_copy_finds_its_rank_among_the_devices_the_runner_names(tmp_path):
    proc = _run_harness(tmp_path, r'''
    PantheonCollectiveGroup g = pantheon_collective_group(3, "2,3,5", "run-1_a");
    if (!g.error.empty() || g.rank != 1 || g.devices.size() != 3 || g.devices[2] != 5 || g.token != "run-1_a") return 1;
    g = pantheon_collective_group(2, "2,3", "tok");
    if (!g.error.empty() || g.rank != 0) return 2;
    return 0;''')
    assert proc.returncode == 0, f"check {proc.returncode} failed: {proc.stderr}"


@needs_gpp
def test_a_copy_without_a_proper_group_says_why_and_has_no_rank(tmp_path):
    proc = _run_harness(tmp_path, r'''
    struct Case { int own; const char* peers; const char* token; const char* fragment; };
    Case cases[] = {
        {0, nullptr, "tok", "not set"},            // started by hand
        {0, "0,1", nullptr, "not set"},
        {0, "", "tok", "not set"},
        {4, "0,1,2", "tok", "not in PANTHEON_PEER_GPUS"},
        {0, "0,0,1", "tok", "twice"},
        {0, "0", "tok", "fewer than two"},
        {0, "0,x", "tok", "not a list"},
        {0, "0,,1", "tok", "not a list"},
        {0, "0,-1", "tok", "not a list"},
        {0, "0,1", "../etc", "may only hold"},      // the token becomes part of a file name
    };
    int n = 0;
    for (const Case& c : cases) {
        n++;
        PantheonCollectiveGroup g = pantheon_collective_group(c.own, c.peers, c.token);
        if (g.error.find(c.fragment) == std::string::npos || g.rank != -1) { printf("case %d: '%s'\n", n, g.error.c_str()); return n; }
    }
    return 0;''')
    assert proc.returncode == 0, f"case {proc.returncode} failed: {proc.stdout}{proc.stderr}"


# --- the arithmetic -------------------------------------------------------------------

@needs_gpp
def test_bus_bandwidth_is_the_figure_nccl_tests_reports(tmp_path):
    proc = _run_harness(tmp_path, r'''
    auto near = [](double a, double b) { return a > b - 1e-9 && a < b + 1e-9; };
    if (!near(pantheon_allreduce_bus_factor(2), 1.0)) return 1;
    if (!near(pantheon_allreduce_bus_factor(4), 1.5)) return 2;
    if (!near(pantheon_allreduce_bus_factor(8), 1.75)) return 3;
    if (pantheon_allreduce_bus_factor(1) != 0.0) return 4;
    // 256 MiB in 2 ms on 8 ranks: 134.2 GB/s of algorithm bandwidth, 234.9 GB/s of bus bandwidth.
    double bytes = 256.0 * 1024 * 1024;
    if (!near(pantheon_allreduce_bus_bandwidth(bytes, 0.002, 8), bytes / 1e9 / 0.002 * 1.75)) return 5;
    if (pantheon_allreduce_bus_bandwidth(bytes, 0.0, 8) != 0.0) return 6;
    return 0;''')
    assert proc.returncode == 0, f"check {proc.returncode} failed: {proc.stderr}"


@needs_gpp
def test_the_expected_sum_is_exact_in_a_float_and_each_rank_sends_its_own_number(tmp_path):
    proc = _run_harness(tmp_path, r'''
    if (pantheon_allreduce_input(0) != 1.0f || pantheon_allreduce_input(7) != 8.0f) return 1;
    if (pantheon_allreduce_expected(2) != 3.0f || pantheon_allreduce_expected(8) != 36.0f) return 2;
    // The sum of the inputs, added in any order, is the expected value.
    for (int ranks = 2; ranks <= 64; ranks++) {
        float forward = 0.0f, backward = 0.0f;
        for (int r = 0; r < ranks; r++) forward += pantheon_allreduce_input(r);
        for (int r = ranks - 1; r >= 0; r--) backward += pantheon_allreduce_input(r);
        if (forward != pantheon_allreduce_expected(ranks) || backward != forward) return 10 + ranks;
    }
    return 0;''')
    assert proc.returncode == 0, f"check {proc.returncode} failed: {proc.stderr}"


@needs_gpp
def test_the_version_code_reads_in_both_of_nccls_schemes(tmp_path):
    proc = _run_harness(tmp_path, r'''
    if (pantheon_nccl_version_string(22005) != "2.20.5") return 1;     // from 2.9: major*10000 + minor*100 + patch
    if (pantheon_nccl_version_string(21903) != "2.19.3") return 2;
    if (pantheon_nccl_version_string(2805) != "2.8.5") return 3;       // before: major*1000 + minor*100 + patch
    return 0;''')
    assert proc.returncode == 0, f"check {proc.returncode} failed: {proc.stderr}"


# --- sharing the unique id ----------------------------------------------------------

@needs_gpp
def test_the_unique_id_reaches_a_copy_that_was_waiting_for_it(tmp_path):
    proc = _run_harness(tmp_path, r'''
    std::string path = pantheon_collective_id_path("pytest-handoff");
    pantheon_collective_remove(path);
    char sent[128];
    for (int i = 0; i < 128; i++) sent[i] = (char)(i * 5 + 1);
    pid_t pid = fork();
    if (pid == 0) {                                   // another rank, there before the id is
        char got[128];
        if (!pantheon_collective_wait(path, got, sizeof got, 20.0)) _exit(3);
        _exit(std::memcmp(got, sent, sizeof got) == 0 ? 0 : 4);
    }
    usleep(200000);
    if (!pantheon_collective_publish(path, sent, sizeof sent)) return 1;
    int status;
    if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) return 2;
    pantheon_collective_remove(path);
    if (access(path.c_str(), F_OK) == 0) return 5;    // removed
    return 0;''')
    assert proc.returncode == 0, f"check {proc.returncode} failed: {proc.stderr}"


@needs_gpp
def test_a_copy_gives_up_when_the_id_never_comes_and_a_half_written_one_is_never_read(tmp_path):
    proc = _run_harness(tmp_path, r'''
    std::string path = pantheon_collective_id_path("pytest-never");
    pantheon_collective_remove(path);
    char got[128];
    auto start = std::chrono::steady_clock::now();
    if (pantheon_collective_wait(path, got, sizeof got, 0.3)) return 1;
    double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (waited < 0.3 || waited > 5.0) return 2;
    // A file that is not yet all there is not an id.
    FILE* f = fopen(path.c_str(), "wb"); fwrite("short", 1, 5, f); fclose(f);
    if (pantheon_collective_wait(path, got, sizeof got, 0.2)) return 3;
    pantheon_collective_remove(path);
    return 0;''')
    assert proc.returncode == 0, f"check {proc.returncode} failed: {proc.stderr}"


# --- the library --------------------------------------------------------------------

@needs_gpp
def test_the_library_is_loaded_and_the_id_is_passed_by_value(tmp_path):
    lib = _fake_library(tmp_path)
    proc = _run_harness(tmp_path, r'''
    PantheonNccl nccl;
    if (!nccl.load(pantheon_nccl_candidates("''' + str(lib) + r'''"))) { printf("%s\n", nccl.error.c_str()); return 1; }
    if (nccl.path != "''' + str(lib) + r'''") return 2;
    int version = 0;
    if (nccl.GetVersion(&version) != 0 || version != 22005) return 3;
    PantheonNcclUniqueId id;
    if (nccl.GetUniqueId(&id) != 0) return 4;
    void* comm = nullptr;
    if (nccl.CommInitRank(&comm, 2, id, 1) != 0 || comm == nullptr) return 5;      // refused if the 128 bytes did not arrive intact
    PantheonNcclUniqueId other = id; other.internal[77] ^= 1;
    if (nccl.CommInitRank(&comm, 2, other, 1) != 2) return 6;                       // and a different id is refused
    if (nccl.describe(2) != "invalid argument (code 2)") return 7;
    float in[4] = {1, 2, 3, 4}, out[4] = {0, 0, 0, 0};
    if (nccl.AllReduce(in, out, 4, PANTHEON_NCCL_FLOAT32, PANTHEON_NCCL_SUM, comm, nullptr) != 0) return 8;
    if (out[3] != 8.0f) return 9;
    if (nccl.CommDestroy(comm) != 0) return 10;
    return 0;''')
    assert proc.returncode == 0, f"check {proc.returncode} failed: {proc.stdout}{proc.stderr}"


@needs_gpp
def test_a_missing_library_or_symbol_is_a_reason_not_a_crash(tmp_path):
    incomplete = _fake_library(tmp_path, "libincomplete.so", FAKE_NCCL.replace("int ncclAllReduce", "int ncclAllReduceX"))
    proc = _run_harness(tmp_path, r'''
    PantheonNccl none;
    if (none.load({"/nonexistent/libnccl.so.2", "libnccl-that-does-not-exist.so"})) return 1;
    if (none.error.find("no NCCL library found (tried /nonexistent/libnccl.so.2, libnccl-that-does-not-exist.so;") == std::string::npos) { printf("%s\n", none.error.c_str()); return 2; }
    if (none.error.find("set PANTHEON_NCCL_LIB") == std::string::npos) return 5;
    PantheonNccl partial;
    if (partial.load({"''' + str(incomplete) + r'''"})) return 3;
    if (partial.error.find("has no ncclAllReduce") == std::string::npos) { printf("%s\n", partial.error.c_str()); return 4; }
    return 0;''')
    assert proc.returncode == 0, f"check {proc.returncode} failed: {proc.stdout}{proc.stderr}"


def test_the_search_order_is_the_named_file_then_the_usual_names():
    source = (HEADER_DIR / "nccl_collective.h").read_text()
    assert source.index("override_path") < source.index('"libnccl.so.2"') < source.index('"libnccl.so"')
    assert '"librccl.so.1"' in source          # the same call on ROCm


# --- the workload -------------------------------------------------------------------

def test_the_copies_stop_together_or_the_last_ones_would_wait_for_ever():
    """Each copy decides for itself that its time is up; if each then stopped on its own, the copies still
    calling the collective would wait for those that had left. They all-reduce a flag and stop on its sum."""
    source = (ROOT / "kernels" / "nccl_all_reduce" / "nccl_all_reduce.cpp").read_text()
    assert "nccl.AllReduce(d_flag, d_flag, 1," in source
    assert source.index("out_of_time") < source.index("stop = out_of_time > 0.0f")
    assert "pantheon_collective_deadline(" in source   # a copy that never arrives cannot hold the rest for ever


def test_nccl_all_reduce_is_registered_in_the_interconnect_suite_and_needs_two_gpus():
    assert pantheon.TEST_REGISTRY["nccl_all_reduce"]["bin"] == "nccl_all_reduce"
    assert "nccl_all_reduce" in pantheon.SUITES["interconnect"]
    assert pantheon.MIN_GPUS_FOR_TEST["nccl_all_reduce"] == 2
    assert "nccl_all_reduce" in pantheon.COLLECTIVE_TESTS
    runnable, skipped = pantheon.filter_gpu_compatible_tests(["nccl_all_reduce", "memory_read"], 1)
    assert runnable == ["memory_read"] and skipped == [("nccl_all_reduce", 2)]


def test_the_header_is_in_the_makefile_dependencies():
    assert "kernels/common/nccl_collective.h" in (ROOT / "Makefile").read_text()


# --- how the runner starts the copies ------------------------------------------------

def test_each_copy_is_told_who_takes_part_and_what_the_run_is_called(monkeypatch):
    monkeypatch.setattr(pantheon, "find_pip_nccl", lambda: None)
    ordinal, environment = pantheon.workload_launch("nccl_all_reduce", 3, "CUDA", environ={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
                                                    peers=[2, 3], token="abc123")
    assert ordinal == 3
    assert environment["PANTHEON_PEER_GPUS"] == "2,3"
    assert environment["PANTHEON_RUN_TOKEN"] == "abc123"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"        # every card stays visible
    assert "PANTHEON_NCCL_LIB" not in environment


def test_the_environment_given_is_not_changed_and_other_workloads_are_not_told_anything(monkeypatch):
    monkeypatch.setattr(pantheon, "find_pip_nccl", lambda: None)
    given = {"CUDA_VISIBLE_DEVICES": "0,1"}
    pantheon.workload_launch("nccl_all_reduce", 0, "CUDA", environ=given, peers=[0, 1], token="t")
    assert given == {"CUDA_VISIBLE_DEVICES": "0,1"}
    # Peer workloads that do not form a group see what they saw before, and so does a collective launched without the group.
    assert pantheon.workload_launch("p2p_matrix", 1, "CUDA", environ=given, peers=[0, 1], token="t") == (1, None)
    assert pantheon.workload_launch("nccl_all_reduce", 1, "CUDA", environ=given) == (1, None)
    # A workload that needs one card keeps its own card only.
    ordinal, isolated = pantheon.workload_launch("memory_read", 1, "CUDA", environ=given, peers=[0, 1], token="t")
    assert ordinal == 0 and isolated["CUDA_VISIBLE_DEVICES"] == "1" and "PANTHEON_PEER_GPUS" not in isolated


def test_the_group_is_told_on_amd_too_but_nccl_is_only_looked_for_on_nvidia(monkeypatch):
    monkeypatch.setattr(pantheon, "find_pip_nccl", lambda: "/pip/libnccl.so.2")
    _, amd = pantheon.workload_launch("nccl_all_reduce", 1, "ROCM", environ={}, peers=[0, 1], token="t")
    assert amd["PANTHEON_PEER_GPUS"] == "0,1" and "PANTHEON_NCCL_LIB" not in amd


def test_a_pip_installed_nccl_is_handed_to_the_workload_unless_the_user_named_one(monkeypatch):
    monkeypatch.setattr(pantheon, "find_pip_nccl", lambda: "/pip/nvidia/nccl/lib/libnccl.so.2")
    _, found = pantheon.workload_launch("nccl_all_reduce", 0, "CUDA", environ={}, peers=[0, 1], token="t")
    assert found["PANTHEON_NCCL_LIB"] == "/pip/nvidia/nccl/lib/libnccl.so.2"
    _, named = pantheon.workload_launch("nccl_all_reduce", 0, "CUDA", environ={"PANTHEON_NCCL_LIB": "/mine/libnccl.so.2"},
                                        peers=[0, 1], token="t")
    assert named["PANTHEON_NCCL_LIB"] == "/mine/libnccl.so.2"


def test_find_pip_nccl_reads_the_package_and_is_quiet_without_it(monkeypatch, tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libnccl.so.2").write_bytes(b"")
    monkeypatch.setattr(pantheon.importlib.util, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(tmp_path)]))
    assert pantheon.find_pip_nccl() == str(lib / "libnccl.so.2")
    monkeypatch.setattr(pantheon.importlib.util, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(tmp_path / "empty")]))
    assert pantheon.find_pip_nccl() is None
    monkeypatch.setattr(pantheon.importlib.util, "find_spec", lambda name: None)
    assert pantheon.find_pip_nccl() is None

    def missing(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(pantheon.importlib.util, "find_spec", missing)
    assert pantheon.find_pip_nccl() is None


# --- the verdict --------------------------------------------------------------------

def test_a_failed_all_reduce_is_an_interconnect_watch_not_a_fault_and_not_a_missing_run():
    row = pantheon.build_result_row("nccl_all_reduce", 3, 60, 99, 0.0, "ERR",
                                    {"max_temp": 60, "max_mem_temp": 0, "throttle_reason": "None"}, status="FAIL")
    row["RAS Status"], row["RAS Error Delta"] = "CLEAN", "None"
    verdict = pantheon.assess_gpu([row], 3, "NVIDIA H100 80GB HBM3")
    assert verdict["verdict"] == "WATCH"
    assert verdict["reasons"] == ["interconnect: nccl_all_reduce failed: the all-reduce result was wrong or the test aborted, see its log"]
    assert verdict["findings"]["incomplete"] == [] and verdict["findings"]["faults"] == []


# --- a group that does not form ---------------------------------------------------------

@needs_gpp
def test_the_id_file_goes_with_rank_zero_and_a_missed_deadline_ends_the_copy_with_a_reason(tmp_path):
    proc = _run_harness(tmp_path, r'''
    std::string kept = pantheon_collective_id_path("pytest-raii");
    {
        PantheonCollectiveIdFile file;
        char id[128] = {1};
        if (!pantheon_collective_publish(kept, id, sizeof id)) return 1;
        file.path = kept;
        if (access(kept.c_str(), F_OK) != 0) return 2;
    }
    if (access(kept.c_str(), F_OK) == 0) return 3;                    // gone with its owner

    std::string hung = pantheon_collective_id_path("pytest-deadline");
    char id[128] = {2};
    if (!pantheon_collective_publish(hung, id, sizeof id)) return 4;
    int quiet[2];
    if (pipe(quiet) != 0) return 5;
    pid_t pid = fork();
    if (pid == 0) {                                                   // a rank 0 whose group never forms
        dup2(quiet[1], 2);
        pantheon_collective_deadline(1, hung);
        for (;;) pause();
    }
    close(quiet[1]);
    char said[200] = {0};
    ssize_t n = read(quiet[0], said, sizeof said - 1);
    int status = 0;
    if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 142) return 6;
    if (n <= 0 || std::string(said).find("did not form or finish in time") == std::string::npos) return 7;
    if (access(hung.c_str(), F_OK) == 0) return 8;                    // removed by the deadline
    return 0;''')
    assert proc.returncode == 0, f"check {proc.returncode} failed: {proc.stdout}{proc.stderr}"
