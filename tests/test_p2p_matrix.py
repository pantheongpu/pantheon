"""p2p_matrix: the arithmetic and turn-taking of the workload, and its place in the verdict.

The transfers themselves need two or more GPUs and are not exercised here; the
header that holds everything else is compiled on its own, and the workload's
deliberate skip on the one-GPU mock backend is covered with the other skips in
test_kernel_contract.py.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

import pantheon

ROOT = Path(__file__).resolve().parent.parent
HEADER_DIR = ROOT / "kernels" / "common"

needs_gpp = pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")


def _run_harness(tmp_path, body, timeout=60):
    source = tmp_path / "harness.cpp"
    source.write_text('#include "p2p_matrix_stats.h"\n#include <cstdio>\n#include <cmath>\n'
                     '#include <sys/wait.h>\nint main() {\n' + body + '\n}\n')
    binary = tmp_path / "harness"
    build = subprocess.run(["g++", "-std=c++14", "-O1", f"-I{HEADER_DIR}", str(source), "-o", str(binary)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr[-3000:]
    return subprocess.run([str(binary)], capture_output=True, text=True, timeout=timeout)


@needs_gpp
def test_median_and_pair_statistics(tmp_path):
    proc = _run_harness(tmp_path, r'''
    if (pantheon_median({}) != 0.0) return 1;
    if (pantheon_median({5.0}) != 5.0) return 2;
    if (pantheon_median({9.0, 1.0, 5.0}) != 5.0) return 3;
    if (pantheon_median({4.0, 1.0, 2.0, 3.0}) != 2.5) return 4;
    PantheonPairStats s = pantheon_pair_stats({230.0, 148.0, 233.0, 231.0, 148.0});
    if (s.min != 148.0 || s.slowest != 1 || s.max != 233.0 || s.median != 230.0) return 5;
    PantheonPairStats none = pantheon_pair_stats({});
    if (none.slowest != -1 || none.min != 0.0) return 6;
    // A stalled batch among healthy ones does not stand for the pair.
    if (pantheon_pair_bandwidth({240.0, 12.0, 241.0, 239.0, 240.0}) != 240.0) return 7;
    return 0;''')
    assert proc.returncode == 0, f"check {proc.returncode} failed: {proc.stderr}"


@needs_gpp
def test_every_gpu_gets_a_share_of_the_duration_and_the_rows_fit_in_it(tmp_path):
    proc = _run_harness(tmp_path, r'''
    if (pantheon_pair_seconds(60, 1) != 0.0) return 1;
    // 8 GPUs, 60 s: 56 directions share 48 s.
    double each = pantheon_pair_seconds(60, 8);
    if (std::fabs(each * 8 * 7 - 48.0) > 1e-9) return 2;
    // A very short run still gets enough time for several copies of the buffer.
    if (pantheon_pair_seconds(5, 8) != 0.1) return 3;
    // Two GPUs: two directions share 0.8 of the time.
    if (std::fabs(pantheon_pair_seconds(10, 2) * 2 - 8.0) > 1e-9) return 4;
    return 0;''')
    assert proc.returncode == 0, f"check {proc.returncode} failed: {proc.stderr}"


@needs_gpp
def test_copies_of_one_run_never_hold_the_interconnect_together(tmp_path):
    """Three children of one parent take turns: their intervals must not overlap."""
    proc = _run_harness(tmp_path, r'''
    const int children = 3;
    int pipes[children][2];
    for (int c = 0; c < children; c++) {
        if (pipe(pipes[c]) != 0) return 1;
        pid_t pid = fork();
        if (pid == 0) {
            PantheonMatrixTurn turn;
            if (!turn.acquire(30.0)) _exit(3);
            double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
            usleep(150000);
            double t1 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
            double pair[2] = {t0, t1};
            if (write(pipes[c][1], pair, sizeof pair) != (ssize_t)sizeof pair) _exit(4);
            _exit(0);
        }
    }
    double span[children][2];
    for (int c = 0; c < children; c++) {
        if (read(pipes[c][0], span[c], sizeof span[c]) != (ssize_t)sizeof span[c]) return 2;
    }
    int status;
    while (wait(&status) > 0) { if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return 5; }
    for (int a = 0; a < children; a++)
        for (int b = a + 1; b < children; b++)
            if (span[a][0] < span[b][1] && span[b][0] < span[a][1]) return 6;   // they overlapped
    return 0;''')
    assert proc.returncode == 0, f"check {proc.returncode} failed: {proc.stderr}"


@needs_gpp
def test_a_copy_gives_up_rather_than_waiting_for_ever(tmp_path):
    # Both copies are children of the harness, as the runner's copies are, so
    # they name the same lock file.
    proc = _run_harness(tmp_path, r'''
    int holding[2];
    if (pipe(holding) != 0) return 1;
    pid_t holder = fork();
    if (holder == 0) {
        PantheonMatrixTurn turn;
        if (!turn.acquire(5.0)) _exit(3);
        char go = 1;
        if (write(holding[1], &go, 1) != 1) _exit(4);
        usleep(1500000);
        _exit(0);
    }
    char go;
    if (read(holding[0], &go, 1) != 1) return 2;
    pid_t waiter = fork();
    if (waiter == 0) {
        PantheonMatrixTurn turn;
        // The holder is alive for 1.5 s more: a 0.3 s patience must run out.
        _exit(turn.acquire(0.3) ? 7 : 0);
    }
    int status = 0, holder_status = 0;
    waitpid(waiter, &status, 0);
    waitpid(holder, &holder_status, 0);
    if (!WIFEXITED(holder_status) || WEXITSTATUS(holder_status) != 0) return 8;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 9;''')
    assert proc.returncode == 0, f"check {proc.returncode} failed: {proc.stderr}"


# --- the registry and the harness ---------------------------------------------

def test_p2p_matrix_is_registered_in_the_interconnect_suite_and_needs_two_gpus():
    assert pantheon.TEST_REGISTRY["p2p_matrix"]["bin"] == "p2p_matrix"
    assert "p2p_matrix" in pantheon.SUITES["interconnect"]
    assert pantheon.MIN_GPUS_FOR_TEST["p2p_matrix"] == 2
    runnable, skipped = pantheon.filter_gpu_compatible_tests(["p2p_matrix", "memory_read"], 1)
    assert runnable == ["memory_read"] and skipped == [("p2p_matrix", 2)]


def test_p2p_matrix_keeps_every_card_visible():
    ordinal, environment = pantheon.workload_launch("p2p_matrix", 3, "CUDA", environ={"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
    assert environment is None, "it sends to the other cards, so it must see them"


def test_the_workload_reports_its_unit_and_its_verification_verdict():
    source = (ROOT / "kernels" / "p2p_matrix" / "p2p_matrix.cpp").read_text()
    assert '"Throughput: "' in source and '" GB/s"' in source
    assert 'Verification: "' in source
    assert "pantheon_skip(" in source


# --- the verdict --------------------------------------------------------------

def _row(gpu, score, status="PASS", unit="GB/s", test="p2p_matrix"):
    row = pantheon.build_result_row(test, gpu, 60, 99, score, unit,
                                    {"max_temp": 60, "max_mem_temp": 0, "throttle_reason": "None"}, status=status)
    row["RAS Status"], row["RAS Error Delta"] = "CLEAN", "None"
    return row


def _node(scores):
    return [_row(gpu, score) for gpu, score in enumerate(scores)]


def test_a_gpu_far_slower_than_the_median_is_named():
    rows = _node([233.0, 233.0, 148.0, 233.0, 233.0, 233.0, 233.0, 233.0])
    reason = pantheon.p2p_matrix_finding(rows, 2)
    assert reason == ("p2p_matrix: the slowest transfer from GPU 2 to a peer ran at 148 GB/s; "
                      "the median GPU's slowest is 233 GB/s")
    assert all(pantheon.p2p_matrix_finding(rows, gpu) is None for gpu in range(8) if gpu != 2)


def test_gpus_that_read_alike_have_no_finding():
    rows = _node([233.0, 232.0, 234.0, 233.0])
    assert [pantheon.p2p_matrix_finding(rows, gpu) for gpu in range(4)] == [None] * 4


def test_there_must_be_three_gpus_to_have_a_median():
    assert pantheon.p2p_matrix_finding(_node([233.0, 40.0]), 1) is None


def test_failed_and_skipped_rows_do_not_count():
    rows = _node([233.0, 232.0, 234.0])
    rows.append(_row(3, 0.0, status="FAIL", unit="ERR"))
    rows.append(_row(4, 0.0, status="SKIP", unit="SKIP"))
    assert pantheon.p2p_matrix_finding(rows, 3) is None
    assert pantheon.p2p_matrix_finding(rows, 0) is None


def test_other_workloads_are_not_read_as_p2p_matrix():
    rows = [_row(gpu, 400.0, test="memory_read") for gpu in range(4)]
    assert pantheon.p2p_matrix_finding(rows, 0) is None


def test_a_slow_gpu_makes_the_verdict_a_watch_listed_as_interconnect():
    rows = _node([233.0, 232.0, 148.0, 234.0])
    rows.append(_row(2, 868.0, test="memory_read"))
    verdict = pantheon.assess_gpu(rows, 2, "NVIDIA A100-SXM4-40GB")
    assert verdict["verdict"] == "WATCH"
    assert verdict["reasons"][0].startswith("interconnect: p2p_matrix: the slowest transfer from GPU 2")
    assert verdict["findings"]["interconnect"] == [pantheon.p2p_matrix_finding(rows, 2)]
    healthy = pantheon.assess_gpu(rows, 0, "NVIDIA A100-SXM4-40GB")
    assert healthy["findings"]["interconnect"] == []


def test_a_failed_p2p_matrix_is_an_interconnect_watch_not_a_fault_and_not_a_missing_run():
    """On 8 H100s with NVSwitch (2026-10-27) an injected bad copy failed verification on every GPU and read "WATCH (1 incomplete)",
    as if the test had never finished. It cannot say which GPU, link or switch is at fault, so it stays a WATCH, filed as interconnect."""
    rows = _node([396.0, 396.0, 396.0])
    rows.append(_row(3, 0.0, status="FAIL", unit="ERR"))
    verdict = pantheon.assess_gpu(rows, 3, "NVIDIA H100 80GB HBM3")
    assert verdict["verdict"] == "WATCH"
    assert verdict["reasons"] == ["interconnect: p2p_matrix failed: a peer copy arrived with wrong data or the test aborted, see its log"]
    assert verdict["findings"]["incomplete"] == []
    assert verdict["findings"]["faults"] == []
    assert verdict["findings"]["interconnect"] == ["p2p_matrix failed: a peer copy arrived with wrong data or the test aborted, see its log"]
    assert verdict["summary"] == "interconnect issue"
    assert pantheon.assess_gpu(rows, 0, "NVIDIA H100 80GB HBM3")["verdict"] == "HEALTHY"


def test_other_failed_workloads_are_still_incomplete_and_failed_memory_tests_still_a_fault():
    rows = [_row(0, 0.0, status="FAIL", unit="ERR", test="memory_read"), _row(0, 0.0, status="FAIL", unit="ERR", test="march_test")]
    verdict = pantheon.assess_gpu(rows, 0, "NVIDIA H100 80GB HBM3")
    assert verdict["verdict"] == "FAULT"
    assert verdict["findings"]["incomplete"] == ["memory_read"]
    assert verdict["findings"]["interconnect"] == []


def test_the_ratio_is_the_documented_85_percent():
    assert pantheon.P2P_MATRIX_WATCH_RATIO == 0.85
    just_above = _node([100.0, 100.0, 100.0, 86.0])
    just_below = _node([100.0, 100.0, 100.0, 84.0])
    assert pantheon.p2p_matrix_finding(just_above, 3) is None
    assert pantheon.p2p_matrix_finding(just_below, 3) is not None


def test_a_node_like_the_v100_mesh_reads_alike():
    """The real scores of 8 V100s on 2026-10-26, each its slowest single-link peer."""
    scores = [24.2536, 24.2542, 24.2545, 24.2545, 24.2533, 24.2542, 24.2544, 24.2548]
    rows = _node(scores)
    assert [pantheon.p2p_matrix_finding(rows, gpu) for gpu in range(8)] == [None] * 8
