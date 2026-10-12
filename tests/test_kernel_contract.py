"""Exit contract and verification coverage of the workload kernels, on the MOCK backend.

Contract with the runner:
  * a deliberate skip prints a line starting with "Skipping" on STDOUT, no
    "Throughput:" line, and exits 0;
  * a failure exits non-zero with the reason on STDERR.

The verification tests compile each kernel's own verifier into a tiny harness
and flip one bit in ONE 32-bit lane of one 128-bit element. A verifier that only
looks at the .x lane passes a flip in y, z or w, which is what these catch.

They need g++ and run in a few seconds each; they skip when there is none.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
KERNELS = ROOT / "kernels"

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++ for the MOCK backend")

MOCK_FLAGS = ["-std=c++14", "-O1", "-DPANTHEON_MOCK", "-Wno-unknown-pragmas", "-pthread",
              f"-I{KERNELS / 'common'}"]


@pytest.fixture(scope="module")
def build_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("mock-kernels")


def _compile(source, out):
    if not out.exists():
        proc = subprocess.run(["g++", *MOCK_FLAGS, str(source), "-o", str(out)],
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-3000:]
    return out


def _kernel(build_dir, name):
    return _compile(KERNELS / name / f"{name}.cpp", build_dir / name)


def _run(binary, *args, timeout=120):
    return subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True, timeout=timeout)


# --- deliberate skips ----------------------------------------------------------

# On the MOCK backend each of these has nothing to run on: no ray tracing
# backend, no NVENC, one GPU, no matrix cores.
SKIPPING = ["rt_virus", "media_enc_virus", "all_reduce", "p2p_thrasher", "mma_virus"]


@pytest.mark.parametrize("name", SKIPPING)
@pytest.mark.parametrize("verify", [False, True])
def test_deliberate_skip_says_so_on_stdout_and_exits_zero(build_dir, name, verify):
    args = [0, 1, 5] + (["--verify"] if verify else [])
    proc = _run(_kernel(build_dir, name), *args)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert any(line.startswith("Skipping: ") and len(line) > len("Skipping: ") for line in lines), proc.stdout
    # A zero throughput next to a skip is a measurement nobody took.
    assert not any("Throughput:" in line for line in lines), proc.stdout
    if verify:
        # A verification that could not run must never read as one that passed.
        assert not any(line.startswith("Verification: PASS") for line in lines), proc.stdout


# --- helpers shared by the verifier harnesses ----------------------------------

def test_uint4_differs_looks_at_every_lane(build_dir, tmp_path):
    source = tmp_path / "lanes.cpp"
    source.write_text("""
#include "common.h"
int main() {
    uint4 want = make_uint4(1, 2, 3, 4);
    if (pantheon_uint4_differs(want, want)) return 1;
    for (int lane = 0; lane < 4; ++lane) {
        uint4 got = want;
        unsigned int* p = (lane == 0) ? &got.x : (lane == 1) ? &got.y : (lane == 2) ? &got.z : &got.w;
        *p ^= 0x80000000u;
        if (!pantheon_uint4_differs(got, want)) return 10 + lane;
        if (pantheon_uint4_first_bad_lane(got, want, 99u) != *p) return 20 + lane;
    }
    return pantheon_uint4_first_bad_lane(want, want, 99u) == 99u ? 0 : 2;
}
""")
    proc = _run(_compile(source, build_dir / "lanes"))
    assert proc.returncode == 0, proc.returncode


LANES = ["x", "y", "z", "w"]


def _verifier_harness(tmp_path, kernel_cpp, body):
    """Build the kernel's own source with main renamed, plus a harness main."""
    source = tmp_path / "harness.cpp"
    source.write_text(f"""
#define main kernel_under_test_main
#include "{kernel_cpp}"
#undef main
int main() {{
    const size_t n = 64;
    uint4* data = nullptr; unsigned int* errors = nullptr;
    hipMalloc(&data, n * sizeof(uint4));
    hipMalloc(&errors, sizeof(unsigned int));
    int rc = 0;
{body}
    hipFree(data); hipFree(errors);
    return rc;
}}
""")
    return source


# (kernel source, initialiser call, verifier call) -- each verifier compares the
# whole element against what the initialiser wrote.
VERIFIERS = {
    "llm_decode": ("llm_decode/llm_decode.cpp", "initialize_decode_kv(data, n, 0);",
                   "verify_decode_kv(data, n, errors);"),
    "llm_prefill": ("llm_prefill/llm_prefill.cpp", "initialize_prefill(data, n, 0);",
                    "verify_prefill(data, n, errors);"),
    "graph_replay": ("graph_replay/graph_replay.cpp", "graph_initialize(data, n, 0);",
                     "graph_verify_input(data, n, errors);"),
    # Every proxy workload built on the shared template verifies the same way.
    "moe_router": ("moe_router/moe_router.cpp", "ai_initialize(data, n, 0);",
                   "ai_verify(data, n, errors);"),
    "kv_cache_churn": ("kv_cache_churn/kv_cache_churn.cpp", "initialize_churn(data, n, 0);",
                       "verify_churn(data, n, errors);"),
}


@pytest.mark.parametrize("lane", LANES)
@pytest.mark.parametrize("name", sorted(VERIFIERS))
def test_verifier_catches_a_flip_in_any_lane(build_dir, tmp_path, name, lane):
    rel, init, verify = VERIFIERS[name]
    body = f"""
    {init}
    hipMemset(errors, 0, sizeof(unsigned int));
    {verify}
    if (*errors != 0) rc = 1;                       // clean data must pass
    data[5].{lane} ^= 0x00000100u;                  // one bit, one lane
    hipMemset(errors, 0, sizeof(unsigned int));
    {verify}
    if (*errors != 1) rc = 2;                       // and the flip must be seen
"""
    source = _verifier_harness(tmp_path, KERNELS / rel, body)
    proc = _run(_compile(source, build_dir / f"verify_{name}_{lane}"))
    assert proc.returncode == 0, f"{name}: harness rc={proc.returncode} (1=clean data failed, 2=flip in .{lane} missed)"


@pytest.mark.parametrize("lane", LANES)
def test_galpat_catches_a_flip_in_any_lane(build_dir, tmp_path, lane):
    body = f"""
    init_galpat_region(data, 0, n, 2);
    hipMemset(errors, 0, sizeof(unsigned int));
    galpat_kernel(data, 0, n, 16, 2, errors, pantheon_fault_log_none());
    if (*errors != 0) rc = 1;
    data[3].{lane} ^= 0x00000100u;
    hipMemset(errors, 0, sizeof(unsigned int));
    galpat_kernel(data, 0, n, 16, 2, errors, pantheon_fault_log_none());
    if (*errors == 0) rc = 2;
"""
    source = _verifier_harness(tmp_path, KERNELS / "galpat/galpat.cpp", body)
    proc = _run(_compile(source, build_dir / f"verify_galpat_{lane}"))
    assert proc.returncode == 0, f"galpat: harness rc={proc.returncode} (1=clean data failed, 2=flip in .{lane} missed)"


# --- whole-binary behaviour ------------------------------------------------------

@pytest.mark.parametrize("name", ["galpat", "llm_prefill", "graph_replay", "kv_cache_churn",
                                  "memory_thermal_asym"])
def test_clean_verify_passes_and_injected_fault_fails(build_dir, name):
    binary = _kernel(build_dir, name)
    clean = _run(binary, 0, 1, 5, "--verify")
    assert clean.returncode == 0, clean.stdout[-500:] + clean.stderr[-500:]
    assert "Verification: PASS" in clean.stdout
    injected = _run(binary, 0, 1, 5, "--verify", "--inject_error")
    assert injected.returncode != 0, injected.stdout[-500:]
    assert "Verification: FAIL" in injected.stdout or "SDC FAULT" in injected.stdout


def _allocation_mib(proc):
    for line in proc.stdout.splitlines():
        if line.strip().startswith("-> Allocation:"):
            return int(line.split(":", 1)[1].split()[0])
    raise AssertionError(proc.stdout)


def test_memory_thermal_asym_honours_mem(build_dir):
    binary = _kernel(build_dir, "memory_thermal_asym")
    small = _allocation_mib(_run(binary, 0, 1, 5, "--warmup_iters", 0))
    large = _allocation_mib(_run(binary, 0, 1, 10, "--warmup_iters", 0))
    # The mock reports 1e9 bytes free, so 5 percent and 10 percent are 47 and 95 MiB.
    assert 40 <= small <= 50, small
    assert 90 <= large <= 100, large


@pytest.mark.parametrize("loops", [1, 100, 500, 2000])
def test_memory_thermal_asym_injects_whatever_the_loop_count(build_dir, loops):
    # Injection used to fire only at loop iteration 500, so it never happened
    # with --kernel_loops 500 or fewer and a verification that could not fail
    # reported PASS.
    proc = _run(_kernel(build_dir, "memory_thermal_asym"), 0, 1, 5, "--verify", "--inject_error",
                "--kernel_loops", loops, "--warmup_iters", 0)
    assert proc.returncode != 0, proc.stdout[-500:]
    assert "Verification: FAIL" in proc.stdout
