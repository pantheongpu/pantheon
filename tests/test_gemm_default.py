import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PROGRAM = r"""
#include "vendor_gemm.h"
#include <cstdio>

int main(int argc, char** argv) {
    int bad = 0;

    PantheonGemmOptions defaults;
    if (!defaults.type.empty()) { std::printf("a format is named by default\n"); ++bad; }

    // The default is BF16 only where the part has BF16 tensor cores: NVIDIA
    // Ampere (8.0) and later. Volta (7.0) and Turing (7.5) do not.
    if (pantheon_gemm_default_has_bf16(7)) { std::printf("Volta/Turing counted as BF16\n"); ++bad; }
    if (pantheon_gemm_default_has_bf16(6)) { std::printf("Pascal counted as BF16\n"); ++bad; }
    for (int major : {8, 9, 10, 12}) {
        if (!pantheon_gemm_default_has_bf16(major)) { std::printf("cc %d.x not BF16\n", major); ++bad; }
    }
    // Not NVIDIA, or unknown: leave the decision to the library.
    if (!pantheon_gemm_default_has_bf16(-1)) { std::printf("unknown part not left to the library\n"); ++bad; }

    // Naming a format still works.
    PantheonGemmOptions named;
    char prog[] = "t", flag[] = "--gemm_type", value[] = "fp16";
    char* args[] = {prog, flag, value};
    int i = 1;
    if (!named.parse(3, args, &i) || named.type != "fp16") { std::printf("--gemm_type not parsed\n"); ++bad; }

    std::printf("%s\n", bad ? "FAIL" : "ok");
    return bad;
}
"""


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs a C++ compiler")
def test_default_gemm_format_follows_bf16_support(tmp_path):
    """A Tesla T4 scored 2.79 TFLOPS on 1.2.3 where 1.2.2 reached 121: the BF16
    default ran on a part with no BF16 tensor cores. The default must come from
    the compute capability, and a named format must still be honoured."""
    source = tmp_path / "gemm_default.cpp"
    source.write_text(PROGRAM, encoding="utf-8")
    exe = tmp_path / "gemm_default"
    build = subprocess.run(
        ["g++", "-std=c++17", "-DPANTHEON_MOCK", "-Wno-unknown-pragmas", "-pthread",
         f"-I{ROOT / 'kernels' / 'common'}", str(source), "-o", str(exe)],
        capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout
    assert run.stdout.strip() == "ok"


def test_setup_falls_back_to_wmma_before_asking_the_library():
    """pantheon_gemm_setup must decide from the part before init() is tried, or
    cuBLASLt's willingness to run BF16 on Turing decides it again."""
    header = (ROOT / "kernels" / "common" / "vendor_gemm.h").read_text(encoding="utf-8")
    setup = header.split("static inline int pantheon_gemm_setup", 1)[1]
    before_init = setup.split("g.init(", 1)[0]

    assert "pantheon_gemm_default_has_bf16(cc_major)" in before_init
    assert "using the portable WMMA tensor stream" in before_init
    assert "return 0;" in before_init
