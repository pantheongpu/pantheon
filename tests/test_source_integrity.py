import re
import os
import subprocess
from pathlib import Path



def test_transformer_build_target_is_portable_by_default():
    makefile = Path("Makefile").read_text(encoding="utf-8")
    transformer = Path("kernels/transformer_virus/transformer_virus.cpp").read_text(encoding="utf-8")

    # A detected architecture can carry feature suffixes; --offload-arch wants
    # the bare name.
    assert "DETECTED_GFX := $(firstword $(subst :, ,$(DETECTED_GFX)))" in makefile
    assert "--offload-arch=$(DETECTED_GFX)" in makefile

    # The WMMA path must stay wired up. It is selected by architecture family
    # from the build system, so no individual model number appears in the
    # source -- but the path itself must not quietly become dead code.
    assert "PANTHEON_AMD_WMMA_TARGET" in transformer
    assert "PANTHEON_ENABLE_EXPERIMENTAL_WMMA" in transformer
    assert "GFX_FAMILY" in makefile and "PANTHEON_AMD_WMMA_TARGET" in makefile

    # Typo guards: a single colon or a misspelled pragma compiles to something
    # silently different rather than failing loudly.
    assert "wmma:fill_fragment" not in transformer
    assert "wmma:mma_sync" not in transformer
    assert "#pragna" not in transformer


def test_no_unreleased_hardware_identifiers_ship():
    """Unannounced parts must not be named anywhere in the tree.

    The patterns are assembled rather than written literally so that this
    guard does not itself become a hit when the tree is swept for them.
    """
    import re
    import subprocess

    patterns = [
        re.compile("gfx" + r"125\d", re.I),      # unannounced AMD architectures
        re.compile("mi" + r"[-_ ]?450", re.I),   # and their product name
    ]
    tracked = subprocess.check_output(
        ["git", "ls-files"], text=True, encoding="utf-8"
    ).split()

    offenders = []
    for name in tracked:
        path = Path(name)
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(p.search(body) for p in patterns):
            offenders.append(name)

    assert offenders == [], f"unreleased hardware named in: {offenders}"


def test_rt_virus_probes_for_optix_rather_than_assuming_it():
    """OptiX headers are not redistributable, so they may be absent.

    An unconditional include makes the whole build fail without them; the AMD
    HIP-RT branch already probed, and the CUDA branch now does too.
    """
    src = Path("kernels/rt_virus/rt_virus.cpp").read_text(encoding="utf-8")
    guard = src[:src.index("#define HIPRT_SUPPORTED 0")]
    assert "__has_include(<optix.h>)" in guard, "OptiX must be probed, not assumed"

    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "OPTIX_PATH" in makefile, "the OptiX include path must be overridable"


def test_no_file_forbids_its_own_redistribution():
    """This tree is meant to be publishable, so nothing in it may carry terms
    that prohibit redistribution.

    The standing example is the NVIDIA OptiX headers, which state that
    distribution without an express licence agreement is prohibited. They are
    not carried here; rt_virus probes for them instead. Without a check, the
    next vendored header to arrive would quietly make the tree unpublishable.
    """
    import re
    import subprocess

    # Licence text wraps mid-sentence, so match on whitespace-normalised
    # content rather than a literal phrase -- a naive grep for the phrase finds
    # nothing and gives false confidence.
    forbids = re.compile(
        r"(distribution|reproduction)[^.]{0,160}(is\s+)?strictly\s+prohibited"
        r"|without\s+an\s+express\s+licen[sc]e\s+agreement[^.]{0,160}prohibited",
        re.I,
    )
    # Files that describe licensing rather than being subject to it. They quote
    # the prohibition on purpose; publishing them is the point.
    describes_licensing = {"NOTICE", "LICENSE", "CONTRIBUTING.md", "SECURITY.md"}

    tracked = subprocess.check_output(["git", "ls-files"], text=True).split()
    offenders = []
    for name in tracked:
        path = Path(name)
        # This file quotes the prohibition in order to test for it.
        if path.resolve() == Path(__file__).resolve() or name in describes_licensing:
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if forbids.search(" ".join(body.split())):
            offenders.append(name)

    assert offenders == [], (
        "these files forbid redistribution and cannot ship in a public tree: "
        f"{offenders}"
    )


def test_cuda_family_target_promotion_is_probed_not_hardcoded():
    """sm_90a and the Blackwell 'a' targets unlock the matrix-core
    instructions; without the suffix nvcc builds the portable subset and the
    tensor pipes never reach their real issue rate.

    Which suffixes exist depends on the toolkit as well as the architecture,
    so the target is probed. The runner has to probe the same way, because its
    answer is part of the build cache key: a mismatch serves a binary compiled
    for the portable target to a build that now asks for the other one.
    """
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "ARCH_SUFFIX" in makefile
    assert "--gpu-architecture=sm_$(DETECTED_ARCH)a" in makefile
    # A trailing space on the fallback would make the concatenation "86 a".
    assert "DETECTED_ARCH := 86\n" in makefile

    runner = Path("pantheon.py").read_text(encoding="utf-8")
    assert "def cuda_arch_suffix(" in runner
    assert "cuda_arch_suffix(detected_arch)" in runner


# The arithmetic furnaces, and what each one's recurrence measured before it
# was fixed. A datapath that stops toggling stops drawing power, and a golden
# pass computed in that state compares a constant against the same constant,
# so --verify cannot fail. Both were measured over each kernel's real default
# loop count; the numbers are in kernels/common/toggle_chaos.h.
ARITHMETIC_FURNACES = (
    "kernels/compute_virus/compute_virus.cpp",
    "kernels/compute_virus/compute_virus_agg.cpp",
    "kernels/fp64_virus/fp64_virus.cpp",
    "kernels/pulse_virus/pulse_virus.cpp",
    "kernels/tensor_virus/tensor_virus.cpp",
    "kernels/sfu_stress/sfu_stress.cpp",
    "kernels/memory_thermal_asym/memory_thermal_asym.cpp",
    "kernels/memory_retention_bake/memory_retention_bake.cpp",
    "kernels/omni_virus/omni_virus.cpp",
)


def test_arithmetic_furnaces_use_the_shared_bounded_state():
    """Every furnace has to draw its state from toggle_chaos.h.

    Each one previously carried its own recurrence, and eight of the nine
    either saturated to infinity within the first ten FMAs or converged onto a
    fixed point within a few dozen steps. Sharing one documented, provably
    closed map is what stops the next copy from drifting back.
    """
    for name in ARITHMETIC_FURNACES:
        src = Path(name).read_text(encoding="utf-8")
        assert "toggle_chaos.h" in src, f"{name} must use the shared bounded state"


def test_no_furnace_reintroduces_the_saturating_chain():
    """Guard the specific shape that overflowed.

    a=fma(a,b,c); b=fma(b,c,a); c=fma(c,a,b) seeded near 1.0 reaches infinity
    after eight FP32 FMAs and seven in FP16. It reads as a perfectly
    reasonable ALU stress loop, which is why it was copied into six kernels.
    """
    coupled = re.compile(
        r"(?:__builtin_)?(?:fmaf?|__hfma2)\s*\(\s*a\s*,\s*b\s*,\s*c\s*\)"
        r"|(?:__builtin_)?(?:fmaf?|__hfma2)\s*\(\s*b\s*,\s*c\s*,\s*[ad]\s*\)"
    )
    offenders = []
    for name in ARITHMETIC_FURNACES:
        for line in Path(name).read_text(encoding="utf-8").splitlines():
            # A mock-build "#define __hfma2(a, b, c)" is a macro signature,
            # not the recurrence.
            if line.lstrip().startswith("#define"):
                continue
            if coupled.search(line):
                offenders.append(f"{name}: {line.strip()}")
    assert offenders == [], f"saturating coupled chain is back in: {offenders}"


def test_sfu_furnaces_issue_to_the_special_function_unit():
    """sinf/cosf are multi-instruction library routines that spend most of
    their cycles in the FMA pipe. A test built on them measures the wrong
    unit, so the SFU paths use the fast intrinsics.
    """
    header = Path("kernels/common/toggle_chaos.h").read_text(encoding="utf-8")
    for fast in ("__sinf(", "__cosf(", "__expf(", "__logf(", "rsqrtf("):
        assert fast in header, f"toggle_chaos.h lost {fast}"

    # The library spellings must not come back in the SFU workloads. The
    # lookbehind keeps __sinf from counting as a hit on sinf.
    library = re.compile(r"(?<![_a-zA-Z])(?:sinf|cosf|expf|logf)\s*\(")
    for name in ("kernels/sfu_stress/sfu_stress.cpp",
                 "kernels/common/toggle_chaos.h",
                 "kernels/omni_virus/omni_virus.cpp"):
        src = Path(name).read_text(encoding="utf-8")
        assert not library.search(src), f"{name} uses the library transcendentals again"

    # And the chains themselves have to come from the shared step.
    for name in ("kernels/sfu_stress/sfu_stress.cpp",
                 "kernels/omni_virus/omni_virus.cpp"):
        src = Path(name).read_text(encoding="utf-8")
        assert "PANTHEON_CHAOS_SFU_STEP" in src, f"{name} lost the shared SFU step"


def test_matrix_furnaces_restage_their_operands():
    """Fragments filled with one constant let the multiplier array recompute
    identical partial products forever, and drive the accumulator to a
    magnitude where the FP32 increment vanishes into rounding.
    """
    for name in ("kernels/mma_virus/mma_virus.cpp",
                 "kernels/omni_virus/omni_virus.cpp"):
        src = Path(name).read_text(encoding="utf-8")
        assert "pantheon_operand_hash" in src, f"{name} must restage operands"
        assert "wmma::load_matrix_sync" in src
        # Several accumulators, because mma_sync into one serialises on its
        # own result exactly as a single FMA chain does. Written either as
        # named fragments or as an array cleared in a loop.
        named = src.count("wmma::fill_fragment(c")
        array = "wmma::fill_fragment(c[k]" in src
        assert named >= 4 or array, f"{name} lost its accumulators"


def test_shared_chaos_header_triggers_rebuilds():
    """toggle_chaos.h now carries the arithmetic of nine workloads, so it has
    to be in the dependency list. Without it, editing the header leaves every
    binary stale and the next run silently measures the old math.
    """
    makefile = Path("Makefile").read_text(encoding="utf-8")
    common = [line for line in makefile.splitlines()
              if line.startswith("COMMON_HEADERS")]
    assert common, "COMMON_HEADERS disappeared"
    assert "kernels/common/toggle_chaos.h" in common[0]


def test_omni_virus_can_drive_the_tensor_cores_through_the_vendor_gemm():
    """The vendor GEMM is loaded at run time and only ever an upgrade."""
    header = Path("kernels/common/vendor_gemm.h").read_text()
    omni = Path("kernels/omni_virus/omni_virus.cpp").read_text()
    makefile = Path("Makefile").read_text()

    # dlopen, not a link dependency: one link line serves every binary, and a
    # hard -lcublasLt would keep unrelated tests from starting where the
    # loader cannot find it.
    assert "dlopen" in header
    assert "-lcublasLt" not in makefile
    assert "vendor_gemm.h" in makefile

    # Absent library or API falls back to WMMA instead of failing the launch.
    assert "using the portable WMMA tensor stream" in header
    assert "PANTHEON_VENDOR_GEMM" in omni

    # The GEMM stream is verified against its own first run.
    assert "GEMM Stream Error" in omni

    # Formats and shape are selectable, and the format is validated up front.
    for flag in ("--gemm_type", "--gemm_size", "--gemm_m", "--gemm_n", "--gemm_k"):
        assert flag in header
    assert "Unknown --gemm_type" in header

    # The default format is chosen by measurement, not hardcoded: BF16 is not
    # available on every part.
    assert 'std::string type = "auto"' in header
    assert "pantheon_gemm_pick" in header


def test_matrix_tests_share_one_vendor_gemm_with_both_backends():
    """omni_virus and mma_virus bring the GEMM up through the same code, and
    the header carries both vendor backends behind one set of aliases."""
    header = Path("kernels/common/vendor_gemm.h").read_text(encoding="utf-8")
    assert "hipblaslt/hipblaslt.h" in header
    assert "cublasLt.h" in header
    assert "libhipblaslt.so" in header
    assert "dlopen" in header

    for name in ("kernels/omni_virus/omni_virus.cpp",
                 "kernels/mma_virus/mma_virus.cpp"):
        src = Path(name).read_text(encoding="utf-8")
        assert "pantheon_gemm_setup" in src, f"{name} must use the shared setup"
        assert "gemm_opt.parse" in src, f"{name} lost the GEMM flags"
        # Both tests fall back to WMMA rather than failing when the BLAS is
        # absent, and say why.
        assert "portable WMMA" in src or "using the portable WMMA" in header


def test_gemm_auto_format_keeps_the_default_unless_clearly_beaten():
    """Probe noise on one part was larger than the gaps between formats, and
    taking the maximum picked different formats on GPUs of the same node. The
    default order plus a margin keeps the choice repeatable."""
    header = Path("kernels/common/vendor_gemm.h").read_text(encoding="utf-8")
    assert "margin_pct" in header
    assert "--gemm_margin" in header
    assert "GEMM_BF16, GEMM_FP16, GEMM_TF32, GEMM_FP8, GEMM_FP32" in header
    margin = re.search(r"int margin_pct = (\d+);", header)
    assert margin and int(margin.group(1)) >= 15, "margin below the measured probe noise"
