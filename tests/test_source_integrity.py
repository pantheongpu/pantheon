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
