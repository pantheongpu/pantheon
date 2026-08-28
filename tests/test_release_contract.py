import re
import os
import subprocess
from pathlib import Path


def test_release_docs_match_version_file():
    version = Path("VERSION").read_text(encoding="utf-8").strip()
    readme = Path("README.md").read_text(encoding="utf-8")
    release_process = Path("docs/release_process.md").read_text(encoding="utf-8")
    release_notes = Path("RELEASE_NOTES.md").read_text(encoding="utf-8")

    assert f"make release VERSION={version}" in readme
    assert f"pantheongpu_{version}_amd64.deb" in readme
    assert f'printf "{version}\\n" > VERSION' in readme
    assert f"make release VERSION={version}" in release_process
    assert f"pantheongpu_{version}_amd64.tar.gz" in release_process
    assert f"Pantheon v{version} Release Notes" in release_notes

    stale_versions = re.findall(r"1[.]0[.][0-9]+", readme + "\n" + release_process)
    assert set(stale_versions) == {version}


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


def test_installed_command_docs_do_not_use_source_entrypoints():
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (Path("README.md"), Path("docs/release_process.md"), Path("packaging/INSTALL.md"))
    )

    assert "python3 pantheon.py" not in docs
    assert "python pantheon.py" not in docs
    assert "python3 tuning.py" not in docs
    assert "python tuning.py" not in docs
    assert "pantheon-tuning" not in docs


def test_glibc_compatibility_check_rejects_newer_symbols(tmp_path):
    target = tmp_path / "pantheon"
    target.write_bytes(b"ELF")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_file = fake_bin / "file"
    fake_objdump = fake_bin / "objdump"
    fake_file.write_text("#!/bin/sh\necho 'ELF 64-bit executable'\n", encoding="utf-8")
    fake_objdump.write_text(
        "#!/bin/sh\n"
        "echo '0000000000000000      DF *UND*  0000000000000000 (GLIBC_2.38) memcpy'\n",
        encoding="utf-8",
    )
    fake_file.chmod(0o755)
    fake_objdump.chmod(0o755)

    result = subprocess.run(
        ["bash", "packaging/check_glibc_compat.sh", str(target), "2.28"],
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 1
    assert "requires GLIBC 2.38, which is newer than 2.28" in result.stderr


def test_build_pantheon_does_not_create_archives_by_default():
    script = Path("build_pantheon.sh").read_text(encoding="utf-8")

    assert 'CREATE_ARCHIVES="${PANTHEON_CREATE_ARCHIVES:-0}"' in script
    assert "PANTHEON_CREATE_ARCHIVES=0" in Path("packaging/build_deb.sh").read_text(encoding="utf-8")


def test_release_artifacts_use_public_readme():
    release_readme = Path("packaging/RELEASE_README.md").read_text(encoding="utf-8")
    build_script = Path("packaging/build_release_bundle.sh").read_text(encoding="utf-8")
    deb_script = Path("packaging/build_deb.sh").read_text(encoding="utf-8")

    assert "binary-only GPU stress" in release_readme
    assert "Website Dashboard" not in release_readme
    assert "Creating a Release" not in release_readme
    assert "packaging/RELEASE_README.md" in build_script
    assert "packaging/RELEASE_README.md" in deb_script


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


def test_public_export_excludes_non_redistributable_files():
    """The export script is what stands between a proprietary header and a
    public repo, so its exclusion list must actually name the offender."""
    script = Path("tools/export_public_tree.sh")
    assert script.is_file(), "public export script is missing"
    body = script.read_text(encoding="utf-8")
    assert "kernels/common/optix" in body, "OptiX is not excluded from the public export"
    assert Path("NOTICE").is_file(), "NOTICE must record bundled third-party terms"


# Note: the workflow contract tests from the upstream repository are not
# present here. This tree ships no .github/workflows, so there is no CI or
# release configuration for them to assert against.
