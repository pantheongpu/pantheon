"""The licensing story has to read the same in every file that tells it.

The project is licensed under the AGPL-3.0-only, offered under commercial
licenses, and keeps the Apache-2.0 text for the releases published under it
(up to v1.2.4). A contributor who signs the CLA is agreeing to the dual
licensing, so the files that explain it must not drift apart, and no file may
go on calling Apache-2.0 the current license.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_license_file_is_the_agpl_version_3():
    head = read("LICENSE")[:400]
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in head
    assert "Version 3, 19 November 2007" in head


def test_apache_text_is_kept_for_the_releases_published_under_it():
    head = read("LICENSES/Apache-2.0.txt")[:300]
    assert "Apache License" in head
    assert "Version 2.0, January 2004" in head


def test_notice_states_the_current_license_and_the_history():
    notice = " ".join(read("NOTICE").split())
    assert "AGPL-3.0-only" in notice
    assert "COMMERCIAL.md" in notice
    assert "v1.2.4" in notice
    assert "Apache License, Version 2.0" in notice
    assert "LICENSES/Apache-2.0.txt" in notice


def test_readme_and_contributing_link_the_license_documents():
    readme = read("README.md")
    for target in ("(LICENSE)", "(COMMERCIAL.md)", "(CLA.md)",
                   "(LICENSES/Apache-2.0.txt)"):
        assert target in readme, f"README.md must link {target}"
    assert "AGPL-3.0-only" in readme

    contributing = read("CONTRIBUTING.md")
    assert "(CLA.md)" in contributing
    assert "COMMERCIAL.md" in contributing
    # The sentence is the record of agreement, so it must match the CLA exactly.
    sentence = "I have read the Pantheon CLA and I agree to its terms."
    assert sentence in contributing
    assert sentence in read("CLA.md")


def test_cla_grants_what_dual_licensing_needs():
    cla = " ".join(read("CLA.md").split()).lower()
    # The right to relicense is the whole point of the agreement.
    assert "sublicense" in cla
    assert "under any license terms" in cla
    assert "irrevocable" in cla
    assert "patent license" in cla
    assert "you keep the copyright" in cla


def test_trademark_policy_names_the_marks_and_is_linked_from_the_license_files():
    policy = " ".join(read("TRADEMARK.md").split())
    for mark in ("Pantheon", "PantheonGPU"):
        assert mark in policy
    assert "saqibkhan@pantheongpu.com" in policy
    # The three rules that give the policy its teeth.
    assert "Logo files are not covered by the AGPL" in policy
    assert "modified version" in policy
    assert "certified" in policy and "verified" in policy
    # A commercial license must not be read as granting the name.
    assert "does not include trademark rights" in policy

    for name in ("README.md", "NOTICE", "COMMERCIAL.md"):
        assert "TRADEMARK.md" in read(name), f"{name} must point at TRADEMARK.md"


def test_no_file_calls_apache_the_current_license():
    """Apache-2.0 may be named only where it describes the earlier releases."""
    historical = {"NOTICE", "README.md", "COMMERCIAL.md",
                  "tests/test_licensing.py"}
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).split("\n")
    offenders = []
    for name in tracked:
        if not name or name in historical:
            continue
        if name.startswith("LICENSES/") or name.startswith("kernels/common/nvenc/"):
            continue
        try:
            body = read(name)
        except (OSError, UnicodeDecodeError):
            continue
        if "Apache-2.0" in body or "Apache License" in body:
            offenders.append(name)
    assert offenders == [], f"still name Apache as the license: {offenders}"


def test_container_label_matches_the_license():
    label = [line for line in read("integrations/apptainer/pantheon.def").splitlines()
             if "org.opencontainers.image.licenses" in line]
    assert label and label[0].split()[-1] == "AGPL-3.0-only"
