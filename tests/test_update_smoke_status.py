import json
import subprocess
import sys
from pathlib import Path


def test_update_smoke_status_replaces_only_generated_readme_region(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "packaging" / "update_smoke_status.py"
    readme = tmp_path / "README.md"
    results = tmp_path / "results"
    (results / "ubuntu").mkdir(parents=True)
    (results / "rhel").mkdir()
    readme.write_text(
        "before\n<!-- DAILY_SMOKE_STATUS:START -->\nold\n<!-- DAILY_SMOKE_STATUS:END -->\nafter\n",
        encoding="utf-8",
    )
    (results / "ubuntu" / "result.json").write_text(
        json.dumps({"name": "Ubuntu 24.04", "status": "success"}), encoding="utf-8"
    )
    (results / "rhel" / "result.json").write_text(
        json.dumps({"name": "RHEL UBI 9", "status": "failure"}), encoding="utf-8"
    )

    subprocess.run(
        [
            sys.executable, str(script), "--readme", str(readme), "--results-dir", str(results),
            "--completed-at", "2026-08-15 20:00 UTC", "--run-url", "https://example.test/run/1",
        ],
        check=True,
    )

    contents = readme.read_text(encoding="utf-8")
    assert contents.startswith("before\n")
    assert contents.endswith("after\n")
    assert "Last completed: [2026-08-15 20:00 UTC](https://example.test/run/1)" in contents
    assert "**1/2 passed**" in contents
    assert "| RHEL UBI 9 | ❌ Failed |" in contents
    assert "| Ubuntu 24.04 | ✅ Passed |" in contents


def test_update_smoke_status_supports_custom_markers_for_the_toolkit_matrix(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "packaging" / "update_smoke_status.py"
    readme = tmp_path / "README.md"
    results = tmp_path / "results"
    (results / "cuda").mkdir(parents=True)
    readme.write_text(
        "smoke\n<!-- DAILY_SMOKE_STATUS:START -->\nkept\n<!-- DAILY_SMOKE_STATUS:END -->\n"
        "<!-- TOOLKIT_MATRIX_STATUS:START -->\nold\n<!-- TOOLKIT_MATRIX_STATUS:END -->\ntail\n",
        encoding="utf-8",
    )
    (results / "cuda" / "result.json").write_text(
        json.dumps({"name": "CUDA 12.8", "status": "success"}), encoding="utf-8"
    )

    subprocess.run(
        [
            sys.executable, str(script), "--readme", str(readme), "--results-dir", str(results),
            "--completed-at", "2026-08-25 21:00 UTC", "--run-url", "https://example.test/run/2",
            "--start-marker", "<!-- TOOLKIT_MATRIX_STATUS:START -->",
            "--end-marker", "<!-- TOOLKIT_MATRIX_STATUS:END -->",
            "--title", "Toolkit compile matrix",
            "--description", "Compile-only builds against real toolchains.",
            "--subject", "Toolkit",
        ],
        check=True,
    )

    contents = readme.read_text(encoding="utf-8")
    assert "<!-- DAILY_SMOKE_STATUS:START -->\nkept\n<!-- DAILY_SMOKE_STATUS:END -->" in contents
    assert "### Toolkit compile matrix" in contents
    assert "| Toolkit | Latest result |" in contents
    assert "| CUDA 12.8 | ✅ Passed |" in contents
    assert contents.endswith("tail\n")
