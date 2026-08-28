import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_debian_package_contains_binary_install_layout(tmp_path):
    app_dir = tmp_path / "release" / "pantheon-9.9.9"
    dist_dir = tmp_path / "dist"
    app_dir.mkdir(parents=True)

    binary = app_dir / "pantheon"
    shutil.copy2("/bin/echo", binary)

    env = {
        **os.environ,
        "BUILD_BINARY": "0",
        "VERSION": "9.9.9",
        "APP_DIR": str(app_dir),
        "DIST_DIR": str(dist_dir),
        "PANTHEON_MAX_GLIBC": "99.0",
    }

    subprocess.run(
        ["bash", str(REPO_ROOT / "packaging/build_deb.sh")],
        check=True,
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    deb = dist_dir / "pantheongpu_9.9.9_amd64.deb"
    assert deb.exists()

    control = subprocess.check_output(["dpkg-deb", "-f", str(deb)], text=True)
    assert "Package: pantheongpu" in control
    assert "Version: 9.9.9" in control
    assert "Depends: make, g++ | build-essential" in control
    assert "python3" not in control

    contents = subprocess.check_output(["dpkg-deb", "-c", str(deb)], text=True)
    assert "./opt/pantheongpu/bin/pantheon" in contents
    assert "./usr/bin/pantheon" in contents
    assert "pantheon-tuning" not in contents
    assert "pantheon.py" not in contents
    assert "tuning.py" not in contents
    assert ".cpp" not in contents

    archive_members = subprocess.check_output(
        ["ar", "t", str(deb)],
        text=True,
    ).splitlines()
    assert "control.tar.gz" in archive_members
    assert "data.tar.gz" in archive_members
    assert not any(member.endswith(".zst") for member in archive_members)

    extract_dir = tmp_path / "extract"
    subprocess.run(["dpkg-deb", "-x", str(deb), str(extract_dir)], check=True)
    control_dir = tmp_path / "control"
    subprocess.run(["dpkg-deb", "-e", str(deb), str(control_dir)], check=True)

    pantheon_wrapper = (extract_dir / "usr/bin/pantheon").read_text(encoding="utf-8")
    postrm = (control_dir / "postrm").read_text(encoding="utf-8")
    package_readme = (
        extract_dir / "opt/pantheongpu/share/doc/pantheongpu/README.md"
    ).read_text(encoding="utf-8")

    assert 'exec /opt/pantheongpu/bin/pantheon "$@"' in pantheon_wrapper
    assert "remove|purge" in postrm
    assert "rm -rf /opt/pantheongpu" in postrm
    assert "--platform" not in pantheon_wrapper
    assert "Pantheon GPU is a binary-only GPU stress" in package_readme
    assert "Website Dashboard" not in package_readme
    assert "Creating a Release" not in package_readme


def test_binary_release_bundle_contains_install_docs_and_no_source(tmp_path):
    app_dir = tmp_path / "release" / "pantheon-9.9.9"
    dist_dir = tmp_path / "dist"
    app_dir.mkdir(parents=True)

    binary = app_dir / "pantheon"
    shutil.copy2("/bin/echo", binary)

    env = {
        **os.environ,
        "BUILD_BINARY": "0",
        "VERSION": "9.9.9",
        "APP_DIR": str(app_dir),
        "DIST_DIR": str(dist_dir),
        "PANTHEON_MAX_GLIBC": "99.0",
    }

    subprocess.run(
        ["bash", str(REPO_ROOT / "packaging/build_release_bundle.sh")],
        check=True,
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    tar_path = dist_dir / "pantheongpu_9.9.9_amd64.tar.gz"
    zip_path = dist_dir / "pantheongpu_9.9.9_amd64.zip"
    assert tar_path.exists()
    assert zip_path.exists()

    with tarfile.open(tar_path) as tf:
        tar_names = set(tf.getnames())
    with zipfile.ZipFile(zip_path) as zf:
        zip_names = set(zf.namelist())

    expected = {
        "pantheongpu_9.9.9_amd64/bin/pantheon",
        "pantheongpu_9.9.9_amd64/packages/pantheongpu_9.9.9_amd64.deb",
        "pantheongpu_9.9.9_amd64/INSTALL.md",
        "pantheongpu_9.9.9_amd64/install.sh",
        "pantheongpu_9.9.9_amd64/uninstall.sh",
        "pantheongpu_9.9.9_amd64/docs/release_process.md",
    }
    assert expected <= tar_names
    assert expected <= zip_names

    joined = "\n".join(sorted(tar_names | zip_names))
    assert "pantheon.py" not in joined
    assert "tuning.py" not in joined
    assert "pantheon-tuning" not in joined
    assert ".cpp" not in joined
    assert ".py" not in joined
    assert "pantheongpu_9.9.9_amd64/kernels/" not in joined
    assert "pantheongpu_9.9.9_amd64/tests/" not in joined
    assert "pantheongpu_9.9.9_amd64/.git/" not in joined

    extract_dir = tmp_path / "pantheongpu_9.9.9_amd64"
    subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(extract_dir.parent)], check=True)
    install_doc = (extract_dir / "INSTALL.md").read_text(encoding="utf-8")
    release_readme = (extract_dir / "README.md").read_text(encoding="utf-8")
    assert "Ubuntu / Debian" in install_doc
    assert "RHEL / Rocky / AlmaLinux / Fedora" in install_doc
    assert "Other Linux Distributions" in install_doc
    assert "## Uninstall" in install_doc
    assert "sudo apt-get remove pantheongpu" in install_doc
    assert "sudo ./uninstall.sh" in install_doc
    assert "pantheon --test baseline_metrics --duration 10" in install_doc
    assert "Pantheon GPU is a binary-only GPU stress" in release_readme
    assert "## Uninstall" in release_readme
    assert "sudo apt-get remove pantheongpu" in release_readme
    assert "sudo ./uninstall.sh" in release_readme
    assert "Website Dashboard" not in release_readme
    assert "Creating a Release" not in release_readme

    install_script = (extract_dir / "install.sh").read_text(encoding="utf-8")
    assert 'cd "${script_dir}"' in install_script

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_id = fake_bin / "id"
    fake_id.write_text("#!/usr/bin/env sh\nprintf '0\\n'\n", encoding="utf-8")
    fake_id.chmod(0o755)

    prefix = tmp_path / "prefix"
    bindir = tmp_path / "cmds"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PREFIX": str(prefix),
        "BINDIR": str(bindir),
    }
    subprocess.run(
        ["sh", str(extract_dir / "install.sh")],
        check=True,
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    result = subprocess.check_output(
        [str(bindir / "pantheon"), "--test", "baseline_metrics"],
        text=True,
    )
    assert result.strip() == "--test baseline_metrics"

    cache_home = tmp_path / "cache"
    (cache_home / "pantheongpu" / "builds").mkdir(parents=True)
    env["PANTHEON_CACHE_HOME"] = str(cache_home)
    subprocess.run(
        ["sh", str(extract_dir / "uninstall.sh")],
        check=True,
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert not prefix.exists()
    assert not (bindir / "pantheon").exists()
    assert not (cache_home / "pantheongpu").exists()


