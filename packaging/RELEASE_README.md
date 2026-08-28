# Pantheon GPU

Pantheon GPU is a binary-only GPU stress and diagnostics tool. This release bundle contains the installed `pantheon` command, installation instructions, release notes, license, and package artifacts.

## Install

Ubuntu and Debian users should install the included package:

```bash
sudo apt install ./packages/pantheongpu_<version>_amd64.deb
```

Uninstall it with:

```bash
sudo apt-get remove pantheongpu
```

Other Linux distributions can use the generic installer:

```bash
sudo ./install.sh
```

Uninstall it with:

```bash
sudo ./uninstall.sh
```

See `INSTALL.md` for distro-specific notes and GPU compiler requirements.

## Uninstall

Debian package installation:

```bash
sudo apt-get remove pantheongpu
```

Portable `install.sh` installation:

```bash
sudo ./uninstall.sh
```

See `INSTALL.md` for custom install paths and optional workload-cache cleanup.

## Run

Pantheon auto-detects CUDA, ROCm/HIP, or mock mode at runtime.

```bash
pantheon --version
pantheon --test baseline_metrics --duration 10
pantheon --test tensor_virus --duration 60 --gpu 0
pantheon --test memory_read_agg --duration 60 --gpu all --mem 90
```

The first native run compiles workload binaries for the detected GPU target and caches them under `${XDG_CACHE_HOME:-$HOME/.cache}/pantheongpu/builds/`.

Official x86-64 binaries are built against GLIBC 2.28. Every release candidate is installed and run on Ubuntu 20.04/22.04/24.04, Debian 11/12, Red Hat UBI 8/9, CentOS Stream 9/10, Rocky Linux 8/9, AlmaLinux 8/9, and current Fedora before publication.

## Included Files

```text
bin/pantheon
packages/pantheongpu_<version>_amd64.deb
install.sh
uninstall.sh
INSTALL.md
RELEASE_NOTES.md
LICENSE
docs/release_process.md
```

This bundle does not include the private source tree, Python entrypoints, tests, or kernel C++ source files.
