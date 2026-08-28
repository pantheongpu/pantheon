# Pantheon GPU Binary Install

This release bundle contains binary-only Pantheon GPU artifacts. It does not include Python entrypoint files, kernel C++ source, or the private source repository.

Official x86-64 release binaries are built in a `manylinux_2_28` environment and require GLIBC 2.28 or newer.

Minimum versions tested by CI:

- Ubuntu 20.04 LTS
- Debian 11
- RHEL 8 userspace through Red Hat UBI 8
- CentOS Stream 9
- RHEL-compatible distributions: Rocky Linux 8 and AlmaLinux 8
- Current Fedora release

Newer Ubuntu, Debian, Red Hat UBI, CentOS Stream, Rocky Linux, and AlmaLinux versions are tested as well. Native GPU workloads still require a compatible CUDA or ROCm driver and compiler stack on the target system.

## Bundle Contents

```text
bin/pantheon
packages/pantheongpu_<version>_amd64.deb
install.sh
uninstall.sh
README.md
RELEASE_NOTES.md
LICENSE
docs/release_process.md
```

Pantheon auto-detects CUDA, ROCm/HIP, or mock mode at runtime. Normal users should not pass `--platform cuda`.

## Ubuntu / Debian

Install the included Debian package:

```bash
sudo apt install ./packages/pantheongpu_<version>_amd64.deb
```

Uninstall it with the native package manager:

```bash
sudo apt-get remove pantheongpu
```

Install GPU build dependencies for your platform:

```bash
# NVIDIA
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit

# AMD ROCm/HIP
sudo apt-get update
sudo apt-get install hipcc
```

Run a smoke test:

```bash
pantheon --version
pantheon --test baseline_metrics --duration 10
pantheon --test fp64_virus --duration 30 --gpu 0
```

## RHEL / Rocky / AlmaLinux / Fedora

Pantheon does not currently ship an RPM in this bundle. Use the generic binary installer:

```bash
sudo ./install.sh
```

Uninstall the portable installation with:

```bash
sudo ./uninstall.sh
```

Install GPU build dependencies for your platform:

```bash
# Generic build tools
sudo dnf install make gcc-c++

# NVIDIA
# Install the CUDA toolkit from NVIDIA's repository for your distro.

# AMD ROCm/HIP
# Install ROCm/HIP packages from AMD's repository for your distro.
```

Run a smoke test:

```bash
pantheon --test baseline_metrics --duration 10
```

## Other Linux Distributions

Use the generic installer:

```bash
sudo ./install.sh
```

Uninstall the portable installation with:

```bash
sudo ./uninstall.sh
```

Then install the CUDA or ROCm compiler stack using your distribution's supported packages. Pantheon uses the local compiler to build workload binaries for the current GPU and caches the result.

## Uninstall

For an installation made from the Debian package:

```bash
sudo apt-get remove pantheongpu
```

Use `sudo apt-get purge pantheongpu` if complete package removal is preferred.
Both commands remove Pantheon's package-owned files and runtime-created files
under `/opt/pantheongpu`. Per-user workload caches are retained.

For an installation made with this bundle's `install.sh`:

```bash
sudo ./uninstall.sh
```

If `install.sh` was run with custom `PREFIX` or `BINDIR` values, pass the same
values to `uninstall.sh`.

The portable uninstaller also removes the invoking user's compiled workload
cache. It does not remove CUDA, ROCm, compilers, or benchmark reports stored
outside Pantheon's installation and cache directories.

## Compiled Workload Cache

The installed Pantheon binary embeds kernel source privately and compiles workload binaries locally for the detected GPU target on first use. Installed builds are cached per user under:

```text
${XDG_CACHE_HOME:-$HOME/.cache}/pantheongpu/builds/<version>/<platform>-<target>/
```

Examples:

```text
$HOME/.cache/pantheongpu/builds/1.0.13/hip-gfx942/
$HOME/.cache/pantheongpu/builds/1.0.13/hip-gfx950/
$HOME/.cache/pantheongpu/builds/1.0.13/cuda-86/
```

If the matching cache already exists and the embedded kernel inputs have not changed, Pantheon skips compilation on the next run. To choose a different writable cache location:

```bash
PANTHEON_BUILD_CACHE_DIR="$HOME/.cache/pantheongpu/builds" pantheon --test baseline_metrics --duration 10
```
