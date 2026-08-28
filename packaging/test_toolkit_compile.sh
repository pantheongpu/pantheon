#!/usr/bin/env bash
# Compile every Pantheon kernel with a real vendor toolchain inside a
# disposable toolkit container. This is a compile-only check: it proves the
# kernels build against a given CUDA or ROCm release, not that they run.
set -euo pipefail

IMAGE="${1:-}"
PLATFORM="${2:-}"
TARGET_GFX="${3:-gfx942}"

if [ -z "${IMAGE}" ] || [ -z "${PLATFORM}" ]; then
    echo "Usage: $0 <container-image> <CUDA|HIP> [target-gfx]" >&2
    exit 2
fi
case "${PLATFORM}" in
    CUDA|HIP) ;;
    *) echo "Platform must be CUDA or HIP." >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
command -v docker >/dev/null 2>&1 || {
    echo "Docker is required for toolkit compile testing." >&2
    exit 1
}

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT
mkdir -p "${work_dir}/source"

tar \
    --exclude=.git \
    --exclude=build \
    --exclude=dist \
    --exclude=release \
    --exclude=results \
    --exclude=database \
    --exclude=.ci-install \
    -C "${REPO_ROOT}" -cf - . | tar -C "${work_dir}/source" -xf -

docker run --rm \
    -e PANTHEON_PLATFORM="${PLATFORM}" \
    -e PANTHEON_TARGET_GFX="${TARGET_GFX}" \
    -e PANTHEON_HOST_UID="$(id -u)" \
    -e PANTHEON_HOST_GID="$(id -g)" \
    -v "${work_dir}/source:/workspace" \
    -w /workspace \
    "${IMAGE}" \
    bash -ec '
        # The container runs as root; return generated files to the host
        # user so the disposable copy can be cleaned up afterwards.
        trap '\''chown -R "${PANTHEON_HOST_UID}:${PANTHEON_HOST_GID}" /workspace || true'\'' EXIT
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends ca-certificates findutils g++ make

        if [ "${PANTHEON_PLATFORM}" = "CUDA" ]; then
            nvcc --version
            # Toolkit containers carry no GPU driver; link the driver API
            # against the stub the toolkit ships for exactly this purpose.
            make PLATFORM=CUDA LDFLAGS="-L/usr/local/cuda/lib64/stubs -lcuda -ldl" -j"$(nproc)"
        else
            export PATH="/opt/rocm/bin:${PATH}"
            hipcc --version
            make PLATFORM=HIP HIPCC="$(command -v hipcc)" ROCM_PATH=/opt/rocm \
                TARGET_GFX="${PANTHEON_TARGET_GFX}" -j"$(nproc)"
        fi

        src_count="$(find kernels -name "*.cpp" ! -path "kernels/common/*" | wc -l)"
        bin_count="$(find build -maxdepth 1 -type f ! -name ".*" | wc -l)"
        echo "Compiled ${bin_count} of ${src_count} kernel binaries."
        test "${src_count}" -gt 0
        test "${bin_count}" -eq "${src_count}"
    '
