#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MANYLINUX_IMAGE="${PANTHEON_MANYLINUX_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64}"
PYTHON_PATH="${PANTHEON_MANYLINUX_PYTHON:-/opt/python/cp311-cp311/bin/python}"

command -v docker >/dev/null 2>&1 || {
    echo "Docker is required to build portable Linux release binaries." >&2
    exit 1
}

docker run --rm \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -e PANTHEON_CREATE_ARCHIVES=0 \
    -e PYTHON_BIN="${PYTHON_PATH}" \
    -v "${REPO_ROOT}:/workspace" \
    -w /workspace \
    "${MANYLINUX_IMAGE}" \
    bash -lc '
        set -euo pipefail
        if [ -f /opt/_internal/static-libs-for-embedding-only.tar.xz ]; then
            (
                cd /opt/_internal
                tar -xf static-libs-for-embedding-only.tar.xz
            )
        fi
        "${PYTHON_BIN}" -m pip install --disable-pip-version-check --no-cache-dir -r requirements.txt
        ./build_pantheon.sh
        chown -R "${HOST_UID}:${HOST_GID}" release build dist 2>/dev/null || true
    '

BUILD_BINARY=0 "${SCRIPT_DIR}/build_release_bundle.sh"
