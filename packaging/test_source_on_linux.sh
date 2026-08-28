#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-}"
TEST_SELECTOR="${PANTHEON_CONTAINER_TEST:-all}"
DURATION="${PANTHEON_CONTAINER_TEST_DURATION:-1}"

if [ -z "${IMAGE}" ]; then
    echo "Usage: $0 <container-image>" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
command -v docker >/dev/null 2>&1 || {
    echo "Docker is required for container compatibility testing." >&2
    exit 1
}

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT
mkdir -p "${work_dir}/source"

# Run from a disposable copy. Containers run as root on several supported
# images, so mounting the developer or runner checkout directly would leave
# root-owned build and result files behind.
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
    -e PANTHEON_MOCK=1 \
    -e PANTHEON_CONTAINER_TEST="${TEST_SELECTOR}" \
    -e PANTHEON_CONTAINER_TEST_DURATION="${DURATION}" \
    -e PANTHEON_HOST_UID="$(id -u)" \
    -e PANTHEON_HOST_GID="$(id -g)" \
    -v "${work_dir}/source:/workspace" \
    -w /workspace \
    "${IMAGE}" \
    sh -lc '
        set -eu
        # Package installation needs root in the container, but return all
        # generated files to the checkout owner before the bind mount closes.
        trap '\''chown -R "${PANTHEON_HOST_UID}:${PANTHEON_HOST_GID}" /workspace || true'\'' EXIT

        # findutils is not part of every minimal base image (rockylinux:8
        # omits it) and the Makefile discovers kernel sources with find.
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update
            DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
                ca-certificates findutils g++ make python3 python3-numpy python3-openpyxl \
                python3-pandas python3-psutil tar
        elif command -v dnf >/dev/null 2>&1; then
            dnf install -y --setopt=install_weak_deps=False \
                ca-certificates findutils gcc-c++ make python3 python3-devel python3-pip tar
            python3 -m pip install --no-cache-dir numpy openpyxl pandas psutil
        elif command -v yum >/dev/null 2>&1; then
            yum install -y ca-certificates findutils gcc-c++ make python3 python3-devel python3-pip tar
            python3 -m pip install --no-cache-dir numpy openpyxl pandas psutil
        else
            echo "Unsupported package manager in container." >&2
            exit 1
        fi

        python3 pantheon.py --platform mock --test "${PANTHEON_CONTAINER_TEST}" \
            --duration "${PANTHEON_CONTAINER_TEST_DURATION}" --gpu 0 --mem 1 --verify

        test -s results/*/summary.csv
        test -s results/*/summary.xlsx
        test -s results/*/time_series.csv
    '
