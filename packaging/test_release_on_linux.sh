#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-}"
TAR_PATH="${2:-}"
DEB_PATH="${3:-}"
INSTALL_METHOD="${4:-bundle}"
DURATION="${PANTHEON_CROSS_OS_SMOKE_DURATION:-1}"

if [ -z "${IMAGE}" ] || [ -z "${TAR_PATH}" ] || [ ! -f "${TAR_PATH}" ]; then
    echo "Usage: $0 <container-image> <release-tar.gz> [release.deb] [bundle|deb]" >&2
    exit 2
fi

archive_path="$(cd "$(dirname "${TAR_PATH}")" && pwd)/$(basename "${TAR_PATH}")"
docker_args=(
    --rm
    -e PANTHEON_MOCK=1
    -e SMOKE_DURATION="${DURATION}"
    -e INSTALL_METHOD="${INSTALL_METHOD}"
    -v "${archive_path}:/tmp/pantheon-release.tar.gz:ro"
)

case "${INSTALL_METHOD}" in
    bundle)
        ;;
    deb)
        if [ -z "${DEB_PATH}" ] || [ ! -f "${DEB_PATH}" ]; then
            echo "A Debian package is required when INSTALL_METHOD=deb." >&2
            exit 2
        fi
        deb_path="$(cd "$(dirname "${DEB_PATH}")" && pwd)/$(basename "${DEB_PATH}")"
        docker_args+=(-v "${deb_path}:/tmp/pantheon-release.deb:ro")
        ;;
    *)
        echo "Unsupported install method: ${INSTALL_METHOD}" >&2
        exit 2
        ;;
esac

docker run "${docker_args[@]}" \
    "${IMAGE}" \
    sh -lc '
        set -eu

        echo "Distribution:"
        cat /etc/os-release
        echo "GLIBC:"
        getconf GNU_LIBC_VERSION
        ldd --version 2>&1 | head -n 1

        if command -v apt-get >/dev/null 2>&1; then
            apt-get update
            DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates findutils g++ make tar gzip
        elif command -v dnf >/dev/null 2>&1; then
            dnf install -y ca-certificates findutils gcc-c++ make tar gzip
        elif command -v yum >/dev/null 2>&1; then
            yum install -y ca-certificates findutils gcc-c++ make tar gzip
        else
            echo "Unsupported package manager in container." >&2
            exit 1
        fi

        case "${INSTALL_METHOD}" in
            deb)
                if ! command -v apt-get >/dev/null 2>&1; then
                    echo "Debian package testing requires apt-get." >&2
                    exit 1
                fi
                DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends /tmp/pantheon-release.deb
                ;;
            bundle)
                mkdir -p /tmp/pantheon-release
                tar -xzf /tmp/pantheon-release.tar.gz -C /tmp/pantheon-release
                bundle_dir="$(find /tmp/pantheon-release -mindepth 1 -maxdepth 1 -type d | head -n 1)"
                cd "${bundle_dir}"
                ./install.sh
                ;;
        esac

        test -x /opt/pantheongpu/bin/pantheon
        pantheon --version
        pantheon --help >/dev/null
        pantheon --platform mock --test baseline_metrics --duration "${SMOKE_DURATION}" --gpu all --mem 1
        test -s results/*/summary.csv
        test -s results/*/summary.xlsx
        test -s results/*/time_series.csv

        mkdir -p /opt/pantheongpu/cache/builds/stale

        case "${INSTALL_METHOD}" in
            deb)
                DEBIAN_FRONTEND=noninteractive apt-get remove -y pantheongpu
                ! command -v pantheon >/dev/null 2>&1
                ! test -e /opt/pantheongpu
                ;;
            bundle)
                ./uninstall.sh
                ! command -v pantheon >/dev/null 2>&1
                ! test -e /opt/pantheongpu

                custom_prefix=/tmp/pantheon-custom-prefix
                custom_bindir=/tmp/pantheon-custom-bin
                PREFIX="${custom_prefix}" BINDIR="${custom_bindir}" ./install.sh
                test -L "${custom_bindir}/pantheon"
                test "$(readlink "${custom_bindir}/pantheon")" = "${custom_prefix}/bin/pantheon"
                PREFIX="${custom_prefix}" BINDIR="${custom_bindir}" ./uninstall.sh
                ! test -e "${custom_prefix}"
                ! test -e "${custom_bindir}/pantheon"

                mkdir -p /tmp/pantheon-command-collision
                printf "foreign command\n" > /tmp/pantheon-command-collision/pantheon
                if PREFIX=/tmp/pantheon-collision-prefix BINDIR=/tmp/pantheon-command-collision ./install.sh; then
                    echo "Installer overwrote an existing pantheon command." >&2
                    exit 1
                fi
                test "$(cat /tmp/pantheon-command-collision/pantheon)" = "foreign command"

                if PREFIX=relative BINDIR=/tmp/pantheon-relative-bin ./install.sh; then
                    echo "Installer accepted a relative prefix." >&2
                    exit 1
                fi
                ;;
        esac
    '
