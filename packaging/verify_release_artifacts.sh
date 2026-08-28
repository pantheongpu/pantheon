#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VERSION="${VERSION:-$(tr -d '[:space:]' < VERSION)}"
ARCH="${ARCH:-amd64}"
PACKAGE_NAME="${PACKAGE_NAME:-pantheongpu}"
DIST_DIR="${DIST_DIR:-dist}"
BUNDLE_NAME="${PACKAGE_NAME}_${VERSION}_${ARCH}"
DEB_PATH="${DIST_DIR}/${BUNDLE_NAME}.deb"
TAR_PATH="${DIST_DIR}/${BUNDLE_NAME}.tar.gz"
ZIP_PATH="${DIST_DIR}/${BUNDLE_NAME}.zip"
SMOKE_DURATION="${PANTHEON_RELEASE_SMOKE_DURATION:-60}"
RUN_INSTALL_SMOKE="${RUN_INSTALL_SMOKE:-1}"
MAX_GLIBC="${PANTHEON_MAX_GLIBC:-2.28}"

fail() {
    echo "[verify-release] ERROR: $*" >&2
    exit 1
}

require_file() {
    [ -f "$1" ] || fail "Missing file: $1"
    [ -s "$1" ] || fail "File is empty: $1"
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

expect_manifest_entry() {
    local manifest="$1"
    local entry="$2"
    grep -Fxq "${entry}" "${manifest}" || fail "Missing ${entry} in ${manifest}"
}

expect_deb_entry() {
    local manifest="$1"
    local entry="$2"
    grep -Eq "^[^ ]+ +[^ ]+ +[^ ]+ +[^ ]+ +[^ ]+ +\\./${entry}$" "${manifest}" || fail "Missing ${entry} in ${manifest}"
}

echo "[verify-release] Checking release artifacts for ${BUNDLE_NAME}"
require_file "${DEB_PATH}"
require_file "${TAR_PATH}"
require_file "${ZIP_PATH}"

require_cmd tar
require_cmd dpkg-deb
require_cmd objdump
require_cmd python3

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

tar_manifest="${tmp_dir}/tar_manifest.txt"
zip_manifest="${tmp_dir}/zip_manifest.txt"
deb_manifest="${tmp_dir}/deb_manifest.txt"

tar -tzf "${TAR_PATH}" | sort > "${tar_manifest}"
if command -v unzip >/dev/null 2>&1; then
    unzip -Z1 "${ZIP_PATH}" | sort > "${zip_manifest}"
else
    python3 - "${ZIP_PATH}" "${zip_manifest}" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as archive:
    names = sorted(archive.namelist())

with open(sys.argv[2], "w", encoding="utf-8") as manifest:
    manifest.write("\n".join(names))
    manifest.write("\n")
PY
fi
dpkg-deb -c "${DEB_PATH}" | sort > "${deb_manifest}"

for entry in \
    "${BUNDLE_NAME}/VERSION" \
    "${BUNDLE_NAME}/LICENSE" \
    "${BUNDLE_NAME}/README.md" \
    "${BUNDLE_NAME}/RELEASE_NOTES.md" \
    "${BUNDLE_NAME}/INSTALL.md" \
    "${BUNDLE_NAME}/install.sh" \
    "${BUNDLE_NAME}/uninstall.sh" \
    "${BUNDLE_NAME}/bin/pantheon" \
    "${BUNDLE_NAME}/docs/release_process.md" \
    "${BUNDLE_NAME}/packages/${BUNDLE_NAME}.deb"
do
    expect_manifest_entry "${tar_manifest}" "${entry}"
    expect_manifest_entry "${zip_manifest}" "${entry}"
done

for entry in \
    "opt/pantheongpu/VERSION" \
    "opt/pantheongpu/bin/pantheon" \
    "opt/pantheongpu/share/doc/pantheongpu/LICENSE" \
    "opt/pantheongpu/share/doc/pantheongpu/README.md" \
    "opt/pantheongpu/share/doc/pantheongpu/RELEASE_NOTES.md" \
    "usr/bin/pantheon"
do
    expect_deb_entry "${deb_manifest}" "${entry}"
done

mkdir -p "${tmp_dir}/tar" "${tmp_dir}/zip"
tar -xzf "${TAR_PATH}" -C "${tmp_dir}/tar"
if command -v unzip >/dev/null 2>&1; then
    unzip -q "${ZIP_PATH}" -d "${tmp_dir}/zip"
else
    python3 - "${ZIP_PATH}" "${tmp_dir}/zip" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as archive:
    archive.extractall(sys.argv[2])
PY
fi

for extracted in "${tmp_dir}/tar/${BUNDLE_NAME}" "${tmp_dir}/zip/${BUNDLE_NAME}"; do
    [ -x "${extracted}/install.sh" ] || fail "install.sh is not executable in ${extracted}"
    [ -x "${extracted}/uninstall.sh" ] || fail "uninstall.sh is not executable in ${extracted}"
    [ -x "${extracted}/bin/pantheon" ] || fail "pantheon is not executable in ${extracted}"
    cmp -s VERSION "${extracted}/VERSION" || fail "VERSION mismatch in ${extracted}"
    "${SCRIPT_DIR}/check_glibc_compat.sh" "${extracted}/bin/pantheon" "${MAX_GLIBC}"
done

if [ -f SHA256SUMS ]; then
    checksum_manifest="${tmp_dir}/SHA256SUMS.current"
    grep -F "  ${DEB_PATH}" SHA256SUMS > "${checksum_manifest}" || true
    grep -F "  ${TAR_PATH}" SHA256SUMS >> "${checksum_manifest}" || true
    grep -F "  ${ZIP_PATH}" SHA256SUMS >> "${checksum_manifest}" || true
    if [ -s "${checksum_manifest}" ]; then
        sha256sum -c "${checksum_manifest}"
    else
        echo "[verify-release] SHA256SUMS has no entries for the selected artifacts; skipping checksum check."
    fi
fi

if [ "${RUN_INSTALL_SMOKE}" != "1" ]; then
    echo "[verify-release] Install smoke disabled; artifact content checks passed."
    exit 0
fi

echo "[verify-release] Installing ${DEB_PATH}"
sudo apt-get update
sudo apt-get install -y --reinstall --no-install-recommends "./${DEB_PATH}"

command -v pantheon >/dev/null 2>&1 || fail "pantheon command was not installed"
pantheon --help >/dev/null
pantheon --version | grep -F "${VERSION}" >/dev/null

echo "[verify-release] Running installed pantheon smoke test for ${SMOKE_DURATION}s"
sudo env PANTHEON_MOCK=1 pantheon --platform mock --test baseline_metrics --duration "${SMOKE_DURATION}" --gpu all --mem 1

echo "[verify-release] Release artifacts passed content, install, and runtime checks."
