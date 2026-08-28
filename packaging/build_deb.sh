#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PACKAGE_NAME="${PACKAGE_NAME:-pantheongpu}"
VERSION="${VERSION:-$(tr -d '[:space:]' < VERSION)}"
ARCH="${ARCH:-amd64}"
RELEASE_ROOT="${RELEASE_ROOT:-release}"
APP_DIR="${APP_DIR:-${RELEASE_ROOT}/pantheon-${VERSION}}"
DIST_DIR="${DIST_DIR:-dist}"
DEB_ROOT="${DIST_DIR}/${PACKAGE_NAME}_${VERSION}_${ARCH}"
DEB_PATH="${DIST_DIR}/${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
BUILD_BINARY="${BUILD_BINARY:-1}"
MAX_GLIBC="${PANTHEON_MAX_GLIBC:-2.28}"

if [ "${BUILD_BINARY}" = "1" ]; then
    PANTHEON_CREATE_ARCHIVES=0 "${REPO_ROOT}/build_pantheon.sh"
fi

if [ ! -x "${APP_DIR}/pantheon" ]; then
    echo "Missing executable: ${APP_DIR}/pantheon" >&2
    exit 1
fi

"${SCRIPT_DIR}/check_glibc_compat.sh" "${APP_DIR}/pantheon" "${MAX_GLIBC}"
if [ -d "${APP_DIR}/pantheon.dist" ]; then
    "${SCRIPT_DIR}/check_glibc_compat.sh" "${APP_DIR}/pantheon.dist" "${MAX_GLIBC}"
fi

rm -rf "${DEB_ROOT}" "${DEB_PATH}"
mkdir -p \
    "${DEB_ROOT}/DEBIAN" \
    "${DEB_ROOT}/opt/pantheongpu/bin" \
    "${DEB_ROOT}/opt/pantheongpu/share/doc/pantheongpu" \
    "${DEB_ROOT}/usr/bin"

install -m 0755 "${APP_DIR}/pantheon" "${DEB_ROOT}/opt/pantheongpu/bin/pantheon"
install -m 0644 VERSION "${DEB_ROOT}/opt/pantheongpu/VERSION"
install -m 0644 LICENSE "${DEB_ROOT}/opt/pantheongpu/share/doc/pantheongpu/LICENSE"
install -m 0644 packaging/RELEASE_README.md "${DEB_ROOT}/opt/pantheongpu/share/doc/pantheongpu/README.md"
install -m 0644 RELEASE_NOTES.md "${DEB_ROOT}/opt/pantheongpu/share/doc/pantheongpu/RELEASE_NOTES.md"

cat > "${DEB_ROOT}/usr/bin/pantheon" <<'EOF'
#!/usr/bin/env sh
exec /opt/pantheongpu/bin/pantheon "$@"
EOF

chmod 0755 "${DEB_ROOT}/usr/bin/pantheon"

cat > "${DEB_ROOT}/DEBIAN/postrm" <<'EOF'
#!/usr/bin/env sh
set -e

case "${1:-}" in
    remove|purge)
        # Pantheon creates workload files below its installation prefix at
        # runtime. They are not tracked by dpkg, so remove the tree explicitly.
        rm -rf /opt/pantheongpu
        ;;
esac
EOF
chmod 0755 "${DEB_ROOT}/DEBIAN/postrm"

installed_size="$(du -sk "${DEB_ROOT}" | awk '{print $1}')"
cat > "${DEB_ROOT}/DEBIAN/control" <<EOF
Package: ${PACKAGE_NAME}
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Maintainer: Pantheon GPU <support@pantheongpu.local>
Depends: make, g++ | build-essential
Recommends: nvidia-cuda-toolkit | hipcc
Installed-Size: ${installed_size}
Description: Universal GPU stress and diagnostics suite
 Pantheon GPU installs binary-only command line tools that auto-detect CUDA,
 ROCm/HIP, or mock mode at runtime and build local GPU workload binaries for
 the current machine. The package does not ship Python or kernel source files.
EOF

# Debian 11's dpkg cannot read the Zstandard-compressed control archive that
# newer dpkg versions produce by default. Gzip keeps the package installable
# across the full supported Ubuntu and Debian matrix.
dpkg-deb --build --root-owner-group -Zgzip "${DEB_ROOT}" "${DEB_PATH}"

echo "Created ${DEB_PATH}"
