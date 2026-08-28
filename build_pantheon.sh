#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- CONFIGURATION ---
EXE_NAME="pantheon"
PYTHON_MAIN="pantheon.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NUITKA_ARGS=()
if [ -n "${PANTHEON_NUITKA_STATIC_LIBPYTHON:-}" ]; then
    NUITKA_ARGS+=("--static-libpython=${PANTHEON_NUITKA_STATIC_LIBPYTHON}")
fi
VERSION="$(tr -d '[:space:]' < VERSION)"
RELEASE_ROOT="release"
RELEASE_NAME="pantheon-${VERSION}"
RELEASE_DIR="${RELEASE_ROOT}/${RELEASE_NAME}"
TAR_PATH="${RELEASE_ROOT}/${RELEASE_NAME}.tar.gz"
ZIP_PATH="${RELEASE_ROOT}/${RELEASE_NAME}.zip"
CREATE_ARCHIVES="${PANTHEON_CREATE_ARCHIVES:-0}"

echo "===================================================="
echo "PANTHEON HIDDEN-JIT FORGE"
echo "===================================================="
echo "Version: ${VERSION}"

rm -rf build/ dist/ "${EXE_NAME}" "${RELEASE_DIR}" "${TAR_PATH}" "${ZIP_PATH}"
mkdir -p "${RELEASE_DIR}"

echo "[*] Embedding kernels and Makefile into standalone pantheon executable..."

"${PYTHON_BIN}" -m nuitka --standalone --onefile \
    "${NUITKA_ARGS[@]}" \
    --output-dir="${RELEASE_DIR}" \
    --output-filename="${EXE_NAME}" \
    --include-data-dir=./kernels=kernels \
    --include-data-file=./Makefile=Makefile \
    --include-data-file=./VERSION=VERSION \
    --follow-imports \
    "${PYTHON_MAIN}"

cp packaging/RELEASE_README.md "${RELEASE_DIR}/README.md"
cp RELEASE_NOTES.md LICENSE "${RELEASE_DIR}/"

if [ "${CREATE_ARCHIVES}" = "1" ]; then
    echo "[*] Creating release archives..."
    tar -czf "${TAR_PATH}" -C "${RELEASE_ROOT}" "${RELEASE_NAME}"

    if command -v zip >/dev/null 2>&1; then
        (cd "${RELEASE_ROOT}" && zip -qr "${RELEASE_NAME}.zip" "${RELEASE_NAME}")
    elif command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        "${PYTHON_BIN}" - <<PY
import shutil
shutil.make_archive("${RELEASE_ROOT}/${RELEASE_NAME}", "zip", "${RELEASE_ROOT}", "${RELEASE_NAME}")
PY
    else
        echo "[!] zip and python3 are unavailable; skipped zip archive."
    fi
else
    echo "[*] Skipping source-style release archives."
fi

echo "===================================================="
echo "SUCCESS:"
echo "  Executable: ${RELEASE_DIR}/${EXE_NAME}"
if [ "${CREATE_ARCHIVES}" = "1" ]; then
    echo "  Tarball:    ${TAR_PATH}"
    if [ -f "${ZIP_PATH}" ]; then
        echo "  Zip:        ${ZIP_PATH}"
    fi
fi
echo "===================================================="
