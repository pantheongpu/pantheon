#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREFIX="${1:-${PANTHEON_SOURCE_WRAPPER_PREFIX:-${REPO_ROOT}/.ci-install}}"
case "${PREFIX}" in
    /*) ;;
    *) PREFIX="${REPO_ROOT}/${PREFIX}" ;;
esac
BIN_DIR="${PREFIX}/bin"

mkdir -p "${BIN_DIR}"

cat > "${BIN_DIR}/pantheon" <<EOF
#!/usr/bin/env sh
exec python3 "${REPO_ROOT}/pantheon.py" "\$@"
EOF

chmod 0755 "${BIN_DIR}/pantheon"
printf '%s\n' "${BIN_DIR}"
