#!/usr/bin/env bash
set -euo pipefail

TARGET_PATH="${1:-}"
MAX_GLIBC="${2:-2.28}"

if [ -z "${TARGET_PATH}" ] || [ ! -e "${TARGET_PATH}" ]; then
    echo "Usage: $0 <executable-or-directory> [max-glibc-version]" >&2
    exit 2
fi

for command_name in file objdump; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "${command_name} is required to inspect GLIBC symbol versions." >&2
        exit 2
    fi
done

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
versions_file="${tmp_dir}/glibc_versions.txt"
files_file="${tmp_dir}/elf_files.txt"

if [ -d "${TARGET_PATH}" ]; then
    find "${TARGET_PATH}" -type f -print0 |
        while IFS= read -r -d '' candidate; do
            if file -b "${candidate}" | grep -q '^ELF '; then
                printf '%s\n' "${candidate}"
            fi
        done > "${files_file}"
else
    if ! file -b "${TARGET_PATH}" | grep -q '^ELF '; then
        echo "${TARGET_PATH} is not an ELF executable or library." >&2
        exit 2
    fi
    printf '%s\n' "${TARGET_PATH}" > "${files_file}"
fi

while IFS= read -r elf_file; do
    objdump -T "${elf_file}" 2>/dev/null |
        sed -n 's/.*GLIBC_\([0-9][0-9.]*\).*/\1/p' >> "${versions_file}" || true
done < "${files_file}"

required_versions="$(sort -Vu "${versions_file}" 2>/dev/null || true)"

if [ -z "${required_versions}" ]; then
    echo "No GLIBC symbol requirements found in ${TARGET_PATH}."
    exit 0
fi

highest_required="$(printf '%s\n' "${required_versions}" | tail -n 1)"
highest_allowed="$(
    printf '%s\n%s\n' "${highest_required}" "${MAX_GLIBC}" |
        sort -Vu |
        tail -n 1
)"

echo "Highest required GLIBC version: ${highest_required}"
echo "Maximum supported build baseline: GLIBC ${MAX_GLIBC}"

if [ "${highest_allowed}" != "${MAX_GLIBC}" ]; then
    echo "${TARGET_PATH} requires GLIBC ${highest_required}, which is newer than ${MAX_GLIBC}." >&2
    exit 1
fi
