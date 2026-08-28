#!/usr/bin/env bash
# Export the publishable subset of this repository.
#
# Copying the working tree by hand is how non-redistributable files end up in a
# public repo. This does the copy and refuses to run if it finds a file whose
# license forbids redistribution outside the known exclusion list.
#
#   tools/export_public_tree.sh ../pantheon-public
#
set -euo pipefail

DEST="${1:?usage: export_public_tree.sh <destination-dir>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Paths that must never be published. Each needs a reason.
EXCLUDE_PREFIXES=(
  "kernels/common/optix"   # NVIDIA proprietary: redistribution expressly prohibited
)

# Files that describe licensing rather than being subject to it. They quote the
# prohibition, so a content scan matches them; publishing them is the point.
SCAN_SKIP=( "NOTICE" "LICENSE" "tools/export_public_tree.sh" )

scan_skipped() {
  local f="$1" s
  for s in "${SCAN_SKIP[@]}"; do [ "$f" = "$s" ] && return 0; done
  return 1
}

excluded() {
  local f="$1" e
  for e in "${EXCLUDE_PREFIXES[@]}"; do
    case "$f" in "$e"/*|"$e") return 0;; esac
  done
  return 1
}

# Licence text wraps across lines, so match on whitespace-normalised content
# rather than on a literal phrase -- the first version of this script matched
# nothing and exported the files it was written to keep out.
forbids_redistribution() {
  tr '\n' ' ' < "$1" | tr -s ' [:space:]' ' ' \
    | grep -qiE "(distribution|redistribution)[^.]{0,120}(is )?strictly prohibited|without an express license agreement[^.]{0,120}prohibited"
}

mapfile -d '' -t TRACKED < <(git ls-files -z)

violations=()
publish=()
for f in "${TRACKED[@]}"; do
  [ -f "$f" ] || continue
  if excluded "$f"; then continue; fi
  if ! scan_skipped "$f" && forbids_redistribution "$f"; then violations+=("$f"); fi
  publish+=("$f")
done

if [ "${#violations[@]}" -gt 0 ]; then
  echo "Refusing to export. These files forbid redistribution but are not excluded:" >&2
  printf '  %s\n' "${violations[@]}" >&2
  echo "Add them to EXCLUDE_PREFIXES in $0, with a reason, or remove them." >&2
  exit 1
fi

mkdir -p "$DEST"
for f in "${publish[@]}"; do
  mkdir -p "$DEST/$(dirname "$f")"
  cp -p "$f" "$DEST/$f"
done

echo "Exported ${#publish[@]} files to $DEST"
echo "Excluded: ${EXCLUDE_PREFIXES[*]}"
echo
echo "rt_virus builds its dummy kernel there unless OPTIX_PATH points at an"
echo "OptiX SDK. See NOTICE."
