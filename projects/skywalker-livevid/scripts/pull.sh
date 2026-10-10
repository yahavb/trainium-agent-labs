#!/usr/bin/env bash
# Usage: scripts/pull.sh <remote_path> <local_path>
# Copies a file or directory out of the seat pod. Relative remote paths are resolved
# against /workspace/livevid. If <local_path> is an existing directory, the item is
# placed inside it; otherwise it is written to <local_path> (replacing it).
set -euo pipefail
source "$(dirname "$0")/_common.sh"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <remote_path> <local_path>" >&2
  exit 2
fi
remote="$1"
local_path="$2"
[[ "$remote" = /* ]] || remote="$REMOTE_ROOT/$remote"
remote="${remote%/}"
rdir="$(dirname "$remote")"
rbase="$(basename "$remote")"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

kubectl exec -i "$SEAT" -c "$CONTAINER" -- tar -cz -C "$rdir" "$rbase" | tar -xz -C "$tmp"

if [[ -d "$local_path" ]]; then
  rm -rf "${local_path%/}/$rbase"
  mv "$tmp/$rbase" "$local_path/"
else
  mkdir -p "$(dirname "$local_path")"
  rm -rf "$local_path"
  mv "$tmp/$rbase" "$local_path"
fi
echo "pulled $SEAT:$remote -> $local_path"
