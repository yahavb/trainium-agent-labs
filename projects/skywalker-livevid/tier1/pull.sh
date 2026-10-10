#!/usr/bin/env bash
# Usage: tier1/pull.sh <remote_path> <local_dir>
# Copies a file or directory out of the seat-233 pod into <local_dir> (created if missing),
# replacing any item of the same name. Relative remote paths resolve against /workspace/livevid.
set -euo pipefail
source "$(dirname "$0")/_seat.sh"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <remote_path> <local_dir>" >&2
  exit 2
fi
remote="$1"
local_dir="${2%/}"
[[ "$remote" = /* ]] || remote="$REMOTE_ROOT/$remote"
remote="${remote%/}"

# kubectl exec streams drop now and then on this cluster, so extract into a temp dir and
# only move the result into place after a clean transfer.
mkdir -p "$local_dir"
rbase="$(basename "$remote")"
for attempt in 1 2 3 4 5 6 7 8; do
  tmp="$(mktemp -d)"
  if kubectl exec -i "$SEAT" -c "$CONTAINER" -- tar -cz -C "$(dirname "$remote")" "$rbase" 2>/dev/null \
      | tar -xz -C "$tmp" 2>/dev/null; then
    rm -rf "${local_dir:?}/$rbase"
    mv "$tmp/$rbase" "$local_dir/"
    rm -rf "$tmp"
    echo "pulled $SEAT:$remote -> $local_dir/ (attempt $attempt)"
    exit 0
  fi
  rm -rf "$tmp"
  sleep 2
done
echo "error: could not pull $SEAT:$remote after 8 attempts" >&2
exit 1
