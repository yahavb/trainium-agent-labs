#!/usr/bin/env bash
# Usage: scripts/pod.sh "<cmd>"   Runs <cmd> in the seat pod and returns its exit code.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 \"<cmd>\"" >&2
  exit 2
fi

rc=0
kubectl exec -i "$SEAT" -c "$CONTAINER" -- bash -lc "$*" || rc=$?
exit "$rc"
