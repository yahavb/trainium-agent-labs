#!/usr/bin/env bash
# Usage: tier1/remote.sh "<cmd>"   Runs <cmd> in the seat-233 pod from /workspace/livevid
# with the tnx venv environment loaded, and returns its exit code.
set -euo pipefail
source "$(dirname "$0")/_seat.sh"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 \"<cmd>\"" >&2
  exit 2
fi

rc=0
kubectl exec -i "$SEAT" -c "$CONTAINER" -- bash -lc \
  "source /workspace/venvs/tnx/neuron_env.sh && mkdir -p $REMOTE_ROOT/logs/tier1 && cd $REMOTE_ROOT && $*" || rc=$?
exit "$rc"
