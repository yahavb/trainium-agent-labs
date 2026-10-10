#!/usr/bin/env bash
# Usage: tier1/push.sh   Syncs only the local tier1/ directory into /workspace/livevid/tier1
# in the seat-233 pod. Outputs produced on the pod (tier1/out) are left alone.
set -euo pipefail
source "$(dirname "$0")/_seat.sh"

cd "$REPO_ROOT"
kubectl exec -i "$SEAT" -c "$CONTAINER" -- mkdir -p "$REMOTE_ROOT/tier1" "$REMOTE_ROOT/logs/tier1" "$REMOTE_ROOT/reports"
COPYFILE_DISABLE=1 tar -cz --no-mac-metadata --no-xattrs --no-acls --no-fflags \
  --exclude=.DS_Store --exclude=__pycache__ --exclude=tier1/out tier1 \
  | kubectl exec -i "$SEAT" -c "$CONTAINER" -- tar -xz --no-same-owner -C "$REMOTE_ROOT"
echo "pushed $REPO_ROOT/tier1 -> $SEAT:$REMOTE_ROOT/tier1"
