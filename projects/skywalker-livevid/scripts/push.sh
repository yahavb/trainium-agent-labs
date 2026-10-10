#!/usr/bin/env bash
# Usage: scripts/push.sh   Syncs the local repo into /workspace/livevid in the seat pod.
# repos/ is excluded because the pod keeps its own clones there.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
is_protected && { echo "error: refusing to push to protected $SEAT" >&2; exit 1; }

cd "$REPO_ROOT"
kubectl exec -i "$SEAT" -c "$CONTAINER" -- mkdir -p "$REMOTE_ROOT/logs" "$REMOTE_ROOT/repos"
COPYFILE_DISABLE=1 tar -cz --no-mac-metadata --no-xattrs --no-acls --no-fflags \
  --exclude=.git --exclude=models --exclude=logs --exclude=repos --exclude=.env --exclude=.DS_Store . \
  | kubectl exec -i "$SEAT" -c "$CONTAINER" -- tar -xz --no-same-owner -C "$REMOTE_ROOT"
echo "pushed $REPO_ROOT -> $SEAT:$REMOTE_ROOT"
