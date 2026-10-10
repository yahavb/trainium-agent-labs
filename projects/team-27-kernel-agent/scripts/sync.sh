#!/usr/bin/env bash
# Push tracked + staged files to the seat pod. Needs AWS creds in the environment.
set -euo pipefail
cd "$(dirname "$0")/.."
POD=${POD:?set POD=seat-NNN}
git ls-files | COPYFILE_DISABLE=1 tar --no-xattrs -czf - -T - \
  | kubectl exec -i "$POD" -c app -- bash -c 'mkdir -p /workspace/stageA && tar xzf - -C /workspace/stageA'
