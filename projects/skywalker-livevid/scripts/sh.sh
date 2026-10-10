#!/usr/bin/env bash
# Usage: scripts/sh.sh   Opens an interactive shell in the seat pod.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
is_protected && { echo "error: refusing interactive shell on protected $SEAT" >&2; exit 1; }

exec kubectl exec -it "$SEAT" -c "$CONTAINER" -- bash
