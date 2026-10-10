# Sourced by the tier1 helpers. Track 1 is pinned to seat-233 and never touches another seat.
TIER1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TIER1_DIR/.." && pwd)"
SEAT="${SEAT:-seat-233}"
if [[ "$SEAT" != "seat-233" ]]; then
  echo "error: tier1 helpers only run against seat-233, got '$SEAT'" >&2
  exit 1
fi
CONTAINER="app"
REMOTE_ROOT="/workspace/livevid"
