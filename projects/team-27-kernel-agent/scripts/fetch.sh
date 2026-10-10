#!/usr/bin/env bash
# Copy a seat pod's run logs into results/<pod>/ (tracked in git, unlike runs/).
#   POD=seat-132 scripts/fetch.sh
set -euo pipefail
cd "$(dirname "$0")/.."
POD=${POD:?set POD=seat-NNN}
mkdir -p "results/$POD"
kubectl exec "$POD" -c app -- tar czf - -C /workspace/stageA/runs . | tar xzf - -C "results/$POD"
ls -la "results/$POD"
