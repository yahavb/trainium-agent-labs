#!/usr/bin/env bash
# Start one benchmark slice in the background on a seat pod; survives a dropped connection.
#   POD=seat-132 scripts/launch.sh naive 1 2 3 4
# Needs AWS creds in this terminal. Push the code first with scripts/sync.sh. The attempt limit is
# the checked-out version's default (v2: 8, v3: 12).
set -euo pipefail
POD=${POD:?set POD=seat-NNN}
MODE=$1; shift
NAME="${MODE}-L$(IFS=; echo "$*")-${POD}"
kubectl exec "$POD" -c app -- bash -c "cd /workspace/stageA && mkdir -p runs && \
  (setsid nohup python3 -u -m kagent.agent --mode $MODE --levels $* --repeat ${REPEAT:-5} \
   --out runs/$NAME.jsonl > runs/$NAME.log 2>&1 < /dev/null &) && sleep 2 && \
  pgrep -af kagent.agent | grep -v pgrep"
echo "started $NAME on $POD; watch with:"
echo "  kubectl exec $POD -c app -- tail -f /workspace/stageA/runs/$NAME.log"
