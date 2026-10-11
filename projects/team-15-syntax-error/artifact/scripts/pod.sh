#!/bin/bash
# Laptop-side helpers for seat-73. Credentials come from the 'hack' AWS profile in ~/.aws (never in this repo).
#   scripts/pod.sh sync                 push projects/02-kernel-agent/ into /workspace in the pod
#   scripts/pod.sh run '<cmd>'          run a command in the pod (cwd = the kernel-agent project)
#   scripts/pod.sh start <label> '<agent args>'   nohup an agent run writing runs/<label>.{log,jsonl} in the pod
#   scripts/pod.sh pull <label>         copy that run back to runs/<HHMM>_<label>/
#   scripts/pod.sh pullall              mirror all pod logs into runs/live/
set -euo pipefail
export PATH="/c/Program Files/Amazon/AWSCLIV2:$PATH" AWS_PROFILE=${AWS_PROFILE:-hack}
POD=${POD:-seat-73}
PROJ=${PROJ:-/workspace/projects/02-kernel-agent}   # PROJ=/workspace/dev for harness work during experiments
ROOT=$(cd "$(dirname "$0")/.." && pwd)
k() { kubectl exec "$POD" -c app -- bash -lc "$1"; }
case "$1" in
  sync)
    tar -C "$ROOT/projects/02-kernel-agent" --exclude='__pycache__' --exclude='runs' -cf - . \
      | kubectl exec -i "$POD" -c app -- bash -c "mkdir -p $PROJ/runs && tar -C $PROJ -xf -"
    echo "synced to $POD:$PROJ" ;;
  run) k "cd $PROJ && $2" ;;
  start)
    label=$2; shift 2
    k "cd $PROJ && mkdir -p runs && (setsid nohup python -u agent.py --tag $label --log runs/$label.jsonl $* > runs/$label.log 2>&1 < /dev/null &) ; sleep 1; pgrep -af 'agent.py --tag $label' | head -1" ;;
  pull)
    label=$2; dest="$ROOT/runs/$(TZ=America/New_York date +%H%M 2>/dev/null)_$label"
    dest="$ROOT/runs/$(python -c 'from datetime import datetime,timezone,timedelta;print((datetime.now(timezone.utc)-timedelta(hours=4)).strftime("%H%M"))')_$label"
    mkdir -p "$dest"
    k "cd $PROJ/runs && tar -cf - $label.log $label.jsonl" | tar -C "$dest" -xf -
    echo "pulled to $dest" ;;
  pullall)
    # mirror every log in the pod's runs/ into runs/live/ (overwritten each time; snapshot later)
    mkdir -p "$ROOT/runs/live"
    k "cd $PROJ/runs && tar -cf - *.log *.jsonl *.txt 2>/dev/null" | tar -C "$ROOT/runs/live" -xf -
    echo "mirrored to runs/live" ;;
  *) echo "usage: $0 sync|run|start|pull|pullall"; exit 1 ;;
esac
