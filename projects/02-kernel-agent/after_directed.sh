#!/usr/bin/env bash
# Hands the chip over automatically, so nobody has to watch the clock.
#
#   bash after_directed.sh           wait (in the background) for the `directed` run in results/ to
#                                    finish, stop what follows it, then run in this order:
#                                      1. directed2            -> results-directed2/   (12 rounds)
#                                      2. enriched, located    -> results-baselines/   (8 rounds)
#   bash after_directed.sh status    one view of all three result folders
#   bash after_directed.sh refresh   after a `git pull` that changed the checker: restart directed2
#                                    with the new code if it has already started
#
# directed2 goes first because it is the best chance of solving a level; the two baselines follow.
# One job at a time: two runs sharing the model were measured at 4 minutes a round instead of 1.
set -uo pipefail
cd "$(dirname "$0")"
D2_ARGS=${D2_ARGS:-"--all --rounds 12 --samples 4 --context 8192"}
BASE_ARGS=${BASE_ARGS:-"--all --rounds 8 --samples 4 --context 8192"}
FOLDERS="results results-directed2 results-baselines"

if [ "${1:-}" = "status" ]; then
  for d in $FOLDERS; do
    echo "==================== $d ===================="
    if [ ! -d "$d" ]; then echo "not started yet"; echo; continue; fi
    if grep -q "^##### waiting" "$d/ablation.log" 2>/dev/null && ! grep -q "started" "$d/ablation.log"; then
      echo "QUEUED: waiting for the job before it."; echo; continue
    fi
    for log in "$d"/*.log; do
      name=$(basename "$log" .log)
      case "$name" in ablation|selftest) continue ;; esac
      echo "--- feedback = $name ---"
      grep -hE "^=========== level|SOLVED on round|STOPPING|not solved in|^  level [0-9]|^  solved " "$log" | cut -c1-110 | tail -14
    done
    grep -h "^#####" "$d/ablation.log" 2>/dev/null | tail -3
    echo
  done
  [ -f handover.log ] && { echo "==================== hand-over ===================="; tail -4 handover.log; }
  exit 0
fi

start_d2_then_baselines() {
  OUT=results-directed2 MODES=directed2 REPEAT=1 AGENT_ARGS="$D2_ARGS" bash go.sh
  GO_WAIT_FOR=results-directed2 OUT=results-baselines MODES="enriched located" REPEAT=1 \
    AGENT_ARGS="$BASE_ARGS" bash go.sh
}

if [ "${1:-}" = "_wait" ]; then
  echo "waiting for 'directed' to finish, since $(date +%H:%M:%S)"
  while ! grep -q "feedback=directed finished" results/ablation.log 2>/dev/null; do sleep 10; done
  sleep 5                                   # let its taxonomy file finish writing
  echo "'directed' finished at $(date +%H:%M:%S); stopping the runs queued behind it"
  bash go.sh stop
  start_d2_then_baselines
  echo "hand-over done at $(date +%H:%M:%S)"
  exit 0
fi

if [ "${1:-}" = "refresh" ]; then
  # Run after `git pull` when the checker's messages changed. A run that has already started keeps
  # the code it loaded, so directed2 is restarted from the top if it is under way; the baselines
  # queued behind it are stopped first so they do not jump in and share the model.
  if [ ! -d results-directed2 ]; then
    echo "directed2 has not started yet, so it will pick up the new code by itself. Nothing to do."
    exit 0
  fi
  OUT=results-baselines bash go.sh stop
  OUT=results-directed2 bash go.sh stop
  echo "Restarting directed2 with the new code; its earlier attempts are kept in a previous-* folder."
  start_d2_then_baselines
  exit 0
fi

if [ ! -f results/ablation.log ]; then
  echo "There is no run in results/ to wait for. Start one with:  bash go.sh"; exit 1
fi
SETSID=$(command -v setsid || true)
nohup $SETSID bash "$(pwd)/$(basename "$0")" _wait > handover.log 2>&1 < /dev/null &
echo "Set. When 'directed' finishes, directed2 starts by itself, then enriched and located."
echo "Nothing more to do. Look in whenever you like with:"
echo "    bash $(pwd)/$(basename "$0") status"
