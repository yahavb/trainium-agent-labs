#!/usr/bin/env bash
# One command for the feedback experiment on the kernel agent.
#
#   bash go.sh           start the model if it is not up, prove the checker, then run the agent
#                        under each feedback mode in the BACKGROUND (survives a dropped connection)
#   bash go.sh status    what is running, the latest lines, and the summary once it is finished
#   bash go.sh stop      stop the runs (the model keeps running)
#
# What it measures: the same levels, the same settings, only the checker's feedback changed.
#   enriched  the repo's original feedback: the exception plus a fix instruction
#   located   the same, plus the model's own failing line quoted back
# Each mode is run REPEAT times, because one run is not a result.
#
# Everything lands in results/:
#   <mode>.log            what the agent printed
#   <mode>.jsonl          every attempt with its score (the attempt log to hand in)
#   TAXONOMY-<mode>.md    failures grouped and counted, token use, how the failure moved
#   SUMMARY.md            the solve rates side by side
#
# Settings can be overridden:  MODES="located" REPEAT=5 bash go.sh
set -uo pipefail
# Resolve this script's own path BEFORE changing directory: it re-runs itself for the background
# job, and a relative path such as projects/02-kernel-agent/go.sh stops resolving after the cd.
cd "$(dirname "$0")"
SELF="$(pwd)/$(basename "$0")"

OUT=results
MODES=${MODES:-"located enriched"}
REPEAT=${REPEAT:-3}
AGENT_ARGS=${AGENT_ARGS:-"--all --rounds 8 --samples 4 --context 8192"}
PIDFILE=$OUT/ablation.pid
PY=${PYTHON:-python}
command -v "$PY" >/dev/null 2>&1 || PY=python3
BASE=${KERNEL_AGENT_BASE_URL:-http://localhost:8000/v1}

# "Running" means: a live process, not a finished one that was never reaped. In a pod whose first
# process is `sleep infinity` nothing reaps orphans, so a finished job can linger as a zombie and
# still answer kill -0. Check the process state and the job's own end marker as well.
running() {
  [ -f "$PIDFILE" ] || return 1
  local pid; pid=$(cat "$PIDFILE")
  kill -0 "$pid" 2>/dev/null || return 1
  grep -q '^State:[[:space:]]*Z' "/proc/$pid/status" 2>/dev/null && return 1
  grep -q '^##### ALL DONE' "$OUT/ablation.log" 2>/dev/null && return 1
  return 0
}

healthy() {
  "$PY" -c "import urllib.request; urllib.request.urlopen('${BASE%/v1}/health', timeout=5)" \
    >/dev/null 2>&1
}

summary() {
  echo "# Feedback experiment: $AGENT_ARGS, $REPEAT runs per mode"
  echo
  for mode in $MODES; do
    echo "## feedback = $mode"
    echo
    echo '```'
    if grep -q "=========== over .* runs" "$OUT/$mode.log" 2>/dev/null; then
      sed -n '/=========== over .* runs/,/all=/p' "$OUT/$mode.log"
    elif grep -q "=========== summary" "$OUT/$mode.log" 2>/dev/null; then
      sed -n '/=========== summary/,/solved [0-9]*\/[0-9]*/p' "$OUT/$mode.log"
    else
      echo "(no result: the run did not finish -- see $OUT/$mode.log)"
    fi
    echo '```'
    grep -h "quoted the failing line" "$OUT/TAXONOMY-$mode.md" 2>/dev/null
    echo
  done
}

case "${1:-}" in
  _run)
    # The background job. Not meant to be called by hand.
    for mode in $MODES; do
      echo "##### feedback=$mode started $(date +%H:%M:%S)"
      # shellcheck disable=SC2086
      "$PY" agent.py $AGENT_ARGS --repeat "$REPEAT" --feedback "$mode" \
        --log "$OUT/$mode.jsonl" > "$OUT/$mode.log" 2>&1
      echo "##### feedback=$mode finished $(date +%H:%M:%S) (exit $?)"
      "$PY" taxonomy.py --log "$OUT/$mode.jsonl" --examples > "$OUT/TAXONOMY-$mode.md" 2>&1
    done
    summary > "$OUT/SUMMARY.md"
    echo "##### ALL DONE $(date +%H:%M:%S)"
    exit 0 ;;

  status)
    if running; then
      echo "RUNNING (started $(date -r "$PIDFILE" +%H:%M 2>/dev/null))."
      cat "$OUT/ablation.log" 2>/dev/null
      current=$(grep "started" "$OUT/ablation.log" 2>/dev/null | tail -1 | sed 's/.*feedback=\([a-z]*\).*/\1/')
      if [ -n "$current" ]; then
        echo
        echo "--- latest from the '$current' run ---"
        grep -E "^(################|=========== |round |  level |  solved |  SOLVED|  STOPPING)" \
          "$OUT/$current.log" 2>/dev/null | tail -15
      fi
      echo
      echo "Check again with:  bash go.sh status"
    elif [ -f "$OUT/SUMMARY.md" ]; then
      echo "FINISHED."
      echo
      cat "$OUT/SUMMARY.md"
      echo "Full details: $OUT/TAXONOMY-*.md   Attempt logs: $OUT/*.jsonl"
    elif [ -f "$OUT/ablation.log" ]; then
      echo "NOT RUNNING, and it did not finish. Last lines:"
      tail -20 "$OUT/ablation.log"
      for mode in $MODES; do
        [ -f "$OUT/$mode.log" ] && { echo "--- $mode.log ---"; tail -15 "$OUT/$mode.log"; }
      done
    else
      echo "Nothing has been started yet. Run:  bash go.sh"
    fi
    exit 0 ;;

  stop)
    if running; then
      pid=$(cat "$PIDFILE")
      # The job leads its own process group (setsid below), so this stops the agent it is
      # running too, not only the loop around it.
      kill -- "-$pid" 2>/dev/null || { pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; }
      rm -f "$PIDFILE"
      echo "Stopped the runs. The model is still up."
    else
      echo "Nothing is running."
    fi
    exit 0 ;;

  "") ;;
  *) echo "usage: bash go.sh [status|stop]"; exit 2 ;;
esac

# ---------------------------------------------------------------- start

if running; then
  echo "The experiment is already running. See how far it is with:  bash go.sh status"
  exit 0
fi

# 1. the model
if [ -z "${GO_SKIP_MODEL:-}" ]; then
  if healthy; then
    echo "[1/3] The model is already answering at ${BASE%/v1}."
  elif [ -x ../../serve.sh ]; then
    echo "[1/3] The model is not up yet. Starting it (about 4 minutes the first time)..."
    ( cd ../.. && ./serve.sh ) || { echo "The model did not start. Find an organiser, or send the lines above."; exit 1; }
  else
    echo "[1/3] No model is answering at ${BASE%/v1} and serve.sh was not found."
    echo "      Start it first:  cd /workspace && ./serve.sh"
    exit 1
  fi
fi

# 2. the checker
if [ -z "${GO_SKIP_SELFTEST:-}" ]; then
  echo "[2/3] Proving the checker (nkibench --selftest)..."
  mkdir -p "$OUT"
  if ! "$PY" nkibench.py --selftest > "$OUT/selftest.log" 2>&1; then
    tail -25 "$OUT/selftest.log"
    echo
    echo "The checker's selftest FAILED, so no score would be trustworthy. Send the lines above."
    exit 1
  fi
  echo "      passed. (full output: $OUT/selftest.log)"
fi

# 3. the runs. Keep any earlier results instead of mixing new attempts into old logs.
mkdir -p "$OUT"
if ls "$OUT"/*.jsonl >/dev/null 2>&1; then
  prev="$OUT/previous-$(date +%H%M%S)"
  mkdir -p "$prev"
  for f in "$OUT"/*.jsonl "$OUT"/*.log "$OUT"/*.md; do
    [ -f "$f" ] && [ "$(basename "$f")" != "selftest.log" ] && mv "$f" "$prev/"
  done
  echo "      (earlier results moved to $prev)"
fi

SETSID=$(command -v setsid || true)
nohup $SETSID bash "$SELF" _run > "$OUT/ablation.log" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "[3/3] Started in the background: modes '$MODES', $REPEAT runs each."
echo
echo "It keeps going if your connection drops. Expect roughly 10-15 minutes per mode."
echo "  See progress, and the summary when it is done:   bash go.sh status"
echo "  Watch it live (Ctrl+C stops watching only):      tail -f $OUT/*.log"
