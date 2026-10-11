#!/usr/bin/env bash
# wait_log.sh — poll a vLLM log every 1s; exit as soon as the run fails or is ready.
# Prints only the useful lines (root-cause error or ready message).
# Usage: bash /workspace/wait_log.sh [logfile] [timeout_s]
LOG=${1:-/tmp/dflash2_test.log}
TIMEOUT=${2:-60}
if (( TIMEOUT > 120 )); then TIMEOUT=120; fi   # hard cap: never block long
start=$(date +%s)

while :; do
    if grep -aqE 'Engine core initialization failed|Worker failed with error|WorkerProc failed to start' "$LOG" 2>/dev/null; then
        echo "=== FAILED after $(( $(date +%s) - start ))s ==="
        # Root cause: the deepest exception lines from worker tracebacks, deduped
        grep -aE '(Error|Exception)(:| )' "$LOG" \
            | grep -vE 'core.py:1231|WorkerProc initialization failed|Engine core initialization failed|Worker failed with error|vllm._C|EFA device' \
            | sed -E 's/^\(Worker[^)]*\) (ERROR [0-9: -]+\[[^]]+\] )?//' \
            | awk '!seen[$0]++' | tail -8
        echo "--- last frames ---"
        grep -aE 'File "/opt/conda/lib/python3.13/site-packages/(vllm|vllm_neuron)/' "$LOG" \
            | sed -E 's/^.*File "\/opt\/conda\/lib\/python3.13\/site-packages\///' \
            | awk '!seen[$0]++' | tail -6
        exit 1
    fi
    if grep -aqE 'Application startup complete|Starting vLLM API server' "$LOG" 2>/dev/null; then
        echo "=== READY after $(( $(date +%s) - start ))s ==="
        grep -aE 'DFlash2|draft|Application startup complete' "$LOG" | tail -5
        exit 0
    fi
    if (( $(date +%s) - start > TIMEOUT )); then
        echo "=== still starting after ${TIMEOUT}s (not an error) ==="
        bash "$(dirname "$0")/progress.sh" "$LOG"
        exit 2
    fi
    sleep 1
done
