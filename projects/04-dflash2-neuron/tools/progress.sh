#!/usr/bin/env bash
# progress.sh — one-line view of where a vLLM Neuron startup is.
LOG=${1:-/tmp/dflash2_test.log}
echo "now: $(date -u +%H:%M:%S)  log age: $(( $(date +%s) - $(stat -c %Y "$LOG") ))s since last write"
echo "HLO graphs compiled: $(grep -ac 'Compiled HLO' "$LOG")"
echo "compiler procs running: $(pgrep -ac -f 'neuronxcc|walrus_driver' || echo 0)"
echo "errors in log: $(grep -acE 'Traceback|Error:' "$LOG")"
echo "--- milestones ---"
grep -aE 'Weight loading completed|DFlash2: loaded|Draft model loading complete|DFlash2 warmup|warmup|Application startup complete|Starting vLLM API' "$LOG" \
  | sed -E 's/^\((Worker_TP0|Worker_TP1|EngineCore|APIServer|Worker)[^)]*\) //' \
  | cut -c1-160 | awk '!seen[$0]++' | tail -10
