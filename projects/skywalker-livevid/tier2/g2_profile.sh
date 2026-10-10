#!/usr/bin/env bash
# Runs inside the seat-234 pod from tier2/. Device profile of the G1 block NEFF (random inputs) on core 0.
set -uo pipefail
NEFF=/workspace/livevid/artifacts/tier2/work/g1_block0_512x512_L1024_cache6144_sink3072_text512/graph.neff
P=/workspace/livevid/artifacts/tier2/profile
mkdir -p "$P"
export NEURON_RT_VISIBLE_CORES=0
neuron-explorer capture -n "$NEFF" -s "$P/block0.ntff" 2>&1 | tail -15
ls -la "$P"
NTFF="$(ls "$P"/*.ntff | head -1)"
neuron-explorer view -n "$NEFF" -s "$NTFF" --output-format summary-text > "$P/summary.txt" 2> "$P/view.err"
echo "view rc=$?"; tail -5 "$P/view.err"
cat "$P/summary.txt"
echo PROFILE DONE
