#!/usr/bin/env bash
# Runs inside a seat pod. Runs smoke_neuron.py on core 0 with the tnx venv and samples
# neuron-monitor / neuron-ls while it is under load.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-/workspace/venvs/tnx}"
LOGS="$ROOT/logs"
mkdir -p "$LOGS" "$ROOT/artifacts"
source "$VENV/neuron_env.sh"

NEURON_RT_VISIBLE_CORES="${NEURON_RT_VISIBLE_CORES:-0}" "$VENV/bin/python" -u "$ROOT/scripts/smoke_neuron.py" "$@" > "$LOGS/smoke.log" 2>&1 &
pid=$!

for _ in $(seq 1 600); do
  grep -q "SOAK START" "$LOGS/smoke.log" 2>/dev/null && break
  kill -0 "$pid" 2>/dev/null || break
  sleep 1
done

if grep -q "SOAK START" "$LOGS/smoke.log"; then
  sleep 2
  neuron-ls > "$LOGS/smoke_neuron_ls.txt" 2>&1
  timeout 12 neuron-monitor > "$LOGS/smoke_neuron_monitor.json" 2> "$LOGS/smoke_neuron_monitor.err"
fi
wait "$pid"
rc=$?
echo "smoke exit code: $rc" >> "$LOGS/smoke.log"
echo "SMOKE DONE rc=$rc"
