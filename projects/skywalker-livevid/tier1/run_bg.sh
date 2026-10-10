#!/usr/bin/env bash
# Runs inside the seat-233 pod. Usage: tier1/run_bg.sh <log-name> <cmd...>
# Starts <cmd> detached from tier1/ with the tnx venv; output goes to logs/tier1/<log-name>.log
# and the exit code is appended to the log as "EXIT <rc>".
# The pod sees 192 CPUs but its cgroup quota is 11, so torch's default thread count (96) gets
# throttled badly; THREADS caps it (default 4).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
name="$1"; shift
mkdir -p "$ROOT/logs/tier1"
source /workspace/venvs/tnx/neuron_env.sh
cd "$ROOT/tier1"
export OMP_NUM_THREADS="${THREADS:-4}" MKL_NUM_THREADS="${THREADS:-4}"
log="$ROOT/logs/tier1/$name.log"
nohup setsid bash -c '"$@"; echo "EXIT $?"' _ "$@" > "$log" 2>&1 < /dev/null &
echo "started $name pid=$! log=$log"
