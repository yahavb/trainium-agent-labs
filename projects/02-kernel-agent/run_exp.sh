#!/usr/bin/env bash
# run_exp.sh -- every experiment, backgrounded, named, reproducible.
#
# usage: ./run_exp.sh <exp_name> <agent.py args...>
#   e.g. ./run_exp.sh base_L4 --level 4 --rounds 8 --samples 4 --context 8192 --repeat 3
#
# Why this exists (all measured in this repo):
#  - a kubectl exec session drops and kills a foreground run, so every run is nohup'd
#  - a --repeat log is only splittable into runs if --exp tags each line (A1 patch), so set it here
#  - a win is only a win if it is attributable, so the git commit + dirty state are recorded
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "usage: ./run_exp.sh <exp_name> <agent.py args...>" >&2
  exit 2
fi
name=$1; shift
mkdir -p logs
commit=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
dirty=$(git diff --quiet 2>/dev/null && echo clean || echo dirty)
echo "$(date -Is) exp=$name commit=$commit($dirty) seat=$(hostname) model=${KERNEL_AGENT_MODEL:-?} args=$*" >> logs/RUNS.txt
nohup python agent.py --exp "$name" --log "logs/$name.jsonl" "$@" > "logs/$name.out" 2>&1 < /dev/null &
echo "started $name (pid $!)."
echo "  watch:   tail -f logs/$name.out"
echo "  running? pgrep -af agent.py"
