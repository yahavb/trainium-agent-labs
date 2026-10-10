#!/bin/bash
# scripts/sweep.sh K M N "mt nt kt" "mt nt kt" ...   (profiles examples/fast_mm.py:fast_env for each config)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
K=$1; M=$2; N=$3; shift 3
for cfg in "$@"; do
  read mt nt kt <<< "$cfg"
  echo -n "MT=$mt NT=$nt KT=$kt: "
  MT=$mt NT=$nt KT=$kt $ROOT/scripts/prof.sh examples/fast_mm.py:fast_env $K $M $N sweep 2>&1 | tail -1
done
