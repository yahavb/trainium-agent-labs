#!/bin/bash
# scripts/sweep.sh K M N "mt nt kt dma order res shard" ...  (use - for none)
# profiles examples/fast_mm.py:fast_env per config; CORES=2 also launches with LNC=2.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
K=$1; M=$2; N=$3; shift 3
export CORES=${CORES:-1}
[ "$CORES" -gt 1 ] && export LNC=$CORES
for cfg in "$@"; do
  read mt nt kt dma order res shard <<< "$cfg"
  [ "$res" = "-" ] && res=""
  [ "$shard" = "-" ] && shard=""
  echo -n "[$cfg] cores=$CORES: "
  MT=$mt NT=$nt KT=$kt DMA=${dma:-sw} ORDER=${order:-mn} RES=$res SHARD=$shard $ROOT/scripts/prof.sh examples/fast_mm.py:fast_env $K $M $N sweep 2>&1 | tail -1
done
