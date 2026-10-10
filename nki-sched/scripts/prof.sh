#!/bin/bash
# Local driver: build -> copy -> compile+check on a free core -> neuron-explorer capture -> summary.
#   scripts/prof.sh examples/fast_mm.py:fast K M N [tag]
# Env: CORE (default 2; cores 0,1 are held by vllm), PY (python with torch).
set -e
TARGET=$1; K=$2; M=$3; N=$4; TAG=${5:-$(basename ${TARGET%%:*} .py)}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/home/luka/aws-hack/.venv/bin/python}
CORE=${CORE:-2}
mkdir -p $ROOT/out
$PY $ROOT/scripts/build.py "$TARGET" $ROOT/out/$TAG.py
$ROOT/scripts/pod.sh put $ROOT/out/$TAG.py /tmp/$TAG.py
$ROOT/scripts/pod.sh put $ROOT/scripts/bench_run.py /tmp/bench_run.py
$ROOT/scripts/pod.sh put $ROOT/scripts/summ.py /tmp/summ.py
$ROOT/scripts/pod.sh "cd /tmp && rm -rf art_$TAG && export NEURON_RT_VISIBLE_CORES=$CORE LNC=${LNC:-1} && ART=/tmp/art_$TAG python bench_run.py $TAG.py mm $K $M $N 2>&1 | grep -E 'max rel|Error|error' | head; \
  neuron-explorer capture -n /tmp/art_$TAG/kernel.neff -s /tmp/$TAG.ntff >/dev/null 2>&1; \
  neuron-explorer view -n /tmp/art_$TAG/kernel.neff -s /tmp/$TAG.ntff --output-format summary-text --disable-ui 2>&1 | python /tmp/summ.py $K $M $N"
