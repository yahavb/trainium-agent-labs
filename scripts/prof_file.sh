#!/bin/bash
# Profile an already-emitted kernel file on the pod:  scripts/prof_file.sh out/x.py K M N
F=$(basename $1 .py); S=$(dirname $0)
$S/pod.sh put $1 /tmp/$F.py; $S/pod.sh put $S/bench_run.py /tmp/bench_run.py; $S/pod.sh put $S/summ.py /tmp/summ.py
$S/pod.sh "cd /tmp && rm -rf art_$F && export NEURON_RT_VISIBLE_CORES=${CORE:-2} && ART=/tmp/art_$F python bench_run.py $F.py mm $2 $3 $4 2>&1 | grep -E 'max rel|Error|error' | head; \
  neuron-explorer capture -n /tmp/art_$F/kernel.neff -s /tmp/$F.ntff >/dev/null 2>&1; \
  neuron-explorer view -n /tmp/art_$F/kernel.neff -s /tmp/$F.ntff --output-format summary-text --disable-ui 2>&1 | python /tmp/summ.py $2 $3 $4"
