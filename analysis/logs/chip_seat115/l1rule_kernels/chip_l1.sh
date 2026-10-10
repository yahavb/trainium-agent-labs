#!/bin/bash
# Level-1 kernels on a NeuronCore: the reference (positive control), the 4 l1fix solves, one held kernel (negative control).
D=/workspace/devtest/l1chip; T=/workspace/projects/02-kernel-agent; K=/workspace/daykit
cd $T
args="1:$T/reference_level1.py"
for f in $D/l1fix_*.py; do args="$args 1:$f"; done
args="$args 1:$D/held_195b96d9.py"
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_LOGICAL_NC_CONFIG=1 PYTHONPATH=$T:$K/agent/nki:$K/agent/nki/check \
  python3 $K/agent/nki/check/device_check.py --reps 2 --out $D/device_l1.json $args > $D/device_l1.log 2>&1
echo "exit $?" >> $D/device_l1.log
