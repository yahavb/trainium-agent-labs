#!/bin/bash
# Today's seat hand-in picks (levels 3, 4, 9) on a NeuronCore, after the level-1 chip run ends.
D=/workspace/devtest/seatpicks; T=/workspace/projects/02-kernel-agent; K=/workspace/daykit
until grep -q "^exit" /workspace/devtest/l1chip/device_l1.log 2>/dev/null; do sleep 10; done
cd $T
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_LOGICAL_NC_CONFIG=1 PYTHONPATH=$T:$K/agent/nki:$K/agent/nki/check \
  python3 $K/agent/nki/check/device_check.py --reps 2 --out $D/device_seat.json 3:$D/l3.py 4:$D/l4.py 9:$D/l9.py > $D/device_seat.log 2>&1
echo "exit $?" >> $D/device_seat.log
