#!/usr/bin/env bash
# Runs inside the seat-234 pod from tier2/. Step 4: resident-cache prototype, 1 layer then 8 layers in one graph.
set -uo pipefail
export NEURON_RT_VISIBLE_CORES=0
python -u g2_resident.py --layers 1 2>&1 | grep -v -i warn
python -u g2_resident.py --layers 8 2>&1 | grep -v -i warn
echo RESIDENT_ALL DONE
