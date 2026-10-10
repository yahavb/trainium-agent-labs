#!/usr/bin/env bash
# Runs inside the seat-234 pod from tier2/. Waits for the 4 stage graphs, warms the caches, then runs the
# pipeline on 4 cores and the same graphs sequentially on 1 core.
set -uo pipefail
A=/workspace/livevid/artifacts/tier2
until [[ $(ls $A/stage_*_resident_bf16.pt 2>/dev/null | wc -l) -ge 4 ]]; do sleep 5; done
sleep 3
NEURON_RT_VISIBLE_CORES=0 python -u g3_pipeline.py warm 2>&1 | grep -v -i warn
python -u g3_pipeline.py run --cores 0,1,2,3 --inflight 6 --chunks 120 --name 4core 2>&1 | grep -v -i warn
python -u g3_pipeline.py run --cores 0,0,0,0 --inflight 1 --chunks 40 --name 1core 2>&1 | grep -v -i warn
echo G3 DONE
