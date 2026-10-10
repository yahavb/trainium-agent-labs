#!/usr/bin/env bash
# Runs inside the seat-234 pod from tier2/. Waits for the CPU reference, then: Neuron run, decode + video,
# and the same host code with the fp32 CPU port (to separate host-logic error from bf16/Neuron error).
set -uo pipefail
export NEURON_RT_VISIBLE_CORES=0
until [[ -s /workspace/livevid/artifacts/tier2/g2/ref.pt ]]; do sleep 5; done
sleep 5
python -u g2_neuron.py --backend neuron 2>&1 | grep -v -i warn
python -u g2_decode.py --backend neuron 2>&1 | grep -v -i warn
python -u g2_neuron.py --backend cpu 2>&1 | grep -v -i warn
echo RUN_ALL DONE
