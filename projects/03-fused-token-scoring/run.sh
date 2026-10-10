#!/usr/bin/env bash
# Run inside the supplied Trainium2 image, after coordinating core availability.
set -euo pipefail
cd "$(dirname "$0")"
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_RT_VISIBLE_CORES="${NEURON_RT_VISIBLE_CORES:-0,1}"
mkdir -p results build
trap 'status=$?; printf "%s\n" "$status" > results/run.exit' EXIT
python -u runtime.py
python -u validate.py
python -u benchmark.py --tune "$@"
python -u validate.py --tuning results/tuning.json --output results/validation-tuned.json
python -u benchmark.py "$@"
python -u audit_results.py
