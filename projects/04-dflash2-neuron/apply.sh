#!/usr/bin/env bash
# apply.sh — install DFlash2 speculative decoding into this seat's vllm_neuron.
#
#   bash apply.sh            # check, then apply
#   bash apply.sh --check    # dry run only
#
# Adds two new files (DFlash2 draft model + proposer) and applies a small patch to
# three existing vllm_neuron files. Safe to re-run: already-applied changes are skipped.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SP="$(python3 -c 'import os, vllm_neuron; print(os.path.dirname(os.path.dirname(vllm_neuron.__file__)))' 2>/dev/null)"
[ -d "$SP/vllm_neuron" ] || { echo "vllm_neuron not found (site-packages: $SP)"; exit 1; }
echo "site-packages: $SP"

if patch -p1 -d "$SP" --dry-run -R -s -f < "$HERE/vllm_neuron-dflash2.patch" >/dev/null 2>&1; then
    echo "patch: already applied"
    PATCH_NEEDED=0
else
    patch -p1 -d "$SP" --dry-run -f < "$HERE/vllm_neuron-dflash2.patch"
    PATCH_NEEDED=1
fi
[ "${1:-}" = "--check" ] && { echo "check only: nothing changed"; exit 0; }

cp -v "$HERE/overlay/vllm_neuron/model/qwen3/dflash_draft.py" "$SP/vllm_neuron/model/qwen3/"
cp -v "$HERE/overlay/vllm_neuron/vllm/spec_decode/dflash2.py" "$SP/vllm_neuron/vllm/spec_decode/"
[ "$PATCH_NEEDED" = 1 ] && patch -p1 -d "$SP" -f < "$HERE/vllm_neuron-dflash2.patch"

python3 -m py_compile \
    "$SP/vllm_neuron/model/qwen3/dflash_draft.py" \
    "$SP/vllm_neuron/vllm/spec_decode/dflash2.py" \
    "$SP/vllm_neuron/model/qwen3/model.py" \
    "$SP/vllm_neuron/model/registry.py" \
    "$SP/vllm_neuron/vllm/worker/neuron_model_runner.py"
echo "DFlash2 installed. Start it with: bash $HERE/serve_dflash2.sh"
