#!/usr/bin/env bash
# Runs inside the seat-234 pod. Idempotent: python deps the original repo imports, the wan_models layout it
# expects (symlink to the HF snapshot), and the TAEHV weights (ckpts/taew2_1.pth).
set -euo pipefail
ROOT=/workspace/livevid/sdv2_root
export UV_CACHE_DIR=/workspace/.uv-cache
uv pip install --python /workspace/venvs/tnx/bin/python einops ftfy regex sentencepiece protobuf omegaconf
mkdir -p "$ROOT/wan_models" "$ROOT/ckpts"
SNAP="$(ls -d /root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/* | head -1)"
ln -sfn "$SNAP" "$ROOT/wan_models/Wan2.1-T2V-1.3B"
[[ -s "$ROOT/ckpts/taew2_1.pth" ]] || curl -L --fail -o "$ROOT/ckpts/taew2_1.pth" https://github.com/madebyollin/taehv/raw/main/taew2_1.pth
ls -laL "$ROOT/wan_models/Wan2.1-T2V-1.3B" "$ROOT/ckpts"
echo SETUP OK
