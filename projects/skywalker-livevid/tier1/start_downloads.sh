#!/usr/bin/env bash
# Runs inside the seat-233 pod. Starts detached HF downloads of only the sd-turbo files the
# tier1 pipeline needs (fp16 safetensors, no full-precision duplicates) plus TAESD.
# vae/ is included because the CPU diffusers reference decodes with the full VAE.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/logs/tier1"
mkdir -p "$LOGS"

# --include takes one pattern per flag; extra bare arguments would be read as filenames.
INCLUDES=(model_index.json "scheduler/*" "tokenizer/*"
          text_encoder/config.json "text_encoder/*.fp16.safetensors"
          unet/config.json "unet/*.fp16.safetensors"
          vae/config.json "vae/*.fp16.safetensors")
INC_ARGS=()
for pat in "${INCLUDES[@]}"; do INC_ARGS+=(--include "$pat"); done

# hf takes one pattern per --include; extra values after it are read as literal filenames.
INCLUDES=(model_index.json "scheduler/*" "tokenizer/*"
          text_encoder/config.json "text_encoder/*.fp16.safetensors"
          unet/config.json "unet/*.fp16.safetensors"
          vae/config.json "vae/*.fp16.safetensors")
ARGS=()
for pat in "${INCLUDES[@]}"; do ARGS+=(--include "$pat"); done
nohup setsid hf download stabilityai/sd-turbo "${ARGS[@]}" > "$LOGS/dl_sd-turbo.log" 2>&1 < /dev/null &
echo "stabilityai/sd-turbo $!"

nohup setsid hf download madebyollin/taesd > "$LOGS/dl_taesd.log" 2>&1 < /dev/null &
echo "madebyollin/taesd $!"
