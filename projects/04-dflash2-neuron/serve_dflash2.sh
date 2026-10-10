#!/usr/bin/env bash
# serve_dflash2.sh — launch Qwen3-8B + DFlash2 speculative decoding on Neuron (dev config).
set -uo pipefail
PORT=${PORT:-8000}

# Stop any existing vLLM and wait until it is really gone (cores can't be shared,
# and a dying server would keep writing into the log).
pkill -f "vllm serve" 2>/dev/null || true
for _ in $(seq 1 20); do pgrep -f "vllm serve" >/dev/null || break; sleep 1; done
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -x walrus_driver 2>/dev/null || true
pkill -9 -x neuronx-cc 2>/dev/null || true
sleep 1

# Remove compile-cache entries that never produced a NEFF (left behind by killed runs).
# The compiler otherwise compiles EVERY unfinished graph in the cache, even ones the
# current config doesn't need (that cost 10+ minutes per start before this fix).
CACHE=/root/.cache/neuron_libtorch/neuron/compile_cache
for d in "$CACHE"/*/; do
    [ -d "$d" ] || continue
    [[ "$(basename "$d")" =~ ^[0-9a-f]{32}$ ]] || continue
    if ! find "$d" -name '*.neff' -print -quit | grep -q .; then
        echo "removing incomplete compile cache entry: $(basename "$d")"
        rm -rf "$d"
    fi
done

LOG=/tmp/dflash2_$(date +%s).log
ln -sfn "$LOG" /tmp/dflash2_test.log

SPEC_CONFIG='{"model": "z-lab/Qwen3-8B-DFlash-b16", "num_speculative_tokens": 4, "method": "dflash"}'
echo "Starting Qwen3-8B + DFlash2. Log: $LOG (linked as /tmp/dflash2_test.log)"

NEURON_SKIP_EFA_AFFINITY=1 PYTHONUNBUFFERED=1 nohup vllm serve Qwen/Qwen3-8B \
    --speculative-config "$SPEC_CONFIG" \
    --tensor-parallel-size 2 \
    --max-model-len "${DFLASH_DEV_MAX_LEN:-256}" \
    --max-num-seqs "${DFLASH_DEV_MAX_SEQS:-1}" \
    --max-num-batched-tokens "${DFLASH_DEV_BATCH_TOKENS:-256}" \
    --block-size 32 \
    --num-gpu-blocks-override "${DFLASH_DEV_BLOCKS:-64}" \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --port "$PORT" > "$LOG" 2>&1 &
echo "Started PID $!"
