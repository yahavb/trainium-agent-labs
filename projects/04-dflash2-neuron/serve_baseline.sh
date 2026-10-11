#!/usr/bin/env bash
# serve_baseline.sh — plain Qwen3-8B with the SAME dev config as serve_dflash2.sh (no DFlash2),
# for an apples-to-apples throughput comparison.
set -uo pipefail
PORT=${PORT:-8000}
pkill -f "vllm serve" 2>/dev/null || true
for _ in $(seq 1 20); do pgrep -f "vllm serve|VLLM::|multiprocessing.resource_tracker" >/dev/null || break; sleep 1; done
pkill -9 -f "vllm serve|VLLM::|multiprocessing.resource_tracker" 2>/dev/null || true
for _ in $(seq 1 30); do
    neuron-ls 2>/dev/null | grep -qE '\| [0-9]{3,} +\|' || break
    sleep 1
done
sleep 1

LOG=/tmp/baseline_$(date +%s).log
ln -sfn "$LOG" /tmp/baseline.log
echo "Starting baseline Qwen3-8B. Log: $LOG (linked as /tmp/baseline.log)"
NEURON_SKIP_EFA_AFFINITY=1 PYTHONUNBUFFERED=1 nohup vllm serve Qwen/Qwen3-8B \
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
