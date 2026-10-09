#!/usr/bin/env bash
# Serve Qwen3-8B on this seat pod's Trainium chip, and wait until it answers.
#
# Runs inside the seat pod, which already runs the vLLM Neuron image — no docker. The server
# runs in the background and survives this shell exiting; run the projects from a second
# `kubectl exec` into the same pod, against http://localhost:8000.
#
# Sized for one s-lnc2 seat: one chip, LNC=2, --tensor-parallel-size 2. This is the exact
# configuration that was measured working.
#
#   ./serve.sh              start it and wait for ready
#   ./serve.sh --logs       follow the log of a running server
#   ./serve.sh --stop       stop it
#
# Override with environment variables, e.g.  MODEL=Qwen/Qwen2.5-7B-Instruct ./serve.sh
# VLLM_EXTRA_ARGS is appended to `vllm serve` as-is (space-separated, so no spaces inside a value).
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-8B}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
BLOCK_SIZE=${BLOCK_SIZE:-32}
TP=${TP:-2}
LOG=${LOG:-/tmp/vllm.log}
read -ra EXTRA <<< "${VLLM_EXTRA_ARGS:-}"

# Required by the NxD vLLM v1 guide, or you get out-of-bounds errors. With chunked prefill on
# (the default) the formula is ceil(max_model_len / block_size) * max_num_seqs.
BLOCKS=${BLOCKS:-$(( ( (MAX_MODEL_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE ) * MAX_NUM_SEQS ))}

healthy() { python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:$PORT/health', timeout=5)" >/dev/null 2>&1; }
running() { pgrep -f "vllm serve" >/dev/null; }

case "${1:-}" in
  --stop) pkill -f "vllm serve" && echo "Stopped." || echo "Not running."; exit 0 ;;
  --logs) exec tail -n 200 -f "$LOG" ;;
esac

if ! ls /dev/neuron* >/dev/null 2>&1; then
  echo "No Trainium device in this pod. Is this a seat pod?" >&2
  exit 1
fi

if running; then
  echo "The model server is already running; waiting for it to answer. Log: $LOG"
else
  mkdir -p "$HOME/.cache/huggingface"
  echo "Starting $MODEL (tensor parallel $TP, context $MAX_MODEL_LEN). Log: $LOG"
  NEURON_SKIP_EFA_AFFINITY=1 PYTHONUNBUFFERED=1 nohup setsid \
    vllm serve \
      --model "$MODEL" \
      --tensor-parallel-size "$TP" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --block-size "$BLOCK_SIZE" \
      --num-gpu-blocks-override "$BLOCKS" \
      --no-enable-prefix-caching \
      --port "$PORT" "${EXTRA[@]}" >"$LOG" 2>&1 </dev/null &
  echo "Started. The first run downloads the weights and compiles the model."
  echo "Measured: about 4 minutes. Allowing up to 40."
fi
echo

start=$(date +%s)
deadline=$(( start + 2400 ))
while :; do
  if healthy; then
    echo "READY. The model is answering on http://localhost:$PORT"
    echo
    echo "Leave this terminal. Open a second one and get another shell in this same pod:"
    echo "    kubectl exec -it $(hostname) -- bash"
    echo
    echo "Then run project 1 there:"
    echo "    cd /workspace/projects/01-heat-rod-pde"
    echo "    python agent.py --level 1 --all"
    exit 0
  fi
  if ! running; then
    echo "The server stopped. Last lines of its log:"
    tail -n 40 "$LOG" 2>&1 | sed 's/^/    /'
    exit 1
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "Still not answering after 40 minutes. Last lines of its log:"
    tail -n 40 "$LOG" 2>&1 | sed 's/^/    /'
    exit 1
  fi
  sleep 20
  printf '  still starting, %d minutes elapsed\n' $(( ( $(date +%s) - start ) / 60 ))
done
