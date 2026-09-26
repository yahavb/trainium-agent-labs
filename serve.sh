#!/usr/bin/env bash
# Serve Qwen3-8B on this instance's Trainium chip, in a container, and wait until it answers.
#
# Sized for one trn2.3xlarge: one chip, LNC=2, so 2 logical NeuronCores, so
# --tensor-parallel-size 2. This is the exact configuration that was measured working.
#
#   ./serve.sh              start it and wait for ready
#   ./serve.sh --logs       follow the log of a running server
#   ./serve.sh --stop       stop and remove it
#
# Override with environment variables, e.g.  MODEL=Qwen/Qwen2.5-7B-Instruct ./serve.sh
set -euo pipefail

IMAGE=${IMAGE:-public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.24.0.1.1.0-neuronx-py313-sdk2.32.0-ubuntu24.04}
NAME=${NAME:-vllm}
MODEL=${MODEL:-Qwen/Qwen3-8B}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
BLOCK_SIZE=${BLOCK_SIZE:-32}
TP=${TP:-2}
NEURON_DEVICE=${NEURON_DEVICE:-/dev/neuron0}
REPO=$(cd "$(dirname "$0")" && pwd)

# Required by the NxD vLLM v1 guide, or you get out-of-bounds errors. With chunked prefill on
# (the default) the formula is ceil(max_model_len / block_size) * max_num_seqs.
BLOCKS=${BLOCKS:-$(( ( (MAX_MODEL_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE ) * MAX_NUM_SEQS ))}

case "${1:-}" in
  --stop) docker rm -f "$NAME" >/dev/null 2>&1 && echo "Stopped." || echo "Not running."; exit 0 ;;
  --logs) exec docker logs -f "$NAME" ;;
esac

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not available. Run ./install-docker.sh first." >&2
  exit 1
fi
if [ ! -e "$NEURON_DEVICE" ]; then
  echo "No Trainium device at $NEURON_DEVICE. Is this a Trainium instance?" >&2
  echo "Devices present: $(ls /dev/neuron* 2>/dev/null || echo none)" >&2
  exit 1
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
mkdir -p "$HOME/.cache/huggingface"

echo "Starting $MODEL on $NEURON_DEVICE (tensor parallel $TP, context $MAX_MODEL_LEN)..."
docker run -d --name "$NAME" \
  --device="$NEURON_DEVICE" \
  --cap-add IPC_LOCK \
  --shm-size 16g \
  -p "$PORT":8000 \
  -v "$REPO":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -e NEURON_SKIP_EFA_AFFINITY=1 \
  -e PYTHONUNBUFFERED=1 \
  "$IMAGE" \
  vllm serve \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --block-size "$BLOCK_SIZE" \
    --num-gpu-blocks-override "$BLOCKS" \
    --no-enable-prefix-caching \
    --port 8000 >/dev/null

echo "Started. The first run downloads the weights and compiles the model."
echo "Measured: about 4 minutes. Allowing up to 40."
echo

deadline=$(( $(date +%s) + 2400 ))
while :; do
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "The server stopped. Last lines of its log:"
    docker logs --tail 40 "$NAME" 2>&1 | sed 's/^/    /'
    exit 1
  fi
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "READY. The model is answering on http://localhost:$PORT"
    echo
    echo "Try it:"
    echo "    curl -s localhost:$PORT/v1/models"
    echo
    echo "Run project 1 inside the container, where the model is on localhost:"
    echo "    docker exec -it $NAME bash"
    echo "    cd /workspace/projects/01-heat-rod-pde"
    echo "    pip install sympy"
    echo "    export HEATROD_BASE_URL=http://localhost:8000/v1"
    echo "    python agent.py --level 1 --all"
    exit 0
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "Still not answering after 40 minutes. Last lines of its log:"
    docker logs --tail 40 "$NAME" 2>&1 | sed 's/^/    /'
    exit 1
  fi
  sleep 20
  printf '  still starting, %d minutes elapsed\n' \
    $(( ( $(date +%s) - (deadline - 2400) ) / 60 ))
done
