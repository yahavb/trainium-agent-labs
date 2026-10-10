#!/usr/bin/env bash
# Generate an image with FLUX.1-dev on this seat pod's Trainium chip.
#
# Uses AWS's NxD Inference Flux code. That needs torch-neuronx 2.9 and transformers 4.57, while the
# pod's vLLM needs torch 2.11 and transformers 5, so the first run builds a separate virtualenv and
# never touches the Python that serve.sh uses.
#
# Flux takes all four NeuronCores by default, so stop the Qwen server first: /workspace/serve.sh --stop
#
# FLUX.1-dev is gated: accept its license at https://huggingface.co/black-forest-labs/FLUX.1-dev,
# then  export HF_TOKEN=hf_...  before running this.
#
#   ./run.sh --prompt "A robot named trn2"     first run installs, downloads + compiles; later runs reuse it
#   ./run.sh --prompt "..." --num 4            four images, written to out/
#
# Metrics: generate.py prints latency, ms/step and images/s and writes out/metrics.json. While it runs,
# neuron-monitor records NeuronCore utilisation and memory to out/neuron-monitor.jsonl, then
# neuron_summary.py prints a summary. Set NO_MONITOR=1 to skip the monitor.
#
# Extra arguments go to generate.py as-is (see --help).
set -euo pipefail
cd "$(dirname "$0")"

VENV=${VENV:-/workspace/.venv-flux}
NEURON_INDEX=https://pip.repos.neuron.amazonaws.com

if ! ls /dev/neuron* >/dev/null 2>&1; then
  echo "No Trainium device in this pod. Is this a seat pod?" >&2
  exit 1
fi
if pgrep -f "vllm serve" >/dev/null && [ "${TP:-4}" -gt 2 ]; then
  echo "The Qwen server holds two of the four NeuronCores. Stop it first: /workspace/serve.sh --stop" >&2
  exit 1
fi
if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then
  echo "FLUX.1-dev is gated. Accept its license on Hugging Face, then: export HF_TOKEN=hf_..." >&2
  exit 1
fi

if [ ! -x "$VENV/bin/neuronx-cc" ]; then
  echo "First run: building $VENV with NxD Inference (torch-neuronx 2.9). Several minutes."
  pip install -q uv
  # uv's own Python, not Ubuntu's: Ubuntu's ships without the libpython3.12.so that torch_xla needs.
  uv venv -q --seed --python 3.12 --python-preference only-managed "$VENV"
  # CPU torch first, so pip doesn't pull the multi-GB CUDA build that PyPI's torch depends on.
  "$VENV/bin/pip" install -q "torch==2.9.*" torchvision --index-url https://download.pytorch.org/whl/cpu
  "$VENV/bin/pip" install -q --extra-index-url "$NEURON_INDEX" "neuronx-distributed-inference[flux]"
  # neuronx-cc breaks on the newest islpy (NCC_ISMP902 "is_subset(): incompatible function arguments").
  # Use the islpy the image's own, working neuronx-cc has.
  "$VENV/bin/pip" install -q "islpy==$(/opt/conda/bin/python -c 'import importlib.metadata as m; print(m.version("islpy"))')"
fi

# torch_xla links against libpython3.12.so, which lives next to the venv's base Python, not on the loader path.
PYLIB=$("$VENV/bin/python" -c 'import sys; print(sys.base_prefix + "/lib")')
if [ ! -e "$PYLIB/libpython3.12.so.1.0" ]; then
  echo "No libpython3.12.so.1.0 in $PYLIB, which torch_xla needs. Delete $VENV and ask for help." >&2
  exit 1
fi

# The venv's neuronx-cc must win over the image's, since NxDI calls the compiler by name.
export PATH="$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$PYLIB:/opt/aws/neuron/lib:${LD_LIBRARY_PATH:-}"
export BASE_COMPILE_WORK_DIR=/workspace/flux-compiled/workdir/

mkdir -p out
if [ -z "${NO_MONITOR:-}" ] && command -v neuron-monitor >/dev/null; then
  neuron-monitor > out/neuron-monitor.jsonl 2>/dev/null &
  MON=$!
  trap 'kill $MON 2>/dev/null || true' EXIT
fi

"$VENV/bin/python" generate.py "$@"

if [ -n "${MON:-}" ]; then
  kill "$MON" 2>/dev/null || true
  wait "$MON" 2>/dev/null || true
  echo
  "$VENV/bin/python" neuron_summary.py out/neuron-monitor.jsonl || true
fi
