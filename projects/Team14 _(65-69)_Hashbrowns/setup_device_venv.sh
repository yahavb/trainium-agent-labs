#!/usr/bin/env bash
# Build a SEPARATE virtualenv that can run NKI kernels on a NeuronCore through torch-xla.
#
# Why: the seat image has no PyTorch->Neuron bridge. Measured on seat-65: torch is 2.11.0+cu130,
# torch_neuronx is not installed, and torch-xla falls back to the CPU ("Defaulting to
# PJRT_DEVICE=CPU", then "No module named 'libneuronxla'" once forced to NEURON).
#
# Why separate: torch-neuronx 2.9.0.2.15 pins torch==2.9, torch-xla==2.9 and libneuronxla 2.2.
# Installing that into the system Python would downgrade the torch vLLM is running on. This venv
# leaves the model server's environment untouched. It lives in /workspace, so a replaced pod
# loses it -- rerun this script.
#
#   ./setup_device_venv.sh                 # build it (several GB of wheels, minutes)
#   source /workspace/nkivenv/bin/activate # ACTIVATE it -- calling bin/python alone is not enough
#   python compare_torch.py --device
#
# The versions of nki and neuronx-cc are pinned to whatever this pod already has, so the venv
# compiles kernels with the same compiler verify_sdk.py checked against.
set -euo pipefail

VENV=${VENV:-/workspace/nkivenv}
REPO=https://pip.repos.neuron.amazonaws.com
PY=${PYTHON:-python3}

NKI_VER=$($PY -c "import nki; print(nki.__version__.split('+')[0])")
CC_VER=$($PY -c "import neuronxcc; print(neuronxcc.__version__.split('+')[0])")
# neuronx-cc asks for islpy~=2026.1, which lets pip pick 2026.2.2 -- and that compiler then dies
# with "[NCC_ISMP902] Simplifier error: is_subset(): incompatible function arguments" (seat-65).
# Match the islpy the pod's own, working compiler uses.
ISL_VER=$($PY -c "import islpy; print(islpy.__version__)" 2>/dev/null || echo "2026.1")
echo "pod has nki $NKI_VER, neuronx-cc $CC_VER, islpy $ISL_VER; matching them in $VENV"

if [ ! -x "$VENV/bin/python" ]; then
  $PY -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install --extra-index-url "$REPO" \
  "torch-neuronx==2.9.*" "neuronx-cc==$CC_VER" "nki==$NKI_VER" "islpy==$ISL_VER" numpy

# torch_neuronx shells out to `libneuronpjrt-path`, a console script installed in the venv's bin/.
# Calling the venv's python by path does not put bin/ on PATH, and the first run on seat-65 died
# with "No such file or directory: 'libneuronpjrt-path'". Activating the venv does.
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo
echo "=============== sanity check: does torch-xla see a NeuronCore? ==============="
# Core 0: vLLM holds the cores TP=2 put it on (README Part 3 measured NC 2-3), so stay off them.
NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-0} PJRT_DEVICE=NEURON "$VENV/bin/python" - <<'EOF'
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
dev = xm.xla_device()
kind = xr.device_type()
x = (torch.arange(4, dtype=torch.float32, device=dev) * 2)
xm.mark_step()
print("device type:", kind)
print("2 * [0,1,2,3] computed there:", x.cpu().tolist())
raise SystemExit(0 if str(kind).upper() == "NEURON" else "torch-xla is NOT on a NeuronCore")
EOF

echo
for tool in neuron-explorer neuron-profile; do
  if command -v "$tool" >/dev/null; then echo "$tool: $(command -v $tool)  -- profiling is possible"; \
  else echo "$tool: not found -- timings only, no per-engine profile"; fi
done
echo
echo "Next, in this shell or any new one:"
echo "  source $VENV/bin/activate"
echo "  python compare_torch.py --device"
