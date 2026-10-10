#!/usr/bin/env bash
# Runs inside a seat pod. Read-only probe of the image's native (system python) Neuron stack.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/reports"
REPORT="$ROOT/reports/native_probe.md"

run() {
  local title="$1"; shift
  local cmd="$*"
  local out rc
  out="$(timeout 90 bash -c "$cmd" 2>&1)"; rc=$?
  {
    echo "## $title"
    echo
    echo '```'
    echo "\$ $cmd"
    echo "$out"
    [[ $rc -ne 0 ]] && echo "[exit code: $rc]"
    echo '```'
    echo
  } >> "$REPORT"
}

: > "$REPORT"
echo -e "# Native stack probe\n\nGenerated $(date -u '+%Y-%m-%d %H:%M:%S UTC') on $(hostname)\n" >> "$REPORT"

run "libtorch-neuronx-lite contents" "pip show -f libtorch-neuronx-lite | head -60"
run "torch neuron attributes" "python3 -c \"import torch; print(torch.__version__, [d for d in dir(torch) if 'neuron' in d.lower()])\""
run "'neuron' device" "python3 -c \"import torch; t=torch.ones(2,2).to('neuron'); print(t.device, (t@t).cpu())\""
run "torch_xla device" "PJRT_DEVICE=NEURON python3 -c \"import torch_xla.core.xla_model as xm; print(xm.xla_device())\""
run "vllm_neuron location" "python3 -c \"import vllm_neuron, os; print(os.path.dirname(vllm_neuron.__file__))\""

pkg_dir() { python3 -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('$1').origin))" 2>/dev/null | tail -1; }
VN="$(pkg_dir vllm_neuron)"
LT="$(pkg_dir libtorch_neuronx_lite)"
if [[ -n "$LT" ]]; then
  run "libtorch_neuronx_lite file tree" "cd '$LT' && find . -name '*.py' -o -name '*.so' | grep -v __pycache__ | sort"
  run "libtorch_neuronx_lite public API" "python3 -c \"import libtorch_neuronx_lite as m; print([n for n in dir(m) if not n.startswith('_')])\" 2>&1 | tail -5"
  run "libtorch_neuronx_lite registers 'neuron' device?" "python3 -c \"import torch, libtorch_neuronx_lite; print(torch._C._get_privateuse1_backend_name()); t=torch.ones(2,2).to('neuron'); print(t.device, (t@t).cpu())\" 2>&1 | tail -8"
  run "libtorch_neuronx_lite: compile backend registration" "cd '$LT' && grep -rnE 'register_backend|rename_privateuse1|privateuse1|def trace|torch\.compile|backend' --include='*.py' . | head -30"
fi
if [[ -n "$VN" ]]; then
  run "vllm_neuron file tree" "cd '$VN' && find . -name '*.py' | sort | head -80"
  run "vllm_neuron imports of Neuron libraries" "cd '$VN' && grep -rhoE '^\s*(from|import) [A-Za-z_.]+' --include='*.py' . | sed -E 's/^\s+//' | sort | uniq -c | sort -rn | grep -iE 'neuron|xla|nxd|nki|torch' | head -40"
  run "vllm_neuron: torch.compile / backend" "cd '$VN' && grep -rnE 'torch\.compile|backend=' --include='*.py' . | head -30"
  run "vllm_neuron: trace / NEFF / nrt" "cd '$VN' && grep -rniE 'torch_neuronx|\.trace\(|neff|nrt|parallel_model_trace|ModelBuilder' --include='*.py' . | head -40"
  run "vllm_neuron: device strings" "cd '$VN' && grep -rnE \"device\(['\\\"](neuron|xla)|'neuron'|\\\"neuron\\\"\" --include='*.py' . | head -30"
fi
run "Neuron python packages installed" "pip list 2>/dev/null | grep -iE 'neuron|nxd|xla|nki'"
run "torch backend / privateuse1 name" "python3 -c \"import torch; print(torch._C._get_privateuse1_backend_name())\""

echo "Wrote $REPORT"
