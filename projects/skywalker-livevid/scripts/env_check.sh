#!/usr/bin/env bash
# Runs inside the seat pod. Collects Neuron container info into reports/env_report.md.
# Every check runs independently; failures are recorded and the script continues.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT"/{reports,logs,repos}
REPORT="$ROOT/reports/env_report.md"
HF_CACHE="${HF_HOME:-/root/.cache/huggingface}"

run() {
  local title="$1"; shift
  local cmd="$*"
  local out rc
  out="$(timeout 120 bash -c "$cmd" 2>&1)"; rc=$?
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
{
  echo "# Environment report"
  echo
  echo "Generated $(date -u '+%Y-%m-%d %H:%M:%S UTC') on $(hostname)"
  echo
} >> "$REPORT"

run "neuron-ls" "neuron-ls"
run "neuron-ls --json-output" "neuron-ls --json-output"
run "Neuron env vars" "env | grep -i neuron"
run "Python and pip location" "which python3 pip; python3 --version"
run "pip list (filtered)" "pip list 2>/dev/null | grep -iE 'neuron|torch|transformers|diffusers|optimum|nki|accelerate|xla|vllm|huggingface|opencv|av'"
run "torch / torch_neuronx versions" "python3 -c \"import torch, torch_neuronx; print(torch.__version__, torch_neuronx.__version__)\""
run "neuronx-cc --version" "neuronx-cc --version"
run "import nki" "python3 -c \"import nki; print('nki ok')\""
run "import neuronxcc.nki" "python3 -c \"import neuronxcc.nki; print('neuronxcc.nki ok')\""
run "import optimum.neuron" "python3 -c \"import optimum.neuron; print('optimum.neuron ok')\""
run "diffusers version" "python3 -c \"import diffusers; print(diffusers.__version__)\""
run "Neuron tools" "which neuron-top neuron-monitor neuron-profile neuron-explorer"
run "nproc" "nproc"
run "free -h" "free -h"
run "Disk usage" "df -h /workspace $HF_CACHE /dev/shm /tmp"
run "HF cache contents (shared across seats on this node)" "ls -la $HF_CACHE/hub"
run "vLLM processes (should be empty)" "pgrep -af '[v]llm'"
run "All processes" "ps -eo pid,ppid,etime,rss,cmd --sort=-rss | head -25"
run "Outbound: huggingface.co" "curl -sI --max-time 15 https://huggingface.co | head -1"
run "Outbound: pypi.org" "curl -sI --max-time 15 https://pypi.org | head -1"
run "uname -a" "uname -a"
run "os-release" "head -3 /etc/os-release"

echo "Wrote $REPORT"
