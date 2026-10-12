#!/usr/bin/env bash
# Everything a fresh seat pod needs for this project, then the checker's self-test.
#
#   cd /workspace/projects/19-verified-rtl-agent && ./setup.sh
#
# Idempotent: re-running skips what is already there. Lost when the pod is replaced (the tools live
# outside /workspace), so run it again on a new pod. DESIGN.md C9-C11.
set -euo pipefail
cd "$(dirname "$0")"

VE_DIR=${VE_DIR:-/root/verilog-eval}
VE_COMMIT=c498220

if ! command -v iverilog >/dev/null || ! command -v yosys >/dev/null; then
  echo "Installing iverilog and yosys (apt)..."
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq iverilog yosys >/dev/null
fi
echo "iverilog: $(iverilog -V 2>&1 | head -1)"
echo "yosys:    $(yosys -V 2>&1 | head -1)"
# `iverilog -V` exits non-zero (no input files), so read the line first: under pipefail the grep
# pipeline itself would fail and print a false warning.
iv_version=$(iverilog -V 2>&1 | head -1 || true)
case "$iv_version" in
  *"version 12."*) ;;
  *) echo "WARNING: VerilogEval's testbenches are written for Icarus 12; this is: $iv_version" >&2 ;;
esac

if [ ! -d "$VE_DIR/.git" ]; then
  echo "Cloning VerilogEval into $VE_DIR..."
  git clone -q https://github.com/NVlabs/verilog-eval.git "$VE_DIR"
fi
git -C "$VE_DIR" -c advice.detachedHead=false checkout -q "$VE_COMMIT" 2>/dev/null \
  || { git -C "$VE_DIR" fetch -q origin && git -C "$VE_DIR" -c advice.detachedHead=false checkout -q "$VE_COMMIT"; }
n=$(ls "$VE_DIR/dataset_spec-to-rtl"/*_prompt.txt | wc -l)
echo "VerilogEval: $VE_DIR @ $(git -C "$VE_DIR" rev-parse --short HEAD), $n problems"
[ "$n" -eq 156 ] || { echo "expected 156 problems" >&2; exit 1; }

python3 -c "import httpx" 2>/dev/null || pip install -q httpx

export VE_ROOT="$VE_DIR/dataset_spec-to-rtl"
python3 checker.py --selftest
python3 translator.py --selftest
python3 workers.py --selftest
python3 agent.py --selftest
