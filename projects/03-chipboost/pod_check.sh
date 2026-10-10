#!/usr/bin/env bash
# P2's one command in the seat pod: harness selftest, Qwen3 config, then every kernel in the simulator.
#   ./pod_check.sh                      dev + held-out shapes
#   ./pod_check.sh --which dev          faster
set -uo pipefail
cd "$(dirname "$0")"
python ../02-kernel-agent/nkibench.py --selftest | tail -3
python shapes.py --verify-config
python check_kernels.py --which dev,heldout "$@"
