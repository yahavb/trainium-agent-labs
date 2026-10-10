#!/usr/bin/env bash
# Device half of the LM-head walkthrough: correctness and a device profile for each kernel version.
# Strictly sequential, one logical core. Logs land in examples/lm-head-walkthrough/results/,
# profiles in megakernel/results/profiles/ex_*.
#   NEURON_RT_VISIBLE_CORES=2 bash examples/lm-head-walkthrough/run_device.sh
set -u
EX="$(cd "$(dirname "$0")" && pwd)"
cd "$EX/../../megakernel"
export NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-2}
R="$EX/results"
step() { local name=$1; shift; echo "== $name: $* ($(date -u +%H:%M:%S))"; "$@" > "$R/$name.log" 2>&1; echo "   rc=$? ($(date -u +%H:%M:%S))"; }
step v0_nkilib_head_check   python kernels/test_head.py
step v2_own_head_check      python kernels/test_head.py --tiled
step v0_nkilib_head_profile python kernels/profile_mega.py --head --tag ex_head
step v1_hwdge_head_profile  python kernels/profile_mega.py --head --hwdge-oproj --tag ex_head_hwdge
step v2_own_head_profile    python kernels/profile_mega.py --head-tiled --tag ex_head_tiled
step bounds                 python kernels/analyze_profile.py ex_head ex_head_hwdge ex_head_tiled
echo DEVICE_DONE
