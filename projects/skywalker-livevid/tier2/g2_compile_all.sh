#!/usr/bin/env bash
# Runs inside the seat-234 pod from tier2/. Extracts all layer weights, then traces the 30 layers 3 at a time.
set -uo pipefail
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
L=/workspace/livevid/logs/tier2
python -u g2_compile.py extract || exit 1
python -u g2_compile.py compile 0 10 > $L/g2_compile_a.log 2>&1 &
python -u g2_compile.py compile 10 20 > $L/g2_compile_b.log 2>&1 &
python -u g2_compile.py compile 20 30 > $L/g2_compile_c.log 2>&1 &
wait
grep -h "COMPILE OK\|skip\|Error\|error" $L/g2_compile_[abc].log
