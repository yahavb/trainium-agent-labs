#!/usr/bin/env bash
# Runs inside the seat-234 pod from tier2/. Compiles the 4 resident-cache stage graphs in parallel.
set -uo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=/workspace/livevid/logs/tier2
python -u g3_stage.py 0 8 > $L/g3_stage_0.log 2>&1 &
python -u g3_stage.py 8 16 > $L/g3_stage_1.log 2>&1 &
python -u g3_stage.py 16 23 > $L/g3_stage_2.log 2>&1 &
python -u g3_stage.py 23 30 > $L/g3_stage_3.log 2>&1 &
wait
grep -h "COMPILE OK\|skip\|Error" $L/g3_stage_[0-3].log
echo STAGES DONE
