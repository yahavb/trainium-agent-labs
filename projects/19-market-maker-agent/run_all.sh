#!/bin/bash
# The final runs, on one seat. B (raw metrics) and C (located diagnosis) run side by side so they
# share the same server conditions: 2 samples each = 4 requests in flight, the seat's maximum.
# A (score only) runs after. Same budget everywhere: ROUNDS rounds x 2 samples per level.
#   REPS=2 ROUNDS=6 setsid nohup ./run_all.sh > run.log 2>&1 < /dev/null &
cd "$(dirname "$0")" && mkdir -p runs
REPS=${REPS:-2}; ROUNDS=${ROUNDS:-6}
python agent.py --feedback raw     --tag B --repeat "$REPS" --rounds "$ROUNDS" --samples 2 > runs/B.log 2>&1 &
python agent.py --feedback located --tag C --repeat "$REPS" --rounds "$ROUNDS" --samples 2 > runs/C.log 2>&1 &
wait
python agent.py --feedback none    --tag A --repeat "$REPS" --rounds "$ROUNDS" --samples 2 > runs/A.log 2>&1
python report.py
echo ALL DONE
