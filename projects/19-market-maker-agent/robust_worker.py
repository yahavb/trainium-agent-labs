"""Child process for the checker's optional robustness layer (MM_ROBUST=1, run C4): the same level
and seeds with the market spread forced to 1 tick, the real market's spread on 99% of the replayed
day. Kept out of mmsim.py so the simulator the frozen runs used stays byte-identical.

    python robust_worker.py CODEFILE LEVEL SEEDS
"""
import json
import sys
import traceback

import mmsim

code = open(sys.argv[1]).read()
level = int(sys.argv[2])
seeds = [int(x) for x in sys.argv[3].split(",")]
mmsim.LEVELS[level] = dict(mmsim.LEVELS[level], spread=1)
try:
    out = mmsim.evaluate(code, level, seeds)
except Exception as e:  # noqa: BLE001
    out = dict(error=dict(step=None, type=type(e).__name__, message=str(e)[:300], line=None,
                          seed=None), episodes=[], violations=[])
sys.stdout.write(json.dumps(out))
