"""Run the baseline and the reference on every level, dev and held-out seeds. Used to set the level
parameters so the baseline fails levels 2-4 and the reference passes all four, with margin."""
import sys
import time

import numpy as np

import mmsim

def summary(name, code, level, seeds):
    t0 = time.perf_counter()
    r = mmsim.evaluate(code, level, seeds)
    if r["error"]:
        print(f"  {name:10s} ERROR {r['error']}")
        return
    e = r["episodes"]
    p = np.array([m["pnl"] for m in e])
    se = p.std(ddof=1) / np.sqrt(len(p))
    g = lambda k: np.mean([m[k] for m in e])
    print(f"  {name:10s} {'PASS' if p.mean() - 2 * se > 0 else 'fail'} pnl {p.mean():8.1f} ± {se:6.1f}  lcb {p.mean() - 2 * se:8.1f}  win {np.mean(p > 0):.2f}"
          f"  fills {g('fills'):6.0f}  edge {g('edge_pnl'):7.1f}  hold {g('hold_pnl'):8.1f}  swept {g('swept'):5.1f}"
          f"  oblig {g('obligation'):.2f}  @lim {g('at_limit'):.2f}  viol {len(r['violations'])}"
          f"  ({time.perf_counter() - t0:.1f}s)")

if __name__ == "__main__":
    levels = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 4]
    names = {"baseline": open("baseline.py").read(), "reference": open("reference.py").read()}
    for lv in levels:
        for split, seeds in (("dev", mmsim.DEV_SEEDS), ("held-out", mmsim.heldout_seeds())):
            print(f"level {lv} {mmsim.LEVELS[lv]['name']} [{split}]")
            for n, c in names.items():
                summary(n, c, lv, seeds)
