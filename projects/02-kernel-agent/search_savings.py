"""
Does the simulator replace brute-force tile search on the device?

For each NEW shape (results_new_*.csv, never used in fitting):
  brute force  = benchmark all 12 tilings on Trainium2 (measured compile+run time per kernel)
  simulator    = rank all 12 on the CPU with the model fitted ONLY on the original 96 kernels,
                 then benchmark just its #1 pick (or its top 3)
and compare search cost and how close the chosen tiling is to the true fastest.

    python search_savings.py
"""

import glob
import time

import numpy as np

import fit
import model as mdl

train = [r for p in ("results_[0-9].csv", "results_long_*.csv") for r in fit.load(p)]
test = fit.load("results_new_*.csv")
if not test:
    raise SystemExit("no results_new_*.csv rows yet")

# compile+run seconds per measured kernel, read straight from the CSVs
import csv
cost_s = {}
for path in glob.glob("results_new_*.csv"):
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["status"] == "ok":
                key = ((r["variant"], int(r["tile_m"]), int(r["tile_k"]), int(r["tile_n"])),
                       (int(r["K"]), int(r["M"]), int(r["N"])))
                cost_s[key] = float(r["compile_run_s"])

coef, _ = fit.fit(train)
shapes = sorted({r["shape"] for r in test})

print(f"Model fitted on {len(train)} original kernels; testing on {len(test)} kernels "
      f"over {len(shapes)} NEW shapes.\n")
print(f"{'shape K,M,N':18s} {'n':>3s} {'brute s':>8s} {'sim ms':>7s} {'top1 s':>7s} "
      f"{'best us':>8s} {'pick us':>8s} {'regret':>7s} {'top3 hit':>8s} {'MAPE':>6s}")

tot_brute = tot_top1 = tot_top3 = tot_sim = 0.0
regrets, regrets3, hits1, hits3, apes = [], [], 0, 0, []
for shp in shapes:
    rows = [r for r in test if r["shape"] == shp]
    meas = np.array([r["y"] for r in rows])

    t0 = time.perf_counter()
    pred = np.array([mdl.predict_sum(r["c"], coef) for r in rows])
    order = np.argsort(pred, kind="stable")
    sim_s = time.perf_counter() - t0

    brute = sum(cost_s[(r["cfg"], shp)] for r in rows)
    top1 = cost_s[(rows[order[0]]["cfg"], shp)]
    top3 = sum(cost_s[(rows[i]["cfg"], shp)] for i in order[:3])
    best = int(np.argmin(meas))
    regret = meas[order[0]] / meas[best] - 1
    regret3 = meas[order[:3]].min() / meas[best] - 1        # benchmark top 3, keep the fastest

    tot_brute += brute; tot_top1 += top1; tot_top3 += top3; tot_sim += sim_s
    regrets.append(regret); regrets3.append(regret3)
    hits1 += int(order[0] == best); hits3 += int(best in order[:3])
    apes += list(np.abs(pred - meas) / meas)
    print(f"{str(shp):18s} {len(rows):3d} {brute:8.0f} {1e3 * sim_s:7.2f} {top1:7.0f} "
          f"{meas[best]:8.0f} {meas[order[0]]:8.0f} {100 * regret:6.1f}% "
          f"{'yes' if best in order[:3] else 'no':>8s} {100 * np.mean(np.abs(pred - meas) / meas):5.1f}%")

n = len(shapes)
print(f"\n{'':18s} SEARCH COST (serial compile+run on the device)        QUALITY OF THE CHOSEN TILING")
print(f"  brute force, all tilings     {tot_brute:7.0f} s  ({tot_brute / 60:.1f} min)       "
      f"exact best by definition")
print(f"  simulator + benchmark top 1  {tot_sim + tot_top1:7.0f} s  ({(tot_sim + tot_top1) / 60:.1f} min)  "
      f"{tot_brute / (tot_sim + tot_top1):4.1f}x cheaper   best found {hits1}/{n}, "
      f"avg {100 * np.mean(regrets):.1f}% slower than best, worst {100 * max(regrets):.0f}%")
print(f"  simulator + benchmark top 3  {tot_sim + tot_top3:7.0f} s  ({(tot_sim + tot_top3) / 60:.1f} min)  "
      f"{tot_brute / (tot_sim + tot_top3):4.1f}x cheaper   best found {hits3}/{n}, "
      f"avg {100 * np.mean(regrets3):.1f}% slower than best, worst {100 * max(regrets3):.0f}%")
print(f"  simulator ranking alone      {1e3 * tot_sim:7.1f} ms on the CPU for all {len(test)} kernels")
print(f"\n  prediction error on the new shapes: {100 * np.mean(apes):.1f}% mean, {100 * max(apes):.0f}% worst")
