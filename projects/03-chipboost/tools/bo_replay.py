#!/usr/bin/env python3
"""
bo_replay.py -- would Bayesian optimisation have beaten random search on arm c's search space?

No chip: the exhaustive sweep (logs/seat-102/sweep-*.jsonl) measured every one of the 62 SBUF-fitting block
settings, so both strategies are replayed against that measured table. The objective is the dashboard's:
speedup over the expert as shipped (m2 n2 k8 at the primary shape), from the referee's A/B speedups. Both
start from attempt 0 = the expert and spend the same budget as search.py's runs (24 evaluations).

- random: the rest of the budget in a uniformly random order, as search.py does.
- bayes: a Gaussian process on log2 of the effective tile counts (RBF kernel, length scale refit by
  marginal likelihood each step), 2 random points, then expected improvement.

The replay treats each setting's one sweep measurement as exact. The real timer has about 0.4% noise between
sessions, which slightly flatters the model-based search.

    python tools/bo_replay.py [--seeds 500] [--budget 24]
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import search  # noqa: E402
import shapes  # noqa: E402


def table():
    tiles = search.tile_counts(shapes.cases("matmul", "timing")[0])
    sp = {}
    for path in glob.glob(os.path.join(ROOT, "logs", "seat-102", "sweep-*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if r.get("speedup"):
                sp[search.effective(search.read_caps(r["code"]), tiles)] = r["speedup"]
    caps = sorted(sp)
    expert = search.effective((16, 2, 8), tiles)
    y = np.array([math.log(sp[c] / sp[expert]) for c in caps])      # log speedup over the expert
    X = np.log2(np.array(caps, float))
    X = X / np.maximum(X.max(0), 1)                                  # each dimension in [0, 1]
    return caps, X, y, caps.index(expert)


def ei_scores(X, y, seen, cand):
    yt = y[seen]
    mu, sd = yt.mean(), (yt.std() or 1.0)
    yt = (yt - mu) / sd
    best = None
    for ls in (0.15, 0.3, 0.6, 1.2):                                 # length scale by marginal likelihood
        d = ((X[seen][:, None] - X[seen][None]) ** 2).sum(-1)
        K = np.exp(-0.5 * d / ls ** 2) + 1e-4 * np.eye(len(seen))
        L = np.linalg.cholesky(K)
        a = np.linalg.solve(L.T, np.linalg.solve(L, yt))
        lml = -0.5 * yt @ a - np.log(np.diag(L)).sum()
        if best is None or lml > best[0]:
            best = (lml, ls, L, a)
    _, ls, L, a = best
    ks = np.exp(-0.5 * ((X[cand][:, None] - X[seen][None]) ** 2).sum(-1) / ls ** 2)
    m = ks @ a
    v = np.linalg.solve(L, ks.T)
    s = np.sqrt(np.clip(1 - (v ** 2).sum(0), 1e-12, None))
    z = (m - yt.max()) / s
    cdf = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    return (m - yt.max()) * cdf + s * np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)


def run(X, y, start, budget, rng, bayes):
    n = len(y)
    seen = [start]
    rest = [i for i in rng.permutation(n) if i != start]
    while len(seen) < budget:
        cand = [i for i in range(n) if i not in seen]
        if not bayes or len(seen) < 3:
            pick = next(i for i in rest if i not in seen)
        else:
            pick = cand[int(np.argmax(ei_scores(X, y, seen, cand)))]
        seen.append(pick)
    return np.maximum.accumulate(y[seen])                            # best so far after each evaluation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--budget", type=int, default=24)
    a = ap.parse_args()
    caps, X, y, start = table()
    top = y.max()
    print(f"{len(caps)} settings measured; the best is {math.exp(top):.3f}x over the expert "
          f"({search.caps_str(caps[int(np.argmax(y))])})\n")
    print(f"{'strategy':<8} {'best at 8':>10} {'best at 12':>11} {'best at 24':>11} {'finds #1':>9} "
          f"{'evals to #1':>12} {'within 2% by':>13}")
    for name, bayes in (("random", False), ("bayes", True)):
        curves = np.array([run(X, y, start, a.budget, np.random.default_rng(s), bayes) for s in range(a.seeds)])
        hit = curves >= top - 1e-12
        first = np.where(hit.any(1), hit.argmax(1) + 1, np.nan)
        near = curves >= top - math.log(1.02)
        first_near = np.where(near.any(1), near.argmax(1) + 1, np.nan)
        med = lambda col: math.exp(np.median(curves[:, col - 1]))
        print(f"{name:<8} {med(8):>9.3f}x {med(12):>10.3f}x {med(24):>10.3f}x {hit[:, -1].mean():>8.0%} "
              f"{np.nanmedian(first):>12.0f} {np.nanmedian(first_near):>13.0f}")
    print("\n'best at n': median over seeds of the best speedup over the expert after n evaluations "
          "(evaluation 1 is the expert itself).")


if __name__ == "__main__":
    main()
