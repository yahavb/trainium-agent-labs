"""
Fit the latency model to measured kernels, evaluate it on held-out shapes, and search tilings.

    python fit.py                       # fit + leave-one-shape-out evaluation + plot
    python fit.py --search 1024 2048 1024    # rank every config for K M N, fitted on all data

Evaluation is leave-one-shape-out: for each shape, fit on the other shapes and predict all of this
shape's configs. That is the search use case -- the model has never seen the shape it ranks.
"""

import argparse
import csv
import glob
import itertools

import numpy as np

import model as mdl

try:
    from scipy.optimize import nnls as _nnls
except ImportError:
    _nnls = None


def nnls(A, y):
    """Non-negative least squares (scipy if present, else an active-set fallback)."""
    if _nnls is not None:
        return _nnls(A, y, maxiter=5000)[0]
    active = list(range(A.shape[1]))
    while True:
        x_act, *_ = np.linalg.lstsq(A[:, active], y, rcond=None)
        if (x_act >= 0).all() or len(active) == 1:
            x = np.zeros(A.shape[1])
            x[active] = np.clip(x_act, 0, None)
            return x
        active.pop(int(np.argmin(x_act)))


# --- data -------------------------------------------------------------------------------------

def load(pattern="results_*.csv"):
    rows = []
    for path in glob.glob(pattern):
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["status"] != "ok":
                    continue
                cfg = (r["variant"], int(r["tile_m"]), int(r["tile_k"]), int(r["tile_n"]))
                shp = (int(r["K"]), int(r["M"]), int(r["N"]))
                c = mdl.counts(*cfg, *shp, dtype=r["dtype"])
                rows.append({"cfg": cfg, "shape": shp, "y": float(r["p50_us"]), "c": c})
    return rows


# --- fitting ----------------------------------------------------------------------------------

def fit_sum(rows):
    """Stage 1: unit costs, no overlap. Rows weighted by 1/y so the fit minimizes relative error."""
    y = np.array([r["y"] for r in rows])
    A = np.array([[1.0] + [r["c"][f] for f in mdl.ALL_FEATS] for r in rows])
    scale = np.abs(A).max(axis=0)
    scale[scale == 0] = 1
    w = 1 / y
    x = nnls((A / scale) * w[:, None], y * w) / scale
    coef = dict(zip(["eps"] + mdl.ALL_FEATS, x))
    return coef


def fit_overlap(rows, coef):
    """Stage 2 (EnergAIzer's lambda correction): how much of DMA and TensorE time overlaps."""
    y = np.array([r["y"] for r in rows])
    feats = []
    for r in rows:
        t_dma, t_pe, t_epi = mdl.engine_times(r["c"], coef)
        feats.append([1.0, max(t_dma, t_pe), min(t_dma, t_pe), t_epi])
    A = np.array(feats)
    w = 1 / y
    x = nnls(A * w[:, None], y * w)
    return dict(zip(["eps", "max", "min", "epi"], x))


def fit(rows):
    coef = fit_sum(rows)
    return coef, fit_overlap(rows, coef)


# --- evaluation -------------------------------------------------------------------------------

def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def evaluate(rows):
    shapes = sorted({r["shape"] for r in rows})
    preds = {"roofline": {}, "ours": {}, "overlap": {}}
    for held in shapes:
        train = [r for r in rows if r["shape"] != held]
        coef, lam = fit(train)
        for i, r in enumerate(rows):
            if r["shape"] != held:
                continue
            preds["roofline"][i] = mdl.predict_roofline(r["c"])
            preds["ours"][i] = mdl.predict_sum(r["c"], coef)
            preds["overlap"][i] = mdl.predict_overlap(r["c"], coef, lam)

    print(f"\nLeave-one-shape-out evaluation: {len(rows)} kernels, {len(shapes)} shapes\n")
    print(f"{'model':10s} {'MAPE':>7s} {'rank corr':>10s} {'top-1':>7s} {'top-3':>7s} {'regret':>8s}")
    for name, p in preds.items():
        ape, rho, top1, top3, regret = [], [], 0, 0, []
        for shp in shapes:
            idx = [i for i, r in enumerate(rows) if r["shape"] == shp]
            meas = np.array([rows[i]["y"] for i in idx])
            pred = np.array([p[i] for i in idx])
            ape += list(np.abs(pred - meas) / meas)
            rho.append(spearman(pred, meas))
            best_true = int(np.argmin(meas))
            order = np.argsort(pred, kind="stable")
            top1 += int(order[0] == best_true)
            top3 += int(best_true in order[:3])
            regret.append(meas[order[0]] / meas[best_true] - 1)
        n = len(shapes)
        print(f"{name:10s} {100 * np.mean(ape):6.1f}% {np.nanmean(rho):10.2f} "
              f"{top1:>3d}/{n:<3d} {top3:>3d}/{n:<3d} {100 * np.mean(regret):7.1f}%")
    print("\n  MAPE: mean abs % error.  rank corr: Spearman within each shape (1 = perfect ranking).")
    print("  top-1/top-3: shapes where the truly fastest config is the model's #1 / in its top 3.")
    print("  regret: how much slower the model's pick is than the true best (0% = picked the best).")
    print("  roofline gives every tiling of a shape the same time, so it cannot rank them at all.")
    return preds


def show_fit(rows):
    coef, lam = fit(rows)
    print("\nFitted on all data -- the measured cost of each unit of work:")
    print(f"  launch overhead (eps)                   {coef['eps']:.1f} us")
    print(f"  per column through the PE array         {coef['mm_cols'] * 1e3:.3f} ns")
    print(f"  extra per column, half-filled array     {coef['cols_half'] * 1e3:.3f} ns  (tile_k 64 vs 128)")
    print(f"  extra per column per 16 chain steps     {coef['cols_chain'] * 1e3:.3f} ns  (chain = K / tile_k)")
    print(f"  per input DMA                           {coef['n_dma_in'] * 1e3:.1f} ns")
    import json
    with open("latency_coef.json", "w") as fh:
        json.dump({k: float(v) for k, v in coef.items()}, fh, indent=1)
    print("  (saved to latency_coef.json -- read by latency_hint.py for the agent's feedback)")
    print(f"  overlap: lam_max {lam['max']:.2f}  lam_min {lam['min']:.2f}"
          "   (lam_min 0 = DMA fully hidden under matmul, 1 = fully serial)")
    return coef, lam


def plot(rows, preds, path="pred_vs_meas.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\n(matplotlib not installed; skipping {path})")
        return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    meas = np.array([r["y"] for r in rows])
    for name, mark in (("roofline", "x"), ("ours", "o")):
        p = np.array([preds[name][i] for i in range(len(rows))])
        ax.scatter(meas, p, marker=mark, s=22, label=name, alpha=0.8)
    lo, hi = meas.min() * 0.5, meas.max() * 1.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(min(lo, 1), hi)
    ax.set_xlabel("measured p50 latency (us)"); ax.set_ylabel("predicted (us), held-out shapes")
    ax.set_title("NKI matmul on Trainium2"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print(f"\nwrote {path}")


# --- search -----------------------------------------------------------------------------------

def search(rows, K, M, N, top=8):
    coef, lam = fit(rows)
    cands = []
    for variant, tm, tk, tn in itertools.product(["naive", "hoist"], [128], [64, 128], [128, 256, 512]):
        if M % tm or K % tk or N % tn:
            continue
        c = mdl.counts(variant, tm, tk, tn, K, M, N)
        cands.append(((variant, tm, tk, tn), mdl.predict_sum(c, coef), c))
    cands.sort(key=lambda x: x[1])
    measured = {r["cfg"]: r["y"] for r in rows if r["shape"] == (K, M, N)}
    print(f"\nSearch K={K} M={M} N={N}: {len(cands)} legal configs ranked in-process\n")
    print(f"{'rank':>4s}  {'config':26s} {'pred us':>8s} {'meas us':>8s} {'bottleneck':>11s} {'HBM redund.':>11s}")
    for i, (cfg, t, c) in enumerate(cands[:top]):
        bn = mdl.bottleneck(c, coef)
        meas = f"{measured[cfg]:.0f}" if cfg in measured else "-"
        print(f"{i + 1:>4d}  {'_'.join(map(str, cfg)):26s} {t:8.1f} {meas:>8s} {bn[0]:>11s} {c['redundancy']:10.2f}x")
    if not measured:
        best = cands[0][0]
        print(f"\nverify the pick on the device:\n  NEURON_RT_VISIBLE_CORES=2 python bench.py --one "
              f"{best[0]} {best[1]} {best[2]} {best[3]} {K} {M} {N}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", nargs=3, type=int, metavar=("K", "M", "N"))
    a = ap.parse_args()
    rows = load()
    if not rows:
        raise SystemExit("no results_*.csv rows with status ok yet")
    if a.search:
        search(rows, *a.search)
    else:
        print(f"loaded {len(rows)} kernels")
        preds = evaluate(rows)
        show_fit(rows)
        plot(rows, preds)
