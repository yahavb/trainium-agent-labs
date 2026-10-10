"""
Speed feedback for the kernel agent, from the latency model in model.py / fit.py.

nkibench.simulate_and_count records the tile shape of every nisa.nc_matmul the kernel issues.
This turns those tiles into a predicted on-device latency -- projected onto a realistic matmul
(2048 x 2048 x 2048), because at the level-4 test shapes the fixed launch cost hides every tiling
difference -- and says which cost dominates and what to change.

The unit costs come from latency_coef.json (written by fit.py from 96 kernels timed on Trainium2).
They were fitted on nl.load / nl.matmul kernels; nisa.dma_copy / nisa.nc_matmul kernels are
assumed to cost the same per unit, which has not been measured.
"""

import collections
import json
import pathlib

import model as mdl

COEF_PATH = pathlib.Path(__file__).with_name("latency_coef.json")
PROJECT = (2048, 2048, 2048)          # (K, M, N) the prediction is reported at
BEST_GRID = [(128, tk, tn) for tk in (64, 128) for tn in (128, 256, 512)]


def _coef():
    try:
        return json.loads(COEF_PATH.read_text())
    except (OSError, ValueError):
        return None


def predict(tm, tk, tn, shape=PROJECT, coef=None):
    coef = coef or _coef()
    c = mdl.counts("naive", tm, tk, tn, *shape)
    parts = {
        "launch": coef["eps"],
        "columns through the PE array": coef["mm_cols"] * c["mm_cols"],
        "half-empty PE array": coef["cols_half"] * c["cols_half"],
        "long accumulation chains": coef["cols_chain"] * c["cols_chain"],
        "DMA loads": coef["n_dma_in"] * c["n_dma_in"],
    }
    return sum(parts.values()), parts


def hint(counter):
    """One feedback sentence for a correct matmul kernel, or '' if nothing to say."""
    coef = _coef()
    tiles = counter.get("matmul_tiles") or []
    if not coef or not tiles:
        return ""
    (tk, tm, tn), _ = collections.Counter(tiles).most_common(1)[0]
    t, parts = predict(tm, tk, tn, coef=coef)
    best_cfg = min(BEST_GRID, key=lambda cfg: predict(*cfg, coef=coef)[0])
    t_best, _ = predict(*best_cfg, coef=coef)
    worst = max((k for k in parts if k != "launch"), key=parts.get)

    advice = []
    if tk < 128:
        advice.append(f"your contraction tile is {tk} rows, so {128 - tk} of the PE array's 128 rows "
                      f"sit idle and every column costs about twice as much: use a K tile of 128")
    if tn < 512:
        advice.append(f"your moving tile is {tn} wide: 512 means fewer, wider matmul calls")
    K, M, N = PROJECT
    msg = (f"SPEED (predicted by a latency model fitted to 96 kernels timed on Trainium2, about "
           f"+/-12%): with your tiles (K {tk}, M {tm}, N {tn}) a {K}x{M}x{N} matmul would take "
           f"about {t:.0f} us; the largest cost is {worst} ({parts[worst]:.0f} us).")
    if t > 1.05 * t_best:
        msg += (f" The best tiling the model knows (K {best_cfg[1]}, N {best_cfg[2]}) takes about "
                f"{t_best:.0f} us, {t / t_best:.1f}x faster.")
        if advice:
            msg += " To get there: " + "; ".join(advice) + "."
    else:
        msg += " That is already the fastest tiling the model knows."
    return msg
