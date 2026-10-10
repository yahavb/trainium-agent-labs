"""
Analytical latency model for tiled NKI matmul on Trainium2, after EnergAIzer (ISPASS'26).

EnergAIzer has to guess a GPU kernel's tiling and estimate its L2 hit rate. On Trainium neither is
needed: the tiling is in the kernel source and SBUF only holds what the kernel loaded, so every
DMA, byte and matmul instruction is counted exactly from the loop nest. What remains unknown is the
cost per unit of each, and how well the engines overlap -- those are fitted (fit.py).

    C[M, N] = lhsT[K, M].T @ rhs[K, N],  tiles tm x tk x tn

Variants (bench.py):
    naive  reloads the lhsT tile for every (m, n, k)
    hoist  loads each lhsT tile once per m and reuses it across n
"""

import math

BYTES = {"bf16": 2, "fp32": 4}

# Spec-sheet estimates per logical NeuronCore (LNC=2) -- used ONLY for the roofline baseline.
PEAK_FLOPS = 167e12   # dense bf16, approx
PEAK_BW = 725e9       # HBM bytes/s, approx (chip ~2.9 TB/s shared by 4 logical cores)

# The unit costs the fit learns, grouped by which engine pays them. Chosen by leave-one-shape-out
# search over feature combinations on 96 measured kernels (8 shapes, K up to 4096):
#   mm_cols     columns streamed through the 128x128 PE array            -- the main cost
#   cols_half   the same columns, weighted by how much of the array's 128 contraction rows sit idle
#               (0 at tile_k = 128, 1x at tile_k = 64): a half-filled array costs ~2x per column
#   cols_chain  the same columns, weighted by accumulation-chain length (K / tile_k matmuls adding
#               into one PSUM tile, /16): long chains cost more per column
#   n_dma_in    fixed cost per input DMA
# Byte counts and output-tile terms fitted to zero: the compiler hides or removes most load traffic.
DMA_FEATS = ["n_dma_in"]                      # HBM -> SBUF loads
PE_FEATS = ["mm_cols", "cols_half", "cols_chain"]  # Tensor Engine
EPI_FEATS = []                                # PSUM -> SBUF copy + store: no measurable cost
ALL_FEATS = DMA_FEATS + PE_FEATS + EPI_FEATS


def counts(variant, tm, tk, tn, K, M, N, dtype="bf16"):
    """Exact action counts for one kernel -- the Trainium version of EnergAIzer's Fig. 4 table."""
    b = BYTES[dtype]
    nM, nN, nK = math.ceil(M / tm), math.ceil(N / tn), math.ceil(K / tk)

    n_mm = nM * nN * nK
    if variant == "naive":
        a_dma = nM * nN * nK            # lhsT tile reloaded for every n
    elif variant == "hoist":
        a_dma = nM * nK                 # lhsT tile loaded once per m
    else:
        raise ValueError(variant)
    b_dma = nM * nN * nK                # rhs tile loaded for every (m, n, k) in both variants

    a_bytes = a_dma * tk * tm * b
    b_bytes = b_dma * tk * tn * b
    bytes_out = M * N * b
    bytes_min = (M * K + K * N + M * N) * b

    return {
        "n_mm": n_mm,
        "mm_cols": n_mm * tn,                       # moving columns streamed through the PE array
        "cols_half": n_mm * tn * (128 / tk - 1),    # same columns, scaled by idle contraction rows
        "cols_chain": n_mm * tn * nK / 16,          # same columns, scaled by accumulation-chain length
        "n_dma_in": a_dma + b_dma,
        "bytes_in": a_bytes + b_bytes,
        "n_out": nM * nN,                           # one PSUM copy + one store per output tile
        "bytes_out": bytes_out,
        "bytes_min": bytes_min,
        "redundancy": (a_bytes + b_bytes + bytes_out) / bytes_min,
        "flops": 2 * M * N * K,
    }


def engine_times(c, coef):
    """Per-engine busy time (us) from counts c and fitted unit costs coef."""
    t_dma = sum(coef[f] * c[f] for f in DMA_FEATS)
    t_pe = sum(coef[f] * c[f] for f in PE_FEATS)
    t_epi = sum(coef[f] * c[f] for f in EPI_FEATS)
    return t_dma, t_pe, t_epi


def predict_sum(c, coef):
    """No overlap: every engine's work adds up."""
    return coef["eps"] + sum(engine_times(c, coef))


def predict_overlap(c, coef, lam):
    """EnergAIzer-style composition: loads and matmuls overlap (max), imperfectly (lam_min * min),
    epilogue mostly exposed. lam_min = 0 is perfect overlap, 1 is fully serial."""
    t_dma, t_pe, t_epi = engine_times(c, coef)
    return (lam["eps"] + lam["max"] * max(t_dma, t_pe) + lam["min"] * min(t_dma, t_pe)
            + lam["epi"] * t_epi)


def bottleneck(c, coef):
    t_dma, t_pe, t_epi = engine_times(c, coef)
    tot = t_dma + t_pe + t_epi or 1.0
    name = max((("DMA", t_dma), ("TensorE", t_pe), ("epilogue", t_epi)), key=lambda x: x[1])[0]
    return name, t_dma / tot, t_pe / tot, t_epi / tot


def predict_roofline(c):
    """Baseline: max(compute floor, minimum-traffic floor). Identical for every tiling of a shape."""
    return max(c["flops"] / PEAK_FLOPS, c["bytes_min"] / PEAK_BW) * 1e6
