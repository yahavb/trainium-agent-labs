"""The operation ladder: reference, error scale, and eval cases per level.

Inputs are float32 (what an accelerator would hold); references run in float64 on those exact
float32 values, so the only error measured is the kernel's own.

Error bound: each level defines `scale`, the magnitude a backward-stable float32 algorithm's error
is proportional to (sum|x| for a row sum, not |sum x|, which can cancel to ~0), and `bound_k`, the
number of roundings on the path to one output. A kernel passes when |got - ref| <= gamma_k * scale
everywhere; see TOLERANCE in harness.py.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .rules import TILE_C, TILE_R

TINY = 1e-30


@dataclass
class Case:
    name: str
    args: tuple                 # positional kernel args: float32 arrays and python scalars
    tags: tuple = ()


@dataclass
class Level:
    num: int
    name: str
    signature: str              # "kernel(x, a, b)"
    task: str                   # one-sentence statement, the only thing attempt 1 needs
    ref: Callable               # float64 args -> float64 output
    scale: Callable             # (float64 args, ref) -> array broadcastable to ref
    bound_k: Callable           # float64 args -> k in the float32 bound gamma_k * scale
    min_flops: Callable         # float64 args -> FLOPs of the direct algorithm, for efficiency
    make_cases: Callable[[bool], list]   # holdout -> cases

    out_tile: tuple = (TILE_R, TILE_C)   # tile size per output axis, for locating errors
    out_axes: tuple = ("row", "col")     # what each output axis means, for error messages
    # (args64, got, ref, bad) -> facts: level-specific hypotheses about a wrong output, e.g. "equals
    # the reference over the last column tile only". Facts are phrased to be shared across cases.
    explain: Callable | None = None
    abs_floor: float = 0.0               # absolute slack added to the bound (e.g. underflow to 0)
    notes: str = ""
    _cache: dict = field(default_factory=dict, repr=False)

    def get_cases(self, holdout=False):
        """Dev cases drive the repair loop. Holdout cases (other shapes, other seed) are never
        shown to the model; they only decide how much to trust a kernel that passed dev."""
        if holdout not in self._cache:
            self._cache[holdout] = self.make_cases(holdout)
        return self._cache[holdout]


def _is_prime(n):
    return n > 2 and all(n % p for p in range(2, int(n ** 0.5) + 1))


def shape_tags(M, N):
    t = []
    if M % TILE_R:
        t.append("partial-row-tile")
    if N % TILE_C:
        t.append("partial-col-tile")
    if M > TILE_R:
        t.append("multi-row-tile")
    if N > TILE_C:
        t.append("multi-col-tile")
    if M == 1 or N == 1:
        t.append("dim-1")
    if _is_prime(M) or _is_prime(N):
        t.append("prime-dim")
    return t


# (M, N): exact tile, tile+1, primes spanning several tiles, dims of exactly 1
SHAPES = [(1, 1), (1, 700), (7, 13), (128, 512), (129, 513), (131, 1031), (300, 1), (37, 2053)]
HOSTILE_SHAPES = [(131, 1031), (129, 513), (1, 700), (300, 1)]


def values(kind, M, N, rng):
    """Value generators. Each name is a tag; failures that concentrate on one are diagnostic."""
    if kind == "normal":
        x = rng.standard_normal((M, N))
    elif kind == "large-magnitude":
        x = rng.standard_normal((M, N)) * 1e4
    elif kind == "all-negative":
        x = -1.0 - np.abs(rng.standard_normal((M, N))) * 5
    elif kind == "huge-negative":
        x = -1e30 * (1.0 + rng.random((M, N)))
    elif kind == "zeros":
        x = np.zeros((M, N))
    elif kind == "constant-rows":
        x = np.repeat(rng.standard_normal((M, 1)) * 3, N, axis=1)
    elif kind == "large-mean":
        x = 1e4 + rng.standard_normal((M, N))
    elif kind == "huge-magnitude":
        x = rng.standard_normal((M, N)) * 1e18
    elif kind == "index-coded":
        x = np.arange(1, M * N + 1, dtype=np.float64).reshape(M, N)   # value encodes its position
    elif kind == "cancelling":
        x = np.where(rng.random((M, N)) < 0.5, 1e6, -1e6) + rng.standard_normal((M, N))
    else:
        raise ValueError(kind)
    return x.astype(np.float32)


def _holdout_shapes(rng, n=6):
    """Random shapes, always including a dim of 1 and a multi-tile prime."""
    shapes = [(int(rng.integers(1, 400)), int(rng.integers(1, 2200))) for _ in range(n - 2)]
    return shapes + [(int(rng.integers(130, 260)), 1), (1, 1543)]


def _cases_2d(value_kinds, extra=lambda x, rng, holdout: (), seed=0, holdout=False):
    """Every shape with normal values, plus each hostile value kind on the awkward shapes."""
    rng = np.random.default_rng(seed + (1000 if holdout else 0))
    cases = []
    if holdout:
        shapes = _holdout_shapes(rng)
        hostile = [shapes[0], shapes[-1], shapes[-2]]
    else:
        shapes, hostile = SHAPES, HOSTILE_SHAPES
    plan = [("normal", s) for s in shapes] + [(k, s) for k in value_kinds for s in hostile]
    for kind, (M, N) in plan:
        x = values(kind, M, N, rng)
        for suffix, rest, tags in extra(x, rng, holdout) or [("", (), ())]:
            cases.append(Case(f"{kind} {M}x{N}{suffix}", (x, *rest),
                              tuple(shape_tags(M, N)) + (kind,) + tuple(tags)))
    return cases


# ---- tile-level hypotheses (explain hooks) -------------------------------------------------------

def _col_tiles(N):
    return [(j0, min(j0 + TILE_C, N)) for j0 in range(0, N, TILE_C)]


def reduction_tile_facts(ref, args, g, bad):
    """Row reduction over several column tiles: does each wrong row hold ONE tile's result?
    (The 'accumulator overwritten per tile' bug, E9.)"""
    x = args[0]
    tiles = _col_tiles(x.shape[1])
    if len(tiles) < 2:
        return []
    rows = np.nonzero(bad)[0]
    for t, (j0, j1) in enumerate(tiles):
        sub = np.asarray(ref(x[:, j0:j1], *args[1:]))
        if np.allclose(g[rows], sub[rows], rtol=1e-5, atol=0):
            which = "last" if t == len(tiles) - 1 else "first" if t == 0 else f"number {t}"
            return [f"each wrong output equals the result over the {which} column tile only"]
    return []


def rowwise_tile_facts(ref_slice, args, g, bad):
    """Row-wise normalisation over several column tiles: was each tile normalised with its own
    statistic (max, sum, mean square) instead of the whole row's?"""
    x = args[0]
    tiles = _col_tiles(x.shape[1])
    if len(tiles) < 2:
        return []
    per_tile = np.concatenate([np.asarray(ref_slice(args, j0, j1)) for j0, j1 in tiles], axis=1)
    if np.allclose(g[bad], per_tile[bad], rtol=1e-4, atol=1e-30):
        return ["each column tile was normalised with a statistic of that tile alone, not of the whole row"]
    return []


def transpose_facts(args, g, bad):
    """Inputs are index-coded (x[i, j] = i*N + j + 1), so a wrong output says where it came from."""
    x = args[0]
    M, N = x.shape
    rr, cc = np.nonzero(bad)
    v = g[rr, cc]
    if not len(v) or not np.all(v == np.round(v)):
        return []
    if np.any(v == 0):
        n0 = int(np.sum(v == 0))
        facts0 = [f"~{n0} of the wrong outputs are 0, i.e. never written"]
        keep = v != 0
        rr, cc, v = rr[keep], cc[keep], v[keep]
        if not len(v):
            return facts0
    else:
        facts0 = []
    if not np.all((v >= 1) & (v <= M * N)):
        return facts0
    src_i, src_j = (v.astype(np.int64) - 1) // N, (v.astype(np.int64) - 1) % N
    facts = []
    k = int(np.argmax(bad[rr, cc])) if len(rr) else 0
    r0, c0 = int(rr[k]), int(cc[k])
    facts.append(f"~e.g. out[{r0}, {c0}] holds x[{src_i[k]}, {src_j[k]}], it should hold x[{c0}, {r0}]")
    if np.all((src_i == rr) & (src_j == cc)):
        facts.append("each wrong out[r, c] holds x[r, c], i.e. it was copied without transposing")
    di, dj = np.unique(src_i - cc), np.unique(src_j - rr)
    if len(di) == 1 and len(dj) == 1 and (di[0], dj[0]) != (0, 0):
        facts.append(f"each wrong out[r, c] holds x[c + {di[0]}, r + {dj[0]}] instead of x[c, r] "
                     f"(an offset in the tile origin)")
    return facts0 + facts


def matmul_facts(args, g, bad):
    """Is the product missing the last partial K tile (or using only the first)?"""
    a, b = args
    K = a.shape[1]
    for step in (128, 512):
        cut = (K // step) * step
        if 0 < cut < K and np.allclose(g[bad], (a[:, :cut] @ b[:cut])[bad], rtol=1e-4, atol=1e-30):
            return [f"each wrong output is the product over K rounded down to a multiple of {step}: "
                    f"the last partial K tile is missing"]
        if step < K and np.allclose(g[bad], (a[:, :step] @ b[:step])[bad], rtol=1e-4, atol=1e-30):
            return [f"each wrong output is the product over the first {step} values of K only"]
    return []


# ---- level 1: y = relu(a*x + b) ---------------------------------------------------------------

def _l1_extra(x, rng, holdout):
    if holdout:
        a, b = float(rng.uniform(0.1, 4)), float(rng.uniform(-2, 2))
        return [(f" a={a:.3g} b={b:.3g}", (a, b), ()),
                (f" a={-a:.3g} b={-b:.3g}", (-a, -b), ("negative-scale",))]
    return [(" a=1.7 b=-0.25", (1.7, -0.25), ()),
            (" a=-3 b=2", (-3.0, 2.0), ("negative-scale",))]


L1 = Level(
    1, "relu_affine", "kernel(x, a, b)",
    "Given a 2-D float32 array x and python floats a, b, return relu(a*x + b) elementwise, "
    "same shape as x.",
    ref=lambda x, a, b: np.maximum(a * x + b, 0.0),
    scale=lambda args, r: np.abs(args[1]) * np.abs(args[0]) + abs(args[2]),
    bound_k=lambda args: 3,
    min_flops=lambda args: 3 * args[0].size,     # mul, add, max
    make_cases=lambda h: _cases_2d(["large-magnitude", "zeros", "all-negative"], _l1_extra,
                                   seed=1, holdout=h),
)

# ---- level 2: row-wise sum ----------------------------------------------------------------------

L2 = Level(
    2, "row_sum", "kernel(x)",
    "Given a 2-D float32 array x of shape (M, N), return the 1-D array of length M whose i-th "
    "entry is the sum of row i.",
    ref=lambda x: x.sum(axis=1),
    scale=lambda args, r: np.abs(args[0]).sum(axis=1),
    bound_k=lambda args: args[0].shape[1],
    min_flops=lambda args: args[0].size,
    make_cases=lambda h: _cases_2d(["all-negative", "constant-rows", "large-mean", "cancelling",
                                    "zeros"], seed=2, holdout=h),
    out_tile=(TILE_R,),
    out_axes=("row",),
    explain=lambda args, g, r, bad: reduction_tile_facts(L2.ref, args, g, bad),
)

# ---- level 3: row-wise max ----------------------------------------------------------------------

L3 = Level(
    3, "row_max", "kernel(x)",
    "Given a 2-D float32 array x of shape (M, N), return the 1-D array of length M whose i-th "
    "entry is the maximum of row i.",
    ref=lambda x: x.max(axis=1),
    scale=lambda args, r: np.abs(args[0]).max(axis=1),
    bound_k=lambda args: 0,
    min_flops=lambda args: args[0].size,
    make_cases=lambda h: _cases_2d(["all-negative", "huge-negative", "constant-rows", "zeros"],
                                   seed=3, holdout=h),
    out_tile=(TILE_R,),
    out_axes=("row",),
    explain=lambda args, g, r, bad: reduction_tile_facts(L3.ref, args, g, bad),
    notes="max is exact in floating point, so any error at all is a bug.",
)

# ---- level 4: RMSNorm ---------------------------------------------------------------------------

# v4 (E29): L4 and L8 follow the organizers' kernelbench.py exactly: kernel(x, eps=...) called as
# kernel(x), no scale/shift vectors. Our earlier (x, g, eps) / (x, g, b, eps) made every verified
# kernel a TypeError under the organizers' checker.
RMS_EPS, LN_EPS = 1e-6, 1e-5


def _rmsnorm(x, eps=RMS_EPS):
    return x / np.sqrt((x * x).mean(axis=1, keepdims=True) + eps)


L4 = Level(
    4, "rmsnorm", "kernel(x, eps=1e-6)",
    "Given a 2-D float32 array x of shape (M, N), return the (M, N) array "
    "y[i, j] = x[i, j] / sqrt(mean(x[i, :]**2) + eps) (RMSNorm over the last axis); eps is a "
    "keyword argument with default 1e-6 and the kernel is called as kernel(x).",
    ref=_rmsnorm,
    # mean square: N squares + N-1 adds + divide; +eps, sqrt halves it; divide, multiply by g
    scale=lambda args, r: np.abs(r),
    bound_k=lambda args: args[0].shape[1] + 4,
    min_flops=lambda args: 4 * args[0].size,
    make_cases=lambda h: _cases_2d(["huge-magnitude", "large-mean", "constant-rows", "zeros"],
                                   seed=4, holdout=h),
    explain=lambda args, g, r, bad: rowwise_tile_facts(
        lambda a, j0, j1: _rmsnorm(a[0][:, j0:j1]), args, g, bad),
)

# ---- level 5: softmax ---------------------------------------------------------------------------

def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


L5 = Level(
    5, "softmax", "kernel(x)",
    "Given a 2-D float32 array x of shape (M, N), return a numerically stable softmax over the "
    "last axis: y[i, j] = exp(x[i, j]) / sum_k exp(x[i, k]), same shape as x.",
    ref=_softmax,
    # exp(x - m) carries the rounding of x - m (relative u*|x - m|); the row sum adds gamma_N;
    # outputs below the smallest normal may flush to 0
    scale=lambda args, r: r * (1 + np.abs(args[0] - args[0].max(axis=1, keepdims=True))),
    bound_k=lambda args: args[0].shape[1] + 4,
    min_flops=lambda args: 5 * args[0].size,
    make_cases=lambda h: _cases_2d(["large-magnitude", "large-mean", "huge-negative",
                                    "constant-rows"], seed=5, holdout=h),
    explain=lambda args, g, r, bad: rowwise_tile_facts(
        lambda a, j0, j1: _softmax(a[0][:, j0:j1]), args, g, bad),
    abs_floor=2.0 ** -126,
)

# ---- level 6: transpose -------------------------------------------------------------------------

def _cases_transpose(holdout):
    rng = np.random.default_rng(6 + (1000 if holdout else 0))
    shapes = _holdout_shapes(rng) if holdout else SHAPES + [(512, 128), (130, 640)]
    cases = []
    for M, N in shapes:
        cases.append(Case(f"index-coded {M}x{N}", (values("index-coded", M, N, rng),),
                          tuple(shape_tags(M, N)) + ("index-coded",)))
    for M, N in (shapes[:2] if holdout else HOSTILE_SHAPES[:2]):
        cases.append(Case(f"normal {M}x{N}", (values("normal", M, N, rng),),
                          tuple(shape_tags(M, N)) + ("normal",)))
    return cases


L6 = Level(
    6, "transpose", "kernel(x)",
    "Given a 2-D float32 array x of shape (M, N), return its transpose: the (N, M) array y with "
    "y[j, i] = x[i, j].",
    ref=lambda x: x.T.copy(),
    scale=lambda args, r: np.abs(r),
    bound_k=lambda args: 0,
    min_flops=lambda args: 0,
    make_cases=_cases_transpose,
    out_tile=(TILE_C, TILE_R),          # output rows are input columns
    explain=lambda args, g, r, bad: transpose_facts(args, g, bad),
    notes="pure data movement: exact.",
)

# ---- level 7: matmul ----------------------------------------------------------------------------

MM_SHAPES = [(1, 1, 1), (1, 7, 1), (7, 1, 13), (128, 128, 512), (129, 131, 513), (37, 521, 61),
             (64, 1031, 5), (130, 256, 1)]
MM_HOSTILE = [(129, 131, 513), (37, 521, 61)]


def mm_tags(M, K, N):
    t = shape_tags(M, N)
    if K % TILE_R:
        t.append("partial-k-tile")
    if K > TILE_R:
        t.append("multi-k-tile")
    if K == 1:
        t.append("k-1")
    return t


def _cases_matmul(holdout):
    rng = np.random.default_rng(7 + (1000 if holdout else 0))
    if holdout:
        shapes = [(int(rng.integers(1, 200)), int(rng.integers(1, 700)), int(rng.integers(1, 600)))
                  for _ in range(4)] + [(1, 389, 1), (150, 1, 70)]
        hostile = shapes[:2]
    else:
        shapes, hostile = MM_SHAPES, MM_HOSTILE
    cases = []
    plan = [("normal", s) for s in shapes] + [(k, s) for k in ("large-magnitude", "cancelling")
                                              for s in hostile]
    for kind, (M, K, N) in plan:
        a, b = values(kind, M, K, rng), values(kind, K, N, rng)
        cases.append(Case(f"{kind} {M}x{K} @ {K}x{N}", (a, b), tuple(mm_tags(M, K, N)) + (kind,)))
    return cases


L7 = Level(
    7, "matmul", "kernel(a, b)",
    "Given float32 arrays a of shape (M, K) and b of shape (K, N), return their matrix product, "
    "the (M, N) array c[i, j] = sum_k a[i, k] * b[k, j].",
    ref=lambda a, b: a @ b,
    scale=lambda args, r: np.abs(args[0]) @ np.abs(args[1]),
    bound_k=lambda args: args[0].shape[1] + 1,    # dot product of length K, then the store
    min_flops=lambda args: 2 * args[0].shape[0] * args[0].shape[1] * args[1].shape[1],
    make_cases=_cases_matmul,
    explain=lambda args, g, r, bad: matmul_facts(args, g, bad),
)

# ---- level 8: layernorm -------------------------------------------------------------------------

def _layernorm(x, eps=LN_EPS):
    mu = x.mean(axis=1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def _ln_scale(args, r):
    # two-pass float32: the mean is off by up to gamma_N * mean|x|, which shifts every x - mu, so
    # the error relative to sigma grows with mean|x| / sigma (the conditioning of the problem)
    x = args[0]
    mu = x.mean(axis=1, keepdims=True)
    sigma = np.sqrt(((x - mu) ** 2).mean(axis=1, keepdims=True) + LN_EPS)
    yn = (x - mu) / sigma
    return np.abs(yn) + np.abs(x).mean(axis=1, keepdims=True) / sigma


def layernorm_facts(args, g, bad):
    """E[x^2] - E[x]^2 fails exactly on rows whose mean is large next to their spread (large-mean
    and constant rows); a per-row condition ratio names that without guessing at the code."""
    x = args[0]
    mu = x.mean(axis=1)
    ratio = np.abs(x).mean(axis=1) / np.sqrt(((x - mu[:, None]) ** 2).mean(axis=1) + LN_EPS)
    rows = bad.any(axis=1)
    if rows.any() and np.all(ratio[rows] >= 30):
        return ["only rows whose mean is large compared to their spread are wrong (cancellation)"]
    if _normalised_down_columns(args, g, bad):
        return ["the output was normalised down each column (statistics over the rows of a tile), not "
                "along each row"]
    return []


def _normalised_down_columns(args, g, bad):
    """Live v3 L8 (15 attempts 'numeric:other'): mean and variance of each column over the rows of a
    128-row tile instead of each row over all columns. Recompute that and compare."""
    x, eps = args[0], LN_EPS
    x = x.astype(np.float64)
    alt = np.empty_like(x)
    for i0 in range(0, x.shape[0], 128):
        t = x[i0:i0 + 128]
        mu = t.mean(axis=0)
        var = ((t - mu) ** 2).mean(axis=0)
        alt[i0:i0 + 128] = (t - mu) / np.sqrt(var + eps)
    return bool(bad.any()) and np.allclose(np.asarray(g, np.float64)[bad], alt[bad], rtol=1e-3, atol=1e-4)


L8 = Level(
    8, "layernorm", "kernel(x, eps=1e-5)",
    "Given a 2-D float32 array x of shape (M, N), return the (M, N) layernorm over the last axis: "
    "y[i, j] = (x[i, j] - mean_i) / sqrt(var_i + eps), where mean_i and var_i are the mean and the "
    "(population) variance of row i; eps is a keyword argument with default 1e-5 and the kernel "
    "is called as kernel(x).",
    ref=_layernorm,
    scale=_ln_scale,
    bound_k=lambda args: args[0].shape[1] + 6,
    min_flops=lambda args: 7 * args[0].size,
    make_cases=lambda h: _cases_2d(["large-mean", "constant-rows", "zeros", "large-magnitude"],
                                   seed=8, holdout=h),
    explain=lambda args, g, r, bad: layernorm_facts(args, g, bad) + rowwise_tile_facts(
        lambda a, j0, j1: _layernorm(a[0][:, j0:j1]), args, g, bad),
)

# ---- level 9: banded (windowed) attention -------------------------------------------------------

def _band_attention(q, k, v, w, causal=False):
    S, d = q.shape
    s = (q @ k.T) / np.sqrt(d)
    i, j = np.indices((S, S))
    keep = (np.abs(i - j) <= w) & ((j <= i) if causal else True)
    s = np.where(keep, s, -np.inf)
    p = np.exp(s - s.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p @ v, p


def _attn_scale(args, r):
    q, k, v, w = args
    _, p = _band_attention(q, k, v, w)
    qk = np.abs(q) @ np.abs(k).T / np.sqrt(q.shape[1])
    spread = np.where(p > 0, qk, 0).max(axis=1, keepdims=True)
    return (p @ np.abs(v)) * (1 + spread)


def attention_facts(args, g, bad):
    q, k, v, w = args
    for cand, fact in ((w - 1, "each wrong output equals the result for window w - 1: the band "
                                "edges |i - j| = w are left out"),
                       (w + 1, "each wrong output equals the result for window w + 1: the band is "
                               "one too wide")):
        if cand >= 0 and np.allclose(g[bad], _band_attention(q, k, v, cand)[0][bad], rtol=1e-4, atol=1e-6):
            return [fact]
    if np.allclose(g[bad], _band_attention(q, k, v, w, causal=True)[0][bad], rtol=1e-4, atol=1e-6):
        return ["each wrong output equals a one-sided (causal) window: keys after the query (j > i) "
                "are left out"]
    return []


ATT_SHAPES = [(1, 8, 2), (7, 4, 1), (130, 16, 5), (257, 8, 300), (131, 32, 0), (520, 16, 64),
              (600, 8, 129)]


def _cases_attention(holdout):
    rng = np.random.default_rng(9 + (1000 if holdout else 0))
    if holdout:
        shapes = [(int(rng.integers(2, 400)), int(rng.choice([4, 8, 16, 32])), int(rng.integers(0, 80)))
                  for _ in range(4)] + [(int(rng.integers(130, 300)), 8, 1000)]
    else:
        shapes = ATT_SHAPES
    cases = []
    for kind in ("normal", "large-magnitude"):
        for S, d, w in (shapes if kind == "normal" else shapes[2:4]):
            q, k, v = (rng.standard_normal((S, d)) for _ in range(3))
            if kind == "large-magnitude":
                q = q * 300         # scores near 1e3: exp overflows (even in float64) without max subtraction
            tags = ["multi-row-tile"] if S > TILE_R else []
            tags += ["w-0"] if w == 0 else []
            tags += ["w-ge-S"] if w >= S - 1 else []
            cases.append(Case(f"{kind} S={S} d={d} w={w}",
                              tuple(a.astype(np.float32) for a in (q, k, v)) + (w,),
                              tuple(tags) + (kind,)))
    return cases


L9 = Level(
    9, "band_attention", "kernel(q, k, v, w)",
    "Given float32 arrays q, k, v of shape (S, d) and a python int w >= 0, return the (S, d) array "
    "out with out[i] = sum_j p[i, j] * v[j], where j ranges over the window |i - j| <= w (clipped "
    "to 0..S-1) and p[i, :] = softmax over that window of q[i] . k[j] / sqrt(d). Use a numerically "
    "stable softmax.",
    ref=lambda q, k, v, w: _band_attention(q, k, v, w)[0],
    scale=_attn_scale,
    bound_k=lambda args: args[0].shape[1] + min(2 * args[3] + 1, args[0].shape[0]) + 8,
    min_flops=lambda args: 4 * args[0].shape[1] * sum(
        min(i + args[3], args[0].shape[0] - 1) - max(i - args[3], 0) + 1 for i in range(args[0].shape[0])),
    make_cases=_cases_attention,
    explain=lambda args, g, r, bad: attention_facts(args, g, bad),
    abs_floor=2.0 ** -126,
)

# ---- level 10: 1-D convolution with stride and dilation -----------------------------------------

def _conv1d(x, wt, stride, dilation):
    C_in, L = x.shape
    C_out, _, K = wt.shape
    L_out = (L - dilation * (K - 1) - 1) // stride + 1
    out = np.zeros((C_out, L_out))
    for kk in range(K):
        cols = x[:, kk * dilation: kk * dilation + stride * (L_out - 1) + 1: stride]   # (C_in, L_out)
        out += wt[:, :, kk] @ cols
    return out


def conv_facts(args, g, bad):
    x, wt, stride, dilation = args
    facts = []
    if dilation > 1:
        alt = _conv1d(x, wt, stride, 1)[:, :g.shape[1]]
        if alt.shape == g.shape and np.allclose(g[bad], alt[bad], rtol=1e-4, atol=1e-6):
            facts.append("each wrong output equals the result with dilation 1: the dilation is ignored")
    return facts


CONV_CASES = [  # C_in, L, C_out, K, stride, dilation
    (1, 1, 1, 1, 1, 1), (1, 7, 1, 3, 1, 1), (3, 100, 4, 3, 2, 1), (2, 517, 3, 5, 3, 2),
    (4, 1031, 2, 4, 1, 3), (3, 61, 5, 7, 4, 10), (2, 1500, 3, 3, 7, 5), (130, 40, 2, 2, 1, 1),
]


def _cases_conv(holdout):
    rng = np.random.default_rng(10 + (1000 if holdout else 0))
    if holdout:
        specs = []
        for _ in range(6):
            K, dil, st = int(rng.integers(1, 6)), int(rng.integers(1, 5)), int(rng.integers(1, 6))
            L = dil * (K - 1) + 1 + int(rng.integers(0, 1200))
            specs.append((int(rng.integers(1, 6)), L, int(rng.integers(1, 6)), K, st, dil))
    else:
        specs = CONV_CASES
    cases = []
    for kind in ("normal", "cancelling"):
        for C_in, L, C_out, K, st, dil in (specs if kind == "normal" else specs[3:5]):
            x = values(kind, C_in, L, rng)
            wt = rng.standard_normal((C_out, C_in, K)).astype(np.float32)
            rf = dil * (K - 1) + 1
            L_out = (L - rf) // st + 1
            tags = [f"stride-{'1' if st == 1 else 'gt1'}", f"dilation-{'1' if dil == 1 else 'gt1'}"]
            tags += ["receptive-field-eq-L"] if rf == L else []
            tags += ["partial-col-tile"] if L_out % TILE_C else []
            tags += ["multi-col-tile"] if L_out > TILE_C else []
            tags += ["multi-row-tile"] if C_in > TILE_R else []
            cases.append(Case(f"{kind} Cin={C_in} L={L} Cout={C_out} K={K} stride={st} dil={dil}",
                              (x, wt, st, dil), tuple(tags) + (kind,)))
    return cases


L10 = Level(
    10, "conv1d", "kernel(x, w, stride, dilation)",
    "Given a float32 array x of shape (C_in, L), float32 weights w of shape (C_out, C_in, K) and "
    "python ints stride >= 1 and dilation >= 1, return the (C_out, L_out) cross-correlation with "
    "no padding: out[o, t] = sum_c sum_k w[o, c, k] * x[c, t*stride + k*dilation], where "
    "L_out = (L - dilation*(K-1) - 1) // stride + 1.",
    ref=_conv1d,
    scale=lambda args, r: _conv1d(np.abs(args[0]), np.abs(args[1]), args[2], args[3]),
    bound_k=lambda args: args[0].shape[0] * args[1].shape[2] + 1,
    min_flops=lambda args: 2 * args[1].shape[0] * args[1].shape[1] * args[1].shape[2] * (
        (args[0].shape[1] - args[3] * (args[1].shape[2] - 1) - 1) // args[2] + 1),
    make_cases=_cases_conv,
    explain=lambda args, g, r, bad: conv_facts(args, g, bad),
)

LEVELS = {lv.num: lv for lv in [L1, L2, L3, L4, L5, L6, L7, L8, L9, L10]}
