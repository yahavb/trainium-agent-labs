#!/usr/bin/env python3
"""
nkibench.py — layer 1 of the kernel-agent harness: the SIMULATOR layer.

Everything here runs on a CPU in seconds and needs no Trainium device. It answers three
questions about a candidate NKI kernel, in this order:

  1. Does it break the rules?        static scan, milliseconds, no execution
  2. Does it compute the right thing? nki.simulate_kernel against a NumPy reference
  3. Is it memory bound or compute bound?  bytes and flops counted during that simulation,
                                           turned into an arithmetic intensity and a verdict

Layers 2 and 3 of the harness -- real latency via nki.baremetal, and a profile showing which
engine owns the time -- need the chip and are not in this file.

    python nkibench.py --list
    python nkibench.py --level 4 --show
    python nkibench.py --level 4 --check my_matmul.py
    python nkibench.py --roofline 512 1024 2048      # what the roofline says before you code
    python nkibench.py --selftest

WHAT IS VERIFIED AND WHAT IS NOT. The rule checker, the references, the shape generators, the
failure messages and the roofline arithmetic are all exercised by --selftest on any machine --
and the roofline model reproduces the tutorial's own published figures exactly, which is the
check that matters. The parts that import `nki` cannot run without the Neuron SDK, so they are
marked NEEDS DEVICE VERIFICATION in the selftest output. Run --selftest on the instance too.
"""

import argparse
import ast
import importlib.util
import sys
import textwrap

import numpy as np

# ---------------------------------------------------------------- hardware facts
#
# From the NKI matmul tutorial and the NKI language tile-size constants. These are properties
# of NeuronCore-v2, not choices.

PMAX = 128                    # nl.tile_size.pmax -- partition axis maximum
GEMM_STATIONARY_FMAX = 128    # nl.tile_size.gemm_stationary_fmax
GEMM_MOVING_FMAX = 512        # nl.tile_size.gemm_moving_fmax

# The roofline ridge point: the arithmetic intensity at which the memory ceiling and the
# compute ceiling meet. Below it a kernel is memory bound, above it compute bound. The
# tutorial states 222 Flops/Byte for bfloat16 on NeuronCore-v2.
# Only the bfloat16 figure is published (NKI matmul tutorial). The others are NOT invented
# here: a made-up ridge silently flips every memory-bound / compute-bound verdict.
RIDGE_FLOPS_PER_BYTE = {"bfloat16": 222.0}

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


def roofline(flops, hbm_bytes, dtype="bfloat16"):
    """Which ceiling does this kernel hit first?

    A chip can move bytes and do arithmetic, each with its own peak. Whichever runs out first
    is the limit. Arithmetic intensity -- flops per byte read from HBM -- says which. SBUF
    traffic is deliberately excluded: SBUF bandwidth is roughly 20x HBM and is not the
    constraint.
    """
    ridge = RIDGE_FLOPS_PER_BYTE[dtype]
    ai = flops / hbm_bytes if hbm_bytes else float("inf")
    bound = "memory_bound" if ai < ridge else "compute_bound"
    return dict(arithmetic_intensity=round(ai, 1), ridge=ridge, bound=bound,
                fraction_of_ridge=round(ai / ridge, 3),
                headroom=f"{ridge / ai:.1f}x more reuse needed" if ai < ridge else "at or above the ridge")


def explain_with_ceiling(flops, measured_bytes, floor_bytes, dtype="bfloat16"):
    """Separate what the KERNEL wastes from what the SHAPE cannot reach.

    Flops are fixed by the problem, so arithmetic intensity is just flops divided by bytes. The byte
    floor -- read each input once, write the output once -- therefore sets a CEILING on intensity that
    no kernel can beat. If that ceiling is already under the ridge, the operation is memory bound at
    this shape no matter how good the kernel is.

    This matters because the previous message was misleading: it told the agent 'memory bound, 2.2x
    more reuse needed' on shapes whose ceiling is 28-73 Flops/Byte against a ridge of 222. The reuse
    it was being asked for did not exist.
    """
    ridge = RIDGE_FLOPS_PER_BYTE[dtype]
    measured = flops / measured_bytes if measured_bytes else float("inf")
    ceiling = flops / floor_bytes if floor_bytes else float("inf")
    waste = measured_bytes / floor_bytes if floor_bytes else 1.0
    lines = [f"arithmetic intensity {measured:.1f} Flops/Byte. "
             f"Ceiling at this shape is {ceiling:.1f} (every byte read once); "
             f"the hardware ridge is {ridge:g}."]
    if waste > 1.15:
        lines.append(f"YOUR KERNEL: moving {waste:.1f}x the necessary bytes, which divides your "
                     f"intensity by {waste:.1f}. Recovering that is entirely in your hands and is "
                     f"the whole of levels 4 to 7.")
    else:
        lines.append("YOUR KERNEL: at the byte floor already, so there is no reuse left to find.")
    if ceiling < ridge:
        lines.append(f"THIS SHAPE: even a perfect kernel reaches only {ceiling:.1f}, which is "
                     f"{ridge / ceiling:.1f}x under the ridge, so the operation is memory bound "
                     f"here however it is written. Do not chase compute-bound at this size -- for a "
                     f"square {dtype} matmul you need about n >= {int(3 * ridge) + 1} before the "
                     f"ceiling clears the ridge.")
    else:
        lines.append(f"THIS SHAPE: the ceiling of {ceiling:.1f} clears the ridge, so a good enough "
                     f"kernel CAN become compute bound here.")
    return "\n".join("    " + l for l in lines)


def explain_roofline(r):
    if r["bound"] == "memory_bound":
        return (f"MEMORY BOUND: {r['arithmetic_intensity']} Flops/Byte against a ridge of "
                f"{r['ridge']:g}, so {r['fraction_of_ridge']:.0%} of what the engine could "
                f"sustain. The engine is idle waiting for data. Find reuse -- make the same "
                f"bytes do more work -- rather than tuning the arithmetic. "
                f"{r['headroom']}.")
    return (f"COMPUTE BOUND: {r['arithmetic_intensity']} Flops/Byte against a ridge of "
            f"{r['ridge']:g}. The loads are keeping up, so cutting HBM traffic buys nothing. "
            f"Any further win has to come from the arithmetic or the engine schedule.")


def matmul_flops(M, K, N):
    return 2 * M * K * N


def matmul_tiled_hbm_bytes(M, K, N, dtype="bfloat16"):
    """HBM bytes read by the straightforward tiled matmul: one lhsT tile and one rhs tile per
    innermost iteration, nothing reused across iterations.

    Verified against the tutorial in --selftest: its innermost loop reads 160 KB per
    16 MFlops, i.e. an arithmetic intensity of 102.
    """
    b = DTYPE_BYTES[dtype]
    iters = (M // GEMM_STATIONARY_FMAX) * (N // GEMM_MOVING_FMAX) * (K // PMAX)
    per_iter = (PMAX * GEMM_STATIONARY_FMAX + PMAX * GEMM_MOVING_FMAX) * b
    return iters * per_iter


# ---------------------------------------------------------------- the references
#
# NumPy, deliberately. The reference has to be obviously right, and these are.

def ref_avgpool2d(x, pool_size):
    """x: [C, H, W] -> [C, H//p, W//p]"""
    C, H, W = x.shape
    p = pool_size
    return x[:, :H // p * p, :W // p * p].reshape(
        C, H // p, p, W // p, p).mean(axis=(2, 4)).astype(x.dtype)


def ref_transpose2d(x, shape2D):
    """Transpose the two FREE axes inside each partition.

    Matches the tutorial's signature: x is [P, F1*F2] and shape2D is (F1, F2), so each
    partition row holds an F1-by-F2 matrix laid out row-major and comes back column-major.
    The partition axis is untouched -- that is the point, since it is the constrained one.
    """
    P, F = x.shape
    F1, F2 = shape2D
    assert F1 * F2 == F, f"shape2D {shape2D} does not match the free size {F}"
    return np.ascontiguousarray(
        x.reshape(P, F1, F2).transpose(0, 2, 1)).reshape(P, F)


def ref_attention(q, k, v):
    """Single-head attention: softmax(Q Kt / sqrt(d)) V, with q, k, v all [seq, dim].

    This level exists to show that the harness is not about matmul. It was added by writing this
    reference and one input builder -- nothing else in the file changed. Do the same for whatever you
    care about.

    Note the trap, which is the whole reason attention is interesting on a chip: a naive exp()
    overflows. The reference subtracts the row max first, and a kernel that does not will return
    plausible numbers on friendly data and NaN on real data.
    """
    d = q.shape[1]
    scores = (q.astype(np.float32) @ k.astype(np.float32).T) / np.sqrt(d)
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    return ((e / e.sum(axis=-1, keepdims=True)) @ v.astype(np.float32)).astype(q.dtype)


def ref_matmul(lhsT, rhs):
    """lhsT: [K, M] (left operand arrives TRANSPOSED), rhs: [K, N] -> [M, N]

    The transposed left operand is not a quirk of this harness. nc_matmul consumes its
    stationary operand down the partition axis, so K has to be the partition dimension of
    both inputs.
    """
    return (lhsT.astype(np.float32).T @ rhs.astype(np.float32)).astype(lhsT.dtype)


# ---------------------------------------------------------------- the ladder

# Each level owns how its inputs are built and labelled, so ADDING AN OPERATION TOUCHES NOTHING
# ELSE. That matters more than elegance: the point of this harness is that a team can point it at
# attention, or a convolution, or whatever they care about, by writing one reference and one input
# generator. See "Adding your own operation" in the README.

def _args_pool(spec, r):
    return (r.standard_normal(spec["shape"]).astype(np.float32), spec["pool_size"])


def _args_transpose(spec, r):
    return (r.standard_normal(spec["shape"]).astype(np.float32), spec["shape2D"])


def _args_matmul(spec, r):
    return (r.standard_normal((spec["K"], spec["M"])).astype(np.float32),
            r.standard_normal((spec["K"], spec["N"])).astype(np.float32))


def _args_attention(spec, r):
    n, d = spec["seq"], spec["dim"]
    return tuple(r.standard_normal((n, d)).astype(np.float32) for _ in range(3))


def make_inputs(spec, level_n, seed=0):
    r = np.random.default_rng(seed + level_n)
    return LEVELS[level_n]["make_args"](spec, r), {}


def label(spec, level_n):
    return LEVELS[level_n]["label"](spec)


LEVELS = {}


def level(n, op, entry, teaches, optimization, ref, shapes, banned, notes="", max_waste=None,
          make_args=None, label=None, tol=None):
    """max_waste: the most HBM traffic this level may move, as a multiple of the byte floor.

    Levels 5 to 7 are optimization levels, so correctness alone cannot tell them apart from level 4 --
    the reference and the shapes are identical, so a level-4 kernel would score 1.0 on all three and
    the ladder would stop meaning anything past 4. A traffic bar is what makes them levels.

    The thresholds come from the shipped tiled kernel, which models at 2.00x the floor on the largest
    test shape: level 5 must beat that, level 6 must beat level 5, level 7 must be near the floor.

    tol: this level's default numerical tolerance for --check, when its dtype needs one other than
    2e-2. --tol on the command line still wins.
    """
    LEVELS[n] = dict(n=n, op=op, entry=entry, teaches=teaches, optimization=optimization,
                     ref=ref, shapes=shapes, banned=banned, notes=notes, max_waste=max_waste,
                     make_args=make_args or _args_matmul,
                     label=label or (lambda sp: f"K={sp['K']} M={sp['M']} N={sp['N']}"),
                     tol=tol)


level(1, "average pooling 2D", "tensor_avgpool_kernel",
     "the programming model: the partition axis is not like the others, and access patterns "
     "are explicit",
     "almost none -- a reduction has inherently low arithmetic intensity. This is the "
     "'can the agent emit legal NKI at all' checkpoint.",
     ref_avgpool2d,
     [dict(shape=(32, 32, 32), pool_size=2), dict(shape=(128, 16, 16), pool_size=4),
      dict(shape=(8, 24, 24), pool_size=3), dict(shape=(64, 8, 8), pool_size=2)],
     {"mean", "average", "avg_pool2d", "avg_pool", "adaptive_avg_pool2d"},
     "nl.sum and nl.mean over a strided access-pattern view are the intended NKI route; "
     "what is banned is handing the whole reduction to numpy or torch.",
     make_args=_args_pool,
     label=lambda sp: f"C,H,W={sp['shape']} pool={sp['pool_size']}")

level(2, "2D transpose", "tensor_transpose2D_kernel_",
     "layout: a transpose has to cross partitions, and the partition axis is the constrained "
     "one",
     "the first real one, and it is NOT about compute. A naive transpose moves tiny pieces "
     "and pays a per-transfer issue cost, so the cost is the NUMBER of transfers rather than "
     "the number of bytes. 'It moves a lot of data' is a misdiagnosis here.",
     ref_transpose2d,
     [dict(shape=(32, 12), shape2D=(3, 4)), dict(shape=(128, 64), shape2D=(8, 8)),
      dict(shape=(64, 128), shape2D=(4, 32)), dict(shape=(8, 35), shape2D=(5, 7))],
     {"transpose", "swapaxes", "moveaxis", "rollaxis", "permute"},
     make_args=_args_transpose,
     label=lambda sp: f"shape={sp['shape']} as {sp['shape2D'][0]}x{sp['shape2D'][1]}")

level(3, "matmul, single tile", "nki_matmul_basic_",
     "the machine: results land in PSUM and must be copied to SBUF, the left operand arrives "
     "transposed, and tiles have hard shape limits",
     "none yet -- this is a correctness level where the layout rules bite.",
     ref_matmul,
     [dict(K=128, M=64, N=512)],
     {"matmul", "dot", "einsum", "tensordot", "inner", "vdot"})

# The last shape exists to expose redundant HBM traffic, which only appears with several tiles in
# each dimension: measured against the floor, the shipped tiled kernel is 1.00x on a single-tile
# shape and 2.0x here. A level-4 kernel can look optimal on small shapes and be badly wasteful on
# real ones, so at least one shape has to be big enough to tell.
_MM_SHAPES = [dict(K=128, M=128, N=512), dict(K=256, M=256, N=1024),
              dict(K=512, M=128, N=512), dict(K=256, M=512, N=1024)]

level(4, "matmul, tiled", "nki_matmul_tiled_",
     "tiling a matmul beyond one tile, in all three dimensions",
     "THE PIVOT LEVEL. Measurably memory bound, and derivably so: its innermost loop reads "
     "160 KB per 16 MFlops, an arithmetic intensity of 102 against a ridge of 222. The agent "
     "should compute that from its own code and classify the kernel BEFORE changing anything.",
     ref_matmul, _MM_SHAPES,
     {"matmul", "dot", "einsum", "tensordot", "inner", "vdot"})

level(5, "matmul, loads hoisted", "nki_matmul_hoist_load_",
     "that the same tiles are re-read every pass of the inner loop",
     "hoist the redundant loads out of the innermost loop. Cheap, mechanical, and the first "
     "measurable win. Arithmetic intensity rises.",
     ref_matmul, _MM_SHAPES,
     {"matmul", "dot", "einsum", "tensordot", "inner", "vdot"}, max_waste=1.6)

level(6, "matmul, M and N blocked", "nki_matmul_block_free_dimension_",
     "spending SBUF capacity to buy reuse",
     "hoisting reuses one row of tiles; SBUF holds far more. Now there is a SEARCH SPACE -- "
     "block sizes bounded by SBUF capacity -- so the agent has to explore rather than derive.",
     ref_matmul, _MM_SHAPES,
     {"matmul", "dot", "einsum", "tensordot", "inner", "vdot"}, max_waste=1.25)

level(7, "matmul, M, N and K blocked", "nki_matmul_fully_optimized_",
     "the full blocking scheme",
     "the top of the ladder. Hard, and a fine place to stop short of.",
     ref_matmul, _MM_SHAPES,
     {"matmul", "dot", "einsum", "tensordot", "inner", "vdot"}, max_waste=1.05)


# ---------------------------------------------------------------- inputs

# ---------------------------------------------------------------- extending the ladder
#
# Level 8 is here as a WORKED EXAMPLE of adding an operation. Everything it needed: the reference
# above, the _args_attention builder, and this one call. No other part of the harness knows it
# exists -- the rule checker, the simulator, the numerics, the traffic measurement and the agent all
# pick it up automatically.

level(8, "single-head attention", "nki_attention_",
      "composition: a matmul, a numerically stable softmax, and a second matmul, with the "
      "intermediate never leaving the chip",
      "the intermediate scores matrix is seq x seq, so writing it out to HBM and reading it back is "
      "the mistake that dominates. Keeping it on-chip is the whole game, and it is why fused "
      "attention kernels exist at all.",
      ref_attention,
      [dict(seq=128, dim=64), dict(seq=64, dim=128), dict(seq=96, dim=32)],
      {"softmax", "scaled_dot_product_attention", "attention", "matmul", "dot", "einsum"},
      "nl.max and nl.sum over tiles are the intended route. Subtract the row maximum before exp() "
      "or large scores overflow: a kernel that skips it looks correct on small test data.",
      make_args=_args_attention,
      label=lambda sp: f"seq={sp['seq']} dim={sp['dim']}")


# ---------------------------------------------------------------- CHIPBOOST: Qwen3-8B's own ops
#
# Levels 9 to 12 are the operations Qwen3-8B is built from, in bfloat16, for projects/03-chipboost.
# Each was added the level-8 way: a reference, an input builder, one level() call.
#
# The `shapes` here are the DEV shapes: small enough for nki.simulate on a CPU in seconds, and
# multi-tile in every dimension so tiling bugs show. The real Qwen3-8B per-core sizes (where speed is
# timed on the chip) and the held-out shapes live in projects/03-chipboost/shapes.py.

QWEN3_RMS_EPS = 1e-6      # Qwen3-8B config.json: rms_norm_eps

# Tolerances for bfloat16 outputs, relative to the output's RMS (describe_mismatch's measure).
# bfloat16 keeps 8 significant bits, so one rounding step at |v| in [4, 8) RMS is 2**-5 = 3.1% of the
# RMS, and these outputs have tails past 4 RMS. A correct matmul accumulates in float32 PSUM and rounds
# once, so it can land one step from the reference: 0.04 admits that. RMSNorm and SwiGLU also go
# through the engine's rsqrt / sigmoid, which may be approximate, so allow two steps: 0.07. The bugs
# these levels exist to catch -- a dropped tile, a missing eps, a ragged row left unwritten -- are
# 10% to 1000%. A copy must be exact.
BF16_MATMUL_TOL = 0.04
BF16_ACTIVATION_TOL = 0.07


def _bf16():
    """NumPy's bfloat16, from ml_dtypes (a Neuron SDK dependency). Imported only when a bf16 level
    builds inputs, so the rest of this file and --selftest still run without it."""
    try:
        import ml_dtypes
    except ImportError as e:
        raise NkiMissing("bfloat16 inputs need the ml_dtypes package, which ships with the Neuron "
                         "SDK. Run this level in the seat pod, or pip install ml_dtypes.") from e
    return ml_dtypes.bfloat16


def _randn(r, shape, scale=1.0):
    return r.standard_normal(shape).astype(np.float32) * np.float32(scale)


def ref_rmsnorm(x, w):
    """Qwen3RMSNorm over the last axis: x * rsqrt(mean(x^2) + eps) * w.

    x: [rows, D], w: [1, D]. Computed in float32 and rounded to x's dtype once, at the end. (Hugging
    Face rounds to bf16 before multiplying by w; a kernel that stays in float32 until the end is at
    most one bf16 step from either, which BF16_ACTIVATION_TOL covers.)

    The trap is eps: on tiny inputs mean(x^2) is far below 1e-6, so a kernel that drops eps looks right
    on ordinary data and is 10x off on quiet rows. The held-out shapes in chipboost include such rows.
    """
    x32 = x.astype(np.float32)
    var = (x32 * x32).mean(axis=-1, keepdims=True)
    y = x32 / np.sqrt(var + np.float32(QWEN3_RMS_EPS)) * w.astype(np.float32)
    return y.astype(x.dtype)


def ref_copy(x):
    """The bandwidth probe: the output is the input, bit for bit."""
    return x.copy()


def ref_swiglu(gate, up):
    """SiLU(gate) * up, as in Qwen3's MLP. Written as g / (1 + exp(-g)), which is safe at both ends:
    a large negative gate sends exp(-g) to inf and the result to 0, a large positive one to g. The trap
    is the other algebra, g * exp(g) / (1 + exp(g)), which returns inf/inf = NaN for a large gate."""
    g = gate.astype(np.float32)
    with np.errstate(over="ignore"):
        silu = g / (np.float32(1.0) + np.exp(-g))
    return (silu * up.astype(np.float32)).astype(gate.dtype)


def _args_matmul_bf16(spec, r):
    bf16 = _bf16()
    s = spec.get("scale", 1.0)
    return (_randn(r, (spec["K"], spec["M"]), s).astype(bf16),
            _randn(r, (spec["K"], spec["N"]), s).astype(bf16))


def _args_rmsnorm(spec, r):
    bf16 = _bf16()
    x = _randn(r, (spec["rows"], spec["dim"]), spec.get("scale", 1.0))
    w = np.float32(1.0) + _randn(r, (1, spec["dim"]), 0.1)
    return (x.astype(bf16), w.astype(bf16))


def _args_copy(spec, r):
    return (_randn(r, (spec["rows"], spec["dim"])).astype(_bf16()),)


def _args_swiglu(spec, r):
    bf16 = _bf16()
    s = spec.get("scale", 1.0)
    return (_randn(r, (spec["rows"], spec["dim"]), s).astype(bf16),
            _randn(r, (spec["rows"], spec["dim"])).astype(bf16))


def _label_rows(sp):
    extra = f" scale={sp['scale']:g}" if sp.get("scale", 1.0) != 1.0 else ""
    return f"rows={sp['rows']} dim={sp['dim']}{extra}"


_MATMUL_BANNED = {"matmul", "dot", "einsum", "tensordot", "inner", "vdot"}

level(9, "Qwen3-8B matmul, bf16", "nki_matmul_tiled_",
      "the projections Qwen3-8B spends most of its time in, as the per-core shapes of tensor "
      "parallelism 2, in bfloat16",
      "the whole matmul ladder at real sizes: hoisting, blocking M and N, blocking K. The start "
      "kernel is the level-4 tiled kernel; the expert ceiling is the tutorial's fully optimised one.",
      ref_matmul,
      # The last shape is 2048 deep, so blocked kernels split K into more than one block and the
      # cross-block accumulation path is exercised in dev, not first met at the held-out shapes.
      [dict(K=512, M=256, N=1024), dict(K=1024, M=128, N=2048), dict(K=256, M=512, N=512),
       dict(K=2048, M=128, N=512)],
      _MATMUL_BANNED,
      "lhsT arrives transposed [K, M], like level 4, and the entry point has level 4's name, so "
      "every matmul kernel and cheat in the team works at either level. K and M are multiples of 128 "
      "and N of 512.",
      make_args=_args_matmul_bf16, tol=BF16_MATMUL_TOL)

level(10, "Qwen3-8B RMSNorm, bf16", "qwen3_rmsnorm",
      "a row-wise reduction whose row count is the number of tokens, so it is ragged: rows are not a "
      "multiple of 128",
      "it is memory bound by nature, so the win is reading each byte once and keeping the "
      "square-sum and the scaling on chip. Its floor is the copy kernel's time for the same bytes.",
      ref_rmsnorm,
      [dict(rows=128, dim=4096), dict(rows=300, dim=4096), dict(rows=256, dim=128)],
      {"rms_norm", "rmsnorm", "norm", "layer_norm", "mean", "var", "std"},
      "x is [rows, D], w is [1, D]; eps is 1e-6. Do not drop eps: quiet rows need it.",
      make_args=_args_rmsnorm, label=_label_rows, tol=BF16_ACTIVATION_TOL)

level(11, "copy, the bandwidth floor", "copy_floor",
      "nothing to compute: it measures how fast bytes go HBM -> SBUF -> HBM on this chip",
      "none. Its time for RMSNorm's byte count is the floor no RMSNorm kernel can beat.",
      ref_copy,
      [dict(rows=128, dim=4096), dict(rows=300, dim=4096)],
      {"copy", "copyto", "clone"},
      make_args=_args_copy, label=_label_rows, tol=0.0)

level(12, "Qwen3-8B SwiGLU, bf16 (stretch)", "qwen3_swiglu",
      "an elementwise activation on the MLP's widest tensor: SiLU(gate) * up",
      "fusing the two reads and one write, and later folding it into the gate/up matmul.",
      ref_swiglu,
      [dict(rows=128, dim=6144), dict(rows=300, dim=1024)],
      {"silu", "swiglu", "sigmoid", "gelu", "softplus"},
      "gate and up are [rows, D]. A large positive gate must not produce NaN.",
      make_args=_args_swiglu, label=_label_rows, tol=BF16_ACTIVATION_TOL)


# ---------------------------------------------------------------- layer 1a: static rules

FRAMEWORK_MODULES = {"np", "numpy", "torch", "jnp", "jax", "F", "nn"}


def _dotted(node):
    """Full dotted name of a call target, e.g. 'np.mean' or 'nisa.nc_matmul'."""
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def check_rules(src, level_n):
    """A scan, not a proof. Catches the cheats and the shape mistakes visible in the text. It
    cannot tell you your kernel is properly tiled -- a human reads that.

    Only FRAMEWORK-level calls are banned: np.mean, torch.matmul, the `@` operator, `.T` on an
    argument. NKI's own primitives are the intended route and are never flagged -- nl.sum over
    a strided view is how the pooling tutorial does it, and nisa.nc_matmul is the whole point
    of the matmul levels. Banning by bare name would reject correct kernels, which is worse
    than missing a cheat.
    """
    spec = LEVELS[level_n]
    bad = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"the file does not parse: {e}"]

    names, decorated, params = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
            for d in node.decorator_list:
                if "jit" in ast.unparse(d):
                    decorated.add(node.name)
            if node.name == spec["entry"]:
                params = {a.arg for a in node.args.args}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dotted = _dotted(node.func)
            head = dotted.split(".")[0] if dotted else ""
            leaf = dotted.split(".")[-1] if dotted else ""
            if leaf in spec["banned"] and (head in FRAMEWORK_MODULES or head == leaf):
                bad.append(f"line {node.lineno}: calls `{dotted}`, which hands the whole "
                           f"operation to a framework. This level is about computing it in "
                           f"the kernel.")
            if leaf == "ndarray" and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.Tuple) and a0.elts:
                    e0 = a0.elts[0]
                    if isinstance(e0, ast.Constant) and isinstance(e0.value, int) \
                            and e0.value > PMAX:
                        bad.append(f"line {node.lineno}: partition dimension {e0.value} "
                                   f"exceeds the maximum of {PMAX}. Tile it.")
        if isinstance(node, ast.Attribute) and node.attr == "T" \
                and isinstance(node.value, ast.Name) and node.value.id in params:
            bad.append(f"line {node.lineno}: uses `{node.value.id}.T`, which transposes the "
                       f"input on the host. The kernel has to do it.")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            bad.append(f"line {node.lineno}: uses the `@` matmul operator, which does the "
                       f"whole job.")

    if spec["entry"] not in names:
        bad.append(f"no function named `{spec['entry']}` is defined; that is the entry point "
                   f"this level is checked through.")
    elif spec["entry"] not in decorated:
        bad.append(f"`{spec['entry']}` is not decorated with `@nki.jit`, so it will run as "
                   f"plain Python rather than compiling for the device.")
    return sorted(set(bad))


# ---------------------------------------------------------------- layer 1b: numerics

def describe_mismatch(got, want, tol=2e-2):
    """The message the agent learns from. Vague here is slow there."""
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    if got.shape != want.shape:
        return (f"WRONG SHAPE: returned {got.shape}, reference is {want.shape}. Check the "
                f"output-size arithmetic, not the values.")
    if not np.all(np.isfinite(got)):
        n_nan, n_inf = int(np.isnan(got).sum()), int(np.isinf(got).sum())
        where = np.argwhere(~np.isfinite(got))[0]
        return (f"NON-FINITE OUTPUT: {n_nan} NaN and {n_inf} Inf, first at "
                f"{tuple(int(i) for i in where)}. Usually an uninitialised PSUM or SBUF "
                f"tile being read before anything wrote to it.")
    scale = float(np.sqrt((want ** 2).mean())) or 1.0
    err = np.abs(got - want)
    worst = float(err.max()) / scale
    if worst <= tol:
        return None

    # An almost-all-zero output is its own diagnosis, and by far the most common one: the kernel
    # ran, so it scores for running, but the results never reached the output tensor. Saying
    # "most elements are wrong" sends the model to re-check arithmetic that was probably fine.
    zero_frac = float((np.abs(got) < 1e-12).mean())
    if zero_frac > 0.9 and float((np.abs(want) < 1e-12).mean()) < 0.5:
        return (f"OUTPUT IS {zero_frac:.0%} ZEROS while the reference is not. The kernel ran but "
                f"its results never reached the output tensor, so the arithmetic is probably not "
                f"the problem. Check that every computed tile is copied all the way out: PSUM to "
                f"SBUF with nisa.tensor_copy, then SBUF to the returned tensor with "
                f"nisa.dma_copy, for every tile of the loop and not only the last one. Also "
                f"check the destination slice indices, since writing every tile to the same "
                f"place leaves the rest zero.")
    if zero_frac > 0.25 and float((np.abs(want) < 1e-12).mean()) < 0.05:
        zi = np.argwhere(np.abs(got) < 1e-12)
        lo, hi = zi.min(axis=0), zi.max(axis=0)
        return (f"{zero_frac:.0%} of the output is zero, in the block from "
                f"{tuple(int(v) for v in lo)} to {tuple(int(v) for v in hi)}, while the reference "
                f"has no zeros there. Some tiles were computed and written and others were not, "
                f"so the loop is covering only part of the output. Check the loop bounds and the "
                f"destination indices for every tile.")
    i = int(np.argmax(err))
    idx = np.unravel_index(i, got.shape)
    msg = [f"NUMERICAL MISMATCH: worst error {worst:.3g} of the output's RMS ({scale:.4g}), "
           f"tolerance {tol:g}.",
           f"  at index {tuple(int(v) for v in idx)}: expected {want.flat[i]:+.6g}, got "
           f"{got.flat[i]:+.6g}",
           f"  {float((err / scale > tol).mean()):.1%} of elements are outside tolerance"]
    if got.ndim >= 2:
        r, c = int(idx[0]), int(idx[-1])
        ragged = []
        if r >= (got.shape[0] // PMAX) * PMAX and got.shape[0] > PMAX:
            ragged.append(f"the final partial PARTITION tile (partition {r} of {got.shape[0]})")
        if c >= (got.shape[-1] // GEMM_MOVING_FMAX) * GEMM_MOVING_FMAX \
                and got.shape[-1] > GEMM_MOVING_FMAX:
            ragged.append(f"the final partial FREE tile (index {c} of {got.shape[-1]})")
        if ragged:
            msg.append(f"  this is in {' and '.join(ragged)} -- the ragged edge is the likely "
                       f"cause, not the core arithmetic")
        elif float((err / scale > tol).mean()) > 0.5:
            msg.append("  most elements are wrong, so this is the core arithmetic or the "
                       "operand layout, not an edge case")
        else:
            msg.append("  inside a full tile, so look at the accumulation rather than the "
                       "edges")
    return "\n".join(msg)


# ---------------------------------------------------------------- layer 1c: simulate + count

class NkiMissing(RuntimeError):
    pass


def itemsize_of(obj, default=4):
    """Bytes per element, without guessing.

    The first cluster run reported exactly half the true byte count for every matmul level,
    because the fallback here assumed 2 bytes while the test inputs are float32. Half the bytes
    doubles the arithmetic intensity, which would have turned a memory-bound kernel into a
    plausible-looking compute-bound one. So resolve the dtype properly and only fall back as a
    last resort.
    """
    dt = getattr(obj, "dtype", None)
    for candidate in (dt, str(dt) if dt is not None else None):
        if candidate is None:
            continue
        got = getattr(candidate, "itemsize", None)
        if isinstance(got, int) and got > 0:
            return got
        try:
            return int(np.dtype(candidate).itemsize)
        except Exception:
            pass
    name = str(dt).lower().replace("nki.", "") if dt is not None else ""
    for key, size in DTYPE_BYTES.items():
        if key in name:
            return size
    return default


def _simulator(nki_mod, kernel):
    """Return a callable that runs `kernel` on the CPU.

    The API moved. nki 0.6.0 exposes `nki.simulate(kernel)(*args)`, while the tutorial examples
    and older docs use `nki.simulate_kernel(kernel, *args)`. Support both rather than pinning a
    version, and say clearly which one was found -- this exact mismatch is what the first
    cluster run caught, with every reference kernel reporting "module 'nki' has no attribute
    'simulate_kernel'".
    """
    if hasattr(nki_mod, "simulate"):
        return lambda *args: nki_mod.simulate(kernel)(*args), "nki.simulate"
    if hasattr(nki_mod, "simulate_kernel"):
        return lambda *args: nki_mod.simulate_kernel(kernel, *args), "nki.simulate_kernel"
    raise NkiMissing(
        f"this nki ({getattr(nki_mod, '__version__', 'unknown')}) exposes neither "
        f"`simulate` nor `simulate_kernel`. Available: "
        f"{sorted(n for n in dir(nki_mod) if not n.startswith('_'))}")


def minimum_hbm_bytes(args, want):
    """The least HBM traffic the operation can possibly need: read each input once, write the
    output once. Anything above this is re-reading, and re-reading is the whole subject of
    levels 4 to 7."""
    total = sum(int(np.prod(a.shape)) * itemsize_of(a) for a in args
                if hasattr(a, "shape"))
    return total + int(np.prod(np.shape(want))) * itemsize_of(np.asarray(want))


def reuse_report(counted, args, want):
    """Name redundant HBM traffic directly, instead of leaving it implied by the intensity.

    A tiled matmul that allocates its PSUM tile INSIDE the contraction loop writes each partial
    product out and reads it back, instead of letting nc_matmul accumulate in PSUM across the whole
    loop. The arithmetic intensity drops, but 'memory bound' does not say why. This does.
    """
    floor = minimum_hbm_bytes(args, want)
    if not floor or not counted.get("bytes"):
        return ""
    ratio = counted["bytes"] / floor
    if ratio < 1.15:
        return (f"    HBM traffic is {counted['bytes']:,} bytes against a floor of {floor:,} "
                f"({ratio:.2f}x) -- essentially optimal. Every byte is read about once.")
    return (f"    HBM traffic is {counted['bytes']:,} bytes against a floor of {floor:,}, so "
            f"{ratio:.1f}x MORE THAN NECESSARY. The same bytes are crossing the bus repeatedly. "
            f"For a tiled matmul the usual cause is allocating the PSUM tile inside the "
            f"contraction loop and copying each partial product out: allocate one PSUM tile "
            f"OUTSIDE that loop and let nisa.nc_matmul accumulate into it across every step, then "
            f"copy out once. Hoisting the operand loads out of the innermost loop cuts the rest.")


def check_traffic_bar(level_n, counted, args, want):
    """Levels 5 to 7 must MOVE FEWER BYTES, not merely be correct.

    Returns None if the bar is met or there is no bar. Otherwise a message that says how far over it
    is and what buys the difference -- which is the only thing separating these levels from level 4.
    """
    bar = LEVELS[level_n].get("max_waste")
    if not bar or not counted.get("bytes"):
        return None
    floor = minimum_hbm_bytes(args, want)
    if not floor:
        return None
    waste = counted["bytes"] / floor
    if waste <= bar:
        return None
    hint = {5: "Hoist the operand loads out of the innermost loop; the same tiles are being re-read "
               "on every pass.",
            6: "Block the M and N dimensions so more tiles stay resident in SBUF and are reused "
               "across iterations, rather than reloaded.",
            7: "Block K as well, and accumulate in one PSUM tile across the whole contraction so "
               "nothing partial is written out."}.get(level_n, "")
    return (f"CORRECT, BUT TOO MUCH HBM TRAFFIC FOR THIS LEVEL: moving {waste:.2f}x the byte floor, "
            f"and level {level_n} requires {bar:.2f}x or better. Correctness alone is level 4; this "
            f"level is about the bytes. {hint}")


def check_inputs_untouched(before, args):
    """A kernel must not write into the tensor it was given.

    The first solved level did exactly that -- dma_copy(dst=x, ...) then return x -- and passed,
    because the reference happens to run before the kernel. That is luck, not correctness: in a real
    graph the caller still owns that buffer.
    """
    for i, (orig, now) in enumerate(zip(before, args)):
        if isinstance(orig, np.ndarray) and not np.array_equal(orig, np.asarray(now)):
            return (f"THE KERNEL MODIFIED ITS INPUT (argument {i}). Allocate a new output with "
                    f"nl.ndarray(shape, dtype=..., buffer=nl.shared_hbm), write the result there, "
                    f"and return that. The input tensor belongs to the caller.")
    return None


def simulate_and_count(kernel, args):
    """Run the kernel on the CPU and count the HBM traffic it asked for.

    NEEDS THE NEURON SDK. The byte count comes from wrapping `nisa.dma_copy` for the duration
    of the simulation, which is what makes an arithmetic intensity measurable without a
    device: flops come from the shapes, bytes come from the copies the kernel actually issued.
    """
    try:
        import nki
        import nki.isa as nisa
    except ImportError as e:
        raise NkiMissing(
            "the Neuron SDK is not installed here, so the kernel cannot be simulated. The "
            "rule check and the roofline arithmetic above still ran. Use the instance for "
            "this layer."
        ) from e

    run, api = _simulator(nki, kernel)
    counter = dict(bytes=0, transfers=0, api=api, dtypes=set())
    original = nisa.dma_copy

    def tensor_bytes(t):
        nbytes = getattr(t, "nbytes", None)
        if not isinstance(nbytes, int) or nbytes <= 0:
            nbytes = int(np.prod(t.shape)) * itemsize_of(t)
        return int(nbytes)

    def counting_dma_copy(dst=None, src=None, **kw):
        try:
            # A DMA can convert dtypes. Counting only src overstated a float32 SBUF -> bf16 HBM store
            # by 2x: measured on the chipboost expert matmul, which read "1.29x the floor" while
            # moving exactly the floor. The HBM side of a converting copy is the narrower one here,
            # since inputs and outputs are bf16.
            nbytes = tensor_bytes(src)
            if dst is not None:
                nbytes = min(nbytes, tensor_bytes(dst))
            counter["bytes"] += nbytes
            counter["dtypes"].add(str(getattr(src, "dtype", "?")))
        except Exception:
            counter["unmeasured"] = counter.get("unmeasured", 0) + 1
        counter["transfers"] += 1
        return original(dst=dst, src=src, **kw)

    # Capture warnings instead of letting them print. The simulator emits one per traced
    # instruction, which meant dozens of identical UserWarning blocks per round drowning the log.
    # They are also not noise: one of them says the pattern produces INCORRECT RESULTS on hardware,
    # which the agent should be told rather than have scrolled past.
    import warnings
    nisa.dma_copy = counting_dma_copy
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = run(*args)
        seen = []
        for w in caught:
            msg = str(w.message).split(". ")[0]
            if msg not in seen:
                seen.append(msg)
        counter["warnings"] = seen[:3]
    finally:
        nisa.dma_copy = original
    return out, counter


def load_kernel(path, entry):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, entry):
        raise AttributeError(f"{path} defines no `{entry}`")
    return getattr(mod, entry)


# ---------------------------------------------------------------- verify

def verify(path, level_n, tol=2e-2, seed=0):
    spec = LEVELS[level_n]
    src = open(path).read()

    violations = check_rules(src, level_n)
    print(f"level {level_n}: {spec['op']}")
    if violations:
        print("\nRULE VIOLATIONS -- this scores zero regardless of speed or correctness:")
        for v in violations:
            print(f"  {v}")
        return 2
    print("  rules      clean")

    try:
        kernel = load_kernel(path, spec["entry"])
    except Exception as e:
        print(f"  FAILED to import: {type(e).__name__}: {e}")
        return 2

    passed, failures, intensities = 0, [], []
    for case in spec["shapes"]:
        args, _ = make_inputs(case, level_n, seed)
        want = spec["ref"](*args)
        try:
            got, counted = simulate_and_count(kernel, args)
        except NkiMissing as e:
            print(f"\n  simulation  SKIPPED: {e}")
            return 3
        except Exception as e:
            failures.append((label(case, level_n),
                             f"RAISED during simulation: {type(e).__name__}: {e}"))
            continue
        m = describe_mismatch(got, want, tol)
        if m:
            failures.append((label(case, level_n), m))
            continue
        passed += 1
        elements = int(np.prod(np.shape(want)))
        if counted["transfers"] > max(8, elements // 64):
            print(f"\n  case {label(case, level_n)}: CORRECT BUT ISSUE-BOUND -- "
                  f"{counted['transfers']:,} transfers for {elements:,} output elements, "
                  f"{counted['bytes'] / max(counted['transfers'], 1):.0f} bytes each. The cost here "
                  f"is the NUMBER of transfers, not the bytes. Move whole tiles, not elements.")
        if counted.get("bytes"):
            rep = reuse_report(counted, args, want)
            if rep:
                print(f"\n  case {label(case, level_n)}:")
                print(rep)
        if level_n >= 3 and counted["bytes"] and {"M", "K", "N"} <= set(case):
            f = matmul_flops(case["M"], case["K"], case["N"])
            intensities.append((label(case, level_n), roofline(f, counted["bytes"]),
                                counted, f, minimum_hbm_bytes(args, want)))

    print(f"  numerics   {passed}/{len(spec['shapes'])} shapes passed")
    for lbl, m in failures[:3]:
        print(f"\n  case {lbl}:")
        print(textwrap.indent(m, "    "))
    if failures:
        print("\n  ^ feed this text back to the model. If it is not enough to locate the bug,")
        print("    improve THIS message before you touch the prompt.")
        return 1

    for lbl, r, counted, flops, floor in intensities:
        dts = ",".join(sorted(counted["dtypes"])) or "?"
        print(f"\n  {lbl}: {counted['bytes']:,} HBM bytes in {counted['transfers']} transfers "
              f"(via {counted['api']}, dtype {dts})")
        if counted.get("unmeasured"):
            print(f"    WARNING: {counted['unmeasured']} transfers could not be sized, so the "
                  f"byte count is a LOWER bound and the intensity an UPPER one.")
        if not any(("bfloat16" in d) or ("bf16" in d) for d in counted["dtypes"]):
            print(f"    NOTE: the {RIDGE_FLOPS_PER_BYTE['bfloat16']:g} Flops/Byte ridge is the "
                  f"published bfloat16 figure, and this ran in {dts}. Treat the verdict as "
                  f"indicative and re-measure in bfloat16 before quoting a number.")
        print(explain_with_ceiling(flops, counted["bytes"], floor))
    return 0


# ---------------------------------------------------------------- selftest

def selftest():
    print("Proving the parts that need no device. Run this on the instance too, for the rest.\n")
    rc = 0

    # The check that matters: reproduce the tutorial's own published numbers.
    per_iter_bytes = (PMAX * GEMM_STATIONARY_FMAX + PMAX * GEMM_MOVING_FMAX) * 2
    per_iter_flops = matmul_flops(GEMM_STATIONARY_FMAX, PMAX, GEMM_MOVING_FMAX)
    ai = per_iter_flops / per_iter_bytes
    ok = abs(per_iter_bytes - 163840) == 0 and abs(ai - 102.4) < 0.1
    print(f"  roofline model vs the tutorial     -> {per_iter_bytes / 1024:.0f} KB per "
          f"{per_iter_flops / 1e6:.1f} MFlops = {ai:.1f} Flops/Byte "
          f"{'(matches the published 160 KB / 16 MFlops / 102)' if ok else 'FAIL'}")
    rc |= 0 if ok else 1

    r = roofline(per_iter_flops, per_iter_bytes)
    ok = r["bound"] == "memory_bound"
    print(f"  the tiled matmul is classified     -> {r['bound']} "
          f"{'ok' if ok else 'FAIL'}  ({r['headroom']})")
    rc |= 0 if ok else 1
    r2 = roofline(per_iter_flops * 4, per_iter_bytes)
    ok = r2["bound"] == "compute_bound"
    print(f"  4x the reuse is classified         -> {r2['bound']} {'ok' if ok else 'FAIL'}")
    rc |= 0 if ok else 1

    # References must be right, since everything is graded against them.
    print()
    x = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    ok = np.allclose(ref_avgpool2d(x, 2),
                     x.reshape(2, 2, 2, 2, 2).mean(axis=(2, 4)))
    print(f"  ref_avgpool2d                      -> {'ok' if ok else 'FAIL'}")
    rc |= 0 if ok else 1
    # The tutorial's own worked example: 3 rows of 4 becomes 4 rows of 3, per partition.
    y = np.array([[0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]], np.float32)
    want = np.array([[0, 10, 20, 1, 11, 21, 2, 12, 22, 3, 13, 23]], np.float32)
    ok = np.array_equal(ref_transpose2d(y, (3, 4)), want)
    print(f"  ref_transpose2d (tutorial example) -> {'ok' if ok else 'FAIL'}")
    rc |= 0 if ok else 1
    lhsT = np.random.default_rng(1).standard_normal((4, 3)).astype(np.float32)
    rhs = np.random.default_rng(2).standard_normal((4, 5)).astype(np.float32)
    ok = np.allclose(ref_matmul(lhsT, rhs), lhsT.T @ rhs, atol=1e-5)
    print(f"  ref_matmul (transposed left)       -> {'ok' if ok else 'FAIL'}")
    rc |= 0 if ok else 1

    # The rule checker has to catch the cheats and accept a plausible kernel.
    print()
    cheats = [
        (4, "import nki\n@nki.jit\ndef nki_matmul_tiled_(a, b):\n    return a.T @ b\n",
         "the @ operator"),
        (2, "import nki\n@nki.jit\ndef tensor_transpose2D_kernel_(in_tensor, shape2D):\n"
            "    return in_tensor.T\n", ".T on an argument"),
        (1, "import nki\nimport numpy as np\n@nki.jit\n"
            "def tensor_avgpool_kernel(x, p):\n    return np.mean(x, axis=(1, 2))\n",
         "np.mean"),
        (4, "import nki\nimport nki.language as nl\n@nki.jit\n"
            "def nki_matmul_tiled_(a, b):\n"
            "    t = nl.ndarray((256, 512), buffer=nl.sbuf)\n    return t\n",
         "a 256-partition tile"),
        (4, "def nki_matmul_tiled_(a, b):\n    return None\n", "a missing @nki.jit"),
        (4, "import nki\n@nki.jit\ndef wrong_name(a, b):\n    return None\n",
         "the wrong entry-point name"),
    ]
    for n, src, what in cheats:
        v = check_rules(src, n)
        print(f"  catches {what:<28}   -> {'caught: ' + v[0][:60] if v else 'UNDETECTED'}")
        rc |= 0 if v else 1

    plausible = ("import nki\nimport nki.isa as nisa\nimport nki.language as nl\n"
                 "@nki.jit\ndef nki_matmul_tiled_(lhsT, rhs):\n"
                 "    K, M = lhsT.shape\n    K_, N = rhs.shape\n"
                 "    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)\n"
                 "    for m in nl.affine_range(M // 128):\n"
                 "        ps = nl.ndarray((128, 512), nl.float32, buffer=nl.psum)\n"
                 "        t = nl.ndarray((128, 128), dtype=lhsT.dtype, buffer=nl.sbuf)\n"
                 "        nisa.dma_copy(dst=t, src=lhsT[0:128, 0:128])\n"
                 "        nisa.nc_matmul(dst=ps, stationary=t, moving=t)\n"
                 "    return result\n")
    v = check_rules(plausible, 4)
    print(f"  accepts a plausible kernel         -> {'ok' if not v else 'FAIL: ' + str(v)}")
    rc |= 0 if not v else 1

    # Byte sizing. A 2x error here silently doubles every arithmetic intensity and can turn a
    # memory-bound kernel into a plausible-looking compute-bound one -- which is exactly what
    # the first cluster run did before itemsize_of existed.
    print()
    class _Fake:
        def __init__(self, dtype):
            self.dtype, self.shape = dtype, (128, 512)
    cases = [("a numpy float32 array", np.zeros(8, np.float32), 4),
             ("dtype as an object", _Fake(np.dtype("float32")), 4),
             ("dtype as a string", _Fake("float32"), 4),
             ("an nki-style name", _Fake("nki.bfloat16"), 2)]
    for what, obj, want in cases:
        got = itemsize_of(obj)
        ok = got == want
        rc |= 0 if ok else 1
        print(f"  itemsize of {what:<24} -> {got} bytes {'ok' if ok else 'FAIL want ' + str(want)}")
    lhsT = np.zeros((128, 128), np.float32)
    rhs = np.zeros((128, 512), np.float32)
    want_bytes = lhsT.nbytes + rhs.nbytes + 128 * 512 * 4
    print(f"  level 3 should move {want_bytes:,} bytes, giving an intensity of "
          f"{matmul_flops(64, 128, 512) / want_bytes:.1f} in float32")

    # Failure messages have to localise, not just complain.
    print()
    want = np.zeros((300, 1100), np.float32) + 1.0
    got = want.copy()
    got[200, 900] = 5.0
    m = describe_mismatch(got, want)
    ok = m and "index (200, 900)" in m
    print(f"  mismatch names the element         -> {'ok' if ok else 'FAIL'}")
    rc |= 0 if ok else 1
    # Genuinely ragged: 300 rows is 2 full partition tiles plus 44, and 1100 columns is
    # 2 full free tiles plus 76. The error is placed in that tail.
    got2 = want.copy()
    got2[256:, 1024:] += 3.0
    m2 = describe_mismatch(got2, want)
    ok = m2 and "ragged edge" in m2
    print(f"  mismatch spots the ragged edge     -> {'ok' if ok else 'FAIL'}")
    rc |= 0 if ok else 1
    ok = describe_mismatch(want, want) is None
    print(f"  a correct result passes            -> {'ok' if ok else 'FAIL'}")
    rc |= 0 if ok else 1
    m3 = describe_mismatch(np.zeros((4, 4)), np.zeros((4, 5)))
    ok = "WRONG SHAPE" in m3
    print(f"  wrong shape is named as such       -> {'ok' if ok else 'FAIL'}")
    rc |= 0 if ok else 1

    print()
    try:
        import nki  # noqa: F401
        api = ("nki.simulate" if hasattr(nki, "simulate")
               else "nki.simulate_kernel" if hasattr(nki, "simulate_kernel") else None)
        print(f"  nki {getattr(nki, '__version__', '?')} is importable; simulation API: "
              f"{api or 'NEITHER — see _simulator()'}")
        rc |= 0 if api else 1
    except ImportError:
        print("  NEEDS DEVICE VERIFICATION: simulate_and_count() imports nki and cannot run")
        print("    here. Its byte counting wraps nisa.dma_copy, which is unverified until")
        print("    someone runs this on the instance. Everything above is verified.")

    print("\nSELFTEST " + ("PASSED" if rc == 0 else "FAILED"))
    return rc


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--level", type=int)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--check", metavar="FILE.py")
    ap.add_argument("--roofline", nargs=3, type=int, metavar=("M", "K", "N"),
                    help="what the tiled matmul's roofline says for this shape")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=sorted(RIDGE_FLOPS_PER_BYTE))
    ap.add_argument("--tol", type=float, default=None,
                    help="numerical tolerance; default is the level's own, else 2e-2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())

    if a.roofline:
        M, K, N = a.roofline
        f = matmul_flops(M, K, N)
        b = matmul_tiled_hbm_bytes(M, K, N, a.dtype)
        print(f"tiled matmul, M={M} K={K} N={N}, {a.dtype}")
        print(f"  {f / 1e9:.2f} GFlop, {b / 1024 ** 2:.1f} MB read from HBM")
        print(textwrap.indent(explain_roofline(roofline(f, b, a.dtype)), "  "))
        return

    if a.level and a.show:
        s = LEVELS[a.level]
        print(f"level {a.level}: {s['op']}")
        print(f"  entry point   {s['entry']}(...)  decorated @nki.jit")
        print(f"\n  teaches       {textwrap.fill(s['teaches'], 84, subsequent_indent='                ')}")
        print(f"\n  optimization  {textwrap.fill(s['optimization'], 84, subsequent_indent='                ')}")
        print(f"\n  reference     {s['ref'].__name__} in this file -- read it")
        print(f"  banned here   {sorted(s['banned'])}")
        print(f"  shapes        {len(s['shapes'])} cases: "
              f"{', '.join(label(c, a.level) for c in s['shapes'])}")
        print(f"\n  tile limits   partition <= {PMAX}, gemm stationary free <= "
              f"{GEMM_STATIONARY_FMAX}, gemm moving free <= {GEMM_MOVING_FMAX}")
        return

    if a.level and a.check:
        own = LEVELS[a.level].get("tol")
        tol = a.tol if a.tol is not None else (own if own is not None else 2e-2)
        sys.exit(verify(a.check, a.level, tol, a.seed))

    print("THE LEVELS — difficulty and optimization headroom rise together.\n")
    for n, s in LEVELS.items():
        tier = ("A correctness" if n <= 2 else "B roofline" if n <= 4
                else "C search" if n <= 7 else "D your own" if n == 8 else "E chipboost")
        bar = s.get("max_waste")
        print(f"  {n}. [{tier:<13}] {s['op']}"
              + (f"   (needs HBM traffic <= {bar:.2f}x the floor)" if bar else ""))
    print(f"\n  ridge point: {RIDGE_FLOPS_PER_BYTE['bfloat16']:g} Flops/Byte for bfloat16 on "
          f"NeuronCore-v2.\n  Below it a kernel is memory bound; above it, compute bound.")
    print("\n  Levels 5-7 are graded on BYTES, not only correctness: the reference and shapes are")
    print("  identical to level 4, so a traffic bar is what makes them levels at all.")
    print("  Level 8 is a worked example of adding your own operation -- see the README.")
    print("\n  python nkibench.py --level 4 --show")
    print("  python nkibench.py --level 4 --check my_matmul.py")
    print("  python nkibench.py --roofline 512 1024 2048")
    print("  python nkibench.py --selftest")


if __name__ == "__main__":
    main()
