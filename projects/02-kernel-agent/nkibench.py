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
import os
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
# On-chip capacity per partition. PSUM is 8 banks of 2 KiB; SBUF is 24 MiB over 128 partitions
# (trn2). A tile spanning several PSUM banks is legal -- the NKI docs allocate all eight at
# once -- so the PSUM limit is the whole of PSUM, not one bank.
PSUM_PARTITION_BYTES = 16 * 1024
SBUF_PARTITION_BYTES = 192 * 1024

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
          make_args=None, label=None):
    """max_waste: the most HBM traffic this level may move, as a multiple of the byte floor.

    Levels 5 to 7 are optimization levels, so correctness alone cannot tell them apart from level 4 --
    the reference and the shapes are identical, so a level-4 kernel would score 1.0 on all three and
    the ladder would stop meaning anything past 4. A traffic bar is what makes them levels.

    The thresholds come from the shipped tiled kernel, which models at 2.00x the floor on the largest
    test shape: level 5 must beat that, level 6 must beat level 5, level 7 must be near the floor.
    """
    LEVELS[n] = dict(n=n, op=op, entry=entry, teaches=teaches, optimization=optimization,
                     ref=ref, shapes=shapes, banned=banned, notes=notes, max_waste=max_waste,
                     make_args=make_args or _args_matmul,
                     label=label or (lambda sp: f"K={sp['K']} M={sp['M']} N={sp['N']}"))


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


# ---------------------------------------------------------------- layer 1d: allocation audit
#
# The simulator enforces the 128-partition limit on dma_copy and tensor_copy, not on allocation.
# Observed on another seat (Yahtze/trainium-agent-labs, branch kernel-feedback): a kernel
# allocated a (256, 256) sbuf tile and a (512, 1024) psum tile, sliced them for nc_matmul, and
# ran -- a kernel the chip cannot execute, which would have scored 1.0 had it been numerically
# right. So every on-chip allocation is recorded during simulation and judged against the limits.
#
# The hook goes in BEFORE the kernel file is imported (load_kernel does it): nki.jit may bind the
# API when it decorates, so a patch applied afterwards would never be seen. It stays installed;
# outside a simulation nothing here calls nl.ndarray. NKIBENCH_NO_ALLOC_AUDIT=1 turns it off.

_ALLOC_LOG = []
_AUDIT = {"installed": False}
# where dtype and buffer sit positionally: ndarray/zeros/ones(shape, dtype, buffer),
# full(shape, fill_value, dtype, buffer). All default to sbuf. *_like is not audited: its buffer
# defaults to its argument's, which the wrapper cannot see reliably.
_ALLOCATORS = {"ndarray": (1, 2), "zeros": (1, 2), "ones": (1, 2), "full": (2, 3)}


def _buffer_name(buf, nl):
    for name in ("sbuf", "psum", "shared_hbm", "private_hbm", "hbm"):
        if buf is getattr(nl, name, object()):
            return "hbm" if "hbm" in name else name
    s = str(getattr(buf, "name", buf)).lower()
    return next((n for n in ("sbuf", "psum", "hbm") if n in s), "")


def _install_alloc_audit():
    if _AUDIT["installed"] or os.environ.get("NKIBENCH_NO_ALLOC_AUDIT"):
        return
    try:
        import nki.language as nl
    except ImportError:
        return

    def wrap(fn_name, original, dtype_at, buffer_at):
        def auditing(*args, **kw):
            try:
                shape = kw.get("shape", args[0] if args else None)
                dtype = kw.get("dtype", args[dtype_at] if len(args) > dtype_at else None)
                buf = kw.get("buffer", args[buffer_at] if len(args) > buffer_at else nl.sbuf)
                _ALLOC_LOG.append(dict(fn=fn_name, shape=tuple(int(s) for s in shape),
                                       itemsize=itemsize_of(type("_D", (), {"dtype": dtype})()),
                                       buffer=_buffer_name(buf, nl)))
            except Exception:
                _ALLOC_LOG.append(dict(fn=fn_name, shape=None, itemsize=0, buffer="?"))
            return original(*args, **kw)
        auditing.__wrapped__ = original
        return auditing

    for fn_name, (dtype_at, buffer_at) in _ALLOCATORS.items():
        original = getattr(nl, fn_name, None)
        if original is None:
            continue
        audited = wrap(fn_name, original, dtype_at, buffer_at)
        setattr(nl, fn_name, audited)
        # also where it is defined, in case the decorator resolves names there
        home = sys.modules.get(getattr(original, "__module__", "") or "")
        if home is not None and getattr(home, fn_name, None) is original:
            setattr(home, fn_name, audited)
    _AUDIT["installed"] = True


def illegal_allocations(log):
    """Distinct allocations the chip would refuse, as sentences. A tile allocated inside a loop is
    logged once per iteration, so each distinct one is reported once."""
    bad = []
    for a in log:
        shape, buf = a.get("shape"), a.get("buffer")
        if not shape or buf not in ("sbuf", "psum"):
            continue
        if shape[0] > PMAX:
            bad.append(f"a {buf} tile of shape {shape}, whose partition dimension (the first) is "
                       f"{shape[0]} where the maximum is {PMAX}")
            continue
        per_partition = int(np.prod(shape[1:], dtype=np.int64)) * (a.get("itemsize") or 4)
        cap = PSUM_PARTITION_BYTES if buf == "psum" else SBUF_PARTITION_BYTES
        if per_partition > cap:
            bad.append(f"a {buf} tile of shape {shape}, which needs {per_partition:,} bytes per "
                       f"partition where {buf} holds {cap:,}")
    return list(dict.fromkeys(bad))


def describe_illegal(counted):
    """The simulator ran it and the chip would refuse it. None when every allocation fits."""
    bad = counted.get("illegal") or []
    if not bad:
        return None
    return ("ILLEGAL ON HARDWARE: the CPU simulator ran this kernel, but it does not check tile "
            "limits when a tile is allocated and the chip does. The kernel allocated "
            + "; and ".join(bad[:3]) + ". Do not allocate one on-chip tile for a whole operand or "
            "the whole result: loop over the rows in chunks of at most 128 and allocate each tile "
            "inside the loop with the chunk's own shape.")


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
    counter = dict(bytes=0, transfers=0, api=api, dtypes=set(), illegal=[], allocations=0)
    original = nisa.dma_copy
    _install_alloc_audit()
    del _ALLOC_LOG[:]

    def counting_dma_copy(dst=None, src=None, **kw):
        try:
            nbytes = getattr(src, "nbytes", None)
            if not isinstance(nbytes, int) or nbytes <= 0:
                nbytes = int(np.prod(src.shape)) * itemsize_of(src)
            counter["bytes"] += int(nbytes)
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
        counter["allocations"] = len(_ALLOC_LOG)
        counter["illegal"] = illegal_allocations(_ALLOC_LOG)
    return out, counter


_LOADED_PATHS = set()
_CANDIDATE_IDS = iter(range(1, 10 ** 9))


def candidate_path(stem):
    """A path no kernel in this process has been loaded from. nki 0.6.0 caches by file path: a
    second candidate of the SAME BYTE SIZE written to the same path is simulated as the FIRST one
    -- numerics and allocations both. Measured: the level-4 reference with its nc_matmul operands
    swapped (same length, raises when run alone) scored 4/4 when loaded from the path the reference
    had just used. Pure Python's import cache does not do this; nki does. Unique per process and
    per call, so agents sharing a pod do not collide either. Any 1.0 from a run on older code must
    be re-graded in a fresh process: scripts/reaudit.py."""
    return f"/tmp/{stem}_{os.getpid()}_{next(_CANDIDATE_IDS)}.py"


def load_kernel(path, entry):
    _install_alloc_audit()        # must precede the import; see layer 1d
    if path in _LOADED_PATHS:
        print(f"nkibench WARNING: {path} was already loaded in this process; nki caches by path, "
              f"so the allocation audit may report the earlier kernel. Use candidate_path().",
              file=sys.stderr)
    _LOADED_PATHS.add(path)
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

    passed, failures, intensities, audited = 0, [], [], []
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
        audited.append(counted.get("allocations", 0))
        m = describe_illegal(counted) or describe_mismatch(got, want, tol)
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
        if level_n >= 3 and counted["bytes"]:
            f = matmul_flops(case["M"], case["K"], case["N"])
            intensities.append((label(case, level_n), roofline(f, counted["bytes"]),
                                counted, f, minimum_hbm_bytes(args, want)))

    print(f"  numerics   {passed}/{len(spec['shapes'])} shapes passed")
    if not _AUDIT["installed"]:
        print("  allocation audit OFF (NKIBENCH_NO_ALLOC_AUDIT set, or nki missing)")
    elif audited and not any(audited):
        # every kernel allocates at least its result, so silence means the hook saw nothing
        print("  allocation audit saw NO allocations -- the hook is not seeing nl.ndarray, so tile "
              "limits are UNCHECKED here")
    elif audited:
        print(f"  allocation audit {sum(audited):,} allocations across {len(audited)} shapes")
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


# ---------------------------------------------------------------- held-out eval set
#
# The loop only ever sees LEVELS[n]["shapes"] filled with standard-normal data from seed 0. The
# judges grade on shapes and values the agent never saw, so passing the loop's shapes is evidence,
# not proof. This set is what the agent checks its own claim against, ONCE, after the loop. Nothing
# here is ever fed back to the model, or it would stop being held out.
#
# Shapes: new ones, inside each level's contract -- the shipped reference kernel passes every one
# (python nkibench.py --level N --eval reference_levelN.py). Level 3's reference asserts
# K=128 M=64 N=512, so its contract is that one shape and only the values change. Shapes outside a
# reference's contract (M not a multiple of 128 on level 4, say) are left out: the reference itself
# rejects them, so failing them would say nothing about the agent.

EVAL_SHAPES = {
    1: [dict(shape=(128, 32, 32), pool_size=2), dict(shape=(16, 30, 30), pool_size=5),
        dict(shape=(1, 64, 64), pool_size=8), dict(shape=(96, 20, 12), pool_size=4),
        dict(shape=(7, 10, 11), pool_size=3)],
    2: [dict(shape=(128, 128), shape2D=(16, 8)), dict(shape=(1, 24), shape2D=(4, 6)),
        dict(shape=(100, 60), shape2D=(6, 10)), dict(shape=(17, 77), shape2D=(7, 11))],
    3: [dict(K=128, M=64, N=512)],
    4: [dict(K=384, M=256, N=512), dict(K=128, M=384, N=1536),
        dict(K=640, M=128, N=1024), dict(K=128, M=128, N=2048)],
}
for _n in (5, 6, 7):
    EVAL_SHAPES[_n] = EVAL_SHAPES[4]

# Values, each chosen to expose a wrong kernel that standard-normal data lets through.
VALUE_KINDS = {
    "normal": "standard normal from a seed the loop never used",
    "ramp": "every element distinct and ordered (linspace -1..1), so an element read from or "
            "written to the wrong place cannot cancel out",
    "large": "standard normal x 1e4: a kernel computing in float16 overflows (max 65504)",
    "float16": "inputs arrive as float16: a kernel that hard-codes float32 returns the wrong dtype "
               "(the simulator casts silently in dma_copy, so only the dtype check catches it)",
}
EVAL_SEED = 1000


def make_eval_inputs(case, level_n, kind):
    args, _ = make_inputs(case, level_n, EVAL_SEED)
    out = []
    for a in args:
        if isinstance(a, np.ndarray) and a.dtype.kind == "f":
            if kind == "ramp":
                a = np.linspace(-1, 1, a.size, dtype=np.float32).reshape(a.shape)
            elif kind == "large":
                a = a * np.float32(1e4)
            elif kind == "float16":
                a = a.astype(np.float16)
        out.append(a)
    return tuple(out)


def check_case(kernel, args, level_n, tol=2e-2):
    """Run one case through every check the loop applies. Returns None if it passes, else why."""
    want = LEVELS[level_n]["ref"](*args)
    before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
    try:
        got, counted = simulate_and_count(kernel, args)
    except NkiMissing:
        raise
    except Exception as e:
        return f"raised {type(e).__name__}: {str(e)[:300]}"
    # Eval only, so the loop's grading stays comparable with the baseline: every reference returns
    # its input's dtype, and the simulator's dma_copy casts silently, so without this a kernel that
    # hard-codes float32 passes float16 inputs.
    wrong_dtype = (f"WRONG DTYPE: returned {np.asarray(got).dtype}, reference returns "
                   f"{np.asarray(want).dtype}" if np.asarray(got).dtype != np.asarray(want).dtype
                   else None)
    m = (check_inputs_untouched(before, args) or describe_illegal(counted) or wrong_dtype
         or describe_mismatch(got, want, tol) or check_traffic_bar(level_n, counted, args, want))
    if not m and any("incorrect results on hardware" in w for w in counted.get("warnings", [])):
        m = "CORRECT ON CPU BUT WRONG ON HARDWARE"
    return m


def evaluate(kernel, level_n, tol=2e-2):
    """The held-out set: every EVAL_SHAPES case with every VALUE_KINDS fill.
    Returns a list of dict(case, kind, ok, why)."""
    results = []
    for case in EVAL_SHAPES.get(level_n, []):
        for kind in VALUE_KINDS:
            why = check_case(kernel, make_eval_inputs(case, level_n, kind), level_n, tol)
            results.append(dict(case=label(case, level_n), kind=kind, ok=why is None,
                                why=(why or "")[:300]))
    return results


def run_eval(path, level_n, tol=2e-2):
    spec = LEVELS[level_n]
    if level_n not in EVAL_SHAPES:
        print(f"level {level_n} has no held-out eval set")
        return 3
    violations = check_rules(open(path).read(), level_n)
    if violations:
        print("RULE VIOLATIONS: " + " ".join(violations))
        return 2
    kernel = load_kernel(path, spec["entry"])
    try:
        res = evaluate(kernel, level_n, tol)
    except NkiMissing as e:
        print(f"SKIPPED: {e}")
        return 3
    print(f"level {level_n} held-out eval: {sum(r['ok'] for r in res)}/{len(res)} cases passed "
          f"({len(EVAL_SHAPES[level_n])} shapes x {len(VALUE_KINDS)} value kinds)")
    for r in res:
        print(f"  {'pass' if r['ok'] else 'FAIL'}  {r['case']:<28} {r['kind']:<8} {r['why'][:120]}")
    return 0 if all(r["ok"] for r in res) else 1


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

    # The allocation audit: the judgement is pure, so it is proven here; the hook that feeds it
    # needs nki (see the end of this selftest).
    print()
    def alloc(shape, buffer, itemsize=4):
        return dict(fn="ndarray", shape=shape, buffer=buffer, itemsize=itemsize)
    reference_l4 = [alloc((256, 1024), "hbm"), alloc((128, 512), "psum"),
                    alloc((128, 128), "sbuf"), alloc((128, 512), "sbuf")]
    audit_cases = [
        ("the level-4 reference's tiles", reference_l4, 0),
        ("a (256, 256) sbuf tile", [alloc((256, 256), "sbuf")], 1),
        ("a (512, 1024) psum tile", [alloc((512, 1024), "psum")], 1),
        ("all 8 psum banks, (128, 4096) f32", [alloc((128, 4096), "psum")], 0),
        ("more than all of psum", [alloc((128, 8192), "psum")], 1),
        ("a huge hbm tensor", [alloc((4096, 4096), "hbm")], 0),
        ("one bad tile in a loop, 4 times", [alloc((256, 256), "sbuf")] * 4, 1),
    ]
    for what, log, want_n in audit_cases:
        got_n = len(illegal_allocations(log))
        ok = got_n == want_n
        rc |= 0 if ok else 1
        print(f"  audit: {what:<34} -> {got_n} illegal {'ok' if ok else 'FAIL want ' + str(want_n)}")
    m = describe_illegal(dict(illegal=illegal_allocations([alloc((256, 256), "sbuf")])))
    ok = bool(m) and m.startswith("ILLEGAL ON HARDWARE") and "(256, 256)" in m
    print(f"  audit message names the tile       -> {'ok' if ok else 'FAIL'}")
    rc |= 0 if ok else 1

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
        _install_alloc_audit()
        import nki.language as nl
        hooked = [n for n in _ALLOCATORS if hasattr(getattr(nl, n, None), "__wrapped__")]
        print(f"  allocation audit hooked onto       -> nl.{', nl.'.join(hooked) or 'NOTHING'}"
              f"{'' if hooked else ' FAIL'}")
        print("    whether nki.jit kernels go through that hook is proven only by --check on a "
              "reference kernel: look for 'allocation audit N allocations'.")
        rc |= 0 if hooked or os.environ.get("NKIBENCH_NO_ALLOC_AUDIT") else 1
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
    ap.add_argument("--eval", metavar="FILE.py",
                    help="run the held-out eval set (new shapes, hostile values) on a kernel")
    ap.add_argument("--roofline", nargs=3, type=int, metavar=("M", "K", "N"),
                    help="what the tiled matmul's roofline says for this shape")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=sorted(RIDGE_FLOPS_PER_BYTE))
    ap.add_argument("--tol", type=float, default=2e-2)
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
        sys.exit(verify(a.check, a.level, a.tol, a.seed))

    if a.level and a.eval:
        sys.exit(run_eval(a.eval, a.level, a.tol))

    print("THE LEVELS — difficulty and optimization headroom rise together.\n")
    for n, s in LEVELS.items():
        tier = ("A correctness" if n <= 2 else "B roofline" if n <= 4
                else "C search" if n <= 7 else "D your own")
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
