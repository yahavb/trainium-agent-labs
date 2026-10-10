#!/usr/bin/env python3
"""
shapes.py -- which shapes each CHIPBOOST op is checked and timed at, and how its inputs are built.
Owner: P2. The referee (speedcheck.py, P1) reads OPS directly; the tools (probe, check_kernels,
search) use the helpers at the bottom.

Qwen3-8B, served on one chip at tensor parallelism 2, so every matmul is a PER-CORE shape. Values from
the model's config.json, verified in the pod with --verify-config: hidden 4096, intermediate 12288,
32 query heads, 8 KV heads, head_dim 128, rms_norm_eps 1e-6. Matmul is timed at 256 prompt tokens,
where P1 measured the timer's noise at 0.0-0.2%. RMSNorm and its copy floor are timed at 2048 tokens:
at 256 a copy took 20.6 us, mostly the ~17 us launch cost, which would hide any speedup.

The referee's speedup is total time over an op's time_shapes, so the biggest shape weighs most: for
matmul, gate_up is 72% of the start kernel's 960 us.

Per op, three shape lists, and they mean different things:

  sim_shapes      what the loop sees. CPU simulator, seconds each, multi-tile in every dimension so
                  tiling bugs show. Single source of truth: nkibench.LEVELS[level]["shapes"].
  time_shapes     real Qwen3-8B per-core sizes. Correctness is re-checked and speed measured ON THE
                  CHIP here. The first one is the primary shape, the one search.py optimises for.
  heldout_shapes  never shown to the agent and never used by search: final evaluation only, always
                  with hostile values. Where a speedup that only works at friendly shapes falls over.

The spec keys are the referee's contract; do not rename them: level, entry, names,
make_inputs(shape, seed, hostile=False) -> {argument name: array}, ref(inputs) -> float32 reference,
out(shape) -> (output shape, dtype), flops(shape), sim_shapes, time_shapes, tol, and
heldout(rng, n, exclude) -> n fresh shapes. The hardened referee asserts every one of them at import, for
every op, and draws its held-out shapes fresh on every check: a fixed list was learnable. heldout_shapes
is the fixed, reproducible list heldout_grid.py uses for the dashboard's end-of-run grid.

Out of scope today: decode-time matmuls, where M is the batch (4 here), not a multiple of 128.

    python shapes.py                     # the table
    python shapes.py --verify-config     # in the pod: check the numbers above against config.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "02-kernel-agent"))
import nkibench  # noqa: E402  dev shapes and the --check levels live there

QWEN3_8B = dict(hidden_size=4096, intermediate_size=12288, num_attention_heads=32,
                num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6, torch_dtype="bfloat16")
TP = 2          # serve.sh: --tensor-parallel-size 2
TOKENS = 256    # prompt tokens at the matmul timing shapes
NORM_TOKENS = 2048   # prompt tokens at the RMSNorm / copy timing shapes
EPS = QWEN3_8B["rms_norm_eps"]

# Per-core projection shapes under TP=2, as (K, N) of y[M, N] = x[M, K] @ W[K, N].
# Column-parallel layers split N, row-parallel layers split K.
PROJ = {
    "q_proj":    (4096, 32 * 128 // TP),       # (4096, 2048)
    "kv_proj":   (4096, 8 * 128 // TP),        # (4096,  512) each of k and v
    "o_proj":    (32 * 128 // TP, 4096),       # (2048, 4096)
    "gate_up":   (4096, 12288 // TP),          # (4096, 6144) each of gate and up
    "down_proj": (12288 // TP, 4096),          # (6144, 4096)
}


def _mm(proj, M):
    K, N = PROJ[proj]
    return (K, M, N)


def _bf16():
    import ml_dtypes   # ships with the Neuron SDK; pip install ml_dtypes on a laptop
    return ml_dtypes.bfloat16


def _randn(r, shape):
    return r.standard_normal(shape).astype(np.float32)


def _pattern(rows):
    """Row classes for hostile inputs: every third row quiet, every third loud, the rest ordinary."""
    i = np.arange(rows)
    return i % 3 == 0, i % 3 == 1


# ---------------------------------------------------------------- matmul

def _matmul_inputs(shape, seed, hostile=False):
    K, M, N = shape
    r = np.random.default_rng(seed)
    a, b = _randn(r, (K, M)), _randn(r, (K, N))
    if hostile:                       # large magnitudes, exact zeros, a sign-flipped block (P1's)
        a[: K // 8] *= 64.0
        b[:, : N // 16] = 0.0
        a[:, M // 2:] *= -1.0
    return {"lhsT": a.astype(_bf16()), "rhs": b.astype(_bf16())}


def _matmul_ref(inp):
    return inp["lhsT"].astype(np.float32).T @ inp["rhs"].astype(np.float32)


# ---------------------------------------------------------------- RMSNorm

def _rmsnorm_inputs(shape, seed, hostile=False):
    rows, dim = shape
    r = np.random.default_rng(seed)
    x = _randn(r, (rows, dim))
    w = np.float32(1.0) + np.float32(0.1) * _randn(r, (1, dim))
    if hostile:
        quiet, loud = _pattern(rows)
        x[quiet] *= np.float32(1e-4)  # mean(x^2) ~ 1e-8, far below eps: a kernel without eps is 10x off
        x[loud] *= np.float32(1e3)
        if rows >= 4:
            x[-1] = 0.0               # a silent row: 0 * rsqrt(eps) must come out 0, not NaN
    return {"x": x.astype(_bf16()), "w": w.astype(_bf16())}


def _rmsnorm_ref(inp):
    x = inp["x"].astype(np.float32)
    var = (x * x).mean(axis=-1, keepdims=True)
    return x / np.sqrt(var + np.float32(EPS)) * inp["w"].astype(np.float32)


# ---------------------------------------------------------------- copy (the bandwidth floor)

def _copy_inputs(shape, seed, hostile=False):
    x = _randn(np.random.default_rng(seed), shape)
    if hostile:
        quiet, loud = _pattern(shape[0])
        x[quiet] *= np.float32(1e-4)
        x[loud] *= np.float32(1e4)
    return {"x": x.astype(_bf16())}


def _copy_ref(inp):
    return inp["x"].astype(np.float32)


# ---------------------------------------------------------------- SwiGLU (stretch)

def _swiglu_inputs(shape, seed, hostile=False):
    r = np.random.default_rng(seed)
    gate, up = _randn(r, shape), _randn(r, shape)
    if hostile:                       # g * exp(g) / (1 + exp(g)) is inf/inf = NaN at a gate of 100
        big, small = _pattern(shape[0])
        gate[big] *= np.float32(100.0)
        gate[small] *= np.float32(-100.0)
    return {"gate": gate.astype(_bf16()), "up": up.astype(_bf16())}


def _swiglu_ref(inp):
    g = inp["gate"].astype(np.float32)
    with np.errstate(over="ignore"):
        return g / (np.float32(1.0) + np.exp(-g)) * inp["up"].astype(np.float32)


# ---------------------------------------------------------------- held-out samplers
#
# heldout(rng, n, exclude) -> n distinct shapes not in exclude, drawn fresh per check by the referee.
# Matmul draws from every legal tile multiple (P1's ranges); the row-wise ops from every row count,
# ragged included, at Qwen3's widths.

def _distinct(draw, n, exclude):
    out = []
    while len(out) < n:
        s = draw()
        if s not in exclude and s not in out:
            out.append(s)
    return out


def _matmul_heldout(r, n, exclude):
    return _distinct(lambda: (128 * int(r.integers(4, 49)), 128 * int(r.integers(1, 5)),
                              512 * int(r.integers(1, 13))), n, exclude)


def _rows_heldout(widths, max_rows):
    def draw(r, n, exclude):
        return _distinct(lambda: (int(r.integers(1, max_rows + 1)), widths[int(r.integers(0, len(widths)))]),
                         n, exclude)
    return draw


def _same_out(shape):
    return (tuple(shape), _bf16())


# ---------------------------------------------------------------- the registry

def _dev(level):
    """Dev shapes as tuples, from nkibench so `nkibench --check` and the referee agree."""
    return [((c["K"], c["M"], c["N"]) if "K" in c else (c["rows"], c["dim"]))
            for c in nkibench.LEVELS[level]["shapes"]]


OPS = {
    "matmul": dict(
        level=9, entry="nki_matmul_tiled_", names=("lhsT", "rhs"),
        make_inputs=_matmul_inputs, ref=_matmul_ref,
        out=lambda s: ((s[1], s[2]), _bf16()), heldout=_matmul_heldout,
        flops=lambda s: 2 * s[0] * s[1] * s[2],
        sim_shapes=_dev(9),
        time_shapes=[_mm("gate_up", TOKENS), _mm("q_proj", TOKENS)],
        heldout_shapes=[_mm("down_proj", TOKENS), _mm("o_proj", TOKENS), _mm("q_proj", 512),
                        _mm("gate_up", 128), (1280, 640, 2560), _mm("kv_proj", 1024)],
        tol=nkibench.LEVELS[9]["tol"],
        labels={_mm("gate_up", TOKENS): "gate_up, 256 tokens",
                _mm("q_proj", TOKENS): "q_proj, 256 tokens",
                _mm("down_proj", TOKENS): "down_proj, 256 tokens: 6 K-blocks",
                _mm("o_proj", TOKENS): "o_proj, 256 tokens",
                _mm("q_proj", 512): "q_proj, 512 tokens",
                _mm("gate_up", 128): "gate_up, 128 tokens: one M-tile",
                (1280, 640, 2560): "odd tile counts 10/5/5",
                _mm("kv_proj", 1024): "kv_proj, 1024 tokens: one N-tile"},
    ),
    "rmsnorm": dict(
        level=10, entry="qwen3_rmsnorm", names=("x", "w"),
        make_inputs=_rmsnorm_inputs, ref=_rmsnorm_ref,
        out=_same_out, heldout=_rows_heldout((128, 4096), 4096),
        flops=lambda s: 4 * s[0] * s[1],
        sim_shapes=_dev(10),
        time_shapes=[(NORM_TOKENS, 4096), (NORM_TOKENS * 32 // TP, 128)],
        heldout_shapes=[(127, 4096), (129, 4096), (1, 4096), (1000, 128), (TOKENS, 4096)],
        tol=nkibench.LEVELS[10]["tol"],
        labels={(NORM_TOKENS, 4096): "input_layernorm, 2048 tokens",
                (NORM_TOKENS * 32 // TP, 128): "q_norm per head, 2048 tokens",
                (127, 4096): "one row short of a tile", (129, 4096): "one row past a tile",
                (1, 4096): "decode: a single token", (1000, 128): "k_norm-like, ragged",
                (TOKENS, 4096): "short prompt, 256 tokens"},
    ),
    "copy": dict(
        level=11, entry="copy_floor", names=("x",),
        make_inputs=_copy_inputs, ref=_copy_ref,
        out=_same_out, heldout=_rows_heldout((128, 4096), 4096),
        flops=lambda s: 0,
        sim_shapes=_dev(11),
        time_shapes=[(NORM_TOKENS, 4096), (NORM_TOKENS * 32 // TP, 128)],   # = RMSNorm's: its floor
        heldout_shapes=[(129, 4096), (1, 4096)],
        tol=nkibench.LEVELS[11]["tol"],
        labels={(NORM_TOKENS, 4096): "= rmsnorm, 2048 tokens",
                (NORM_TOKENS * 32 // TP, 128): "= q_norm, 2048 tokens",
                (129, 4096): "one row past a tile", (1, 4096): "a single row"},
    ),
    "swiglu": dict(
        level=12, entry="qwen3_swiglu", names=("gate", "up"),
        make_inputs=_swiglu_inputs, ref=_swiglu_ref,
        out=_same_out, heldout=_rows_heldout((12288 // TP,), 2048),
        flops=lambda s: 5 * s[0] * s[1],
        sim_shapes=_dev(12),
        time_shapes=[(TOKENS, 12288 // TP)],
        heldout_shapes=[(129, 12288 // TP), (1, 12288 // TP), (512, 12288 // TP)],
        tol=nkibench.LEVELS[12]["tol"],
        labels={(TOKENS, 12288 // TP): "MLP, 256 tokens", (129, 12288 // TP): "one row past a tile",
                (1, 12288 // TP): "decode: a single token", (512, 12288 // TP): "MLP, 512 tokens"},
    ),
}


# ---------------------------------------------------------------- helpers for the tools
#
# Cases are dicts so a tool can print and log them: matmul {K, M, N}, the rest {rows, dim}, plus `name`
# and `hostile` (True for every held-out case, as the referee builds them).

WHICH = ("dev", "timing", "heldout")
_LIST = {"dev": "sim_shapes", "timing": "time_shapes", "heldout": "heldout_shapes"}


def _case(op, shape, hostile):
    keys = ("K", "M", "N") if op == "matmul" else ("rows", "dim")
    c = dict(zip(keys, shape))
    c.update(hostile=hostile)
    name = OPS[op]["labels"].get(tuple(shape))
    if name:
        c["name"] = name
    return c


def shape_of(op, case):
    return (case["K"], case["M"], case["N"]) if op == "matmul" else (case["rows"], case["dim"])


def cases(op, which):
    """The cases for one op: which is 'dev', 'timing' or 'heldout'. Held-out cases are hostile."""
    if which not in WHICH:
        raise ValueError(f"which must be one of {WHICH}, got {which!r}")
    return [_case(op, s, which == "heldout") for s in OPS[op][_LIST[which]]]


def entry(op):
    """The function name every kernel for this op must define."""
    return OPS[op]["entry"]


def make_inputs(op, case, seed=0):
    """Deterministic bf16 NumPy inputs, as a tuple in the kernel's argument order."""
    inp = OPS[op]["make_inputs"](shape_of(op, case), seed, hostile=case.get("hostile", False))
    return tuple(inp[n] for n in OPS[op]["names"])


def reference(op, args):
    """The float32 reference (unrounded, as the referee compares against)."""
    return OPS[op]["ref"](dict(zip(OPS[op]["names"], args)))


def tolerance(op):
    """nkibench's per-level gate, relative to the output RMS. The referee also checks bf16 ulps."""
    return OPS[op]["tol"]


def label(op, case):
    base = " ".join(f"{k}={case[k]}" for k in (("K", "M", "N") if op == "matmul" else ("rows", "dim")))
    extra = [case["name"]] if case.get("name") else []
    if case.get("hostile"):
        extra.append("hostile")
    return f"{base} ({', '.join(extra)})" if extra else base


def work(op, case):
    """(flops, minimum HBM bytes) for one case in bf16: the numbers a roofline or a floor needs."""
    s = shape_of(op, case)
    if op == "matmul":
        K, M, N = s
        return 2 * M * K * N, 2 * (K * M + K * N + M * N)
    rows, dim = s
    nbytes = {"rmsnorm": 2 * (2 * rows * dim + dim), "copy": 2 * 2 * rows * dim,
              "swiglu": 2 * 3 * rows * dim}[op]
    return OPS[op]["flops"](s), nbytes


# ---------------------------------------------------------------- config check

def find_config():
    roots = [os.environ.get("HF_HOME", ""), os.path.expanduser("~/.cache/huggingface"),
             "/root/.cache/huggingface"]
    for root in filter(None, roots):
        hits = sorted(glob.glob(os.path.join(root, "hub", "models--Qwen--Qwen3-8B", "snapshots",
                                             "*", "config.json")))
        if hits:
            return hits[-1]
    return None


def verify_config(path=None):
    path = path or find_config()
    if not path:
        print("Qwen3-8B config.json not found under the Hugging Face cache. Start the model once "
              "with ./serve.sh (it downloads it), or pass the path: --verify-config PATH")
        return 2
    cfg = json.load(open(path))
    bad = 0
    print(f"checking {path}")
    for k, want in QWEN3_8B.items():
        got = cfg.get(k)
        ok = got == want or (isinstance(want, float) and isinstance(got, (int, float))
                             and abs(got - want) < 1e-12)
        bad += not ok
        print(f"  {k:<22} {got!s:<12} {'ok' if ok else f'MISMATCH, shapes.py assumes {want}'}")
    print("config matches shapes.py" if not bad else
          f"{bad} mismatch(es): fix QWEN3_8B and PROJ in shapes.py before timing anything")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-config", nargs="?", const="", metavar="CONFIG_JSON")
    a = ap.parse_args()
    if a.verify_config is not None:
        sys.exit(verify_config(a.verify_config or None))
    for op, s in OPS.items():
        print(f"{op}: level {s['level']}, entry {s['entry']}({', '.join(s['names'])}), "
              f"--check tolerance {s['tol']}")
        for which in WHICH:
            for i, c in enumerate(cases(op, which)):
                flops, nbytes = work(op, c)
                star = " *primary*" if which == "timing" and i == 0 else ""
                print(f"  {which:<8} {label(op, c):<62} {flops / 1e9:7.2f} GFlop "
                      f"{nbytes / 2 ** 20:7.2f} MiB{star}")
        print()


if __name__ == "__main__":
    main()
