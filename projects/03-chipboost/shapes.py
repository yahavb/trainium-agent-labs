#!/usr/bin/env python3
"""
shapes.py -- which shapes each CHIPBOOST op is checked and timed at. Owner: P2.

Qwen3-8B, served on one chip at tensor parallelism 2, so every matmul here is a PER-CORE shape.
Values from the model's config.json (verify in the pod with --verify-config):
hidden 4096, intermediate 12288, 32 query heads, 8 KV heads, head_dim 128, rms_norm_eps 1e-6.

Three lists per op, and they mean different things:

  dev      what the loop sees. Small enough for nki.simulate on the CPU in seconds, but multi-tile in
           every dimension so tiling bugs show. Single source of truth: nkibench.LEVELS[level]["shapes"].
  timing   real Qwen3-8B per-core sizes. Where speed is measured on the chip, and correctness re-checked
           on the chip. Too big to simulate on every attempt. The FIRST one is the primary shape: the
           one the progress curve and search.py optimise for.
  heldout  never shown to the agent and never used by search. Final evaluation only. They are where a
           speedup that only works at friendly shapes falls over.

Out of scope today: decode-time matmuls, where M is the batch (4 here) rather than a multiple of 128.

    python shapes.py                     # the table
    python shapes.py --verify-config     # in the pod: check the numbers above against config.json

    import shapes
    case = shapes.OPS["matmul"]["timing"][0]
    args = shapes.make_inputs("matmul", case)       # bf16 NumPy arrays, deterministic
    want = shapes.reference("matmul", args)
"""

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "02-kernel-agent"))
import nkibench  # noqa: E402  references, input builders and dev shapes live there

QWEN3_8B = dict(hidden_size=4096, intermediate_size=12288, num_attention_heads=32,
                num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6, torch_dtype="bfloat16")
TP = 2   # serve.sh: --tensor-parallel-size 2

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
    return dict(K=K, M=M, N=N, name=f"{proj}, {M} tokens")


OPS = {
    "matmul": dict(
        level=9, entry="qwen3_matmul",
        timing=[_mm("gate_up", 512), _mm("q_proj", 512), _mm("down_proj", 512),
                _mm("gate_up", 2048)],
        # Other tile multiples, chosen to break assumptions: 10/5/5 tiles (odd, not powers of two),
        # a single moving tile, three stationary tiles.
        heldout=[dict(K=1280, M=640, N=2560, name="odd tile counts 10/5/5"),
                 _mm("kv_proj", 1024),
                 _mm("o_proj", 384),
                 dict(K=384, M=128, N=1536, name="small, 3/1/3 tiles")],
    ),
    "rmsnorm": dict(
        level=10, entry="qwen3_rmsnorm",
        timing=[dict(rows=512, dim=4096, name="input_layernorm, 512 tokens"),
                dict(rows=2048, dim=4096, name="input_layernorm, 2048 tokens"),
                dict(rows=512 * 32 // TP, dim=128, name="q_norm per head, 512 tokens")],
        heldout=[dict(rows=127, dim=4096, name="one row short of a tile"),
                 dict(rows=129, dim=4096, name="one row past a tile"),
                 dict(rows=1, dim=4096, name="decode: a single token"),
                 dict(rows=200, dim=4096, scale=1e-4, name="quiet rows: needs eps"),
                 dict(rows=128, dim=4096, scale=1e3, name="loud rows"),
                 dict(rows=1000, dim=128, name="k_norm-like, ragged")],
    ),
    "copy": dict(
        level=11, entry="copy_floor",
        # Same bytes as rmsnorm's timing shapes: this is RMSNorm's floor.
        timing=[dict(rows=512, dim=4096, name="= rmsnorm 512 tokens"),
                dict(rows=2048, dim=4096, name="= rmsnorm 2048 tokens"),
                dict(rows=512 * 32 // TP, dim=128, name="= q_norm 512 tokens")],
        heldout=[dict(rows=129, dim=4096, name="one row past a tile"),
                 dict(rows=1, dim=4096, name="a single row")],
    ),
    "swiglu": dict(
        level=12, entry="qwen3_swiglu",
        timing=[dict(rows=512, dim=12288 // TP, name="MLP, 512 tokens"),
                dict(rows=2048, dim=12288 // TP, name="MLP, 2048 tokens")],
        heldout=[dict(rows=129, dim=12288 // TP, name="one row past a tile"),
                 dict(rows=128, dim=12288 // TP, scale=100.0, name="large gates: NaN trap"),
                 dict(rows=1, dim=12288 // TP, name="decode: a single token")],
    ),
}
for _op, _s in OPS.items():
    _s["dev"] = nkibench.LEVELS[_s["level"]]["shapes"]

WHICH = ("dev", "timing", "heldout")


def cases(op, which):
    """The shape dicts for one op: which is 'dev', 'timing' or 'heldout'."""
    if which not in WHICH:
        raise ValueError(f"which must be one of {WHICH}, got {which!r}")
    return OPS[op][which]


def entry(op):
    """The function name every kernel for this op must define."""
    return OPS[op]["entry"]


def make_inputs(op, case, seed=0):
    """Deterministic bf16 NumPy inputs, the same ones nkibench --check builds for this seed."""
    args, _ = nkibench.make_inputs(case, OPS[op]["level"], seed)
    return args


def reference(op, args):
    return nkibench.LEVELS[OPS[op]["level"]]["ref"](*args)


def tolerance(op):
    """nkibench's per-level gate for --check (relative to the output RMS). The referee may be stricter."""
    return nkibench.LEVELS[OPS[op]["level"]]["tol"]


def label(op, case):
    base = nkibench.label(case, OPS[op]["level"])
    return f"{base} ({case['name']})" if case.get("name") else base


def work(op, case):
    """(flops, minimum HBM bytes) for one case in bf16: the numbers a roofline or a floor needs."""
    if op == "matmul":
        M, K, N = case["M"], case["K"], case["N"]
        return 2 * M * K * N, 2 * (K * M + K * N + M * N)
    rows, dim = case["rows"], case["dim"]
    if op == "rmsnorm":
        return 4 * rows * dim, 2 * (2 * rows * dim + dim)       # read x and w, write y
    if op == "copy":
        return 0, 2 * (2 * rows * dim)
    if op == "swiglu":
        return 5 * rows * dim, 2 * (3 * rows * dim)
    raise KeyError(op)


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
        print(f"{op}: level {s['level']}, entry {s['entry']}(...), --check tolerance {tolerance(op)}")
        for which in WHICH:
            for i, c in enumerate(s[which]):
                flops, nbytes = work(op, c)
                star = " *primary*" if which == "timing" and i == 0 else ""
                print(f"  {which:<8} {label(op, c):<58} {flops / 1e9:7.2f} GFlop "
                      f"{nbytes / 2 ** 20:8.2f} MiB{star}")
        print()


if __name__ == "__main__":
    main()
