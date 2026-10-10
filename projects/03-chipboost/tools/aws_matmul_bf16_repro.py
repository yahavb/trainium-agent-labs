#!/usr/bin/env python3
"""
aws_matmul_bf16_repro.py -- AWS's NKI matmul tutorial kernel rounds its running sum to bf16 once per K-block.

The kernel is nki_matmul_fully_optimized_ (NKI_EXAMPLE_21) in aws-neuron/aws-neuron-sdk,
nki/examples/matrix_multiplication/matrix_multiplication_nki_kernels.py (master is byte-identical to tag
v2.32.0). Its SBUF accumulator, result_m_tile, is allocated with dtype=result.dtype. With bf16 inputs every
K-block's fp32 PSUM result is added into a bf16 tile, so each output is rounded K/1024 - 1 extra times. The
tutorial's own test has K=1024, one block, so it never takes that path.

This runs AWS's file unmodified, and the same file with that one line changed to dtype=nl.float32 (the
final dma_copy then converts fp32 to bf16 once), against a float64 reference of the same bf16 inputs.
Standalone: numpy and ml_dtypes; nki 0.6.0 for --sim; a Trainium device and ../timing.py for --chip.

    python aws_matmul_bf16_repro.py --emulate                   # numpy model of both kernels; runs anywhere
    python aws_matmul_bf16_repro.py --sim                       # nki.simulate on CPU: AWS's actual kernel
    CHIPBOOST_CORE=2 python aws_matmul_bf16_repro.py --chip     # on the chip: errors and time, A/B
"""

import argparse
import hashlib
import importlib.util
import os
import re
import sys
import tempfile
import time
import urllib.request

import numpy as np

URL = ("https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/"
       "nki/examples/matrix_multiplication/matrix_multiplication_nki_kernels.py")
KNOWN_SHA256 = "033dec767d298df1"          # master and v2.32.0, checked 2026-10-10
ENTRY = "nki_matmul_fully_optimized_"
BLOCK_K = 8 * 128                           # the kernel's default TILES_IN_BLOCK_K * TILE_K
# The one line the fix changes: the SBUF accumulator of each result M-tile.
ACC_LINE = re.compile(r"(result_m_tile = nl\.ndarray\(\s*shape=\(TILE_M, TILES_IN_BLOCK_N, TILE_N\),\s*)"
                      r"dtype=result\.dtype,")
# (M, K, N), legal for the default blocking: M % 2048, N % 1024, K % 1024. K=1024 is the tutorial's own
# test depth (one K-block); 4096 is Qwen3-8B's and Llama-3-8B's hidden size; 8192 is the K of the
# tutorial's benchmark, [4096, 8192] @ [8192, 8192].
SHAPES = [(2048, 1024, 1024), (2048, 4096, 1024), (2048, 8192, 1024)]
CHIP_SHAPES = [(4096, 1024, 2048),          # the tutorial's correctness test
               (4096, 8192, 8192),          # the tutorial's benchmark
               (8192, 4096, 2048)]          # M where fp32 accumulators pass the 24 MB SBUF budget


def bf16():
    import ml_dtypes
    return ml_dtypes.bfloat16


def source(path):
    if path:
        src = open(path).read()
    else:
        with urllib.request.urlopen(URL, timeout=30) as r:
            src = r.read().decode()
    sha = hashlib.sha256(src.encode()).hexdigest()[:16]
    note = "the file the issue is about" if sha == KNOWN_SHA256 else f"NOT {KNOWN_SHA256}: the file changed"
    print(f"kernel file: {path or URL}\n  sha256 {sha} ({note})")
    return src


def fixed(src):
    out, n = ACC_LINE.subn(r"\1dtype=nl.float32,", src)
    if n != 1:
        sys.exit(f"expected exactly one accumulator line to change, found {n}: the file differs from v2.32.0")
    return out


def load(src, name, tmp):
    path = os.path.join(tmp, f"{name}.py")     # @nki.jit reads the function's source, so it needs a file
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, ENTRY)


def make_inputs(M, K, N, dist, seed=0):
    rng = np.random.default_rng(seed)
    draw = rng.random if dist == "uniform" else rng.standard_normal
    return draw((K, M)).astype(bf16()), draw((K, N)).astype(bf16())


def finite(x):
    # numpy 2 on macOS (Accelerate) can raise spurious overflow warnings inside matmul; the check is this.
    assert np.isfinite(np.asarray(x, np.float64)).all(), "non-finite values"
    return x


def reference(lhsT, rhs):
    with np.errstate(all="ignore"):
        return finite(lhsT.astype(np.float64).T @ rhs.astype(np.float64))


def emulate(lhsT, rhs, acc_bf16):
    """The kernel's arithmetic: fp32 PSUM within a K-block, then a tensor_tensor add into the accumulator,
    which is rounded to bf16 after every block if acc_bf16 (AWS), else kept in fp32 (the fix)."""
    a, b = lhsT.astype(np.float32), rhs.astype(np.float32)
    acc = np.zeros((a.shape[1], b.shape[1]), np.float32)
    for k0 in range(0, a.shape[0], BLOCK_K):
        with np.errstate(all="ignore"):
            acc = acc + a[k0:k0 + BLOCK_K].T @ b[k0:k0 + BLOCK_K]
        if acc_bf16:
            acc = acc.astype(bf16()).astype(np.float32)
    return finite(acc.astype(bf16()))


def bf16_ulps(got, want):
    """Worst error in bf16 units-in-the-last-place of the reference (the CHIPBOOST referee's measure; values
    within 1e-3 of the output RMS use that floor instead, so near-zero outputs cannot inflate it)."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    ulp = np.exp2(np.floor(np.log2(np.maximum(np.abs(want), 1e-30))) - 7)
    floor = 1e-3 * (np.sqrt((want ** 2).mean()) or 1.0)
    return float((np.abs(got - want) / np.maximum(ulp, floor)).max())


def errors(got, want):
    got = np.asarray(got, np.float64)
    rel = float(np.linalg.norm(got - want) / np.linalg.norm(want))
    # The tutorial's own check, torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2), where
    # output_torch is torch.matmul in bf16: the exact product rounded to bf16 once.
    want_bf16 = np.asarray(want).astype(bf16()).astype(np.float64)
    aws_check = bool(np.allclose(want_bf16, got, atol=1e-4, rtol=1e-2))
    return bf16_ulps(got, want), rel, aws_check


def row(shape, dist, e_aws, e_fix, extra=""):
    M, K, N = shape
    return (f"| {M}x{K}x{N} | {K // BLOCK_K} | {dist} | {e_aws[0]:.1f} | {e_aws[1]:.2e} | "
            f"{'pass' if e_aws[2] else 'FAIL'} | {e_fix[0]:.1f} | {e_fix[1]:.2e} | "
            f"{'pass' if e_fix[2] else 'FAIL'} |{extra}")


HEADER = ("| MxKxN | K-blocks | inputs | AWS: worst ulps | AWS: rel. RMS err | AWS: tutorial check | "
          "fixed: worst ulps | fixed: rel. RMS err | fixed: tutorial check |")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emulate", action="store_true", help="numpy model of both kernels (no nki)")
    mode.add_argument("--sim", action="store_true", help="nki.simulate on CPU (nki 0.6.0)")
    mode.add_argument("--chip", action="store_true", help="on a NeuronCore, with timing (needs ../timing.py)")
    ap.add_argument("--file", help="a local copy of matrix_multiplication_nki_kernels.py (default: download)")
    ap.add_argument("--rounds", type=int, default=3, help="--chip: interleaved A/B timing rounds")
    a = ap.parse_args()

    if a.emulate:
        print("numpy emulation of the kernels' arithmetic (not the kernels themselves)\n")
        print(HEADER + "\n" + "|---" * 9 + "|")
        for shape in SHAPES:
            for dist in ("uniform", "normal"):
                lhsT, rhs = make_inputs(*shape, dist)
                want = reference(lhsT, rhs)
                print(row(shape, dist, errors(emulate(lhsT, rhs, True), want),
                          errors(emulate(lhsT, rhs, False), want)))
        return 0

    import nki
    src = source(a.file)
    tmp = tempfile.mkdtemp(prefix="aws_mm_")
    k_aws, k_fix = load(src, "aws_as_published", tmp), load(fixed(src), "aws_fp32_accumulator", tmp)
    print(f"nki {getattr(nki, '__version__', '?')}; the fix changes one line: result_m_tile dtype=nl.float32\n")

    if a.sim:
        print(HEADER + " seconds |\n" + "|---" * 10 + "|")
        for shape in SHAPES:
            for dist in ("uniform", "normal"):
                lhsT, rhs = make_inputs(*shape, dist)
                want = reference(lhsT, rhs)
                t0 = time.time()
                got_aws = nki.simulate(k_aws)(lhsT, rhs)
                got_fix = nki.simulate(k_fix)(lhsT, rhs)
                print(row(shape, dist, errors(got_aws, want), errors(got_fix, want), f" {time.time() - t0:.0f} |"),
                      flush=True)
        return 0

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import timing
    print(f"NeuronCore {timing._pick_core()}; times are device-clock medians, AWS and fixed interleaved\n")
    print(HEADER + " AWS us | fixed us | fixed/AWS time |\n" + "|---" * 12 + "|")
    for shape in CHIP_SHAPES:
        lhsT, rhs = make_inputs(*shape, "normal")
        want = reference(lhsT, rhs)
        inp = {"lhsT": lhsT, "rhs": rhs}
        try:
            L_aws = timing.Loaded(timing.compile_kernel(k_aws, inp), inp)
            L_fix = timing.Loaded(timing.compile_kernel(k_fix, inp), inp)
            e_aws, e_fix = errors(L_aws.run()[0], want), errors(L_fix.run()[0], want)
            ab = timing.time_ab(L_aws, L_fix, rounds=a.rounds)
            t_aws, t_fix = ab["a"]["median_us"], ab["b"]["median_us"]
            print(row(shape, "normal", e_aws, e_fix, f" {t_aws:.1f} | {t_fix:.1f} | {t_fix / t_aws:.3f} |"),
                  flush=True)
        except Exception as e:
            print(f"| {'x'.join(map(str, shape))} | failed: {type(e).__name__}: {str(e)[:200]} |", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
