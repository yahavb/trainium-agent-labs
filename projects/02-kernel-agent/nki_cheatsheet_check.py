#!/usr/bin/env python3
"""
nki_cheatsheet_check.py — proves every line of nki_cheatsheet.md on the installed nki.

Each check is a tiny kernel plus the outcome the cheat-sheet claims for it: it runs and matches
NumPy ("ok"), it fails with a message containing a phrase ("error"), or it runs and returns the
wrong numbers ("wrong").

NEVER SHOW THIS FILE TO THE KERNEL AGENT: some test kernels (the K-loop matmul, the pooling view) are
close to ladder answers. Run it in a seat pod before trusting the sheet on a new nki version:

    PYTHONDONTWRITEBYTECODE=1 python nki_cheatsheet_check.py

It then confirms that every card in the sheet names its checks and that each of those held. The exit
status is the number of claims that did not hold plus the number of cards left unbacked.
"""

import re
import sys
from pathlib import Path

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

rng = np.random.default_rng(0)


def rnd(*shape):
    return rng.standard_normal(shape).astype(np.float32)


CHECKS = []


def check(cid, claim, expect, args, ref=None, phrase=""):
    def deco(kernel):
        CHECKS.append(dict(cid=cid, claim=claim, expect=expect, kernel=kernel, args=args, ref=ref,
                           phrase=phrase))
        return kernel
    return deco


def hbm_like(x, shape=None):
    return nl.ndarray(shape or x.shape, dtype=x.dtype, buffer=nl.shared_hbm)


def load(x, shape=None):
    t = nl.ndarray(shape or x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    return t


X = rnd(128, 64)
Y = rnd(128, 64)
POS = np.abs(rnd(128, 64)) + 0.5
LHST, RHS = rnd(128, 64), rnd(128, 512)
LHST2, RHS2 = rnd(256, 64), rnd(256, 512)


# ---------------------------------------------------------------- memory and tiles

@check("out-hbm", "allocate the output with buffer=nl.shared_hbm and return it", "ok", (X,),
       ref=lambda x: x)
@nki.jit
def k_copy(x):
    out = hbm_like(x)
    nisa.dma_copy(dst=out, src=load(x))
    return out


@check("tile-1d", "an on-chip tile needs at least 2 dimensions", "error", (rnd(64),))
@nki.jit
def k_tile_1d(x):
    out = hbm_like(x)
    nisa.dma_copy(dst=out, src=load(x))
    return out


@check("tile-3d", "a tile may have more than 2 dimensions; the first is the partition", "ok",
       (rnd(8, 6, 4),), ref=lambda x: x)
@nki.jit
def k_tile_3d(x):
    out = hbm_like(x)
    nisa.dma_copy(dst=out, src=load(x))
    return out


@check("pmax", "the first (partition) dimension of a tile is at most 128", "error", (rnd(256, 8),))
@nki.jit
def k_pmax(x):
    out = hbm_like(x)
    nisa.dma_copy(dst=out, src=load(x))
    return out


@check("dma-count", "dma_copy does not broadcast: dst and src hold the same number of elements",
       "error", (X,), phrase="same number of elements")
@nki.jit
def k_dma_count(x):
    out = hbm_like(x)
    t = nl.ndarray((128, 32), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    nisa.dma_copy(dst=out[:, 0:32], src=t)
    return out


@check("ragged", "loop with range(), clamp the last tile with min(), slice with nl.ds or a:b", "ok",
       (rnd(200, 32),), ref=lambda x: x)
@nki.jit
def k_ragged(x):
    P, F = x.shape
    out = hbm_like(x)
    for i in range((P + 127) // 128):
        lo = i * 128
        n = min(128, P - lo)
        t = nl.ndarray((n, F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[nl.ds(lo, n), 0:F])
        nisa.dma_copy(dst=out[lo:lo + n, 0:F], src=t)
    return out


@check("affine-range", "nl.affine_range(n) is the same as range(n)", "ok", (X,), ref=lambda x: x)
@nki.jit
def k_affine_range(x):
    out = hbm_like(x)
    for j in nl.affine_range(2):
        t = nl.ndarray((128, 32), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[:, j * 32:(j + 1) * 32])
        nisa.dma_copy(dst=out[:, j * 32:(j + 1) * 32], src=t)
    return out


@check("reshape", "reshaping an HBM input works when the element count is unchanged", "ok", (X,),
       ref=lambda x: x)
@nki.jit
def k_reshape(x):
    out = hbm_like(x)
    t = nl.ndarray((64, 128), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x.reshape((64, 128)))
    nisa.dma_copy(dst=out.reshape((64, 128)), src=t)
    return out


@check("region-call", "nl.sbuf / nl.psum are memory regions, not functions", "error", (X,),
       phrase="not callable")
@nki.jit
def k_region_call(x):
    out = hbm_like(x)
    t = nl.sbuf(x.shape, nl.float32)
    nisa.dma_copy(dst=t, src=x)
    nisa.dma_copy(dst=out, src=t)
    return out


@check("py-operator", "Python operators (+, *, +=) do not work on tiles", "error", (X,))
@nki.jit
def k_py_operator(x):
    out = hbm_like(x)
    t = load(x)
    nisa.dma_copy(dst=out, src=t + t)
    return out


@check("nl-load-store", "nl.load / nl.store also work (the AWS rules file bans them)", "ok", (X,),
       ref=lambda x: x)
@nki.jit
def k_nl_load_store(x):
    out = hbm_like(x)
    t = nl.load(x)
    nl.store(out, value=t)
    return out


# ---------------------------------------------------------------- matmul

def _mm_out(lhsT, rhs, psum):
    M, N = psum.shape
    out = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    s = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=s, src=psum)
    nisa.dma_copy(dst=out, src=s)
    return out


@check("matmul", "nc_matmul(dst=psum[M,N], stationary=sbuf[K,M], moving=sbuf[K,N]) = lhsT.T @ rhs",
       "ok", (LHST, RHS), ref=lambda a, b: a.T @ b)
@nki.jit
def k_matmul(lhsT, rhs):
    K, M = lhsT.shape
    N = rhs.shape[1]
    p = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=p, stationary=load(lhsT), moving=load(rhs))
    return _mm_out(lhsT, rhs, p)


def _mm_k_loop(lhsT, rhs, mode):
    K, M = lhsT.shape
    N = rhs.shape[1]
    p = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    for k in range(K // 128):
        a = nl.ndarray((128, M), dtype=lhsT.dtype, buffer=nl.sbuf)
        b = nl.ndarray((128, N), dtype=rhs.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a, src=lhsT[k * 128:(k + 1) * 128, 0:M])
        nisa.dma_copy(dst=b, src=rhs[k * 128:(k + 1) * 128, 0:N])
        if mode == "default":
            nisa.nc_matmul(dst=p, stationary=a, moving=b)
        elif mode == "first":
            nisa.nc_matmul(dst=p, stationary=a, moving=b, accumulate=(k > 0))
        else:
            nisa.nc_matmul(dst=p, stationary=a, moving=b, accumulate=False)
    return _mm_out(lhsT, rhs, p)


@check("k-accumulate", "K > 128: loop K in chunks of 128, nc_matmul into ONE psum tile; it adds up",
       "ok", (LHST2, RHS2), ref=lambda a, b: a.T @ b)
@nki.jit
def k_k_default(lhsT, rhs):
    return _mm_k_loop(lhsT, rhs, "default")


@check("k-accumulate-flag", "accumulate=(k > 0) also works", "ok", (LHST2, RHS2),
       ref=lambda a, b: a.T @ b)
@nki.jit
def k_k_first(lhsT, rhs):
    return _mm_k_loop(lhsT, rhs, "first")


@check("k-overwrite", "accumulate=False overwrites, so only the last K chunk survives", "wrong",
       (LHST2, RHS2), ref=lambda a, b: a.T @ b)
@nki.jit
def k_k_overwrite(lhsT, rhs):
    return _mm_k_loop(lhsT, rhs, "overwrite")


@check("moving-fmax", "the moving operand's free size N is at most 512", "error",
       (LHST, rnd(128, 1024)))
@nki.jit
def k_moving_fmax(lhsT, rhs):
    K, M = lhsT.shape
    N = rhs.shape[1]
    p = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=p, stationary=load(lhsT), moving=load(rhs))
    return _mm_out(lhsT, rhs, p)


@check("stationary-fmax", "the stationary operand's free size M is at most 128", "error",
       (rnd(128, 256), RHS))
@nki.jit
def k_stationary_fmax(lhsT, rhs):
    K, M = lhsT.shape
    N = rhs.shape[1]
    p = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=p, stationary=load(lhsT), moving=load(rhs))
    return _mm_out(lhsT, rhs, p)


@check("matmul-dst-sbuf", "nc_matmul's dst must be in nl.psum", "error", (LHST, RHS),
       phrase="psum")
@nki.jit
def k_matmul_dst_sbuf(lhsT, rhs):
    K, M = lhsT.shape
    N = rhs.shape[1]
    s = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_matmul(dst=s, stationary=load(lhsT), moving=load(rhs))
    out = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=s)
    return out


@check("psum-to-hbm", "PSUM cannot be DMA'd to HBM: tensor_copy it to SBUF first", "error", (X,))
@nki.jit
def k_psum_to_hbm(x):
    out = hbm_like(x)
    p = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.psum)
    nisa.tensor_copy(dst=p, src=load(x))
    nisa.dma_copy(dst=out, src=p)
    return out


@check("transpose", "nc_transpose(dst=psum[F,P], data=sbuf[P,F]) transposes a tile", "ok",
       (rnd(64, 128),), ref=lambda x: x.T)
@nki.jit
def k_transpose(x):
    P, F = x.shape
    out = nl.ndarray((F, P), dtype=x.dtype, buffer=nl.shared_hbm)
    p = nl.ndarray((F, P), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=p, data=load(x))
    s = nl.ndarray((F, P), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=s, src=p)
    nisa.dma_copy(dst=out, src=s)
    return out


@check("transpose-sbuf", "nc_transpose into SBUF only handles up to 32x32", "error",
       (rnd(64, 128),), phrase="32, 32")
@nki.jit
def k_transpose_sbuf(x):
    P, F = x.shape
    out = nl.ndarray((F, P), dtype=x.dtype, buffer=nl.shared_hbm)
    s = nl.ndarray((F, P), dtype=x.dtype, buffer=nl.sbuf)
    nisa.nc_transpose(dst=s, data=load(x))
    nisa.dma_copy(dst=out, src=s)
    return out


# ---------------------------------------------------------------- elementwise and reductions

def _unary(x, fn):
    out = hbm_like(x)
    t = load(x)
    r = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    fn(r, t)
    nisa.dma_copy(dst=out, src=r)
    return out


@check("tensor-scalar", "tensor_scalar(dst, data, op0=nl.multiply, operand0=0.5)", "ok", (X,),
       ref=lambda x: x * 0.5)
@nki.jit
def k_tensor_scalar(x):
    return _unary(x, lambda r, t: nisa.tensor_scalar(dst=r, data=t, op0=nl.multiply, operand0=0.5))


@check("tensor-tensor", "tensor_tensor(dst, data1, data2, op=nl.add)", "ok", (X, Y),
       ref=lambda x, y: x + y)
@nki.jit
def k_tensor_tensor(x, y):
    out = hbm_like(x)
    r = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=r, data1=load(x), data2=load(y), op=nl.add)
    nisa.dma_copy(dst=out, src=r)
    return out


@check("activation-exp", "activation(dst, op=nl.exp, data)", "ok", (X,), ref=np.exp)
@nki.jit
def k_activation_exp(x):
    return _unary(x, lambda r, t: nisa.activation(dst=r, op=nl.exp, data=t))


@check("activation-rsqrt", "rsqrt is activation(op=nl.rsqrt); there is no nisa.rsqrt", "ok", (POS,),
       ref=lambda x: 1 / np.sqrt(x))
@nki.jit
def k_activation_rsqrt(x):
    return _unary(x, lambda r, t: nisa.activation(dst=r, op=nl.rsqrt, data=t))


@check("nisa-rsqrt", "nisa.rsqrt does not exist", "error", (POS,), phrase="rsqrt")
@nki.jit
def k_nisa_rsqrt(x):
    return _unary(x, lambda r, t: nisa.rsqrt(dst=r, data=t))


@check("reciprocal", "reciprocal(dst, data)", "ok", (POS,), ref=lambda x: 1 / x)
@nki.jit
def k_reciprocal(x):
    return _unary(x, lambda r, t: nisa.reciprocal(dst=r, data=t))


@check("nisa-multiply", "ops are nl.add, nl.multiply, ... passed as op=; nisa.multiply does not exist",
       "error", (X,), phrase="multiply")
@nki.jit
def k_nisa_multiply(x):
    return _unary(x, lambda r, t: nisa.multiply(dst=r, data=t, operand0=0.5))


@check("no-dst", "nisa calls write into dst= and return nothing", "error", (X,), phrase="dst")
@nki.jit
def k_no_dst(x):
    out = hbm_like(x)
    r = nisa.activation(op=nl.exp, data=load(x))
    nisa.dma_copy(dst=out, src=r)
    return out


def _reduce(x, op, keepdims):
    P = x.shape[0]
    out = nl.ndarray((P, 1), dtype=x.dtype, buffer=nl.shared_hbm)
    r = nl.ndarray((P, 1), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=r, op=op, data=load(x), axis=(1,), keepdims=keepdims)
    nisa.dma_copy(dst=out, src=r)
    return out


@check("reduce-sum", "tensor_reduce(dst=[P,1], op=nl.add, data=[P,F], axis=(1,), keepdims=True)",
       "ok", (X,), ref=lambda x: x.sum(axis=1, keepdims=True))
@nki.jit
def k_reduce_sum(x):
    return _reduce(x, nl.add, True)


@check("reduce-no-keepdims", "tensor_reduce into a [P,1] dst also works without keepdims", "ok", (X,),
       ref=lambda x: x.sum(axis=1, keepdims=True))
@nki.jit
def k_reduce_no_keepdims(x):
    return _reduce(x, nl.add, False)


@check("reduce-max", "row max: tensor_reduce(op=nl.maximum, ...)", "ok", (X,),
       ref=lambda x: x.max(axis=1, keepdims=True))
@nki.jit
def k_reduce_max(x):
    return _reduce(x, nl.maximum, True)


@check("reduce-nl-max", "nl.max also works as a tensor_reduce op (AWS bans it)", "ok", (X,),
       ref=lambda x: x.max(axis=1, keepdims=True))
@nki.jit
def k_reduce_nl_max(x):
    return _reduce(x, nl.max, True)


@check("nl-sum", "nl.sum(tile, axis=[1], keepdims=True) returns a new tile", "ok", (X,),
       ref=lambda x: x.sum(axis=1, keepdims=True))
@nki.jit
def k_nl_sum(x):
    P = x.shape[0]
    out = nl.ndarray((P, 1), dtype=x.dtype, buffer=nl.shared_hbm)
    s = nl.sum(load(x), axis=[1], keepdims=True)
    nisa.dma_copy(dst=out, src=s)
    return out


@check("row-broadcast", "a [P,1] tile as operand0 applies per row: x - rowmax", "ok", (X,),
       ref=lambda x: x - x.max(axis=1, keepdims=True))
@nki.jit
def k_row_broadcast(x):
    P = x.shape[0]
    out = hbm_like(x)
    t = load(x)
    m = nl.ndarray((P, 1), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=m, op=nl.maximum, data=t, axis=(1,), keepdims=True)
    r = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=r, data=t, op0=nl.subtract, operand0=m)
    nisa.dma_copy(dst=out, src=r)
    return out


@check("ap-every-other", "t.ap([[stride, count], ...]) is a strided view: every other column", "ok",
       (X,), ref=lambda x: x[:, ::2])
@nki.jit
def k_ap_every_other(x):
    P, F = x.shape
    out = nl.ndarray((P, F // 2), dtype=x.dtype, buffer=nl.shared_hbm)
    r = nl.ndarray((P, F // 2), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r, src=load(x).ap([[F, P], [2, F // 2]]))
    nisa.dma_copy(dst=out, src=r)
    return out


# Level 1's answer in miniature, so it backs no card: the cards must not hand the agent a level.
@check("strided-view", "pooling: .ap([[stride, count], ...]) view, then nl.sum over the window axes",
       "ok", (rnd(4, 8, 8),),
       ref=lambda x: x.reshape(4, 4, 2, 4, 2).sum(axis=(2, 4)))
@nki.jit
def k_strided_view(x):
    C, H, W = x.shape
    out = nl.ndarray((C, H // 2, W // 2), dtype=x.dtype, buffer=nl.shared_hbm)
    t = load(x)
    view = t.ap([[H * W, C], [2 * W, H // 2], [2, W // 2], [W, 2], [1, 2]])
    s = nl.sum(view, axis=[3, 4])
    nisa.dma_copy(dst=out, src=s)
    return out


# ---------------------------------------------------------------- run

def run(c):
    try:
        got = nki.simulate(c["kernel"])(*c["args"])
    except Exception as e:
        msg = f"{type(e).__name__}: {e}".splitlines()[0][:160]
        return "error", msg
    if c["ref"] is None:
        return "ran", ""
    want = c["ref"](*c["args"])
    if np.shape(got) == np.shape(want) and np.allclose(got, want, rtol=1e-4, atol=1e-4):
        return "ok", ""
    return "wrong", f"shape {np.shape(got)} vs {np.shape(want)}"


CARD = re.compile(r"<!-- card (\S+) checks=(\S+)(?: withhold=(\S+))? -->\n(.*?)<!-- /card -->", re.S)


def load_cards(path=Path(__file__).with_name("nki_cheatsheet.md")):
    """{name: (card text, [check ids], {levels it must not be shown at})} from the cheat-sheet."""
    return {m[1]: (m[4], m[2].split(","), {int(n) for n in (m[3] or "").split(",") if n})
            for m in CARD.finditer(path.read_text())}


def main():
    print(f"nki {getattr(nki, '__version__', '?')}")
    failed, held = 0, set()
    for c in CHECKS:
        outcome, msg = run(c)
        if c["expect"] == "probe":
            verdict = "probe"
        else:
            holds = outcome == c["expect"] and (c["expect"] != "error"
                                                or c["phrase"].lower() in msg.lower())
            verdict = "holds" if holds else "DOES NOT HOLD"
            failed += not holds
            if holds:
                held.add(c["cid"])
        print(f"{verdict:13} {c['cid']:20} expect {c['expect']:5} got {outcome:5}  {msg}")
    print(f"\n{len(CHECKS)} checks, {failed} claims did not hold")

    # Every card must name its checks, and every check it names must have held just now.
    cards = load_cards()
    for name, (_, ids, _) in cards.items():
        unbacked = [i for i in ids if i not in held]
        if unbacked:
            failed += 1
            print(f"card {name}: not backed by a passing check: {', '.join(unbacked)}")
    print(f"{len(cards)} cards, each backed by passing checks" if cards else "no cards found")
    return failed


if __name__ == "__main__":
    sys.exit(main())
