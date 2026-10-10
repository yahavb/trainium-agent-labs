#!/usr/bin/env python3
"""cards_check.py -- proves every card in agents2/cards.md on the installed nki, the way
../nki_cheatsheet_check.py proves the cheat-sheet. Run it in a seat pod:

    PYTHONDONTWRITEBYTECODE=1 python agents2/cards_check.py

NEVER SHOW THIS FILE TO THE KERNEL AGENT. The exit status is the number of claims that did not hold
plus the number of cards not backed by a check that held.
"""

import re
import sys
from pathlib import Path

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

X = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
CHECKS = []


def check(cid, expect, ref=None, phrase=""):
    def deco(kernel):
        CHECKS.append(dict(cid=cid, expect=expect, kernel=kernel, ref=ref, phrase=phrase))
        return kernel
    return deco


def _load(x):
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    return t


def _out(x, shape, src):
    o = nl.ndarray(shape, dtype=x.dtype, buffer=nl.shared_hbm)
    s = nl.ndarray(shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=s, src=src)
    nisa.dma_copy(dst=o, src=s)
    return o


@check("reshape-row-major", "ok", ref=lambda x: x.reshape(4, 2, 4))
@nki.jit
def k_reshape(x):
    return _out(x, (4, 2, 4), _load(x).reshape((4, 2, 4)))


@check("permute", "ok", ref=lambda x: x.reshape(4, 2, 4).transpose(0, 2, 1))
@nki.jit
def k_permute(x):
    return _out(x, (4, 4, 2), _load(x).reshape((4, 2, 4)).permute((0, 2, 1)))


@check("sum-permuted", "ok", ref=lambda x: x.reshape(4, 2, 4).sum(axis=1))
@nki.jit
def k_sum_permuted(x):
    o = nl.ndarray((4, 4), dtype=x.dtype, buffer=nl.shared_hbm)
    s = nl.sum(_load(x).reshape((4, 2, 4)).permute((0, 2, 1)), axis=[2])
    nisa.dma_copy(dst=o, src=s)
    return o


@check("mean-free", "ok", ref=lambda x: x.mean(axis=1, keepdims=True))
@nki.jit
def k_mean_free(x):
    o = nl.ndarray((4, 1), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=o, src=nl.mean(_load(x), axis=[1], keepdims=True))
    return o


@check("mean-partition", "error", phrase="partition")
@nki.jit
def k_mean_partition(x):
    o = nl.ndarray((1, 8), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=o, src=nl.mean(_load(x), axis=[0], keepdims=True))
    return o


@check("dma-transpose-2d", "ok", ref=lambda x: x.T)
@nki.jit
def k_dma_transpose(x):
    o = nl.ndarray((8, 4), dtype=x.dtype, buffer=nl.shared_hbm)
    s = nl.ndarray((8, 4), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_transpose(dst=s, src=_load(x), axes=(1, 0))
    nisa.dma_copy(dst=o, src=s)
    return o


@check("dma-transpose-axes", "error", phrase="axes must be one of")
@nki.jit
def k_dma_transpose_axes(x):
    o = nl.ndarray((4, 4, 2), dtype=x.dtype, buffer=nl.shared_hbm)
    s = nl.ndarray((4, 4, 2), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_transpose(dst=s, src=_load(x).reshape((4, 2, 4)), axes=(0, 2, 1))
    nisa.dma_copy(dst=o, src=s)
    return o


def run(c):
    try:
        got = nki.simulate(c["kernel"])(X.copy())
    except Exception as e:
        return "error", f"{type(e).__name__}: {e}".splitlines()[0][:160]
    want = c["ref"](X) if c["ref"] else None
    if want is not None and np.shape(got) == np.shape(want) and np.allclose(got, want):
        return "ok", ""
    return "wrong", f"shape {np.shape(got)}"


CARD = re.compile(r"<!-- card (\S+) checks=(\S+)(?: withhold=(\S+))? -->\n(.*?)<!-- /card -->", re.S)


def main():
    failed, held = 0, set()
    for c in CHECKS:
        outcome, msg = run(c)
        ok = outcome == c["expect"] and (c["expect"] != "error" or c["phrase"].lower() in msg.lower())
        failed += not ok
        held |= {c["cid"]} if ok else set()
        print(f"{'holds' if ok else 'DOES NOT HOLD':13} {c['cid']:20} expect {c['expect']:5} got {outcome:5} {msg}")
    for m in CARD.finditer(Path(__file__).with_name("cards.md").read_text()):
        missing = [i for i in m[2].split(",") if i not in held]
        if missing:
            failed += 1
            print(f"card {m[1]}: not backed by a passing check: {', '.join(missing)}")
    print(f"{len(CHECKS)} checks; {failed} problem(s)")
    return failed


if __name__ == "__main__":
    sys.exit(main())
