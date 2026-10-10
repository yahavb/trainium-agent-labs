#!/usr/bin/env python3
"""
agent.py — the loop: a model writes an NKI kernel, the harness grades it, the reason goes back,
it tries again.

    controller  picks a level                          nkibench.py
    generator   the model writes a kernel             the shared gpt-oss endpoint
    checker     rules, then simulate, then roofline   nkibench.py
    loop        the failure becomes the next prompt   this file

Everything here runs on the CPU, so it does NOT need the Trainium device -- which is the point,
because the device is what Project 2 needs later for real timings, and a Neuron device cannot be
shared by two processes. The model therefore comes from the shared endpoint, not from a server on
your own chip.

    export GPTOSS_BASE_URL="https://..."
    python agent.py --level 1
    python agent.py --level 4 --rounds 6 --samples 4
    python agent.py --all
    python agent.py --offline --level 4        # no model; replays the reference kernel

Every attempt is appended to a JSONL file with its reward, so the log is the deliverable.
"""

import argparse
import ast
import hashlib
import inspect
import json
import os
import re
import sys
import tempfile
import textwrap
import time

import numpy as np

import nkibench

MODEL = os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-8B")

# The model writes to a hidden reasoning channel before it writes any answer. Measured on this
# endpoint: a coding task burned 900 tokens thinking and returned EMPTY content. See gptoss/README.
MIN_ANSWER_TOKENS = 2500

REASONING_KEYS = ("reasoning", "reasoning_content")


# ---------------------------------------------------------------- reward
#
# Graded, not pass/fail, so a near miss is distinguishable from nonsense and the loop has
# something to climb. Matches Project 1's shape: correctness dominates, and nothing else counts
# until the kernel is right.

WEIGHTS = dict(parses=0.1, rules=0.2, runs=0.2, correct=0.5)


def grade(source, level):
    """Returns (reward, parts, feedback, progress). Feedback is an INSTRUCTION, never just a verdict.

    progress breaks ties between samples with the same reward, so the loop repairs the attempt that
    got furthest. Measured on level 3: all four samples scored 0.30 every round, and the repair
    always went to whichever came back first -- once a kernel failing on its first line, while
    another sample had loaded both operands correctly. 0..1 is how far into the kernel the simulator
    got before raising; 1..2 means it ran, plus the fraction of the output that is right.
    """
    parts = dict(parses=False, rules=False, runs=False, correct=False)

    if not source.strip():
        return 0.0, parts, ("No code came back. Reply with one python code block containing the "
                            "kernel and nothing else."), 0.0
    try:
        compile(source, "<candidate>", "exec")
        parts["parses"] = True
    except SyntaxError as e:
        return (WEIGHTS["parses"] * 0, parts,
                f"The code does not parse: {e.msg} on line {e.lineno}. Send one complete python "
                f"code block.", 0.0)

    violations = nkibench.check_rules(source, level)
    if violations:
        extra = ""
        if any("no function named" in v for v in violations):
            # Measured: this repeated 15 rounds running, because "there is no function named X"
            # never said what the function should look like. Hand over the exact line.
            ref = nkibench.LEVELS[level]["ref"]
            args = ", ".join(inspect.signature(ref).parameters)
            extra = (f" Start the function with exactly this line:  "
                     f"def {nkibench.LEVELS[level]['entry']}({args}):  "
                     f"and put @nki.jit on the line above it.")
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                "Rule violations, which score zero however fast the kernel is. Fix exactly "
                "these: " + " ".join(violations) + extra, 0.0)
    parts["rules"] = True

    spec = nkibench.LEVELS[level]
    # One file per distinct kernel. Every candidate used to go to the same /tmp path, and Python
    # reuses a cached .pyc when the source has the same size and the same mtime second -- so a
    # same-length repair graded within a second of the previous sample (swapping stationary= and
    # moving=, or (N, M) for (M, N)) was silently graded as the OLD code. Two agents running at
    # once also overwrote each other's file.
    path = os.path.join(_scratch_dir(), f"level{level}_"
                        f"{hashlib.sha1(source.encode()).hexdigest()[:16]}.py")
    with open(path, "w") as f:
        f.write(source)
    try:
        kernel = nkibench.load_kernel(path, spec["entry"])
    except ModuleNotFoundError as e:
        # Two very different causes, and conflating them aborted a whole run: either the SDK is
        # absent (an environment problem the model cannot fix), or the model imported a module it
        # invented -- observed: `import nki.nl`. Tell them apart by asking whether nki itself is
        # importable.
        try:
            import nki  # noqa: F401
            sdk_present = True
        except ImportError:
            sdk_present = False
        if not sdk_present:
            raise SystemExit(
                f"cannot import {e.name!r}, so no NKI kernel can be loaded here. Run this where "
                f"the Neuron SDK exists -- k8s/kernel-agent-job.yaml does that -- or use "
                f"--offline to exercise the loop without a kernel ever running.") from e
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                f"There is no module named {e.name!r}. The only imports that exist are: "
                f"`import nki`, `import nki.language as nl`, and `import nki.isa as nisa`. "
                f"Use exactly those three.", 0.0)
    except Exception as e:
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                f"The file imports but {spec['entry']} could not be loaded: "
                f"{type(e).__name__}: {e}", 0.0)

    failures, passed, intensity, progress = [], 0, None, []
    busiest = None      # (transfers, output elements, label) of the passing shape with most DMAs
    for case in spec["shapes"]:
        args, _ = nkibench.make_inputs(case, level)
        before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
        want = spec["ref"](*args)
        try:
            got, counted = nkibench.simulate_and_count(kernel, args)
        except nkibench.NkiMissing as e:
            return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                    f"CANNOT SIMULATE: {e}", 0.0)
        except Exception as e:
            msg, got_to = explain_exception(e, path, source, level, np.shape(want))
            failures.append((nkibench.label(case, level), msg))
            progress.append(got_to)
            continue
        parts["runs"] = True
        m = (nkibench.check_inputs_untouched(before, args)
             or nkibench.describe_mismatch(got, want)
             or nkibench.check_traffic_bar(level, counted, args, want))
        # A simulator warning about a hardware-correctness hazard counts as a failure even when the
        # numbers happen to match on CPU: the kernel would be wrong on the device.
        hazards = [w for w in counted.get("warnings", [])
                   if "incorrect results on hardware" in w]
        if hazards and not m:
            m = ("CORRECT ON CPU BUT WRONG ON HARDWARE: " + hazards[0]
                 + ". Fix that before anything else -- the simulator agrees with the reference here "
                   "and the device would not.")
        if m and is_matmul(level):
            m += psum_accumulation_hint(counted, case)
        if m and is_transpose(level):
            m += transpose_hint(got, args)
        if m and is_pool(level):
            m += pool_hint(got, want, args[1])
        if m:
            failures.append((nkibench.label(case, level), m))
            progress.append(1.0 + fraction_right(got, want))
            continue
        progress.append(2.0)
        passed += 1
        elements = int(np.prod(np.shape(want)))
        if busiest is None or counted["transfers"] > busiest[0]:
            busiest = (counted["transfers"], elements, nkibench.label(case, level))
        if level >= 3 and counted["bytes"]:
            intensity = nkibench.roofline(
                nkibench.matmul_flops(case["M"], case["K"], case["N"]), counted["bytes"])

    if failures:
        lbl, first = failures[0]
        return (sum(WEIGHTS[k] for k, v in parts.items() if v)
                + WEIGHTS["correct"] * passed / len(spec["shapes"]), parts,
                f"{passed} of {len(spec['shapes'])} shapes passed. On {lbl}: {first}",
                sum(progress) / len(progress))

    parts["correct"] = True
    reward = sum(WEIGHTS.values())
    note = "Correct on every shape."
    if intensity:
        note += " " + nkibench.explain_roofline(intensity)
    elif busiest:
        # Off the matmul levels the cost that matters is the number of DMAs, not the bytes, so say
        # it on success too -- an element-at-a-time kernel is correct and still the wrong answer.
        transfers, elements, lbl = busiest
        note += f" At most {transfers} DMA transfers per shape ({lbl}: {elements} output elements)."
        if nkibench.issue_bound(transfers, elements):
            note += (" ISSUE-BOUND: that is close to one DMA per element. Move whole tiles, not "
                     "elements.")
    return reward, parts, note, 2.0


def psum_accumulation_hint(counted, case):
    """Name the two PSUM mistakes that run cleanly and return wrong numbers.

    nc_matmul accumulates into a PSUM tile across calls, so a tiled matmul must send exactly K/128
    matmuls to each output block's tile. nkibench counts them per PSUM allocation.
    """
    calls = list(counted.get("psum_matmuls", {}).values())
    k_tiles = -(-case["K"] // nkibench.PMAX)
    if not calls:
        return ""
    if max(calls) > k_tiles:
        return (f"\n  One psum tile received {max(calls)} nc_matmul calls, but K={case['K']} needs "
                f"only {k_tiles} per output block, so different output blocks are adding into the "
                f"same psum tile. Allocate a fresh psum tile for each (m, n) block: inside the n "
                f"loop, before the k loop.")
    if k_tiles > 1 and max(calls) < k_tiles:
        return (f"\n  Each psum tile received only {max(calls)} nc_matmul call(s), but K={case['K']} "
                f"is {k_tiles} chunks of 128 that must add up in ONE psum tile. Allocate the psum "
                f"tile before the k loop, not inside it, and nc_matmul every k chunk into it.")
    return ""


def transpose_hint(got, args):
    """Name the two wrong transposes that run cleanly: no transpose at all, and the other one.

    Measured on level 2's row-copy kernel: storing the loaded tile instead of the transposed one
    reads as "81.8% of elements are wrong, the core arithmetic or the operand layout" -- true, and
    no help in finding the one-word slip.
    """
    x, (f1, f2) = args
    got = np.asarray(got)
    p, f = x.shape
    if got.shape != x.shape:
        return ""
    if np.allclose(got, x):
        return ("\n  The output is the INPUT, unchanged: nothing was transposed. Store the tile the "
                "row copies wrote into, not the tile you loaded the input into.")
    if f1 != f2 and np.allclose(got, x.reshape(p, f2, f1).transpose(0, 2, 1).reshape(p, f)):
        return ("\n  This transposes each row as an F2 x F1 matrix; it is F1 x F2 = shape2D, so F1 "
                "and F2 are swapped somewhere in your slices.")
    return ""


def pool_hint(got, want, p):
    """Name the pooling mistakes that run cleanly: the wrong scale, and rows swapped with columns."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    if got.shape != want.shape or not np.isfinite(got).all():
        return ""
    scale = float(np.sqrt((want ** 2).mean())) or 1.0
    close = lambda a, b: float(np.abs(a - b).max()) / scale <= 2e-2  # noqa: E731
    if close(got, want * p * p):
        return (f"\n  Every output is exactly {p * p} times the mean: these are the window SUMS. "
                f"Multiply them by 1.0 / (p * p) before storing.")
    if p > 1 and close(got, want * p):
        return (f"\n  Every output is exactly {p} times the mean: the sums were divided by p, but a "
                f"window holds p * p = {p * p} elements. Multiply by 1.0 / (p * p).")
    if want.ndim == 3 and want.shape[1] == want.shape[2] and close(got, want.transpose(0, 2, 1)):
        return ("\n  The output has rows and columns swapped. In the access pattern the window-row "
                "entry, stride p * W, comes before the window-column entry, stride p.")
    return ""


_SCRATCH = []


def _scratch_dir():
    if not _SCRATCH:
        _SCRATCH.append(tempfile.mkdtemp(prefix="kernel_agent_"))
    return _SCRATCH[0]


def fraction_right(got, want, tol=2e-2):
    """Share of output elements within tolerance; 0 when the shape is wrong."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    if got.shape != want.shape:
        return 0.0
    scale = float(np.sqrt((want ** 2).mean())) or 1.0
    with np.errstate(invalid="ignore"):
        return float((np.abs(got - want) / scale <= tol).mean())


# ---------------------------------------------------------------- prompting

# Every name here appears in the three shipped tutorial kernels, so none of it is invented. The
# model does not know this API and guesses plausible names -- nl.scalar, nl.value, nl.dot,
# tile.mean -- none of which exist. A short card of what IS real costs ~200 tokens and is
# documentation rather than the answer.
API_CARD = """Available NKI functions:

  @nki.jit                                  decorate the entry point
  nl.ndarray(shape, dtype=..., buffer=b)    allocate; b is nl.sbuf, nl.psum or nl.shared_hbm
  nl.affine_range(n)                        the loop
  nl.sum(view, axis=[i, j])                 reduce; axis is a list
  nl.float32, nl.bfloat16                   dtypes
  nisa.dma_copy(dst=, src=)                 move data between HBM and SBUF
  nisa.nc_matmul(dst=, stationary=, moving=)   matmul into a PSUM tile
  nisa.tensor_copy(dst=, src=)              copy, e.g. PSUM to SBUF
  nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=0.5)   scale by a constant
  tile.ap([[stride, count], ...])           a strided view, for reductions

nisa.nc_matmul has strict memory rules: dst must live in nl.psum, while stationary and moving must
both live in nl.sbuf. So the pattern is: dma_copy both operands from HBM into sbuf tiles, allocate a
psum tile for the result, call nc_matmul(dst=psum_tile, stationary=..., moving=...), then
tensor_copy from psum into an sbuf tile, then dma_copy that out to the shared_hbm output.
The left operand arrives already transposed, with K on the partition axis.

Slice tiles with ranges, e.g. a[0:128, 0:64]. A complete kernel looks like this:

import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def copy_kernel(a):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    tile = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=a)
    nisa.dma_copy(dst=out, src=tile)
    return out

"""


# ---------------------------------------------------------------- strategy cards
#
# One card per handled level, appended to the generation and repair prompts by card_for(). A card
# says HOW this level's operation maps onto NKI -- which tiles, which copies, in which order --
# because each wall below was a missing idiom rather than a missing rule. Levels without a card
# (8, and levels added later) see exactly the prompt they were measured with.

# Level 1. Every sample of the final five-run measurement scored 0.30, and the samples show why: 54
# of 80 reached for nc_matmul -- the operation the API card spends most of its words on -- to compute
# a mean, none used the strided .ap() view the card lists "for reductions", and the rest moved one
# element per DMA or invented names (nisa.multiply). The pieces were on the card; how they compose
# into a pooling was not. This is the tutorial's composition -- a DMA in, one reduction over a view
# that groups each window's elements into its last two axes, one scale, a DMA out -- run over
# chunks of channels and bands of rows. The first version loaded each image whole, as the tutorial
# does: 5/5, 28 samples of one kernel, and wrong past 128 channels or ~180x180 pixels, sizes the
# tests did not reach until level 1 got a 200-channel and a 240x240 shape.
POOL_CARD = """How to do this pooling: let C, H, W = x.shape, p = pool_size, Ho = H // p and Wo = W // p.
Channels go on the partition axis, at most 128 at a time, and no matmul is involved. A whole image
may not fit on chip, so move the input through in bands of output rows, R rows per band:
  - Allocate out as nl.ndarray((C, Ho, Wo), dtype=x.dtype, buffer=nl.shared_hbm) and set
    R = max(1, min(Ho, 16384 // (p * W))).
  - Loop with plain Python ranges, since the last chunk and band can be partial:
    for c0 in range(0, C, 128), with cs = min(128, C - c0), and inside it
    for r0 in range(0, Ho, R), with rs = min(R, Ho - r0).
  - Per band, load its input rows: a = nl.ndarray((cs, rs * p, W), dtype=x.dtype, buffer=nl.sbuf)
    and nisa.dma_copy(dst=a, src=x[c0:c0 + cs, r0 * p:(r0 + rs) * p, :]).
  - View the band as windows with an access pattern of [stride, count] pairs -- partition, window
    row, window column, row inside the window, column inside the window -- and sum every window
    in one reduction:
    view = a.ap([[rs * p * W, cs], [p * W, rs], [p, Wo], [W, p], [1, p]])
    s = nl.sum(view, axis=[3, 4])   # shape (cs, rs, Wo)
  - Scale to the mean: res = nl.ndarray((cs, rs, Wo), dtype=x.dtype, buffer=nl.sbuf), then
    nisa.tensor_scalar(dst=res, data=s, op0=nl.multiply, operand0=1.0 / (p * p)).
  - Store the band: nisa.dma_copy(dst=out[c0:c0 + cs, r0:r0 + rs, :], src=res). Return out.
Two DMAs and two instructions per band. Never move single elements."""


# Level 2. Level 2 was the one level whose result was luck: 2/5 in one five-run measurement and 4/5
# in another, 0.30 to 1.00 run to run, and 2/3 with the level-3/4 loop before any card. The reverted
# chunked-copy example showed what an open-ended transpose invites -- the model reached for
# dma_transpose and got it wrong five ways -- so the card fixes one strategy instead.
#
# The first card, from the level2 branch, assigned out[p, j*F1 + i] = sbuf[p, i, j] element by
# element. It solved 5/5, and every solve was the pattern this level exists to expose: each
# assignment into HBM is its own one-element DMA, 8,193 transfers for the 128x64 shape. (The byte
# counter missed them until nkibench learned to count assignments, and called it "essentially
# optimal".) This card keeps the fixed strategy and moves whole tiles: one DMA in, one row of every
# partition's matrix per on-chip copy, one DMA out -- F1 copies where the shipped reference makes
# F1*F2.
TRANSPOSE_CARD = """How to do this transpose with whole-tile moves: let P, F = x.shape and F1, F2 = shape2D, with
F1 * F2 == F. Each partition holds an F1 x F2 matrix stored row by row, so row i is
x[:, i*F2:(i+1)*F2]; after the transpose its F2 elements sit F1 apart, at i, i+F1, i+2*F1, ...
  - Allocate out as nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm), and two (P, F) sbuf
    tiles, a and b.
  - Load the whole input once: nisa.dma_copy(dst=a, src=x).
  - Loop i over F1 with nl.affine_range and move one whole row per copy:
    nisa.tensor_copy(dst=b[:, i:F:F1], src=a[:, i*F2:(i+1)*F2]).
  - Store once: nisa.dma_copy(dst=out, src=b), then return out.
That is two DMAs and F1 copies in total. Never assign single elements, and do not reshape."""


# What nc_matmul DOES, as shapes. Measured on level 3 before it existed: the memory rules above were
# followed and the shapes were not. Samples sized the output as lhsT.shape[1:] -- a 1-D (64,) -- or
# poured both operands into one (128, 512) tile, and never once got past the first copy. The
# reference's docstring said "-> [M, N]" and that was not enough: it never said which operand
# supplies M and which N.
MATMUL_CARD = """How nc_matmul uses shapes: stationary is [K, M] and moving is [K, N], two sbuf tiles that share
the partition axis K (K <= 128, M <= 128, N <= 512). dst is [M, N], a float32 psum tile, and
receives stationary.T @ moving. lhsT already is [K, M] and rhs is [K, N], so load each into its OWN
sbuf tile of exactly its own shape. The result, and the shared_hbm output you return, is
(M, N) = (lhsT.shape[1], rhs.shape[1]). PSUM never goes straight to HBM: tensor_copy the psum tile
into a new (M, N) sbuf tile, then dma_copy that sbuf tile to the output."""


# The single-tile card above is WRONG past level 3, and measured so: on level 4 it said "load each
# into its own sbuf tile of exactly its own shape", and every round-0 sample did exactly that and
# died on "dma_copy dst partition dimension 256 exceeds maximum 128". Repairs then taught one
# dimension per round -- K tiling arrived in round 2, M and N never did -- and four rounds is not
# enough to meet three walls one at a time. So the levels whose shapes need tiling get the tiling
# contract up front: which loop owns which slice, and where the PSUM tile lives.
TILED_MATMUL_CARD = """How nc_matmul uses shapes: stationary is [K, M] and moving is [K, N], two sbuf tiles that share
the partition axis K; dst is [M, N], a float32 psum tile, and receives stationary.T @ moving. One
call takes at most K=128, M=128, N=512. These inputs are bigger, so tile all three dimensions;
every test shape is a whole number of tiles (K and M multiples of 128, N a multiple of 512).
  - Loop m over M // 128 and n over N // 512. For each (m, n) allocate ONE (128, 512) psum tile,
    before the k loop.
  - Loop k over K // 128: load lhsT[k*128:(k+1)*128, m*128:(m+1)*128] into a (128, 128) sbuf tile
    and rhs[k*128:(k+1)*128, n*512:(n+1)*512] into a (128, 512) sbuf tile, then nc_matmul into that
    same psum tile. Repeated nc_matmul calls into one psum tile add up.
  - After the k loop, tensor_copy the psum tile into a (128, 512) sbuf tile and dma_copy that to
    out[m*128:(m+1)*128, n*512:(n+1)*512]. PSUM never goes straight to HBM.
lhsT has shape (K, M), so unpack it as `K, M = lhsT.shape`; rhs is (K, N). The output you return is
(M, N) = (lhsT.shape[1], rhs.shape[1]), allocated in shared_hbm."""


def is_matmul(level):
    return nkibench.LEVELS[level]["ref"] is nkibench.ref_matmul


def is_transpose(level):
    return nkibench.LEVELS[level]["ref"] is nkibench.ref_transpose2d


def is_pool(level):
    return nkibench.LEVELS[level]["ref"] is nkibench.ref_avgpool2d


def needs_tiling(level):
    """A matmul level whose test shapes do not all fit one nc_matmul call."""
    return is_matmul(level) and any(
        c["K"] > nkibench.PMAX or c["M"] > nkibench.GEMM_STATIONARY_FMAX
        or c["N"] > nkibench.GEMM_MOVING_FMAX for c in nkibench.LEVELS[level]["shapes"])


def card_for(level):
    """This level's strategy card, or "" for a level that has none yet."""
    if is_matmul(level):
        return TILED_MATMUL_CARD if needs_tiling(level) else MATMUL_CARD
    if is_transpose(level):
        return TRANSPOSE_CARD
    if is_pool(level):
        return POOL_CARD
    return ""


def strategy_card(level):
    """The card as the first prompt carries it: with the shapes it will be tested on."""
    card = card_for(level)
    if not card:
        return ""
    shapes = ", ".join(nkibench.label(c, level) for c in nkibench.LEVELS[level]["shapes"])
    return f"{card}\nIt is tested on: {shapes}.\n\n"


def available_names(dotted):
    """Turn 'no attribute X' into 'here are the real ones'.

    Measured: told only `module 'nki.language' has no attribute 'value'`, the model guessed
    another invented name every round -- value, scalar, sbuf_scalar, dot. A verdict names the
    mistake and never the fix, so list what actually exists and let it choose.
    """
    import difflib
    import importlib
    mod_name, _, attr = dotted.rpartition(".")
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        return ""
    # Measured on level 1, 10 samples of 80: op0=nisa.multiply. The name is real -- in the OTHER
    # module -- and "nothing similar exists" in nki.isa sent the model hunting through the list.
    sibling = {"nki.isa": ("nki.language", "nl"), "nki.language": ("nki.isa", "nisa")}.get(mod_name)
    if sibling:
        try:
            if hasattr(importlib.import_module(sibling[0]), attr):
                short = "nisa" if mod_name == "nki.isa" else "nl"
                return (f" `{attr}` is not in {mod_name} but in {sibling[0]}: write "
                        f"{sibling[1]}.{attr}, not {short}.{attr}.")
        except Exception:
            pass
    names = [n for n in dir(mod) if not n.startswith("_")]
    close = difflib.get_close_matches(attr, names, n=6, cutoff=0.4)
    if close:
        return (f" `{mod_name}` has no `{attr}`. The closest real names are: "
                f"{', '.join(close)}. Pick one of those or use a different approach.")
    return (f" `{mod_name}` has no `{attr}`, and nothing similar exists. Its real names include: "
            f"{', '.join(sorted(names)[:25])}.")


def real_signature(func_name):
    """The actual signature of an NKI function, for when the model invents arguments."""
    for mod_name in ("nki.language", "nki.isa", "nki"):
        try:
            mod = __import__(mod_name, fromlist=["x"])
        except Exception:
            continue
        fn = getattr(mod, func_name, None)
        if fn is None:
            continue
        try:
            return f"{mod_name.split('.')[-1]}.{func_name}{inspect.signature(fn)}"
        except (TypeError, ValueError):
            return f"{mod_name.split('.')[-1]}.{func_name}"
    return ""


def enrich(error_text):
    """Add the real names when the failure is an invented API call."""
    if "'MemoryRegion' object is not callable" in error_text:
        return (error_text + " nl.sbuf, nl.psum and nl.shared_hbm are memory regions, not "
                "functions. Do not call them. Allocate with "
                "nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf) and pass the region as the "
                "buffer= argument.")
    m = re.search(r"(\w+)\(\) got an unexpected keyword argument '(\w+)'", error_text)
    if m:
        sig = real_signature(m.group(1))
        return (error_text + f" Remove the `{m.group(2)}=` argument."
                + (f" The real signature is {sig}." if sig else ""))
    if "unsupported operand type(s) for" in error_text and "NkiTensor" in error_text:
        return (error_text + " A tile is not a number, so Python operators like += do not work on "
                "one. Accumulate by allocating a PSUM tile with "
                "nl.ndarray(shape, nl.float32, buffer=nl.psum) and letting nisa.nc_matmul add into "
                "it across the loop, or combine two tiles with a nisa op rather than a Python "
                "operator.")
    m = re.search(r"dma_copy requires src and dst to have the same number of elements, "
                  r"got src=(\d+), dst=(\d+)", error_text)
    if m:
        src, dst = int(m.group(1)), int(m.group(2))
        return (error_text + f" The tile you allocated holds {dst} elements but you copied {src} "
                f"into it. nisa.dma_copy does not slice or broadcast: allocate the destination with "
                f"EXACTLY the shape of the slice you are moving. If you want a 128x512 piece of a "
                f"bigger tensor, write "
                f"t = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf) and then "
                f"nisa.dma_copy(dst=t, src=a[0:128, 0:512]) -- the slice on the right must have the "
                f"same shape as the tile on the left.")
    m = re.search(r"dma_copy (\w+) partition dimension (\d+) exceeds maximum (\d+)", error_text)
    if m:
        which, got, mx = m.group(1), int(m.group(2)), int(m.group(3))
        return (error_text + f" A tile may have at most {mx} rows, and you asked for {got}. Do not "
                f"allocate one tile for the whole tensor: loop over the partition dimension in "
                f"chunks of at most {mx} with nl.affine_range, allocate the tile inside the loop with "
                f"the chunk's own size, and copy one chunk at a time, e.g. "
                f"src=a[i*{mx}:(i+1)*{mx}, :]. If a dimension is already {mx} or smaller, use it "
                f"whole -- do NOT pad it up to {mx}, that reads past the end of the tensor. The same "
                f"applies to where you write the result back.")
    m = re.search(r"value array of shape \((\d+),?\) could not be broadcast to "
                  r"indexing result of shape \((\d+),?\)", error_text)
    if m:
        val, dst = int(m.group(1)), int(m.group(2))
        return (error_text + f" You assigned {val} elements into a slice that holds {dst}. Assignment "
                f"does not reshape or broadcast either: the slice on the left and the value on the "
                f"right must have the SAME shape. If the value is bigger, you are writing a whole tile "
                f"where a slice belongs -- index the destination to match, e.g. "
                f"out[i*128:(i+1)*128, :] = tile. If it is smaller, you are looping over the wrong "
                f"dimension.")
    m = re.search(r"Out-of-bound access for tensor .*? on dimension (\d+): "
                  r"index range \[(\d+), (\d+)\] exceed dimension size of (\d+)",
                  error_text)
    if m:
        dim, hi, size = m.group(1), int(m.group(3)), int(m.group(4))
        return (error_text + f" You indexed up to {hi} on dimension {dim}, which is only {size} "
                f"long. Tile limits are a MAXIMUM, not a target. Derive every bound from the tensor's "
                f"own shape -- use min(limit, size) and let the final chunk be partial -- rather than "
                f"writing a fixed number. Note the two limits differ: the partition dimension (first) "
                f"allows at most 128, the free dimension allows more.")
    m = re.search(r"Matmul contraction dimension (\d+) exceeds pmax=(\d+)", error_text)
    if m:
        k, mx = int(m.group(1)), int(m.group(2))
        return (error_text + f" The contraction dimension K is {k} and one nc_matmul can only "
                f"contract {mx}. Split K into chunks of {mx} and accumulate: allocate ONE psum tile "
                f"OUTSIDE the K loop, call nisa.nc_matmul into that same psum tile once per chunk so "
                f"the partial products add up there, and only after the loop copy it out with "
                f"nisa.tensor_copy. Do not allocate a new psum tile per chunk and do not write partial "
                f"results to HBM.")
    m = re.search(r"(\w+) (?:dst|src)? ?must be in \['sbuf', 'psum'\], got shared_hbm", error_text)
    if m:
        return (error_text + f" `nisa.{m.group(1)}` only moves data between on-chip buffers, sbuf and "
                f"psum. To reach HBM -- the tensor you allocated with buffer=nl.shared_hbm and will "
                f"return -- use nisa.dma_copy instead. The usual sequence is nc_matmul into psum, "
                f"tensor_copy psum to sbuf, then dma_copy sbuf to the shared_hbm output.")
    m = re.search(r"(\w+) must be in \['(\w+)'\], got (\w+)", error_text)
    if m:
        which, needed, got = m.groups()
        place = {"psum": "nl.psum", "sbuf": "nl.sbuf"}.get(needed, needed)
        return (error_text + f" Allocate the `{which}` tile with buffer={place} instead of "
                f"nl.{got}. For nisa.nc_matmul: dst must be in nl.psum, and stationary and moving "
                f"must both be in nl.sbuf. Copy between them with nisa.tensor_copy.")
    if "got multiple values for argument" in error_text:
        return (error_text + " Pass every argument by keyword, e.g. "
                "nisa.nc_matmul(dst=..., stationary=..., moving=...), so none is bound twice.")
    if "must have at least 2 dimensions" in error_text:
        return (error_text + " Every SBUF and PSUM tile needs two dimensions: a partition dimension "
                "first, then a free dimension. A 1-D tile is not allowed, so write "
                "nl.ndarray((rows, cols), ...) and give a length-N vector the shape (1, N) or "
                "(N, 1) depending on which axis you are reducing over.")
    if "cannot reshape array of size" in error_text:
        return (error_text + " Do not reshape. Work with the shapes you were given and slice "
                "them into tiles, e.g. src=a[0:128, 0:64].")
    m = re.search(r"module '([\w.]+)' has no attribute '(\w+)'", error_text)
    if m:
        return error_text + available_names(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"'(\w+)' object has no attribute '(\w+)'", error_text)
    if m:
        return (error_text + f" A {m.group(1)} is not a numpy array, so it has no "
                f"`{m.group(2)}`. Use the nl/nisa functions instead.")
    return error_text

# ---------------------------------------------------------------- locating a failure
#
# Measured on level 3, four runs out of four: every repair round sent the kernel back BYTE-FOR-BYTE
# unchanged, all four samples. The feedback named the error -- "dma_copy requires src and dst to have
# the same number of elements, got src=8192, dst=65536" -- but not the line, nor which tile was the
# wrong shape, so "change exactly what the checker names" was satisfied by changing nothing. Worse,
# two of the canned hints answered the wrong question: "cannot reshape array of size 32768" came from
# a PSUM tile shaped (128, 512) for a (64, 512) result, and the hint said "do not reshape" to a kernel
# that never called reshape.
#
# The simulator runs the kernel as plain Python, so the traceback still holds the failing line and
# the live tiles. Read them, and name the change in the model's own variable names.

def _shape_of(t):
    try:
        return tuple(int(s) for s in t.shape)
    except Exception:
        return None


def _region_name(region):
    return str(region).rsplit(".", 1)[-1] or "?"      # MemoryRegion.sbuf -> sbuf


def _buffer_of(t):
    return _region_name(getattr(t, "buffer", ""))


def _is_tile(v):
    return hasattr(v, "shape") and hasattr(v, "buffer") and not isinstance(v, type)


def _entry_span(tree, entry):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry:
            return node.lineno, node.end_lineno
    return None


def locate_failure(exc, path, source, entry):
    """The line of the candidate that raised, the call on it, and that call's real operands.

    Returns None when the exception never passed through the candidate's own code.
    """
    tb, frame, lineno, wrapper = exc.__traceback__, None, None, None
    while tb is not None:
        f = tb.tb_frame
        if f.f_code.co_filename == path:
            frame, lineno, wrapper = f, tb.tb_lineno, None
        elif frame is not None and wrapper is None and {"func", "args", "kwargs"} <= set(f.f_locals):
            # nki routes every public call through one context wrapper; its locals are the call.
            wrapper = dict(f.f_locals)
        tb = tb.tb_next
    if frame is None:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    stmt = None
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(node, (ast.FunctionDef, ast.For, ast.If,
                                                                ast.While, ast.With)) \
                and node.lineno <= lineno <= (node.end_lineno or node.lineno):
            if stmt is None or node.lineno >= stmt.lineno:
                stmt = node
    code = " ".join((ast.get_source_segment(source, stmt) or "").split()) if stmt else ""
    local_vars = dict(frame.f_locals)

    op, args, exprs = None, {}, {}
    if wrapper is not None:
        func = wrapper["func"]
        op = getattr(func, "__name__", None)
        try:
            params = list(inspect.signature(func).parameters)
            args = dict(inspect.signature(func).bind_partial(*wrapper["args"],
                                                             **wrapper["kwargs"]).arguments)
        except (TypeError, ValueError):
            params, args = [], dict(wrapper["kwargs"])
        for node in ast.walk(stmt or tree):
            if isinstance(node, ast.Call) and nkibench._dotted(node.func).split(".")[-1] == op:
                for i, arg in enumerate(node.args):
                    if i < len(params):
                        exprs[params[i]] = ast.get_source_segment(source, arg)
                for kw in node.keywords:
                    if kw.arg:
                        exprs[kw.arg] = ast.get_source_segment(source, kw.value)
                break

    names = []
    for node in ast.walk(stmt) if stmt else []:
        if isinstance(node, ast.Name) and node.id not in names and _is_tile(local_vars.get(node.id)):
            names.append(node.id)
    tiles = {n: local_vars[n] for n in names}

    span = _entry_span(tree, entry)
    progress = 0.0
    if span and span[1] > span[0]:
        progress = min(1.0, max(0.0, (lineno - span[0]) / (span[1] - span[0])))
    return dict(line=lineno, code=code, op=op, args=args, exprs=exprs, tiles=tiles,
                progress=progress)


BLOCKS = ("lhsT[k*128:(k+1)*128, m*128:(m+1)*128] as a (128, 128) sbuf tile, "
          "rhs[k*128:(k+1)*128, n*512:(n+1)*512] as a (128, 512) sbuf tile, and the result as one "
          "(128, 512) psum tile per (m, n), written to out[m*128:(m+1)*128, n*512:(n+1)*512]")


def shape_advice(loc, want_shape, level):
    """One named change for the shape mistakes the simulator reports obliquely, or None.

    Tiled matmul levels get block-level advice. Measured on level 4 with the single-tile wording:
    "stationary free dimension 256 exceeds 128 -- check they are not swapped" went to a kernel whose
    operands were the right way round and whose M simply needed tiling.
    """
    op, args, exprs = loc["op"], loc["args"], loc["exprs"]
    T = {k: v for k, v in args.items() if _is_tile(v)}
    matmul, tiled = is_matmul(level), needs_tiling(level)

    def nm(k):
        return f"`{exprs.get(k) or k}`"

    if op == "ndarray":
        shape = args.get("shape")
        buf = _region_name(args.get("buffer"))
        try:
            shape = tuple(int(s) for s in shape)
        except Exception:
            return None
        if len(shape) < 2 and buf in ("sbuf", "psum"):
            msg = (f"This allocates a {buf} tile of shape {shape}, which is 1-D. SBUF and PSUM tiles "
                   f"are 2-D: (partition, free).")
            if tiled:
                msg += f" In this tiled matmul the tiles are: {BLOCKS}."
            elif matmul and want_shape:
                msg += (f" For this matmul the result is (M, N) = (lhsT.shape[1], rhs.shape[1]) = "
                        f"{want_shape}, so a tile or output meant to hold it needs that 2-D shape.")
            return msg
        return None

    if op == "nc_matmul" and {"dst", "stationary", "moving"} <= set(T):
        d, s, m = (_shape_of(T[k]) for k in ("dst", "stationary", "moving"))
        if not (d and s and m and len(s) == 2 and len(m) == 2):
            return None
        if s[0] != m[0]:
            return (f"nc_matmul contracts over the FIRST axis of both operands, so they must share it: "
                    f"stationary {nm('stationary')} is {s} and moving {nm('moving')} is {m}. "
                    f"stationary is [K, M] (the left matrix, already transposed: lhsT) and moving is "
                    f"[K, N] (rhs), with K on the partition axis of both.")
        swapped = s[1] > nkibench.GEMM_STATIONARY_FMAX and m[1] <= nkibench.GEMM_STATIONARY_FMAX
        if tiled and not swapped and (s[0] > nkibench.PMAX or s[1] > nkibench.GEMM_STATIONARY_FMAX
                                      or m[1] > nkibench.GEMM_MOVING_FMAX):
            too_big = [f"{what} {have} exceeds {cap}" for what, have, cap in
                       (("K", s[0], nkibench.PMAX), ("M", s[1], nkibench.GEMM_STATIONARY_FMAX),
                        ("N", m[1], nkibench.GEMM_MOVING_FMAX)) if have > cap]
            return (f"stationary {nm('stationary')} is {s} and moving {nm('moving')} is {m}, so "
                    f"{', '.join(too_big)} -- one nc_matmul takes at most K=128, M=128, N=512. Tile "
                    f"every dimension and pass nc_matmul one block at a time: {BLOCKS}. Load the "
                    f"blocks inside the m, n and k loops, and allocate the psum tile inside the n "
                    f"loop but before the k loop, so the k partial products add up in it.")
        if s[1] > nkibench.GEMM_STATIONARY_FMAX:
            return (f"stationary {nm('stationary')} is {s}, and its free dimension {s[1]} exceeds "
                    f"{nkibench.GEMM_STATIONARY_FMAX}. stationary must be the [K, M] tile loaded from "
                    f"lhsT and moving the [K, N] tile loaded from rhs -- check they are not swapped.")
        if m[1] > nkibench.GEMM_MOVING_FMAX:
            return (f"moving {nm('moving')} is {m}, and its free dimension {m[1]} exceeds "
                    f"{nkibench.GEMM_MOVING_FMAX}. Split N into chunks of at most "
                    f"{nkibench.GEMM_MOVING_FMAX}.")
        if d != (s[1], m[1]):
            return (f"nc_matmul writes stationary.T @ moving into dst. stationary {nm('stationary')} "
                    f"is {s} = [K, M] and moving {nm('moving')} is {m} = [K, N], so dst must be "
                    f"[M, N] = {(s[1], m[1])}, but dst {nm('dst')} is {d}. Allocate dst as "
                    f"nl.ndarray({(s[1], m[1])}, dtype=nl.float32, buffer=nl.psum).")
        return None

    if op in ("tensor_scalar", "tensor_tensor") and "dst" in T:
        # Element-wise: dst gets one element per input element, so its shape is the input's.
        d = _shape_of(T["dst"])
        for k in ("data", "data1", "data2"):
            if k in T and _shape_of(T[k]) and _shape_of(T[k]) != d:
                s = _shape_of(T[k])
                return (f"{op} writes one element per input element, so dst {nm('dst')} must have "
                        f"the shape of {k} {nm(k)}, {s}; it is {d}. Allocate dst as "
                        f"nl.ndarray({s}, dtype=..., buffer=nl.sbuf).")
        return None

    if op in ("dma_copy", "tensor_copy") and {"dst", "src"} <= set(T):
        d, s = _shape_of(T["dst"]), _shape_of(T["src"])
        db, sb = _buffer_of(T["dst"]), _buffer_of(T["src"])
        if sb == "psum" and "hbm" in db:
            # Measured on level 3: four fresh samples of four got the loads, the PSUM tile and the
            # matmul right, then copied PSUM straight to the output -- with tensor_copy in some,
            # dma_copy in the repairs. Each copy's own error says only half of it. Name both hops.
            dst, src = exprs.get("dst") or "out", exprs.get("src") or "psum_tile"
            dtype = f"{dst}.dtype" if dst.isidentifier() else "nl.float32"
            return (f"PSUM cannot be copied straight to HBM by either copy. Replace this line with "
                    f"three: `res_sb = nl.ndarray({s}, dtype={dtype}, buffer=nl.sbuf)`, then "
                    f"`nisa.tensor_copy(dst=res_sb, src={src})`, then "
                    f"`nisa.dma_copy(dst={dst}, src=res_sb)`.")
        if op == "dma_copy" and "psum" in (db, sb):
            return (f"dma_copy cannot read or write PSUM ({nm('src')} is in {sb}, {nm('dst')} is in "
                    f"{db}). First nisa.tensor_copy the PSUM tile into an sbuf tile of the same shape, "
                    f"then dma_copy that sbuf tile to the shared_hbm output.")
        if d is None or s is None:
            return None
        if tiled:
            # The measured level-4 wall: a whole 256-row operand loaded into one tile.
            big = [(k, sh) for k, sh, b in (("dst", d, db), ("src", s, sb))
                   if b in ("sbuf", "psum") and sh and sh[0] > nkibench.PMAX]
            if big:
                k, sh = big[0]
                return (f"{nm(k)} is {sh}, but an sbuf or psum tile holds at most {nkibench.PMAX} "
                        f"partitions (its first dimension). Do not load or store whole tensors: "
                        f"inside the m, n and k loops, move one block at a time -- {BLOCKS}.")
            if d != s:
                return (f"{nm('dst')} is {d} but {nm('src')} is {s}; a copy needs the same shape on "
                        f"both sides, and here every block has a fixed shape: {BLOCKS}. Make both "
                        f"sides of this copy the same block.")
            return None
        if d == s:
            return None
        if is_transpose(level) and db == sb == "sbuf":
            # The generic advice -- "copy into a new tile of exactly the source's shape" -- is wrong
            # here: both sides are views of tiles that already exist, and the view is the mistake.
            return (f"{nm('dst')} is {d} but {nm('src')} is {s}. Both sides of a row copy are one "
                    f"row of the F1 x F2 matrix, F2 elements per partition: the source is the "
                    f"contiguous a[:, i*F2:(i+1)*F2] and the destination is the strided "
                    f"b[:, i:F:F1] (start i, step F1). Check which of F1 and F2 each slice uses.")
        if "hbm" in db and want_shape and d != want_shape and s == want_shape:
            return (f"The output {nm('dst')} is {d}, but the result is {want_shape}. Allocate the "
                    f"output you return as nl.ndarray({want_shape}, dtype=..., buffer=nl.shared_hbm).")
        if "hbm" in db and want_shape and s != want_shape:
            return (f"You are writing {nm('src')}, shape {s}, to the output {nm('dst')}, shape {d}. "
                    f"The result is {want_shape}: write the sbuf tile that holds the result (copied "
                    f"out of PSUM with nisa.tensor_copy), and make the output that shape too.")
        if "hbm" in sb:
            return (f"{nm('dst')} is {d} but {nm('src')} is {s}; a copy needs the same shape on both "
                    f"sides. Give the tensor you load its own sbuf tile of exactly {s}, e.g. "
                    f"nl.ndarray({s}, dtype=nl.float32, buffer=nl.sbuf), instead of one shared tile.")
        return (f"{nm('dst')} is {d} but {nm('src')} is {s}; a copy needs the same shape on both "
                f"sides. Copy into a new tile allocated with exactly {s}, e.g. "
                f"nl.ndarray({s}, dtype=nl.float32, buffer=nl.sbuf) -- do not reuse an input tile to "
                f"hold a result of a different shape.")
    return None


def swapped_unpack(source, level):
    """`M, K = lhsT.shape` -- the one-line slip that passes every shape with K == M.

    Measured on level 4: four fresh samples of four were correct kernels apart from this line, and
    two of the four test shapes have K == M, so it surfaced only as an out-of-bounds read on
    K=512 M=128 with advice about tile limits that had nothing to do with it.
    """
    if not is_matmul(level):
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    entry = nkibench.LEVELS[level]["entry"]
    params = next((n.args.args for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == entry), [])
    if not params:
        return None
    first = params[0].arg
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple) \
                and len(node.targets[0].elts) == 2 \
                and all(isinstance(e, ast.Name) for e in node.targets[0].elts) \
                and ast.unparse(node.value) == f"{first}.shape":
            a, b = (e.id for e in node.targets[0].elts)
            if a.upper().startswith("M") and b.upper().startswith("K"):
                return (f"Line {node.lineno}, `{a}, {b} = {first}.shape`, has them backwards: {first} "
                        f"arrives TRANSPOSED, shape (K, M), so write `{b}, {a} = {first}.shape`. The "
                        f"kernel only worked on shapes where K == M.")
    return None


def pool_size_advice(raw, level):
    """The two ways a level-1 kernel outgrows the chip, answered in the card's terms.

    The generic hints are wrong here: the partition hint says to loop with nl.affine_range in fixed
    chunks of 128, and 200 channels leave a partial chunk of 72 that a fixed slice reads past.
    """
    if not is_pool(level):
        return None
    if re.search(r"partition dimension \d+ exceeds maximum", raw):
        return ("C is larger than the 128 partitions. Loop over channel chunks with a plain Python "
                "range -- for c0 in range(0, C, 128), cs = min(128, C - c0) -- and move "
                "x[c0:c0 + cs, ...] and out[c0:c0 + cs, ...] one chunk at a time.")
    if "SBUF capacity exceeded" in raw:
        return ("The image does not fit on chip whole. Move it through in bands of R output rows, "
                "R = max(1, min(Ho, 16384 // (p * W))): inside for r0 in range(0, Ho, R), with "
                "rs = min(R, Ho - r0), load x[c0:c0 + cs, r0 * p:(r0 + rs) * p, :] into a "
                "(cs, rs * p, W) tile and store the band to out[c0:c0 + cs, r0:r0 + rs, :].")
    return None


def explain_exception(exc, path, source, level, want_shape):
    """(feedback, progress) for a kernel that raised in the simulator."""
    entry = nkibench.LEVELS[level]["entry"]
    raw = f"raised {type(exc).__name__}: {exc}"
    loc = locate_failure(exc, path, source, entry)
    if loc is None:
        return enrich(raw), 0.0
    where = f"\n  The failing line is {loc['line']}: `{loc['code'][:200]}`"
    if loc["tiles"]:
        where += "\n  where " + "; ".join(f"`{n}` is {_shape_of(t)} in {_buffer_of(t)}"
                                          for n, t in loc["tiles"].items())
    advice = (swapped_unpack(source, level) or pool_size_advice(raw, level)
              or shape_advice(loc, want_shape, level))
    if advice:
        # The specific diagnosis replaces the generic hint, which for these errors points the wrong way.
        return f"{raw}{where}\n  FIX: {advice}", loc["progress"]
    return enrich(raw) + where, loc["progress"]


def first_prompt(level, terse=0):
    """Deliberately short, and it does NOT list the rules.

    Measured twice in this repo: hand a model an enumerated list of prohibitions and it audits
    itself against each one and returns nothing, while a bigger budget only buys more thinking.
    So the rules live in the checker. Generate freely, let the checker object, then send back one
    named change.
    """
    s = nkibench.LEVELS[level]
    import inspect
    level_hint = ""
    if level == 2:
        level_hint = level_hint = (
                                        "Use this exact implementation strategy for Level 2:\n"
                                        "- Let P, F = x.shape and F1, F2 = shape2D, with F1 * F2 == F.\n"
                                        "- Allocate out as nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm).\n"
                                        "- Allocate sbuf as nl.ndarray((P, F1, F2), dtype=x.dtype, buffer=nl.sbuf).\n"
                                        "- Copy the full input with nisa.dma_copy(dst=sbuf, src=x).\n"
                                        "- Loop p over P, i over F1, and j over F2 using nl.affine_range.\n"
                                        "- For each element, assign out[p, j * F1 + i] = sbuf[p, i, j].\n"
                                        "- Return out directly.\n"
                                        "- Do NOT reshape tensors. Do NOT dma_copy sbuf into out after the loop.\n"
                                        "Follow this structure exactly rather than inventing a different transpose strategy.\n\n"
    ) 
    if terse >= 2:
        # Last resort. Measured on this endpoint: one-sentence prompts answered in 300-700
        # tokens while every structured, rule-carrying prompt spiralled.
        return (f"Write a Python function `{s['entry']}` decorated with @nki.jit that computes "
                f"the same thing as this, using nki.language as nl and nki.isa as nisa:\n\n"
                f"{inspect.getsource(s['ref'])}\n"
                f"Reply with one python code block.")
    if terse >= 1:
        # The matmul memory rules are the substance of levels 3 and 4, and the short prompt has to
        # carry them: measured, the agent cycled between "dst must be in ['psum']" and "moving must
        # be in ['sbuf']" because nothing told it where the operands live.
        mm = ("nisa.nc_matmul(dst=, stationary=, moving=) needs dst in nl.psum and both operands "
              "in nl.sbuf. So: dma_copy the operands HBM->sbuf, allocate a psum tile, nc_matmul "
              "into it, tensor_copy psum->sbuf, then dma_copy sbuf->the shared_hbm output you "
              "return. The left operand is already transposed, with K on the partition axis.\n"
              if level >= 3 else "")
        return (f"Write an AWS Neuron NKI kernel: a function `{s['entry']}` decorated with "
                f"@nki.jit that computes what this reference computes.\n\n"
                f"{inspect.getsource(s['ref'])}\n"
                f"Allocate with nl.ndarray(shape, dtype=..., buffer=nl.sbuf), move data with "
                f"nisa.dma_copy(dst=, src=), loop with nl.affine_range(n). A tile's partition "
                f"dimension is at most {nkibench.PMAX}.\n{mm}\n"
                f"Reply with one python code block.")
    return (
        f"Write an AWS Neuron NKI kernel.\n\n"
        f"{level_hint}"
        f"Operation: {s['op']}\n"
        f"Entry point: a function named `{s['entry']}`, decorated with `@nki.jit`.\n"
        f"It must compute exactly what this NumPy reference computes:\n\n"
        f"{inspect.getsource(s['ref'])}\n"
        f"Hardware limits: a tile's partition dimension is at most {nkibench.PMAX}. For matmul, "
        f"the stationary free dimension is at most {nkibench.GEMM_STATIONARY_FMAX} and the "
        f"moving free dimension at most {nkibench.GEMM_MOVING_FMAX}.\n\n"
        f"Import nki, nki.language as nl, and nki.isa as nisa.\n\n{API_CARD}\n\n"
        f"{strategy_card(level)}"
        f"Reply with ONE python code block containing the imports and the function. No prose.")


def repair_prompt(level, source, feedback, echoed=False):
    """One named change, and the previous code. No rules list, no reference re-sent.

    The lesson this whole repo keeps re-learning: feeding a verifier's report back verbatim
    reproduces the same mistake, because a report says what is wrong and never what to do.

    Measured on level 3: "change exactly what the checker names and keep everything else identical"
    got the kernel back byte-for-byte, 16 samples of 16 across four runs -- keeping everything
    identical is the easiest way to obey it. So the prompt now says the code as written fails, and
    says so twice as loudly when the last answer was an echo.
    """
    echo_note = ("You already sent this exact code back once, unchanged, and it failed the same way. "
                 "It has to change, starting with the failing line.\n\n" if echoed else "")
    # The card rides along so a repair cannot drift off the level's strategy.
    card = card_for(level)
    return (
        f"This NKI kernel for {nkibench.LEVELS[level]['op']} fails.\n\n"
        f"```python\n{source}\n```\n\n"
        f"A checker ran it and reports:\n{feedback}\n\n"
        f"{card + chr(10) + chr(10) if card else ''}"
        f"{echo_note}"
        f"Make the change the checker names, then reply with the complete corrected kernel in ONE "
        f"python code block. Sending the code back unchanged fails again.")


CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_code(text):
    """Pull out the code, and return NOTHING rather than prose.

    The old fallback returned the whole reply whenever it contained "def ", so a numbered list
    or a sentence reached the compiler and produced "invalid decimal literal on line 2" -- a
    parse error that blamed the model for the extractor's mistake.
    """
    text = text or ""
    blocks = CODE_BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    # No fence: start at the first line that can legally begin a module and keep the rest.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*(import |from |@nki|def )", line):
            return "\n".join(lines[i:]).strip()
    return ""


# ---------------------------------------------------------------- the model

def ask(a, prompt):
    import httpx
    # enable_thinking=False matters. Qwen3 reasons before answering, and with thinking on it
    # spent the whole budget there: the first cluster run returned "No code came back" at 54.7s
    # over and over, plus truncated fragments (invalid decimal literal, unterminated string).
    # Keep prompt + answer inside the server's context, or the answer is silently cut off and
    # every parse error below is really a budget error. Repair prompts grow with the kernel.
    est_prompt = len(prompt) // 4
    budget = min(a.max_tokens, max(256, a.context - est_prompt - 64))
    if budget < a.max_tokens:
        print(f"    (prompt is ~{est_prompt} tokens, so the answer budget is capped at {budget} "
              f"to stay inside the {a.context}-token context)")
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=budget, temperature=0.6, top_p=0.95,
                chat_template_kwargs={"enable_thinking": a.think})
    r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body,
                   timeout=900, verify=False)
    if r.status_code != 200:
        raise SystemExit(f"the endpoint returned HTTP {r.status_code}:\n{r.text[:600]}")
    payload = r.json()
    ch = payload["choices"][0]
    msg = ch.get("message", {})
    reasoning = next((msg[k] for k in REASONING_KEYS if msg.get(k)), "")
    content = msg.get("content") or ""
    finish = ch.get("finish_reason")
    if finish == "length":
        # Do not let a budget problem look like a model failure.
        print(f"    (TRUNCATED: finish_reason=length after {len(content)} chars. The answer was "
              f"cut off, so any parse error below is the budget, not the model. Prompt is "
              f"{len(prompt)} chars; server context is the ceiling.)")
    if not content.strip() and reasoning:
        # The single most common surprise on this endpoint, so name it rather than reporting
        # an empty answer as a model failure.
        print(f"    (empty answer, {len(reasoning)} chars of hidden reasoning, "
              f"finish={ch.get('finish_reason')} — shorten the prompt rather than raising the "
              f"budget)")
    return content


def ask_parallel(a, prompts):
    """One request per prompt, all at once; replies come back in the same order."""
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        return [f.result() for f in [ex.submit(ask, a, p) for p in prompts]]


def offline_answers(level, n, rnd):
    """No model. Replays the shipped reference, preceded by a deliberately broken version, so the
    loop and the feedback path can be exercised with no endpoint. Never report a number."""
    ref = open(f"reference_level{level}.py").read()
    if rnd == 0:
        broken = ref.replace("@nki.jit", "", 1)
        return [f"```python\n{broken}\n```"] * n
    return [f"```python\n{ref}\n```"] * n


# ---------------------------------------------------------------- the loop

def _headline(feedback):
    """The checker's error in one line, without the shape bookkeeping in front of it."""
    first = feedback.split("\n", 1)[0]
    return re.sub(r"^\d+ of \d+ shapes passed\. On [^:]*: ", "", first)[:160]


def _normalized(src):
    return "\n".join(line.rstrip() for line in (src or "").strip().splitlines())


def solve(a, level, log):
    print(f"\n=========== level {level}: {nkibench.LEVELS[level]['op']} ===========")
    terse = a.terse
    n = a.samples
    prompts = [first_prompt(level, terse)] * n
    strategies = ["fresh"] * n
    best = (0.0, None, "")
    tried, streak, seen = [], 0, {}
    graded_before = {}      # normalized code -> (reward, parts, feedback, progress)
    latest = None           # the attempt the next repair starts from
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        replies = (offline_answers(level, n, rnd) if a.offline else ask_parallel(a, prompts))
        earlier = set(graded_before)
        graded = []
        for prompt, strategy, reply in zip(prompts, strategies, replies):
            src = extract_code(reply)
            key = _normalized(src)
            # An echo is code this level has already graded. Measured on level 3: every repair
            # sample of every round was one, so the loop spent three rounds of four regrading a
            # kernel it already knew failed. Reuse the grade, and say so in the next prompt.
            # Two samples of one round agreeing is not an echo, so only earlier rounds count.
            echo = bool(key) and key in earlier
            if key in graded_before:
                reward, parts, feedback, progress = graded_before[key]
            else:
                reward, parts, feedback, progress = grade(src, level)
                graded_before[key] = (reward, parts, feedback, progress)
            graded.append(dict(reward=reward, src=src, feedback=feedback, parts=parts,
                               progress=progress, echo=echo))
            log.write(json.dumps(dict(level=level, round=rnd, reward=reward, parts=parts,
                                      progress=round(progress, 3), echo=echo, strategy=strategy,
                                      prompt_chars=len(prompt), reply_chars=len(reply),
                                      code=src, feedback=feedback)) + "\n")
        log.flush()
        # Rank by reward, then by how far the kernel got, then prefer new code over an echo.
        # Reward alone ties constantly below 0.5 -- every level-3 sample scored 0.30 -- and the
        # tie went to whichever reply came back first.
        graded.sort(key=lambda g: (g["reward"], g["progress"], not g["echo"]), reverse=True)
        top = graded[0]
        if top["reward"] > best[0]:
            best = (top["reward"], top["src"], top["feedback"])
        # Repair the LATEST round's best attempt, not the all-time best. Rebuilding from the best
        # attempt with the best attempt's feedback is a fixed point: once a round scores worse, the
        # prompt stops changing, and a greedy model then returns the same answer forever.
        # Measured: level 2 stuck at 0.10 for four rounds while the prompt still carried the 0.50
        # code.
        if top["src"].strip():
            latest = top
        echoes = sum(g["echo"] for g in graded)
        same = top["feedback"] == (tried[-1] if tried else None)
        if same:
            # Collapse. Fifteen identical multi-line blocks is noise, not information.
            print(f"round {rnd}: same failure again ({top['reward']:.2f}, best so far "
                  f"{best[0]:.2f})" + (f"  [{echoes}/{n} echoed earlier code]" if echoes else ""))
        else:
            print(f"round {rnd}: this round {top['reward']:.2f}  best so far {best[0]:.2f}  "
                  f"({time.perf_counter() - t0:.1f}s)"
                  + (f"  [{echoes}/{n} echoed earlier code]" if echoes else ""))
            print(f"  {top['feedback'][:700]}")
        if top["reward"] >= sum(WEIGHTS.values()) - 1e-9:
            print(f"  SOLVED on round {rnd}. {top['feedback']}")
            print("  ---------------- the kernel ----------------")
            print(textwrap.indent(top["src"], "  "))
            print("  -------------------------------------------")
            return top["reward"], rnd + 1
        seen[top["feedback"]] = seen.get(top["feedback"], 0) + 1
        streak = streak + 1 if same else 1
        if seen[top["feedback"]] >= a.give_up_after:
            how = ("the identical failure %d rounds running" % streak if streak >= a.give_up_after
                   else "this failure for the %dth time, alternating with %d other(s)"
                        % (seen[top["feedback"]], len(seen) - 1))
            print(f"  STOPPING this level: {how}. The agent is cycling between a fixed set of "
                  f"mistakes rather than converging, so more rounds will not help. Failures seen:")
            for f, k in sorted(seen.items(), key=lambda kv: -kv[1]):
                print(f"    {k}x  {f[:110]}")
            return best[0], rnd + 1
        tried.append(top["feedback"])

        if latest is None:
            # Nothing came back to repair. Asking it to "fix" an empty code block produced a
            # 202-character prompt and, under greedy sampling, the identical non-answer six
            # rounds running. Shorten and re-ask instead.
            terse = min(terse + 1, 2)
            prompts, strategies = [first_prompt(level, terse)] * n, ["fresh"] * n
            print(f"  no code yet, so re-asking with a shorter prompt (terseness {terse})")
            continue
        repair = repair_prompt(level, latest["src"], latest["feedback"], echoed=latest["echo"])
        if streak >= 2:
            # Sampling on the gpt-oss endpoint is greedy, so an unchanged prompt returns an
            # unchanged answer. Measured: the same TypeError 19 rounds running. Changing the
            # prompt is the only thing that can change the answer, so say what has already failed.
            ledger = "\n".join(f"- {t[:160]}" for t in dict.fromkeys(tried))
            repair += (f"\n\nThese approaches have already failed, so do something different:\n"
                       f"{ledger}")
            print(f"  same failure {streak}x — adding a ledger of {len(set(tried))} failed "
                  f"attempts to break the repeat")
        # Half the samples repair, half start over from the first prompt. Repairs converge on one
        # kernel; fresh samples are how a round escapes a kernel whose structure is wrong. Round 0
        # on level 3 produced up to four DISTINCT kernels, while repairs produced one. With one
        # sample (the greedy endpoint, where a fresh prompt would replay round 0) it only repairs.
        # The fresh prompt carries the errors seen so far, one line each. Measured on level 4: a
        # verbatim first prompt replayed round 0's kernels, 6 fresh samples of 6 echoes.
        n_repair = max(1, (n + 1) // 2)
        heads = list(dict.fromkeys(_headline(t) for t in tried))[-3:]
        fresh = (first_prompt(level, terse) + "\n\nEarlier attempts failed with these errors, so "
                 "avoid them:\n" + "\n".join(f"- {h}" for h in heads))
        prompts = [repair] * n_repair + [fresh] * (n - n_repair)
        strategies = ["repair"] * n_repair + ["fresh"] * (n - n_repair)
    print(f"  not solved in {a.rounds} rounds; best reward {best[0]:.2f}")
    return best[0], a.rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, choices=sorted(nkibench.LEVELS))
    ap.add_argument("--all", action="store_true", help="levels 1 to 4")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=MIN_ANSWER_TOKENS)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL")
                    or os.environ.get("GPTOSS_BASE_URL"))
    ap.add_argument("--path", default="", help="path to append, e.g. /agg/v1 for gpt-oss")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run the whole thing N times and report a solve RATE. One run is not a "
                         "result: measured, the same config scored 1.00, 1.00 and 0.50 on level 2 "
                         "across three runs with no code change.")
    ap.add_argument("--log", default="attempts.jsonl")
    ap.add_argument("--give-up-after", type=int, default=4,
                    help="stop a level after this many identical failures in a row. Measured: 15 "
                         "was pure waste, because the prompt had stopped changing.")
    ap.add_argument("--terse", type=int, default=0, choices=(0, 1, 2),
                    help="starting prompt length. Measured on gpt-oss-20b: 0 produced 13,245 "
                         "chars of hidden reasoning and no answer, while 1 answered with code. "
                         "Qwen3-8B is fine at 0.")
    ap.add_argument("--context", type=int, default=4096,
                    help="the server's max-model-len; prompt + answer must fit inside it")
    ap.add_argument("--think", action="store_true",
                    help="let the model reason first; costs budget, and it ran out")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()

    if not a.offline:
        # Validate before the first request. An empty or scheme-less value produces a hostname
        # starting with "." and 30 lines of httpx/idna traceback that say nothing about the cause.
        raw = (a.base or "").strip()
        if not raw:
            sys.exit("KERNEL_AGENT_BASE_URL is empty or unset. Point it at a model:\n"
                     "  export KERNEL_AGENT_BASE_URL=http://localhost:8000/v1\n"
                     "  export KERNEL_AGENT_MODEL=Qwen/Qwen3-8B\n"
                     "That is the server ./serve.sh started. The seat pods set these for you; if they are "
                     "missing, you are not in a seat pod. Or pass --offline to run with no model.\n"
                     "(Separately: the shared gpt-oss endpoint is reached by setting "
                     "GPTOSS_BASE_URL and passing --path /agg/v1 -- not needed for the local run.)")
        from urllib.parse import urlparse
        u = urlparse(raw)
        if u.scheme not in ("http", "https") or not u.netloc:
            sys.exit(f"base URL {raw!r} is not usable. It needs a scheme and a host, e.g. "
                     f"http://qwen3-8b:8000/v1.")
        a.base = raw.rstrip("/") + a.path
        print(f"endpoint {a.base}  model {a.model}")
    else:
        print("*** OFFLINE: replaying the reference kernel. Numbers are meaningless. ***")

    levels = sorted(nkibench.LEVELS)[:4] if a.all else [a.level or 1]
    full = sum(WEIGHTS.values())
    history = {lv: [] for lv in levels}

    with open(a.log, "a") as log:
        for rep in range(a.repeat):
            if a.repeat > 1:
                print(f"\n################ run {rep + 1} of {a.repeat} ################")
            results = []
            for level in levels:
                results.append((level,) + solve(a, level, log))
                history[level].append(results[-1][1])

            print("\n=========== summary ===========")
            for level, reward, rounds in results:
                print(f"  level {level}  reward {reward:.2f} after {rounds} round(s)"
                      + ("  SOLVED" if reward >= full - 1e-9 else ""))
            print(f"  solved {sum(1 for _, r, _ in results if r >= full - 1e-9)}/{len(results)}")

    if a.repeat > 1:
        # The number that actually means something. A solve rate over N runs survives the variance
        # that makes any single run uninterpretable.
        print(f"\n=========== over {a.repeat} runs ===========")
        for lv in levels:
            got = history[lv]
            solves = sum(1 for r in got if r >= full - 1e-9)
            print(f"  level {lv}: solved {solves}/{len(got)}  "
                  f"best {max(got):.2f}  worst {min(got):.2f}  "
                  f"mean {sum(got) / len(got):.2f}  all={[round(r, 2) for r in got]}")
        print("\n  Report the rate, not your best run. A level that solves 1 in 3 times is not solved.")
    print(f"\nattempts logged to {a.log}")


if __name__ == "__main__":
    main()
