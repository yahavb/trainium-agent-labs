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
import json
import os
import re
import sys
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


def lint_count(feedback):
    """How many static-check problems a grade's feedback reports (0 when the kernel passed the check)."""
    m = re.match(r"A static check found (\d+) problem", feedback or "")
    return int(m.group(1)) if m else 0


def grade(source, level):
    """Returns (reward, parts, feedback). Feedback is an INSTRUCTION, never just a verdict."""
    parts = dict(parses=False, rules=False, runs=False, correct=False)

    if not source.strip():
        return 0.0, parts, ("No code came back. Reply with one python code block containing the "
                            "kernel and nothing else.")
    try:
        compile(source, "<candidate>", "exec")
        parts["parses"] = True
    except SyntaxError as e:
        return (WEIGHTS["parses"] * 0, parts,
                f"The code does not parse: {e.msg} on line {e.lineno}. Send one complete python "
                f"code block.")

    violations = nkibench.check_rules(source, level)
    if violations:
        extra = ""
        if any("no function named" in v for v in violations):
            # Measured: this repeated 15 rounds running, because "there is no function named X"
            # never said what the function should look like. Hand over the exact line.
            import inspect
            ref = nkibench.LEVELS[level]["ref"]
            args = ", ".join(inspect.signature(ref).parameters)
            extra = (f" Start the function with exactly this line:  "
                     f"def {nkibench.LEVELS[level]['entry']}({args}):  "
                     f"and put @nki.jit on the line above it.")
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                "Rule violations, which score zero however fast the kernel is. Fix exactly "
                "these: " + " ".join(violations) + extra)
    parts["rules"] = True

    if LINT and level >= 3:
        # --lint: the simulator stops at the FIRST error, so each round fixed one placement mistake and
        # revealed the next. Measured on 108 logged level-8 attempts: a failing kernel had 3-6 such
        # mistakes at once; the static check found all of them (and flags none in any verified kernel).
        from lint import lint_kernel, doc_notes
        issues = lint_kernel(source)
        if issues:
            notes = doc_notes(issues)
            return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                    f"A static check found {len(issues)} problem(s) before running the kernel. Fix ALL "
                    f"of them in this reply:\n" + "\n".join(f"- {i}" for i in issues)
                    + ("\nThe rules, from the AWS NKI docs:\n" + "\n".join(f"- {n}" for n in notes) if notes else ""))

    spec = nkibench.LEVELS[level]
    path = f"/tmp/_agent_level{level}_{os.getpid()}.py"
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
                f"Use exactly those three.")
    except Exception as e:
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                f"The file imports but {spec['entry']} could not be loaded: "
                f"{type(e).__name__}: {e}")

    failures, passed, intensity = [], 0, None
    for case in spec["shapes"]:
        args, _ = nkibench.make_inputs(case, level)
        before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
        want = spec["ref"](*args)
        try:
            got, counted = nkibench.simulate_and_count(kernel, args)
        except nkibench.NkiMissing as e:
            return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                    f"CANNOT SIMULATE: {e}")
        except Exception as e:
            loc = ""
            if LINT:   # name the line that raised, so the model knows WHERE, not only what
                import traceback
                frames = [fr for fr in traceback.extract_tb(e.__traceback__) if fr.filename == path]
                if frames:
                    # Read the line from THIS candidate's source, not Python's line cache: every candidate
                    # in a process is written to the same temp path, so the cache can hold a previous one.
                    src_lines = source.splitlines()
                    n = frames[-1].lineno
                    text = src_lines[n - 1].strip() if 0 < n <= len(src_lines) else ""
                    loc = f" (at line {n}: `{text[:100]}`)"
            failures.append((nkibench.label(case, level),
                             enrich(f"raised {type(e).__name__}: {e}{loc}", level)))
            continue
        if got is None or np.ndim(got) == 0:
            # Measured on level 8: a reply that looped on one comment line until the token limit left
            # a body that does nothing; it "ran", returned nothing, and scored 0.50 -- above every
            # honest attempt at 0.30, so the loop carried it forward. Returning nothing is not running.
            failures.append((nkibench.label(case, level),
                             "returned nothing: the kernel must compute into an nl.shared_hbm tensor and "
                             "end with `return out`."))
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
        if m and LEVEL_HINTS and level in (5, 6, 7):
            m = level_hint(m, level) or m
        if m:
            failures.append((nkibench.label(case, level), m))
            continue
        passed += 1
        # Level 8 (attention) cases are seq/dim, not M/K/N: a CORRECT level-8 kernel used to
        # raise KeyError 'M' here and crash the run before it could print SOLVED.
        if level >= 3 and counted["bytes"] and "M" in case:
            intensity = nkibench.roofline(
                nkibench.matmul_flops(case["M"], case["K"], case["N"]), counted["bytes"])

    if failures:
        lbl, first = failures[0]
        return (sum(WEIGHTS[k] for k, v in parts.items() if v)
                + WEIGHTS["correct"] * passed / len(spec["shapes"]), parts,
                f"{passed} of {len(spec['shapes'])} shapes passed. On {lbl}: {first}")

    parts["correct"] = True
    reward = sum(WEIGHTS.values())
    note = "Correct on every shape."
    if intensity:
        note += " " + nkibench.explain_roofline(intensity)
    return reward, parts, note


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


# Levels 8-10 use calls the base API_CARD never shows (it covers the matmul levels). Measured over
# 866 attempts: ~15% of all failures were invented names/arguments, and on levels 8-10 every round
# hit a NEW one (nisa.subtract, dma_copy(offset=), nc_transpose(src=), .buffer on a memory region).
# The repo already found that an API card "fixed the invented names" for the matmul levels -- so show
# the real call for each new operation UP FRONT, one line each. Calls, not a kernel.
ATTN_API_CARD = """More NKI calls, exactly as they are spelled (x_sb is an sbuf tile of shape (n, d)):

  t_ps = nl.ndarray((d, n), dtype=nl.float32, buffer=nl.psum)
  nisa.nc_transpose(dst=t_ps, data=x_sb)            transpose on the Tensor engine, into psum
  t_sb = nl.ndarray((d, n), dtype=nl.float32, buffer=nl.sbuf)
  nisa.tensor_copy(dst=t_sb, src=t_ps)              psum -> sbuf (matmul operands must be sbuf)
  m = nl.max(x_sb, axis=[1], keepdims=True)         row max, shape (n, 1); nl.sum is the same form
  nisa.tensor_scalar(dst=y, data=x_sb, op0=nl.subtract, operand0=m)   per-row subtract (op0=nl.multiply to scale)
  nisa.activation(dst=y, data=y, op=nl.exp)         elementwise exp
  nisa.reciprocal(dst=r, data=s)                    r and s are (n, 1) sbuf tiles

Operators live in nl (nl.subtract, nl.multiply, nl.exp), never in nisa. Allocate every tile with
nl.ndarray(shape, dtype=..., buffer=...); never index a tile with a tuple of Python ranges.
"""


def api_extra(level):
    return f"\n\n{ATTN_API_CARD}" if LEVEL_HINTS and level in (8, 9, 10, 11) else ""


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
    names = [n for n in dir(mod) if not n.startswith("_")]
    close = difflib.get_close_matches(attr, names, n=6, cutoff=0.4)
    if close:
        return (f" `{mod_name}` has no `{attr}`. The closest real names are: "
                f"{', '.join(close)}. Pick one of those or use a different approach.")
    return (f" `{mod_name}` has no `{attr}`, and nothing similar exists. Its real names include: "
            f"{', '.join(sorted(names)[:25])}.")


def real_signature(func_name):
    """The actual signature of an NKI function, for when the model invents arguments."""
    import inspect
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


LINT = False             # --lint: static memory/operator check before simulating; line numbers on errors
LEVEL_HINTS = False      # --level-hints: the level-aware messages below; off = the original ladder


def _hoisted_structure(level):
    """Levels 5-7 in words: the level-4 three-loop matmul, with the loads hoisted so each chunk
    crosses the bus once. Measured on the harness shapes: lhsT-only hoisting is still 1.86x on
    K=256 M=512 N=1024 (fails the 1.60 bar); n outermost with the rhs column block loaded once per
    n is 1.14x (clears 1.60 and 1.25); 1.05 needs both inputs loaded exactly once.

    v8: measured, the level-7 text ("load every chunk of both inputs once, before the loops") was
    SOLVED on round 1 when seeded, while the levels 5/6 text (n outermost, reload per n) stalled on
    UnboundLocalError and copy-size walls. The level-7 structure clears every bar (1.00x at these
    shapes) and is simpler to follow, so it is now used for levels 5 and 6 too. The older text is
    kept below for the record."""
    if level in (5, 6, 7):
        return (" This is a matmul and this level grades HBM bytes, so tile all three dimensions "
                "AND load every chunk of BOTH inputs exactly once, before the loops: build nested "
                "Python lists lhsT_tiles[k // 128][m // 128], each a (128, 128) sbuf tile filled by "
                "dma_copy from lhsT[k:k+128, m:m+128], and rhs_tiles[k // 128][n // 512], each a "
                "(128, 512) sbuf tile filled by dma_copy from rhs[k:k+128, n:n+512]. Then loop m "
                "over M in steps of 128 and n over N in steps of 512: allocate ONE psum tile of "
                "shape (128, 512), float32, for this (m, n) output tile, then loop k over K in "
                "steps of 128 and nc_matmul(dst=that psum tile, stationary=lhsT_tiles[k // 128]"
                "[m // 128], moving=rhs_tiles[k // 128][n // 512]) into that SAME psum tile so the "
                "k chunks add up. After the k loop, tensor_copy psum into a fresh (128, 512) sbuf "
                "tile and dma_copy it to result[m:m+128, n:n+512] -- that is the only dma_copy "
                "inside the loops. nc_matmul reads ONLY sbuf tiles: never pass lhsT or rhs (the "
                "HBM inputs) or a slice of them to it. Every sbuf tile has exactly its chunk's 2-D "
                "shape; never reuse one scratch tile for different things.")
    return (" This is a matmul and this level grades HBM bytes, so tile all three dimensions AND "
            "load each rhs chunk only once. Make n the OUTERMOST loop, over N in steps of 512. "
            "Right after `for n`, before any other loop, load the whole rhs column block once: "
            "rhs_tiles = [], and for each k in range(0, K, 128) allocate a (128, 512) sbuf tile, "
            "dma_copy rhs[k:k+128, n:n+512] into it and append it. Then loop m over M in steps of "
            "128: allocate ONE psum tile of shape (128, 512), float32, for this (m, n) output "
            "tile, then loop k over K in steps of 128: dma_copy the lhsT chunk lhsT[k:k+128, "
            "m:m+128] into a fresh (128, 128) sbuf tile and nc_matmul(dst=that psum tile, "
            "stationary=that lhsT tile, moving=rhs_tiles[k // 128]) into that SAME psum tile so "
            "the k chunks add up. After the k loop, tensor_copy psum into a fresh (128, 512) sbuf "
            "tile and dma_copy it to result[m:m+128, n:n+512]. Do not dma_copy rhs inside the m or "
            "k loop. nc_matmul reads ONLY sbuf tiles: never pass lhsT or rhs (the HBM inputs) or a "
            "slice of them to it. Every sbuf tile has exactly its chunk's 2-D shape; never reuse "
            "one scratch tile for different things.")


def level_hint(error_text, level):
    """The ONE structural change a level's measured wall needs, when the generic message is not it.

    Levels 1, 3 and 4 sat at 0.30 / 0.30 / 0.62 with zero spread over five runs: each is one idiom
    the model lacks, and the generic per-error message names a local fix (chunk this copy) where
    the kernel needs a different SHAPE. This names that shape in words -- no worked kernel, which
    was measured to make every level worse.
    """
    # Every SHAPE error at the matmul levels is the same missing idea, whichever symptom shows up
    # first. Measured: with the hint on only one symptom, level 3 hit "at least 2 dimensions" then
    # "cannot reshape", and level 4 an out-of-bound index, and the hint never fired.
    size_wall = re.search(r"partition dimension (\d+) exceeds maximum|dma_copy requires src and dst "
                          r"to have the same number of elements|Out-of-bound access|cannot reshape "
                          r"array|must have at least 2 dimensions|could not be broadcast", error_text)
    # Measured at levels 3 and 5: a kernel correct in every other line tensor_copies the (M, N)
    # result into the (K, N) rhs tile. The numbers in the error say exactly that, so say it.
    bc = re.search(r"value array of shape \((\d+),?\) could not be broadcast to indexing result of "
                   r"shape \((\d+),?\)", error_text)
    if level >= 3 and bc and bc.group(1) != bc.group(2):
        error_text = (error_text + f" You copied {bc.group(1)} elements into a tile that holds "
                      f"{bc.group(2)}: a result was written into a tile allocated for something "
                      f"else -- for example tensor_copy of the (M, N) psum result into the (K, N) rhs "
                      f"tile. Every copy destination needs its OWN tile with the source's shape: "
                      f"allocate a NEW sbuf tile for the result, e.g. "
                      f"res = nl.ndarray((M, N), dtype=..., buffer=nl.sbuf), tensor_copy the psum "
                      f"into res, and dma_copy res to the output.")
    # Level 2: measured on Qwen3-8B, 24 of 24 baseline attempts: the model allocates the tile as
    # (F1, F2) -- it takes the partition axis for one of the two transposed dimensions -- and
    # swaps elements with Python assignment tile[i, j] = tile[j, i]. The transpose is INSIDE each
    # row: x is (P, F1*F2) and every row is its own flattened F1 x F2 matrix. Same trigger set
    # as the matmul levels, because the symptom moves (copy size -> out-of-bound) as it edits.
    if level == 2 and size_wall:
        sh = re.search(r"shape=\((\d+), (\d+)\) as (\d+)x(\d+)", error_text)
        nums = ""
        if sh:
            P, F, F1, F2 = (int(g) for g in sh.groups())
            nums = (f" Here x is ({P}, {F}) and shape2D is ({F1}, {F2}): the tile that holds x "
                    f"must be ({P}, {F}), not ({F1}, {F2}); {P} is the partition size and the "
                    f"{F} columns are the {F1}x{F2} matrix. Column i*{F2}+j of a row moves to "
                    f"column j*{F1}+i of the same row.")
        return (error_text + " The partition axis is NOT one of the two transposed dimensions. "
                "x has shape (P, F1*F2) and EVERY ROW of x is its own flattened F1 x F2 matrix, "
                "so the transpose only reorders the columns within each row; P stays where it "
                "is. Do this: dma_copy the WHOLE x into one sbuf tile of shape (P, F1*F2) -- P "
                "is at most 128 at this level -- and allocate a second sbuf tile of the same "
                "shape (P, F1*F2) for the result. Then loop i over F1 and j over F2 with "
                "nl.affine_range and move ONE column at a time with nisa.tensor_copy: the "
                "source is all partitions of column i*F2+j of the input tile and the "
                "destination is all partitions of column j*F1+i of the result tile, each "
                "selected as tile[:, nl.ds(start, 1)]. Never swap elements with Python "
                "assignment like tile[i, j] = tile[j, i]: data moves only with "
                "nisa.tensor_copy and nisa.dma_copy. Finally dma_copy the result tile to the "
                "shared_hbm output." + nums)
    if level == 3 and size_wall:
        # Measured on Qwen3-8B: one (128, 512) scratch tile reused for lhsT, rhs AND the result,
        # so every copy is the wrong size. 14 of 16 level-3 attempts.
        return (error_text + " Give every operand its OWN tile with exactly its own shape, and never "
                "reuse one scratch tile for different things: lhsT is (K, M) so copy it into an sbuf "
                "tile of shape (K, M); rhs is (K, N) so copy it into an sbuf tile of shape (K, N); "
                "nc_matmul into a psum tile of shape (M, N), float32; tensor_copy that into an sbuf "
                "tile of shape (M, N); dma_copy that to the (M, N) output. Read K, M, N from the "
                "input shapes. Move data ONLY with nisa.dma_copy and nisa.tensor_copy -- never with "
                "Python assignment like out[...] = tile, which cannot move data between memories.")
    # Level 4: the first hint makes it tile K, after which the wall becomes a copy-size mismatch
    # (measured) -- so the structural hint has to keep firing on that error too.
    if level in (5, 6, 7):
        # Measured (v5, 36 attempts): the name "hoist_load" makes the model load whole tensors
        # once and feed the HBM input rhs straight to nc_matmul (13), or reuse one scratch tile
        # (14). The generic message says "allocate the moving tile in sbuf", which is not what
        # happened. And the traffic bar is a loop-order problem the generic ladder never names.
        bar = re.search(r"TOO MUCH HBM TRAFFIC FOR THIS LEVEL: moving ([\d.]+)x the byte floor, "
                        r"and level \d requires ([\d.]+)x", error_text)
        if bar:
            return (error_text + f" The numbers are right; the loop ORDER is the problem: at "
                    f"{bar.group(1)}x, a chunk of one input is dma_copy-ed again for every chunk "
                    f"of the other instead of once. Keep the same arithmetic and change only "
                    f"where the loads happen." + _hoisted_structure(level))
        hbm_operand = re.search(r"(stationary|moving) must be in \['sbuf'\], got (private|shared)_hbm",
                                error_text)
        if hbm_operand:
            return (error_text + f" You passed the HBM input itself as `{hbm_operand.group(1)}`. "
                    f"nc_matmul cannot read HBM, and no whole-tensor copy fits a tile either."
                    + _hoisted_structure(level))
        if size_wall:
            n = re.search(r"got src=(\d+), dst=(\d+)", error_text)
            if n:  # v7: name the numbers, as the level-3 fix did
                error_text += (f" You allocated a tile of {n.group(2)} elements for a chunk of "
                               f"{n.group(1)}: allocate each sbuf tile with the 2-D shape of the exact "
                               f"slice you copy into it, e.g. (128, 128) for an lhsT chunk and (128, 512) "
                               f"for an rhs chunk -- both dimensions, never one.")
            return error_text + _hoisted_structure(level)
    if level == 4 and size_wall:
        return (error_text + " This is a matmul, so chunking one copy is not enough: tile ALL THREE "
                "dimensions. Loop m over M in steps of 128 and n over N in steps of 512. Inside, "
                "allocate ONE psum tile of shape (128, 512), float32, for this (m, n) output tile. "
                "Then loop k over K in steps of 128: dma_copy the lhsT chunk [k-chunk, m-chunk] into a "
                "(128, 128) sbuf tile and the rhs chunk [k-chunk, n-chunk] into a (128, 512) sbuf tile, "
                "and nc_matmul into that SAME psum tile, so the k chunks add up. Only after the k loop, "
                "tensor_copy psum to sbuf and dma_copy it to result[m-chunk, n-chunk]. Allocate every "
                "sbuf tile inside its loop with the chunk's shape.")
    # Level 8 (attention). Measured on Qwen3-8B, round 0 of the baseline: reshape of the (seq, dim)
    # input into (128, 128), out-of-bound index 64 on dim 64, an invented offset= on dma_copy. The
    # kernel it needs is a composition every shape of which fits ONE tile, so the hint names the
    # composition in words. Structure verified in this simulator on all three shapes at the byte floor.
    api_wall = re.search(r"unexpected keyword argument|has no attribute|not callable|must be in \[|transpose requires shape",
                         error_text)
    # Levels 9 and 10: attention's two pieces as their own levels (curriculum). Same error-first
    # style as level 8: one named fix, the exact signatures when an API name is invented, a short plan.
    if level in (9, 10) and (size_wall or api_wall or "unsupported operand" in error_text
                             or "NUMERICAL MISMATCH" in error_text or "NON-FINITE" in error_text):
        sig = (" The ONLY argument names are: nisa.nc_transpose(dst=, data=); nisa.tensor_copy(dst=, "
               "src=); nisa.dma_copy(dst=, src=); nisa.activation(dst=, data=, op=); "
               "nisa.reciprocal(dst=, data=); nisa.tensor_scalar(dst=, data=, op0=, operand0=); "
               "nl.max(tile, axis=[1], keepdims=True); nl.sum(tile, axis=[1], keepdims=True).")
        fix = ""
        if "unexpected keyword argument" in error_text or "has no attribute" in error_text:
            fix = sig
        elif "transpose requires shape" in error_text:
            fix = (" That transpose went to the Vector engine (32x32 limit) because its destination was "
                   "sbuf. nisa.nc_transpose(dst=t_ps, data=x_sb) with t_ps allocated in nl.psum, float32, "
                   "shape (dim, seq) uses the Tensor engine (up to 128x128); then tensor_copy t_ps into an "
                   "sbuf tile of the same shape.")
        elif "unsupported operand" in error_text:
            fix = (" A tile is not a number: Python * / + - do not work on it. Subtract the row max with "
                   "nisa.tensor_scalar(dst=e, data=x, op0=nl.subtract, operand0=m) and scale rows with "
                   "op0=nl.multiply, operand0=r, where m and r are (seq, 1) tiles.")
        elif "must be in" in error_text:
            fix = (" That instruction cannot read or write that memory. Compute only on sbuf/psum tiles: "
                   "dma_copy the input into sbuf first, and dma_copy the sbuf result to the output last.")
        plan = (" Plan: dma_copy x whole into an sbuf tile of shape (seq, dim) read from x.shape; "
                "nisa.nc_transpose into a psum tile (dim, seq) float32; tensor_copy it into an sbuf tile "
                "(dim, seq); dma_copy that to the (dim, seq) shared_hbm output."
                if level == 9 else
                " Plan, all in sbuf, shapes from x.shape: dma_copy x into an sbuf tile (seq, dim); "
                "m = nl.max(x_sb, axis=[1], keepdims=True); nisa.tensor_scalar(dst=e, data=x_sb, "
                "op0=nl.subtract, operand0=m); nisa.activation(dst=e, data=e, op=nl.exp); "
                "s = nl.sum(e, axis=[1], keepdims=True); nisa.reciprocal(dst=r, data=s) with r an sbuf "
                "tile (seq, 1); nisa.tensor_scalar(dst=p, data=e, op0=nl.multiply, operand0=r); "
                "dma_copy p to the (seq, dim) shared_hbm output.")
        return error_text + fix + plan
    if level in (8, 11) and (size_wall or api_wall or "NON-FINITE" in error_text
                       or "NUMERICAL MISMATCH" in error_text or "unsupported operand" in error_text
                       or "positional argument" in error_text or "contraction dimension" in error_text):
        extra = ""
        if "unexpected keyword argument" in error_text or "has no attribute" in error_text:
            # Measured v2 round 1: 4 of 4 attempts invented nc_matmul(a=, src0=). The level hint
            # pre-empts the generic signature message in enrich(), so name the signatures here.
            extra = (" The ONLY argument names are: nisa.nc_matmul(dst=, stationary=, moving=); "
                     "nisa.nc_transpose(dst=, data=); nisa.tensor_copy(dst=, src=); "
                     "nisa.dma_copy(dst=, src=); nisa.activation(dst=, data=, op=); "
                     "nisa.reciprocal(dst=, data=); nisa.tensor_scalar(dst=, data=, op0=, operand0=); "
                     "nl.max(tile, axis=[1], keepdims=True); nl.sum(tile, axis=[1], keepdims=True). "
                     "Nothing else exists; no a=, b=, src0=, lhs=, rhs=, x=, out=, transposed=.")
        elif "transpose requires shape" in error_text:
            # Measured v1 round 1: the transpose ran on the Vector engine (32x32 limit) because its
            # dst was an sbuf tile. A psum dst is the Tensor engine, which does 128x128.
            extra = (" That transpose went to the Vector engine, which is limited to 32x32, because "
                     "its destination was an sbuf tile (or nl.transpose was used). Use "
                     "nisa.nc_transpose(dst=t_ps, data=x_sb) where t_ps is allocated in nl.psum, "
                     "float32, with the transposed shape (columns of x, rows of x): that is the "
                     "Tensor engine, which transposes up to 128x128. Then nisa.tensor_copy t_ps into "
                     "an sbuf tile of the same shape and use THAT as the matmul operand.")
        elif re.search(r"(stationary|moving) must be in \['sbuf'\], got psum", error_text):
            # Measured v3fixed round 2: transposed operand passed to nc_matmul straight from psum.
            extra = (" nc_matmul reads ONLY sbuf tiles. That operand is still in psum: tensor_copy the "
                     "psum tile into a NEW sbuf tile of the same shape, and pass the sbuf tile.")
        elif "unsupported operand" in error_text:
            extra = (" A tile is not a number, so Python * / + - do not work on it. To scale every row "
                     "by its own value, use nisa.tensor_scalar(dst=out, data=tile, op0=nl.multiply, "
                     "operand0=row_tile) where row_tile has shape (seq, 1); use op0=nl.subtract to "
                     "subtract the row max the same way.")
        elif re.search(r"contraction dimension mismatch", error_text):
            # Audit: the current level-8 wall on two seats; it matched no trigger before.
            cm = re.search(r"stationary\[0\]=(\d+) != moving\[0\]=(\d+)", error_text)
            nums = f" (yours: stationary has {cm.group(1)} rows, moving has {cm.group(2)})" if cm else ""
            extra = (f" nc_matmul contracts over the PARTITION (first) axis of BOTH operands, so their "
                     f"first dimensions must be equal{nums}. For the scores Q K^T both operands need dim "
                     f"on the partition axis: stationary=qT and moving=kT, each (dim, seq). For P V the "
                     f"keys are the partition axis: stationary=pT (seq_k, seq_q), moving=v (seq_k, dim).")
        elif "positional argument" in error_text:
            # Measured: seeded from the level-3 matmul, which takes (lhsT, rhs).
            extra = (" The entry point takes THREE inputs: def nki_attention_(q, k, v): -- each of "
                     "shape (seq, dim). Keep the matmul pattern, but it now runs twice: scores = "
                     "Q K^T, then out = P V.")
        elif "Out-of-bound access" in error_text:
            # Measured v3fixed round 0: "index 64 exceed dimension size of 64".
            extra = (" You indexed past a tile's edge. q, k, v are (seq, dim); the scores are (seq, "
                     "seq); their transposes swap the two. Read every size from q.shape and allocate "
                     "each tile with its exact shape -- never hard-code 128 or 64, and do not index "
                     "inside a tile at all: every operand here is used whole.")
        elif "NON-FINITE" in error_text:
            extra = (" NaN here means exp() overflowed or a tile was read before it was written: "
                     "subtract the row maximum from the scaled scores BEFORE exp, and never read a "
                     "psum tile you did not matmul or transpose into.")
        elif "NUMERICAL MISMATCH" in error_text:
            extra = (" The numbers are wrong, not the structure: check that the scores are "
                     "multiplied by 1/sqrt(dim) (dim is the second axis of q), that the max and the "
                     "sum are taken along axis=[1] (the free axis, one value per query row), and "
                     "that the operand you pass as stationary is the TRANSPOSED one.")
        if level == 11:   # the scores step only: Q K^T / sqrt(d)
            return (error_text + extra + " Plan, everything whole in one tile, sizes from q.shape: "
                    "dma_copy q and k into sbuf (seq, dim); nc_transpose each into a psum tile (dim, seq) "
                    "float32 and tensor_copy to sbuf qT, kT; nc_matmul(dst=psum (seq, seq) float32, "
                    "stationary=qT, moving=kT); tensor_scalar(op0=nl.multiply, operand0=1/sqrt(dim)) into "
                    "an sbuf (seq, seq) tile; dma_copy it to the (seq, seq) output.")
        if extra:
            # v4, from an independent review (Codex) of the attempts: the full recipe appended every
            # round buried the one fix that mattered. Name the failing step first, then a short plan.
            return (error_text + extra + " Plan, everything whole in one tile: qT and kT via "
                    "nc_transpose into psum then tensor_copy to sbuf; scores = nc_matmul(stationary=qT, "
                    "moving=kT) into a (seq, seq) psum; scale, subtract the row max, exp, row sum, "
                    "reciprocal and multiply, all in sbuf; pT the same way; out = nc_matmul(stationary=pT, "
                    "moving=v) into a (seq, dim) psum; tensor_copy to sbuf; dma_copy to the output.")
        return (error_text + " This is attention and every shape here fits in ONE tile (seq <= 128 "
                "on the partition axis), so do not reshape, tile, chunk or offset anything: dma_copy "
                "q, k and v whole into three sbuf tiles of shape (seq, dim), read from q.shape. "
                "nc_matmul contracts over the PARTITION axis, so the scores Q K^T need Q^T and K^T "
                "with dim on the partition axis: nisa.nc_transpose(dst=psum_tile, data=q_sb) into a "
                "psum tile of shape (dim, seq) float32, then nisa.tensor_copy it into an sbuf tile of "
                "shape (dim, seq); same for k. Then nc_matmul(dst=psum (seq, seq) float32, "
                "stationary=qT, moving=kT) gives the scores; scale them into an sbuf tile with "
                "nisa.tensor_scalar(dst=s, data=that psum, op0=nl.multiply, operand0=1/sqrt(dim)). "
                "Softmax along the free axis, entirely in sbuf: m = nl.max(s, axis=[1], keepdims=True) "
                "(shape (seq, 1)); nisa.tensor_scalar(dst=e, data=s, op0=nl.subtract, operand0=m); "
                "nisa.activation(dst=e, data=e, op=nl.exp); ssum = nl.sum(e, axis=[1], keepdims=True); "
                "nisa.reciprocal(dst=r, data=ssum) with r an sbuf tile (seq, 1); "
                "nisa.tensor_scalar(dst=p, data=e, op0=nl.multiply, operand0=r). The second matmul "
                "P V contracts over the keys, so transpose p the same way (nc_transpose into a psum "
                "tile (seq, seq), tensor_copy to sbuf pT) and nc_matmul(dst=psum (seq, dim) float32, "
                "stationary=pT, moving=v_sb); tensor_copy that to an sbuf tile (seq, dim) and dma_copy "
                "it to the (seq, dim) shared_hbm output. The scores never touch HBM. Allocate every "
                "tile with its own exact shape, move data only with dma_copy and tensor_copy, no "
                "Python assignment and no loops." + extra)
    # Level 1, v4. Measured (attempts-l1-hints-v3.jsonl, seat-95): the v3 hint moved the wall from
    # the copy-size error to "ap() pattern has invalid partition stride. Partition step 1 must equal
    # tensor free dimension size 1024. Pattern: [[1, 32], [2, 16], [2, 16], [1, 2], [1, 2]]" in 32 of
    # 40 attempts -- the model writes strides as index steps, not element distances -- and the hint
    # never fired on that error. The same kernels scale with tensor_scalar(dst=sum, data=1/(p*p),
    # op0=nisa.multiply), which is the next wall. So: fire on both, name the stride mistake with the
    # numbers in the error, and give the stride formulas and the real tensor_scalar signature in words.
    if level == 1 and re.search(r"dma_copy requires src and dst to have the same number of elements|"
                                r"ap\(\) pattern|invalid partition stride|tensor_scalar|"
                                r"module 'nki.isa' has no attribute|"
                                r"has no attribute '(sum|reduce|reduce_sum|multiply|scalar_mul|mul|divide|scalar_div)'|"
                                r"sum\(\) got an unexpected keyword|"
                                r"tensor_scalar\(\) got an unexpected keyword", error_text):
        ap = re.search(r"Partition step (\d+) must equal tensor free dimension size (\d+)", error_text)
        stride_note = ""
        if ap:
            stride_note = (f" Your ap() pattern used a partition stride of {ap.group(1)} where it must "
                           f"be {ap.group(2)} = H*W: in ap() every stride is a DISTANCE in elements of "
                           f"the flattened (C, H, W) tile, not an index step of 1.")
        sum_note = ""
        if re.search(r"has no attribute '(sum|reduce|reduce_sum|add)'|sum\(\) got an unexpected keyword", error_text):
            # v4 round 1 (attempts-l1-v4.jsonl): all 4 samples wrote nisa.sum(dst=, src=view, axis=).
            sum_note = (" The reduction is nl.sum, not nisa.sum, and it RETURNS a new tile: write "
                        "sum_tile = nl.sum(view, axis=[3, 4]) with no dst= and no src=.")
        scale_note = ""
        na = re.search(r"module '(nki\.\w+)' has no attribute '(\w+)'", error_text)
        if na and na.group(2) not in ("sum", "reduce", "reduce_sum", "add"):
            # v5 rounds 2-3 (attempts-l1-v5.jsonl): nl.tensor_scalar, then nisa.scalar_mul -- the
            # kernel was otherwise correct. Name the one real call and its module up front.
            scale_note = (f" There is no {na.group(1)}.{na.group(2)}. The ONLY scaling call is "
                          f"nisa.tensor_scalar -- module nki.isa (nisa), function tensor_scalar -- "
                          f"so write exactly nisa.tensor_scalar(dst=out_tile, data=sum_tile, "
                          f"op0=nl.multiply, operand0=1.0 / (pool * pool)) and change nothing else.")
        return (error_text + stride_note + sum_note + scale_note + " The kernel shape is: load the WHOLE input in one dma_copy "
                "into an sbuf tile in_tile with the input's own shape (C, H, W) -- C is the partition "
                "dimension and fits in 128. Build the pooling windows as ONE strided view with five "
                "[stride, count] pairs, view = in_tile.ap([[H*W, C], [pool*W, H//pool], "
                "[pool, W//pool], [W, pool], [1, pool]]): channel (one channel is H*W elements "
                "apart), output row (one output row skips pool input rows = pool*W elements), output "
                "column (pool elements), row inside the window (W elements), column inside the window "
                "(1 element). Then sum_tile = nl.sum(view, axis=[3, 4]) is a new tile of shape "
                "(C, H//pool, W//pool). Scale it with nisa.tensor_scalar(dst=out_tile, data=sum_tile, "
                "op0=nl.multiply, operand0=1.0 / (pool * pool)) where out_tile is a NEW sbuf tile of "
                "sum_tile.shape -- data= is the tile, operand0= is the number, and the op is "
                "nl.multiply (there is no nisa.multiply or nisa.scalar_mul). Finally dma_copy out_tile "
                "to the shared_hbm output once.")
    return None


def enrich(error_text, level=None):
    """Add the real names when the failure is an invented API call."""
    if LEVEL_HINTS and level is not None:
        hinted = level_hint(error_text, level)
        if hinted:
            return hinted
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

def first_prompt(level, terse=0):
    """Deliberately short, and it does NOT list the rules.

    Measured twice in this repo: hand a model an enumerated list of prohibitions and it audits
    itself against each one and returns nothing, while a bigger budget only buys more thinking.
    So the rules live in the checker. Generate freely, let the checker object, then send back one
    named change.
    """
    s = nkibench.LEVELS[level]
    import inspect
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
        f"Operation: {s['op']}\n"
        f"Entry point: a function named `{s['entry']}`, decorated with `@nki.jit`.\n"
        f"It must compute exactly what this NumPy reference computes:\n\n"
        f"{inspect.getsource(s['ref'])}\n"
        f"Hardware limits: a tile's partition dimension is at most {nkibench.PMAX}. For matmul, "
        f"the stationary free dimension is at most {nkibench.GEMM_STATIONARY_FMAX} and the "
        f"moving free dimension at most {nkibench.GEMM_MOVING_FMAX}.\n\n"
        f"Import nki, nki.language as nl, and nki.isa as nisa.\n\n{API_CARD}\n\n"
        f"Reply with ONE python code block containing the imports and the function. No prose."
        + api_extra(level) + blocks_text())


BLOCKS = ""   # --blocks: the agent's own verified kernels from earlier levels, shown as building blocks


def blocks_text():
    return (f"\n\nVerified building blocks you wrote earlier -- each passes its own checker. Reuse "
            f"their calls and patterns exactly:\n{BLOCKS}" if BLOCKS else "")


def repair_prompt(level, source, feedback):
    """One named change, and the previous code. No rules list, no reference re-sent.

    The lesson this whole repo keeps re-learning: feeding a verifier's report back verbatim
    reproduces the same mistake, because a report says what is wrong and never what to do.
    """
    return (
        f"This NKI kernel for {nkibench.LEVELS[level]['op']} is not right yet.\n\n"
        f"```python\n{source}\n```\n\n"
        f"A checker reports:\n{feedback}\n\n"
        f"Change exactly what the checker names and keep everything else identical. Reply with "
        f"ONE python code block." + api_extra(level) + blocks_text())


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

class Reply(str):
    """The model's text, carrying telemetry (real token counts, finish reason, seconds) for the log."""
    meta = {}


def ask(a, prompt, temperature=0.6):
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
                max_tokens=budget, temperature=temperature, top_p=0.95,
                chat_template_kwargs={"enable_thinking": a.think})
    t_start = time.perf_counter()
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
    out = Reply(content)
    usage = payload.get("usage") or {}
    out.meta = dict(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                    finish=finish, seconds=round(time.perf_counter() - t_start, 2))
    return out


PORTFOLIO_TEMPS = (0.2, 0.6, 0.9, 1.2)


def sample_temps(a, n):
    """--portfolio: one temperature per sample. Measured over 866 attempts: in 70% of rounds all four
    samples at temperature 0.6 were the SAME kernel, character for character -- four generations
    paid for one attempt. Spreading the temperatures makes the four samples four real attempts."""
    if getattr(a, "portfolio", False):
        return [PORTFOLIO_TEMPS[i % len(PORTFOLIO_TEMPS)] for i in range(n)]
    return [0.6] * n


# --prompt-portfolio: measured, different temperatures still gave 1-3 distinct kernels out of 4 -- this
# model is too sure of itself for sampling noise to matter. Different FRAMINGS of the same request are a
# stronger lever (an untested assumption until measured). Sample k gets framing k: fix related uses too /
# plan every tile first / rewrite cleanly / smallest change.
FRAMINGS = (
    "\n\nFix what the checker names, then check every OTHER place in the kernel that uses the same tile or "
    "the same call in the same way, and fix those too.",
    "\n\nBefore the code, write a comment block that lists every tile you will allocate: its name, its "
    "shape, and its memory (sbuf, psum or hbm). Then write the kernel so that every instruction reads and "
    "writes tiles from the memory that instruction requires.",
    "\n\nIf the kernel above has several problems, do not patch it: rewrite it cleanly from the start, "
    "using the same calls.",
    "\n\nMake the smallest possible change: fix exactly what the checker names and nothing else.",
)


def sample_prompts(a, prompt, n):
    if getattr(a, "prompt_portfolio", False):
        return [prompt + FRAMINGS[i % len(FRAMINGS)] for i in range(n)]
    return [prompt] * n


def ask_parallel(a, prompt, n):
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        return [f.result() for f in [ex.submit(ask, a, p, t)
                                     for p, t in zip(sample_prompts(a, prompt, n), sample_temps(a, n))]]


def offline_answers(level, n, rnd):
    """No model. Replays the shipped reference, preceded by a deliberately broken version, so the
    loop and the feedback path can be exercised with no endpoint. Never report a number."""
    ref = open(f"reference_level{level}.py").read()
    if rnd == 0:
        broken = ref.replace("@nki.jit", "", 1)
        return [f"```python\n{broken}\n```"] * n
    return [f"```python\n{ref}\n```"] * n


# ---------------------------------------------------------------- the loop

def solve(a, level, log):
    print(f"\n=========== level {level}: {nkibench.LEVELS[level]['op']} ===========")
    terse = a.terse
    prompt = first_prompt(level, terse)
    best = (0.0, None, "")
    tried, streak, seen = [], 0, {}
    latest = ("", "")
    if getattr(a, "seed_from", None):
        # Build on the agent's OWN verified earlier kernel instead of starting from nothing.
        # Levels 5-7 are level 4's matmul plus a traffic bar; from scratch the model relearns
        # tiling, hits the correctness walls again and never reaches the optimization. Seeded, the
        # checker's real verdict on the seed ("correct, but 2.0x the byte floor") is round 0's
        # repair prompt, and the only change asked for is where the loads happen.
        seed = open(a.seed_from).read()
        entry = nkibench.LEVELS[level]["entry"]
        seed = re.sub(r"^def \w+\(", f"def {entry}(", seed, count=1, flags=re.M)
        s_reward, _, s_feedback = grade(seed, level)
        print(f"  seeded from {a.seed_from}: reward {s_reward:.2f} at this level. {s_feedback[:300]}")
        if s_reward >= sum(WEIGHTS.values()) - 1e-9:
            print("  the seed already passes this level")
            return s_reward, 0
        latest = (seed, s_feedback)
        prompt = repair_prompt(level, seed, s_feedback)
    echoed = False
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        if echoed:
            prompt = ("Your last reply returned the kernel below UNCHANGED, so nothing was fixed. "
                      "This time the code must differ: make the change the checker names.\n\n" + prompt)
        replies = (offline_answers(level, a.samples, rnd) if a.offline
                   else ask_parallel(a, prompt, a.samples))
        graded, records, cache = [], [], {}
        temps = sample_temps(a, len(replies))
        for k, reply in enumerate(replies):
            src = extract_code(reply)
            dup = src in cache
            if not dup:   # review finding: identical samples were each simulated again
                cache[src] = grade(src, level)
            reward, parts, feedback = cache[src]
            if getattr(reply, "meta", {}).get("finish") == "length":
                # Cut off at the token limit: measured, these are repetition loops (one comment line
                # written ~100 times), never a finished kernel. Keep them below honest attempts.
                reward = min(reward, WEIGHTS["parses"] + WEIGHTS["rules"])
                feedback = ("Your reply was cut off at the token limit: it repeated the same lines instead "
                            "of finishing the kernel. Send the complete kernel, with short or no comments. "
                            "Last checker result: " + feedback)
            graded.append((reward, src, feedback, parts, k))
            records.append(dict(level=level, run=getattr(a, "_run", 0), round=rnd, sample=k,
                                temperature=temps[k], framing=(k % len(FRAMINGS)) if getattr(a, "prompt_portfolio", False) else 0,
                                reward=reward, parts=parts,
                                prompt_chars=len(prompt), reply_chars=len(reply), dup=dup,
                                **getattr(reply, "meta", {}), **getattr(a, "_provenance", {}),
                                echo=bool(latest[0].strip()) and src.strip() == latest[0].strip(),
                                code=src, feedback=feedback))
        # Ties: every failing level-8 round scores 0.30, and a stable sort then always picked sample 0
        # (the 0.2-temperature one), so --portfolio diversity never reached the next round (audit:
        # seats 95 and 98 ran byte-identical traces). Prefer a sample that changed the kernel, then one
        # whose error differs from last round's.
        prev_fb = tried[-1] if tried else None
        if LINT:
            # Every failing level-8 round scores 0.30, so the reward cannot tell a kernel with one problem
            # left from one with seven. Prefer fewer static-check problems (a kernel that passed the static
            # check counts as zero), then a changed kernel, then a new error.
            graded.sort(key=lambda g: (g[0], -lint_count(g[2]), (g[1] or '').strip() != (latest[0] or '').strip(),
                                       g[2] != prev_fb), reverse=True)
        elif getattr(a, "portfolio", False) or getattr(a, "echo_check", False):
            graded.sort(key=lambda g: (g[0], (g[1] or '').strip() != (latest[0] or '').strip(),
                                       g[2] != prev_fb), reverse=True)
        else:   # original behaviour, so baseline runs stay comparable
            graded.sort(key=lambda g: g[0], reverse=True)
        top = graded[0]
        for rec in records:   # provenance: which sample the loop actually carried forward
            rec["selected"] = rec["sample"] == top[4]
            log.write(json.dumps(rec) + "\n")
        log.flush()
        # --echo-check. Measured: in 39% of repair rounds the best reply WAS the kernel in the prompt,
        # unchanged -- the feedback had no effect, and the loop graded it again as if it were new.
        echoed = (getattr(a, "echo_check", False) and bool((latest[0] or "").strip())
                  and (top[1] or "").strip() == latest[0].strip())
        if echoed:
            print("  (echo: the best reply is the kernel we sent, unchanged)")
        if top[0] > best[0]:
            best = (top[0], top[1], top[2])
        # Repair the LATEST attempt, not the best one. Rebuilding from the best attempt with the
        # best attempt's feedback is a fixed point: once a round scores worse, the prompt stops
        # changing, and a greedy model then returns the same answer forever. Measured: level 2
        # stuck at 0.10 for four rounds while the prompt still carried the 0.50 code.
        if (top[1] or "").strip():
            latest = (top[1], top[2])
        same = top[2] == (tried[-1] if tried else None)
        if same:
            # Collapse. Fifteen identical multi-line blocks is noise, not information.
            print(f"round {rnd}: same failure again ({top[0]:.2f}, best so far {best[0]:.2f})")
        else:
            print(f"round {rnd}: this round {top[0]:.2f}  best so far {best[0]:.2f}  "
                  f"({time.perf_counter() - t0:.1f}s)")
            print(f"  {top[2][:400]}")
        if top[0] >= sum(WEIGHTS.values()) - 1e-9:
            print(f"  SOLVED on round {rnd}. {top[2]}")
            print("  ---------------- the kernel ----------------")
            print(textwrap.indent(top[1], "  "))
            print("  -------------------------------------------")
            return top[0], rnd + 1
        seen[top[2]] = seen.get(top[2], 0) + 1
        streak = streak + 1 if same else 1
        if seen[top[2]] >= a.give_up_after:
            how = ("the identical failure %d rounds running" % streak if streak >= a.give_up_after
                   else "this failure for the %dth time, alternating with %d other(s)"
                        % (seen[top[2]], len(seen) - 1))
            print(f"  STOPPING this level: {how}. The agent is cycling between a fixed set of "
                  f"mistakes rather than converging, so more rounds will not help. Failures seen:")
            for f, n in sorted(seen.items(), key=lambda kv: -kv[1]):
                print(f"    {n}x  {f[:110]}")
            return best[0], rnd + 1
        tried.append(top[2])
        repeats = streak
        if repeats >= 2 and (best[1] or "").strip():
            # Sampling on this endpoint is greedy, so an unchanged prompt returns an unchanged
            # answer. Measured: the same TypeError 19 rounds running. Changing the prompt is the
            # only thing that can change the answer, so say what has already been tried.
            ledger = "\n".join(f"- {t[:160]}" for t in dict.fromkeys(tried))
            prompt = (repair_prompt(level, latest[0], latest[1])
                      + f"\n\nThese approaches have already failed, so do something different:\n"
                        f"{ledger}")
            print(f"  same failure {repeats}x — adding a ledger of {len(set(tried))} failed "
                  f"attempts to break the repeat")
            continue
        if not (latest[0] or "").strip():
            # Nothing came back to repair. Asking it to "fix" an empty code block produced a
            # 202-character prompt and, under greedy sampling, the identical non-answer six
            # rounds running. Shorten and re-ask instead.
            terse = min(terse + 1, 2)
            prompt = first_prompt(level, terse)
            print(f"  no code yet, so re-asking with a shorter prompt (terseness {terse})")
        else:
            prompt = repair_prompt(level, latest[0], latest[1])
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
    ap.add_argument("--seed-from", default=None,
                    help="start from a kernel the agent already verified at an earlier level (e.g. "
                         "solved/level4_*.py for levels 5-7): its function is renamed to this level's "
                         "entry point and the checker's verdict on it becomes round 0's repair prompt")
    ap.add_argument("--blocks", default="",
                    help="comma-separated kernels the agent already verified at other levels, shown in "
                         "every prompt as building blocks (e.g. its matmul, transpose and softmax "
                         "for attention)")
    ap.add_argument("--portfolio", action="store_true",
                    help="give each sample its own temperature (0.2/0.6/0.9/1.2) so they differ")
    ap.add_argument("--echo-check", action="store_true",
                    help="tell the model when its reply is the kernel it was sent, unchanged")
    ap.add_argument("--prompt-portfolio", action="store_true",
                    help="give each sample a different framing (as-is / plan the tiles first / rewrite / "
                         "smallest change) -- measured to be a stronger lever than temperature")
    ap.add_argument("--lint", action="store_true",
                    help="check every tile's memory and every operator BEFORE simulating, and report all "
                         "problems at once with line numbers (the simulator stops at the first)")
    ap.add_argument("--level-hints", action="store_true",
                    help="name the structural change each stuck level needs (levels 1, 3, 4) "
                         "instead of the generic per-error message. Off by default, so baseline "
                         "runs stay comparable with the repo's measurements.")
    a = ap.parse_args()
    # Provenance on every logged attempt: which session, which exact code, which switches. A commit alone
    # misses uncommitted edits, so hash the source files actually used.
    import hashlib, subprocess, uuid
    here = os.path.dirname(os.path.abspath(__file__))
    digest = hashlib.sha256()
    for name in ("agent.py", "nkibench.py", "lint.py"):
        fp = os.path.join(here, name)
        if os.path.exists(fp):
            digest.update(open(fp, "rb").read())
    try:
        commit = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=5).stdout.strip() or None
    except Exception:
        commit = None
    a._provenance = dict(session=uuid.uuid4().hex[:8], source_hash=digest.hexdigest()[:12], commit=commit,
                         flags=" ".join(sys.argv[1:]))
    global LEVEL_HINTS, BLOCKS, LINT
    LEVEL_HINTS = a.level_hints
    LINT = a.lint
    if a.blocks:
        BLOCKS = "\n\n".join(f"```python\n{open(f).read().strip()}\n```" for f in a.blocks.split(","))

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
                a._run = rep   # logged with every attempt, so runs never have to be reconstructed
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
