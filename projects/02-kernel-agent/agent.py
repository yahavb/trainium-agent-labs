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

    spec = nkibench.LEVELS[level]
    path = f"/tmp/_agent_level{level}.py"
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
            failures.append((nkibench.label(case, level),
                             enrich(f"raised {type(e).__name__}: {e}")))
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
        if m:
            failures.append((nkibench.label(case, level), m))
            continue
        passed += 1
        if level >= 3 and counted["bytes"]:
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


def available_names(dotted):
    """Turn 'no attribute X' into 'here are the real ones'.

    Measured: told only `module 'nki.language' has no attribute 'value'`, the model guessed
    another invented name every round -- value, scalar, sbuf_scalar, dot. A verdict names the
    mistake and never the fix, so list what actually exists and let it choose.
    """
    import difflib
    import importlib
    mod_name, _, attr = dotted.rpartition(".")
    # Measured on level 1: the model wrote op0=nisa.multiply, and the reply said "nki.isa has no
    # multiply, and nothing similar exists" -- true of nki.isa, but nl.multiply is real and is
    # exactly what the API card shows. The reply then listed nki.isa's first 25 names
    # alphabetically (bn_aggr, dropout, iota ... max8), none relevant, and the model froze for three
    # rounds. So search all three modules, and never dump an alphabetical slice.
    aliases = {"nki.language": "nl", "nki.isa": "nisa", "nki": "nki"}
    here = aliases.get(mod_name, mod_name)

    # Every public name, written the way a kernel writes it: bare name -> ["nl.x", "nisa.x", ...]
    import types
    names = {}
    for mod, alias in aliases.items():
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        for n in dir(m):
            # Skip submodules: `nki.language` is an attribute of nki, not something to call.
            if not n.startswith("_") and not isinstance(getattr(m, n, None), types.ModuleType):
                names.setdefault(n, []).append(f"{alias}.{n}")
    if not names:
        return ""

    # 1. The exact name exists, only in a different module.
    elsewhere = [q for q in names.get(attr, []) if q != f"{here}.{attr}"]
    if elsewhere:
        return (f" `{attr}` is not in `{mod_name}` but it IS real: write `{elsewhere[0]}`, not "
                f"`{here}.{attr}`.")

    # 2. The closest names across nki, nl and nisa, each written with its module. Word parts
    #    first: scalar_mul shares "scalar" with tensor_scalar and "mul" starts multiply, yet
    #    spelling similarity alone ranks it 0.52, below junk like dot -> dropout at 0.60.
    parts = [p for p in attr.lower().split("_") if len(p) >= 3]
    close = [n for n in names
             if any(q == p or q.startswith(p) for p in parts for q in n.lower().split("_"))]
    close += [n for n in difflib.get_close_matches(attr, list(names), n=6, cutoff=0.6)
              if n not in close]
    close = close[:6]
    if close:
        found = [q for n in close for q in names[n] if q != f"{here}.{attr}"][:6]
        return (f" `{here}` has no `{attr}`. The closest real names in nki, nl and nisa are: "
                f"{', '.join(found)}. Pick one of those or use a different approach.")

    # 3. Nothing close anywhere. Point back at the card, which lists the calls these levels use.
    return (f" `{attr}` does not exist in nki, nl or nisa, and nothing similar does. Use only the "
            f"functions listed under 'Available NKI functions'.")


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


def argument_list(func_name):
    """Every argument of an NKI function, split into required and optional, written as a kernel
    writes the call (nisa.x / nl.x). Returns (call_name, required, optional) or None."""
    import inspect
    for mod_name, alias in (("nki.isa", "nisa"), ("nki.language", "nl"), ("nki", "nki")):
        try:
            mod = __import__(mod_name, fromlist=["x"])
        except Exception:
            continue
        fn = getattr(mod, func_name, None)
        if fn is None:
            continue
        try:
            params = inspect.signature(fn).parameters.values()
        except (TypeError, ValueError):
            return None
        plain = [p for p in params if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        required = [p.name for p in plain if p.default is p.empty]
        optional = [f"{p.name}={p.default!r}" for p in plain if p.default is not p.empty]
        return f"{alias}.{func_name}", required, optional
    return None


def enrich(error_text):
    """Add the real names when the failure is an invented API call."""
    if "'MemoryRegion' object is not callable" in error_text:
        return (error_text + " nl.sbuf, nl.psum and nl.shared_hbm are memory regions, not "
                "functions. Do not call them. Allocate with "
                "nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf) and pass the region as the "
                "buffer= argument.")
    # Measured on level 1: told only "tensor_scalar() missing 1 required positional argument:
    # 'operand0'", the model added operand0 and left data=0.5 alone, so the next round failed on
    # data. Name every argument the function takes, so all of them can be checked in one round.
    m = re.search(r"(\w+)\(\) missing \d+ required (?:positional |keyword-only )?arguments?: (.+)",
                  error_text)
    if m:
        missing = re.findall(r"'(\w+)'", m.group(2))
        info = argument_list(m.group(1))
        if not info:
            return error_text + f" Add the missing argument(s): {', '.join(missing)}."
        call, required, optional = info
        return (error_text + f" Add {', '.join(f'`{a}=`' for a in missing)}. {call} takes these "
                f"REQUIRED arguments: {', '.join(required)}"
                + (f"; and these optional ones: {', '.join(optional)}" if optional else "")
                + f". Pass every argument by keyword, and check that each one holds the right kind "
                  f"of value: a tile where the function works on data, a number where it takes a "
                  f"constant.")
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
    # KNOWN GAP, deliberately left as in the original for the --plan vs --plan-merge comparison: the
    # NKI simulator says "cannot reshape TENSOR of size ...", so this numpy-worded rule never fires
    # on tiles. Do NOT fix it as "tiles cannot be reshaped": measured under --plan, reshaping a WHOLE
    # (C, H, W) tile to 5D works; the failure was reshaping a 2D slice with too few elements.
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
        f"Reply with ONE python code block containing the imports and the function. No prose.")


def repair_prompt(level, source, feedback, card=True):
    """One named change, and the previous code. No rules list, no reference re-sent.

    The lesson this whole repo keeps re-learning: feeding a verifier's report back verbatim
    reproduces the same mistake, because a report says what is wrong and never what to do.

    The closing line used to be "Change exactly what the checker names and keep everything else
    identical". Measured on level 1, the model then refused to touch anything the feedback did not
    name -- data=0.5 survived three rounds -- and returned identical code when the feedback was
    vague. It now asks only for the necessary changes.

    card=True re-sends API_CARD (~400 tokens). Measured on level 1 without it: from round 1 on the
    model no longer saw the card, and guessed nisa.multiply, op=nisa.multiply and nisa.scalar_mul
    for three rounds, although the card's nisa.tensor_scalar line is the real call.
    Turn it off with --repair-card 0 to compare.
    """
    api = f"{API_CARD}\n" if card else ""
    return (
        f"This NKI kernel for {nkibench.LEVELS[level]['op']} is not right yet.\n\n"
        f"```python\n{source}\n```\n\n"
        f"{api}"
        f"A checker reports:\n{feedback}\n\n"
        f"Make only the changes that are necessary to fix this. Reply with ONE python code "
        f"block.")


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

THINK_TAGS = re.compile(r"<think>(.*?)(?:</think>|$)", re.S)


def split_thinking(msg, think):
    """Return (thinking, answer) from one chat message.

    Where the reasoning lands depends on how vLLM was started. With a reasoning parser it arrives in
    its own field; without one (serve.sh passes none) it is inline in `content`, as
    <think>...</think> -- and Qwen3's template may already have opened the tag in the prompt, so the
    reply can hold only a closing </think>, or no tag at all if the budget ran out mid-thought.
    """
    content = msg.get("content") or ""
    thinking = next((msg[k] for k in REASONING_KEYS if msg.get(k)), "")
    if "<think>" in content:
        thinking += "".join(THINK_TAGS.findall(content))
        content = THINK_TAGS.sub("", content)
    elif "</think>" in content:
        before, _, content = content.partition("</think>")
        thinking += before
    elif think and not thinking:
        # Thinking was on, the budget ran out before </think>: everything we got is thinking.
        thinking, content = content, ""
    return thinking.strip(), content.strip()


def chat(a, prompt, think, max_tokens, temperature=0.6):
    """One request. Returns dict(thinking, content, finish, seconds, temperature)."""
    import httpx
    # Keep prompt + answer inside the server's context, or the answer is silently cut off and
    # every parse error below is really a budget error. Repair prompts grow with the kernel.
    est_prompt = len(prompt) // 4
    budget = min(max_tokens, max(256, a.context - est_prompt - 64))
    if budget < max_tokens:
        print(f"    (prompt is ~{est_prompt} tokens, so the budget is capped at {budget} "
              f"to stay inside the {a.context}-token context)")
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=budget, temperature=temperature, top_p=0.95,
                chat_template_kwargs={"enable_thinking": think})
    t0 = time.perf_counter()
    r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body,
                   timeout=1800, verify=False)
    if r.status_code != 200:
        raise SystemExit(f"the endpoint returned HTTP {r.status_code}:\n{r.text[:600]}")
    ch = r.json()["choices"][0]
    thinking, content = split_thinking(ch.get("message", {}), think)
    return dict(thinking=thinking, content=content, finish=ch.get("finish_reason"),
                seconds=round(time.perf_counter() - t0, 1), temperature=temperature)


def ask(a, prompt):
    """The original single request. Returns the same dict shape as ask_planned()."""
    # enable_thinking=False matters. Qwen3 reasons before answering, and with thinking on it
    # spent the whole budget there: the first cluster run returned "No code came back" at 54.7s
    # over and over, plus truncated fragments (invalid decimal literal, unterminated string).
    r = chat(a, prompt, a.think, a.max_tokens)
    if r["finish"] == "length":
        # Do not let a budget problem look like a model failure.
        print(f"    (TRUNCATED: finish_reason=length after {len(r['content'])} chars. The answer "
              f"was cut off, so any parse error below is the budget, not the model.)")
    if not r["content"] and r["thinking"]:
        print(f"    (empty answer, {len(r['thinking'])} chars of thinking, finish={r['finish']} "
              f"-- shorten the prompt rather than raising the budget)")
    return dict(reply=r["content"], thinking=r["thinking"], summary="",
                stages=[dict(stage="answer", think=a.think, finish=r["finish"],
                             seconds=r["seconds"])])


SUMMARY_PROMPT = """You were asked to do the task below, and you thought about it first. Your notes are
below the task. They may stop mid-sentence.

=== TASK ===
{task}

=== YOUR NOTES ===
{thinking}

=== NOW ===
Summarize your notes into a short, concrete plan of at most 10 bullet points:
- what the kernel must compute, and the approach that computes it
- if there is existing code: what is wrong with it, and every change it needs
- the exact NKI calls to use, with their arguments
Do not write the full kernel. Reply with the bullet points only."""

CODE_PROMPT = """{task}

A plan worked out for this task:
{summary}

Follow the plan. Reply with ONE python code block."""


def ask_planned(a, prompt, temperature=0.6):
    """--plan: think with a fixed budget, summarize the thinking, then write code without thinking.

    1. thinking ON, at most --think-tokens. Only the thinking is kept; a truncated thought is fine.
    2. thinking OFF: the model condenses its own thinking into a short plan.
    3. thinking OFF: the original prompt plus that plan, answered with code.

    Why this shape. Measured with plain --think: every sample ran out of budget mid-thought and
    returned no code, at 446 s a round. Capping the thought and then asking for the answer in a
    separate request with thinking off guarantees an answer, and the summary keeps the third prompt
    short -- long prompts are what pushed both models in this repo into reasoning instead of answering.
    """
    t = chat(a, prompt, True, a.think_tokens, temperature)
    s = chat(a, SUMMARY_PROMPT.format(task=prompt, thinking=t["thinking"] or "(no notes)"),
             False, a.summary_tokens)
    summary = s["content"]
    c = chat(a, CODE_PROMPT.format(task=prompt, summary=summary or "(no plan)"),
             False, a.max_tokens)
    if c["finish"] == "length":
        print(f"    (TRUNCATED code answer after {len(c['content'])} chars)")
    return dict(reply=c["content"], thinking=t["thinking"], summary=summary,
                stages=[dict(stage=name, think=(name == "think"), finish=r["finish"],
                             seconds=r["seconds"], temperature=r["temperature"],
                             chars=len(r["thinking"] or r["content"]))
                        for name, r in (("think", t), ("summary", s), ("code", c))])


MERGE_PROMPT = """You were asked to do the task below. {n} separate attempts at thinking about it
follow the task. They were written independently, may disagree, and may stop mid-sentence.

=== TASK ===
{task}

{thoughts}

=== NOW ===
Combine the useful parts of ALL the attempts into one short, concrete plan of at most 10 bullet
points. Where they disagree, choose the option that is correct for the task and the NKI functions
listed in it, and drop the rest.
- what the kernel must compute, and the approach that computes it
- if there is existing code: what is wrong with it, and every change it needs
- the exact NKI calls to use, with their arguments
Do not write the full kernel. Reply with the bullet points only."""


def clip_middle(text, max_chars):
    """Keep the opening (the plan the model starts with) and the end (where it had got to)."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 3
    return text[:head] + "\n[... middle omitted ...]\n" + text[-(max_chars - head):]


def ask_merged(a, prompt, n):
    """--plan-merge: n thinkings in parallel, ONE summary of all of them, n code attempts from it.

    1. thinking ON, n requests at once, each at its own temperature from --think-temps.
    2. thinking OFF, one request: every thought, clipped to fit the context, merged into one plan.
    3. thinking OFF, n requests at once: the original prompt plus that one plan. They also cycle
       through --think-temps: with thinking off and an identical prompt, Qwen3 returned the same
       code for every sample on level 1, which would make n attempts cost n and count as one.
    """
    import concurrent.futures as cf
    temps = [a.think_temps[i % len(a.think_temps)] for i in range(n)]
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        thoughts = list(ex.map(lambda t: chat(a, prompt, True, a.think_tokens, t), temps))

    # Fit every thought into what the context leaves after the task and the summary's own budget.
    # Estimated at 3 characters a token rather than chat()'s 4, with 600 tokens spare: an overlong
    # request is rejected by the server, and that ends the whole run, not just this sample.
    room = a.context - len(prompt) // 3 - a.summary_tokens - 600
    per = max(200, room // max(1, n)) * 3                       # characters per thought
    blocks = [f"=== THINKING ATTEMPT {i + 1} (temperature {t['temperature']}) ===\n"
              f"{clip_middle(t['thinking'] or '(no notes)', per)}"
              for i, t in enumerate(thoughts)]
    s = chat(a, MERGE_PROMPT.format(n=n, task=prompt, thoughts="\n\n".join(blocks)),
             False, a.summary_tokens)
    summary = s["content"]

    code_prompt = CODE_PROMPT.format(task=prompt, summary=summary or "(no plan)")
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        codes = list(ex.map(lambda t: chat(a, code_prompt, False, a.max_tokens, t), temps))

    thinking = "\n\n".join(blocks)        # as the summary saw it, clipped
    out = []
    for i, c in enumerate(codes):
        if c["finish"] == "length":
            print(f"    (TRUNCATED code answer {i} after {len(c['content'])} chars)")
        out.append(dict(
            reply=c["content"], thinking=thinking, summary=summary,
            stages=[dict(stage=f"think{j}", think=True, finish=t["finish"], seconds=t["seconds"],
                         temperature=t["temperature"], chars=len(t["thinking"]))
                    for j, t in enumerate(thoughts)]
                   + [dict(stage="summary", think=False, finish=s["finish"], seconds=s["seconds"],
                           chars=len(summary)),
                      dict(stage="code", think=False, finish=c["finish"], seconds=c["seconds"],
                           temperature=c["temperature"], chars=len(c["content"]))]))
    return out


def ask_parallel(a, prompt, n):
    import concurrent.futures as cf
    if a.plan_merge:
        return ask_merged(a, prompt, n)
    if a.plan:
        temps = [a.think_temps[i % len(a.think_temps)] for i in range(n)]
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            return list(ex.map(lambda t: ask_planned(a, prompt, t), temps))
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        return [f.result() for f in [ex.submit(ask, a, prompt) for _ in range(n)]]


def offline_answers(level, n, rnd):
    """No model. Replays the shipped reference, preceded by a deliberately broken version, so the
    loop and the feedback path can be exercised with no endpoint. Never report a number."""
    ref = open(f"reference_level{level}.py").read()
    code = ref.replace("@nki.jit", "", 1) if rnd == 0 else ref
    return [dict(reply=f"```python\n{code}\n```", thinking="", summary="", stages=[])] * n


# ---------------------------------------------------------------- the transcript

def write_transcript(out, run, level, rnd, prompt, records):
    """Append one round to the human-readable transcript: the prompt once, then every DISTINCT
    sample with its thinking, summary, reply and feedback. Identical samples are printed once.
    Under --plan-merge every sample shares one thinking and one summary, so those print once."""
    if out is None:
        return
    w = lambda s="": print(s, file=out)
    stage_line = lambda st: "stages: " + ", ".join(
        f"{s['stage']} {s['seconds']}s" + (f" t={s['temperature']}" if "temperature" in s else "")
        + f" finish={s['finish']}" for s in st)
    rewards = [round(r["reward"], 2) for r in records]
    keys = [(r["thinking"], r["summary"], r["reply"]) for r in records]
    shared = len({(r["thinking"], r["summary"]) for r in records}) == 1 and records[0]["summary"]
    w("=" * 80)
    w(f"run {run}  level {level}  round {rnd}  rewards {rewards}  "
      f"({len(set(keys))} distinct of {len(records)} samples)")
    w("-" * 30 + " PROMPT " + "-" * 30)
    w(prompt)
    if shared:
        r = records[0]
        if r["stages"]:
            w(stage_line([s for s in r["stages"] if s["stage"] != "code"]))
        w("-" * 30 + " THINKING (all attempts, as the summary saw them) " + "-" * 5)
        w(r["thinking"])
        w("-" * 30 + " SUMMARY (shared by every sample) " + "-" * 5)
        w(r["summary"])
    shown = set()
    for i, (r, k) in enumerate(zip(records, keys)):
        if k in shown:
            continue
        shown.add(k)
        same = [j for j, x in enumerate(keys) if x == k and j != i]
        w("#" * 30 + f" SAMPLE {i}" + (f" (same as {same})" if same else "") + " " + "#" * 20)
        if r["stages"]:
            w(stage_line([s for s in r["stages"] if not shared or s["stage"] == "code"]))
        if r["thinking"] and not shared:
            w("-" * 30 + " THINKING " + "-" * 28)
            w(r["thinking"])
        if r["summary"] and not shared:
            w("-" * 30 + " SUMMARY " + "-" * 29)
            w(r["summary"])
        w("-" * 30 + " REPLY " + "-" * 31)
        w(r["reply"])
        w("-" * 30 + f" FEEDBACK (reward {round(r['reward'], 2)}) " + "-" * 15)
        w(r["feedback"])
    out.flush()


# ---------------------------------------------------------------- the loop

def solve(a, level, log, transcript=None, run=0):
    print(f"\n=========== level {level}: {nkibench.LEVELS[level]['op']} ===========")
    terse = a.terse
    prompt = first_prompt(level, terse)
    best = (0.0, None, "")
    tried, streak, seen = [], 0, {}
    latest = ("", "")
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        replies = (offline_answers(level, a.samples, rnd) if a.offline
                   else ask_parallel(a, prompt, a.samples))
        graded, records = [], []
        for r in replies:
            reply = r["reply"]
            src = extract_code(reply)
            reward, parts, feedback = grade(src, level)
            graded.append((reward, src, feedback, parts))
            rec = dict(run=run, level=level, round=rnd, reward=reward, parts=parts,
                       prompt_chars=len(prompt), reply_chars=len(reply),
                       prompt=prompt, thinking=r["thinking"], summary=r["summary"],
                       reply=reply, stages=r["stages"], code=src, feedback=feedback)
            records.append(rec)
            log.write(json.dumps(rec) + "\n")
        log.flush()
        write_transcript(transcript, run, level, rnd, prompt, records)
        graded.sort(key=lambda g: g[0], reverse=True)
        top = graded[0]
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
            prompt = (repair_prompt(level, latest[0], latest[1], a.repair_card)
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
            prompt = repair_prompt(level, latest[0], latest[1], a.repair_card)
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
    ap.add_argument("--repair-card", type=int, default=1, choices=(0, 1),
                    help="1 (default) puts the API card in every repair prompt as well as the first "
                         "one; 0 is the original behaviour, for comparison")
    ap.add_argument("--think", action="store_true",
                    help="let the model reason first; costs budget, and it ran out")
    ap.add_argument("--plan", action="store_true",
                    help="three requests per sample: think (capped at --think-tokens), summarize "
                         "that thinking into a plan, then write the code with thinking OFF")
    ap.add_argument("--think-tokens", type=int, default=2000,
                    help="--plan: the thinking budget of step 1")
    ap.add_argument("--plan-merge", action="store_true",
                    help="like --plan, but all --samples thinkings feed ONE summary, and every code "
                         "attempt is written from that same summary")
    ap.add_argument("--think-temps", default=None,
                    help="comma-separated temperatures, cycled across samples for the thinking (and, "
                         "under --plan-merge, the code) requests. Default: 0.6 for --plan, "
                         "0.6,0.75,0.9,1.0 for --plan-merge")
    ap.add_argument("--summary-tokens", type=int, default=700,
                    help="--plan: the budget for the summary of step 2")
    ap.add_argument("--transcript", default=None,
                    help="readable transcript of this run (prompt, thinking, summary, reply, "
                         "feedback per round). Default: the --log name with .txt instead of "
                         ".jsonl. Overwritten each time, unlike the .jsonl log, which appends.")
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

    if a.transcript is None:
        a.transcript = os.path.splitext(a.log)[0] + ".txt"
    if a.think_temps is None:
        a.think_temps = "0.6,0.75,0.9,1.0" if a.plan_merge else "0.6"
    a.think_temps = [float(t) for t in a.think_temps.split(",")]
    if a.plan_merge:
        print(f"plan-merge mode: {a.samples} thinkings of {a.think_tokens} tokens at temperatures "
              f"{a.think_temps} -> one summary -> {a.samples} code attempts")
    elif a.plan:
        print(f"plan mode: think {a.think_tokens} tokens -> summary {a.summary_tokens} -> code "
              f"{a.max_tokens}, three requests per sample")

    with open(a.log, "a") as log, open(a.transcript, "w", encoding="utf-8") as transcript:
        print(f"command: {' '.join(sys.argv)}", file=transcript)
        print(f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}  model {a.model}", file=transcript)
        for rep in range(a.repeat):
            if a.repeat > 1:
                print(f"\n################ run {rep + 1} of {a.repeat} ################")
            results = []
            for level in levels:
                results.append((level,) + solve(a, level, log, transcript, rep))
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
    print(f"readable transcript written to {a.transcript}")


if __name__ == "__main__":
    main()