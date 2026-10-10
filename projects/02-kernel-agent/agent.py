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

# Knowledge layer (opt-in). "v0" leaves every prompt and every piece of feedback byte-identical to
# the organizer's loop; "v1" adds the matmul rules card, installed-nki signatures on exceptions and,
# for levels 5-7, a per-tensor traffic instruction. See nkiknow/.
KNOWLEDGE = "v0"
SEED_KERNEL = None
SECTIONS = {}          # char counts for the attempt just graded / prompt just built
NATIVE_TOOLS = False
_DOC_CACHE = {}        # query string -> lookup text, per process (a run)
_LOG_LOCK = __import__("threading").Lock()


def _kv():
    """Knowledge version as an int: "v0" -> 0 ... "v3" -> 3. Each includes the previous."""
    return int(KNOWLEDGE[1:])


def _reset_sections():
    SECTIONS.update(lookup=0, traffic=0, docs=0)   # "card" belongs to the prompt, not to a grade


SECTIONS["card"] = 0
_reset_sections()


def _lookup_note(error_text, source):
    """v1 only: append real signatures when the failure raised an exception."""
    if _kv() < 1:
        return ""
    from nkiknow import api_lookup
    note = api_lookup.signatures_for(error_text, source)
    SECTIONS["lookup"] += len(note)
    return (" " + note) if note else ""

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



ALLOC_CHECK = False   # --alloc-check: reject SBUF/PSUM tiles allocated with more than 128 lanes


class _AllocWatch:
    """During simulation, record SBUF/PSUM allocations whose first (lane) dimension exceeds 128.
    The simulator only checks the slices each instruction touches, so an over-sized allocation passes
    on CPU but is illegal on the device (gotcha V8)."""
    def __enter__(self):
        import nki.language as _nl
        self.nl, self.orig, self.bad = _nl, _nl.ndarray, []
        def wrapped(shape, *a, **k):
            buf = k.get("buffer", a[1] if len(a) > 1 else None)
            try:
                lanes = int(tuple(shape)[0])
            except Exception:
                lanes = 0
            if lanes > 128 and buf is not None and buf in (_nl.sbuf, _nl.psum):
                self.bad.append((tuple(shape), "psum" if buf is _nl.psum else "sbuf"))
            return self.orig(shape, *a, **k)
        _nl.ndarray = wrapped
        return self
    def __exit__(self, *exc):
        self.nl.ndarray = self.orig
        return False

def grade(source, level):
    """Returns (reward, parts, feedback). Feedback is an INSTRUCTION, never just a verdict."""
    parts = dict(parses=False, rules=False, runs=False, correct=False)
    _reset_sections()

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
                f"{type(e).__name__}: {e}" + _lookup_note(f"{type(e).__name__}: {e}", source))

    failures, passed, intensity = [], 0, None
    for case in spec["shapes"]:
        args, _ = nkibench.make_inputs(case, level)
        before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
        want = spec["ref"](*args)
        watch = _AllocWatch() if ALLOC_CHECK else None
        try:
            if watch:
                watch.__enter__()
            if _kv() >= 1 and level >= 5:
                from nkiknow import traffic
                import inspect as _inspect
                got, counted, attrib = traffic.simulate_and_attribute(
                    kernel, args, list(_inspect.signature(spec["ref"]).parameters))
            else:
                got, counted = nkibench.simulate_and_count(kernel, args)
        except nkibench.NkiMissing as e:
            if watch:
                watch.__exit__()
            return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                    f"CANNOT SIMULATE: {e}")
        except Exception as e:
            if watch:
                watch.__exit__()
            # The lookup is attached later, to the one failure the feedback shows; doing it per
            # shape wasted lookups and over-counted SECTIONS["lookup"] (measured: 3.7k vs ~1.2k).
            failures.append((nkibench.label(case, level),
                             enrich(f"raised {type(e).__name__}: {e}"),
                             f"{type(e).__name__}: {e}"))
            continue
        if watch:
            watch.__exit__()
        parts["runs"] = True
        m = (nkibench.check_inputs_untouched(before, args)
             or nkibench.describe_mismatch(got, want)
             or nkibench.check_traffic_bar(level, counted, args, want))
        if (_kv() >= 1 and level >= 5 and m
                and m.startswith("CORRECT, BUT TOO MUCH HBM TRAFFIC")):
            note = traffic.explain(attrib, args)
            if note:
                m += " " + note
        # A simulator warning about a hardware-correctness hazard counts as a failure even when the
        # numbers happen to match on CPU: the kernel would be wrong on the device.
        hazards = [w for w in counted.get("warnings", [])
                   if "incorrect results on hardware" in w]
        if hazards and not m:
            m = ("CORRECT ON CPU BUT WRONG ON HARDWARE: " + hazards[0]
                 + ". Fix that before anything else -- the simulator agrees with the reference here "
                   "and the device would not.")
        if watch and watch.bad and not m:
            shp, mem = watch.bad[0]
            m = (f"ILLEGAL ON HARDWARE: a {mem.upper()} tile was allocated with shape {shp}, i.e. {shp[0]} lanes; "
                 f"every SBUF/PSUM tile has at most 128 lanes. The simulator only checked the slices you used. "
                 f"Allocate tile-sized buffers (at most 128 in the first dimension) inside the loops instead of "
                 f"one buffer for the whole tensor.")
        if m:
            failures.append((nkibench.label(case, level), m, None))
            continue
        passed += 1
        if level >= 3 and counted["bytes"] and all(k in case for k in ("M", "K", "N")):
            intensity = nkibench.roofline(
                nkibench.matmul_flops(case["M"], case["K"], case["N"]), counted["bytes"])

    if failures:
        lbl, first, raised = failures[0]
        # Measured on v2 level 4 (2026-10-10): every NaN kernel wrote `nl.ndarray(...)` inside a
        # tensor_copy/dma_copy call, so the result went to one fresh tile and the output was filled
        # from another, empty one. Name that, ahead of everything else.
        if _kv() >= 1 and re.search(r"(dst|src)\s*=\s*nl\.ndarray\(", source):
            first = ("Your code allocates a tile inside a call (`dst=nl.ndarray(...)` or "
                     "`src=nl.ndarray(...)`). Every nl.ndarray(...) creates a NEW, EMPTY tile, so the "
                     "data copied into one is never seen by the next call. Allocate each tile once, "
                     "give it a name (`t = nl.ndarray(...)`), and pass that same name to both calls. "
                     + first)
        if raised:
            first += _lookup_note(raised, source)
        if " was read " in first:   # the traffic note (v1+) on the shown failure only
            SECTIONS["traffic"] = len(first.split(" was read ", 1)[1]) + 10
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
    m = re.search(r"Matmul (stationary|moving) free dimension (\d+) exceeds (?:gemm_stationary_fmax=|max )(\d+)",
                  error_text)
    if m and _kv() >= 1:
        which, got, mx = m.group(1), int(m.group(2)), int(m.group(3))
        dim = "M (the stationary tile's second dimension)" if which == "stationary" else \
              "N (the moving tile's second dimension)"
        return (error_text + f" One nc_matmul accepts {dim} of at most {mx}, and yours is {got}. Add a loop "
                f"over that dimension in chunks of at most {mx}, with tiles sized by the chunk, and keep "
                f"every other loop. Check the other matmul dimensions against their limits too.")
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
    if _kv() >= 1 and "does not support the context manager protocol" in error_text:
        return (error_text + " NKI loops are ordinary for-loops: write `for k in nl.affine_range(n):`, "
                "not `with nl.affine_range(n) as k:`.")
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


SYSTEM_GUIDE = False   # --system-guide (v1d): guide goes in a system message
TAGGED = False         # --tagged (v1d): XML-like tags around prompt parts
EXAMPLES3 = False      # --examples3 (v1d): guide with three worked examples
DIVERSE_SAMPLES = False  # --diverse-samples: rotate general planning cues across samples
SAMPLING = {"organizer": dict(temperature=0.6, top_p=0.95),
            "qwen": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0, presence_penalty=1.0)}
DIVERSITY_CUES = (
    "Map each logical axis to an appropriate NKI tile axis.",
    "Choose the output tile first, then derive matching input tiles.",
    "Check loop coverage, including tail tiles and boundary conditions.",
    "Match DMA source and destination tile shapes and element counts.",
)


def diverse_sample_prompts(prompt, n, enabled=False):
    if not enabled:
        return [prompt] * n
    instruction = prompt.rfind("Reply with")
    prompts = []
    for i in range(n):
        cue = DIVERSITY_CUES[i % len(DIVERSITY_CUES)]
        note = f"Planning focus for this sample: {cue}\n\n"
        prompts.append(prompt[:instruction] + note + prompt[instruction:] if instruction >= 0
                       else prompt.rstrip() + "\n\n" + note.rstrip())
    return prompts


def tag_task(prompt):
    """--tagged: wrap the task text in <task>, with the closing instruction line kept last."""
    if not TAGGED:
        return prompt
    i = prompt.rfind("Reply with")
    if i <= 0:
        return f"<task>\n{prompt.strip()}\n</task>"
    return f"<task>\n{prompt[:i].strip()}\n</task>\n\n{prompt[i:]}"


def repair_prompt(level, source, feedback, ledger=None):
    """One named change, and the previous code. No rules list, no reference re-sent.

    The lesson this whole repo keeps re-learning: feeding a verifier's report back verbatim
    reproduces the same mistake, because a report says what is wrong and never what to do.
    """
    last = ("Change exactly what the checker names and keep everything else identical. Reply with "
            "ONE python code block.")
    if TAGGED:
        out = (f"This NKI kernel for {nkibench.LEVELS[level]['op']} is not right yet.\n\n"
               f"<previous_kernel>\n```python\n{source}\n```\n</previous_kernel>\n\n"
               f"A checker reports:\n<checker_feedback>\n{feedback}\n</checker_feedback>\n\n")
        if ledger is not None:
            out += (f"These approaches have already failed, so do something different:\n"
                    f"<already_tried>\n{ledger}\n</already_tried>\n\n")
        return out + last
    out = (f"This NKI kernel for {nkibench.LEVELS[level]['op']} is not right yet.\n\n"
           f"```python\n{source}\n```\n\n"
           f"A checker reports:\n{feedback}\n\n" + last)
    if ledger is not None:
        out += (f"\n\nThese approaches have already failed, so do something different:\n"
                f"{ledger}")
    return out


def matmul_card(level):
    """v1: the matmul rules card for levels >= 3, else ''."""
    if _kv() < 1 or level < 3:
        return ""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nkiknow", "cards",
                        "matmul_rules.md")
    with open(path) as f:
        return f.read().strip() + "\n\n"


def seed_prompt(level, source):
    return (f"This NKI kernel for {nkibench.LEVELS[level]['op']}:\n\n```python\n{source}\n```\n\n"
            f"This kernel is correct. Make it move fewer HBM bytes while staying correct. "
            f"Reply with ONE python code block.")


PRIMER = False   # --primer (v1b): NKI primer + a verified multi-tile example, before the card


def nki_primer():
    if not PRIMER or _kv() < 1:
        return ""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nkiknow", "cards",
                        "nki_primer.md")
    with open(path) as f:
        return f.read().strip() + "\n\n"


GUIDE = False    # --guide (v1c): one general NKI guide for every level, replacing primer + matmul card


def nki_guide():
    name = "nki_guide_examples3.md" if EXAMPLES3 else "nki_guide.md"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nkiknow", "cards", name)
    with open(path) as f:
        text = f.read().strip()
    if TAGGED:
        text = f"<nki_guide>\n{text}\n</nki_guide>"
    return text + "\n\n"


def system_guide():
    """--system-guide (with --guide, knowledge >= v1): the guide text for the system message, else ''."""
    return nki_guide() if (SYSTEM_GUIDE and GUIDE and _kv() >= 1) else ""


def with_knowledge(level, prompt):
    """Prepend knowledge: --guide gives the general NKI guide (no per-operation card); otherwise the v1
    matmul card (and, with --primer, the primer). v0 returns the prompt untouched."""
    if GUIDE and _kv() >= 1:
        entry = nkibench.LEVELS[level]["entry"]
        prompt = (prompt.rstrip() + "\n\n" + PLAN_LAST
                  + f"\nThe function must be named exactly `{entry}` and decorated with `@nki.jit`.")
        card = nki_guide()
        if SYSTEM_GUIDE:
            SECTIONS["card"] = len(card)
            card = ""        # sent as the system message by ask()
            if _kv() >= 3:
                card = LOOKUP_PARAGRAPH + "\n\n"
            return card + prompt
    else:
        card = nki_primer() + matmul_card(level)
    SECTIONS["card"] = len(card)
    if _kv() >= 3:
        card += LOOKUP_PARAGRAPH + "\n\n"
    return card + prompt


PLAN_LAST = ("Begin your code block with comment lines that plan the kernel: every tensor and tile with its shape, "
             "each instruction's dimension limits (lanes <= 128; matmul K <= 128, M <= 128, N <= 512), and one loop "
             "for every dimension that can exceed its limit. Then write the code. The examples above are for "
             "other operations: reuse their patterns, never their slices or variable lists.\n"
             "Before replying, check your code: (1) every tile, including result and accumulator tiles, is tile-sized (never the whole output's shape), has at most 128 lanes, and every operand is within "
             "its instruction's limit; (2) each reduction loop is inside the output-tile loops and the output is "
             "written after it; (3) every tile is allocated once by name, never inside a call; (4) every tensor is "
             "sliced by its own dimensions; (5) loops are `for ... in nl.affine_range(...)` with no `if` on the "
             "loop index.")


LOOKUP_PARAGRAPH = (
    "If you need NKI documentation before writing code, reply with only lines of the form "
    "`LOOKUP <api name or topic>` (at most 3) and nothing else; the answers will be sent to you. "
    "Otherwise reply with the code block.")
LOOKUP_RE = re.compile(r"^\s*`?LOOKUP\s+(.+?)`?\s*$", re.M)
MAX_LOOKUP_EXCHANGES = 2
LOOKUP_TOTAL_TOKENS = 700


def doc_lookup(query, max_tokens=300):
    """retrieve.lookup with a per-run cache. Never returns text sourced from a downloads/ dir."""
    key = (query, max_tokens)
    if key not in _DOC_CACHE:
        from nkiknow import retrieve
        text = retrieve.lookup(query, max_tokens=max_tokens) or ""
        assert not re.search(r"\[source: [^\]]*downloads/", text), "lookup leaked a downloads/ file"
        _DOC_CACHE[key] = text
    return _DOC_CACHE[key]


def docs_query(feedback):
    """Retrieval query from a failure: the exception line if there is one, else the first line."""
    lines = [l.strip() for l in (feedback or "").splitlines() if l.strip()]
    pick = next((l for l in lines if re.search(r"\b\w*(Error|Exception)\b", l)),
                lines[0] if lines else "")
    return re.sub(r"\s+", " ", pick)[:200]


def docs_note(feedback):
    """v2: documentation retrieved for this failure, or ''."""
    q = docs_query(feedback)
    text = doc_lookup(q, 300).strip() if q else ""
    return ("\n\nRelevant NKI documentation:\n" + text) if text else ""


def parse_lookups(reply):
    """v3: 1-3 LOOKUP lines and no code block -> the queries; otherwise []."""
    if CODE_BLOCK.search(reply or ""):
        return []
    qs = [m.strip() for m in LOOKUP_RE.findall(reply or "")]
    return qs if 1 <= len(qs) <= 3 else []


def run_lookups(queries):
    """Answers for one exchange, total capped at LOOKUP_TOTAL_TOKENS (~4 chars per token)."""
    left, out = LOOKUP_TOTAL_TOKENS, []
    for q in queries:
        if left <= 0:
            break
        t = doc_lookup(q, min(300, left)).strip()
        left -= max(1, len(t) // 4)
        out.append(f"## {q}\n{t or '(nothing found)'}")
    return "\n\n".join(out)


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

def ask(a, prompt, messages=None, tools=None, raw=False):
    import httpx
    # enable_thinking=False matters. Qwen3 reasons before answering, and with thinking on it
    # spent the whole budget there: the first cluster run returned "No code came back" at 54.7s
    # over and over, plus truncated fragments (invalid decimal literal, unterminated string).
    # Keep prompt + answer inside the server's context, or the answer is silently cut off and
    # every parse error below is really a budget error. Repair prompts grow with the kernel.
    sysmsg = system_guide()
    if messages is None:
        messages = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + \
                   [{"role": "user", "content": prompt}]
    prompt_chars = sum(len(m.get("content") or "") +
                       (len(json.dumps(m["tool_calls"])) if m.get("tool_calls") else 0)
                       for m in messages)
    if tools:
        prompt_chars += len(json.dumps(tools))
    est_prompt = prompt_chars // 4
    remaining = a.context - est_prompt - 64
    if remaining < 256:
        raise SystemExit(f"prompt is ~{est_prompt} tokens; fewer than 256 answer tokens fit "
                         f"in the {a.context}-token context")
    budget = min(a.max_tokens, remaining)
    if budget < a.max_tokens:
        print(f"    (prompt is ~{est_prompt} tokens, so the answer budget is capped at {budget} "
              f"to stay inside the {a.context}-token context)")
    body = dict(model=a.model, messages=messages,
                max_tokens=budget, **SAMPLING[getattr(a, "sampling", "organizer")],
                chat_template_kwargs={"enable_thinking": a.think})
    if tools:
        body["tools"] = tools
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
    if raw:
        return content, msg.get("tool_calls") or [], msg
    return content


LOOKUP_TOOL = [{"type": "function", "function": {
    "name": "lookup", "parameters": {"type": "object",
                                     "properties": {"query": {"type": "string"}},
                                     "required": ["query"]}}}]


def _log_lookup(log, level, rnd, queries, text):
    if log is None:
        return
    with _LOG_LOCK:
        log.write(json.dumps(dict(type="lookup", level=level, round=rnd, queries=queries,
                                  tokens=len(text) // 4, docs=text, knowledge=KNOWLEDGE)) + "\n")
        log.flush()


def ask_native(a, prompt, level=None, rnd=None, log=None):
    """--native-tools: OpenAI tool calls, max 2 tool turns. Falls back to plain ask() if the
    server rejects tools or never returns tool_calls (the text protocol then applies)."""
    sysmsg = system_guide()
    messages = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + \
               [{"role": "user", "content": prompt}]
    for _ in range(MAX_LOOKUP_EXCHANGES):
        try:
            content, tcs, msg = ask(a, prompt, messages=messages, tools=LOOKUP_TOOL, raw=True)
        except SystemExit:
            return ask(a, prompt)          # server lacks tool support
        if not tcs:
            return content
        messages.append({"role": "assistant", "content": content or None, "tool_calls": tcs})
        qs, got = [], ""
        for tc in tcs[:3]:
            try:
                q = str(json.loads(tc["function"]["arguments"] or "{}").get("query", ""))
            except (ValueError, KeyError, TypeError, AttributeError):
                q = ""
            text = run_lookups([q]) if q else "(no query)"
            qs.append(q)
            got += text
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": text})
        _log_lookup(log, level, rnd, qs, got)
    return ask(a, prompt, messages=messages)   # no tools offered: force the answer


def ask_one(a, prompt, level=None, rnd=None, log=None):
    """One sample. v3 answers LOOKUP replies (not graded rounds, at most 2 exchanges)."""
    if _kv() < 3:
        return ask(a, prompt)
    if NATIVE_TOOLS:
        content = ask_native(a, prompt, level, rnd, log)
    else:
        content = ask(a, prompt)
    docs = ""
    for _ in range(MAX_LOOKUP_EXCHANGES):
        qs = parse_lookups(content)
        if not qs:
            break
        text = run_lookups(qs)
        _log_lookup(log, level, rnd, qs, text)
        docs += ("\n\n" if docs else "") + text
        content = ask(a, prompt + "\n\nDocumentation you requested:\n" + docs)
    return content


def ask_parallel(a, prompt, n, level=None, rnd=None, log=None):
    import concurrent.futures as cf
    prompts = prompt if isinstance(prompt, list) else [prompt] * n
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        return [f.result() for f in [ex.submit(ask_one, a, prompts[i], level, rnd, log)
                                     for i in range(n)]]


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
    prompt = (seed_prompt(level, open(SEED_KERNEL).read()) if SEED_KERNEL
              else tag_task(first_prompt(level, terse)))
    prompt = with_knowledge(level, prompt)
    best = (0.0, None, "")
    tried, streak, seen = [], 0, {}
    latest = ("", "")
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        sample_prompts = diverse_sample_prompts(prompt, a.samples, DIVERSE_SAMPLES)
        replies = (offline_answers(level, a.samples, rnd) if a.offline
                   else ask_parallel(a, sample_prompts, a.samples, level, rnd, log))
        graded = []
        for sample_prompt, reply in zip(sample_prompts, replies):
            src = extract_code(reply)
            reward, parts, feedback = grade(src, level)
            docs = ""
            if _kv() >= 2 and reward < sum(WEIGHTS.values()) - 1e-9:
                docs = docs_note(feedback)
                feedback += docs
            sec = dict(card=SECTIONS["card"], lookup=SECTIONS["lookup"],
                       traffic=SECTIONS["traffic"], code=len(src), feedback=len(feedback))
            if _kv() >= 2:
                sec["docs"] = len(docs)
            graded.append((reward, src, feedback, parts))
            log.write(json.dumps(dict(level=level, round=rnd, reward=reward, parts=parts,
                                      prompt_chars=len(sample_prompt), reply_chars=len(reply),
                                      prompt=sample_prompt, reply=reply, system=system_guide(),
                                      code=src, feedback=feedback,
                                      knowledge=KNOWLEDGE, sections=sec)) + "\n")
        log.flush()
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
            prompt = with_knowledge(level, repair_prompt(level, latest[0], latest[1], ledger))
            print(f"  same failure {repeats}x — adding a ledger of {len(set(tried))} failed "
                  f"attempts to break the repeat")
            continue
        if not (latest[0] or "").strip():
            # Nothing came back to repair. Asking it to "fix" an empty code block produced a
            # 202-character prompt and, under greedy sampling, the identical non-answer six
            # rounds running. Shorten and re-ask instead.
            terse = min(terse + 1, 2)
            prompt = with_knowledge(level, tag_task(first_prompt(level, terse)))
            print(f"  no code yet, so re-asking with a shorter prompt (terseness {terse})")
        else:
            prompt = with_knowledge(level, repair_prompt(level, latest[0], latest[1]))
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
    ap.add_argument("--knowledge", choices=("v0", "v1", "v2", "v3"), default="v0",
                    help="v0: unchanged prompts and feedback. v1: matmul rules card (levels >= 3), "
                         "installed-nki signatures on exceptions, per-tensor traffic note (5-7). "
                         "v2: + retrieved NKI docs appended to failure feedback. "
                         "v3: + model may reply LOOKUP <topic> to fetch docs (not a graded round).")
    ap.add_argument("--native-tools", action="store_true",
                    help="v3: offer a `lookup` function via OpenAI tool calls (max 2 turns); "
                         "falls back to the text protocol if the server has no tool support")
    ap.add_argument("--seed-kernel", default=None, metavar="PATH",
                    help="start from this correct kernel and ask for fewer HBM bytes")
    ap.add_argument("--alloc-check", action="store_true",
                    help="reject SBUF/PSUM tiles allocated with more than 128 lanes (illegal on hardware)")
    ap.add_argument("--guide", action="store_true",
                    help="v1c: general NKI guide for every level instead of primer + matmul card (needs v1+)")
    ap.add_argument("--sampling", choices=("organizer", "qwen"), default="organizer",
                    help="v1d: organizer = temp 0.6/top_p 0.95; qwen = Qwen3 non-thinking card values + presence_penalty 1")
    ap.add_argument("--system-guide", action="store_true",
                    help="v1d: send the guide as a system message (needs --guide and v1+)")
    ap.add_argument("--tagged", action="store_true", help="v1d: XML-like tags around prompt parts")
    ap.add_argument("--examples3", action="store_true",
                    help="v1d: guide with three worked examples (needs --guide)")
    ap.add_argument("--diverse-samples", action="store_true",
                    help="add a different general NKI planning cue to each sample prompt")
    ap.add_argument("--primer", action="store_true",
                    help="v1b: prepend an NKI primer and a verified multi-tile example (needs v1+)")
    ap.add_argument("--ladder-fix", action="store_true",
                    help="use the corrected levels 5-7 (larger shapes, per-shape limits); "
                         "see nkiknow/ladder_fix.py")
    a = ap.parse_args()
    global KNOWLEDGE, SEED_KERNEL, NATIVE_TOOLS
    KNOWLEDGE, SEED_KERNEL, NATIVE_TOOLS = a.knowledge, a.seed_kernel, a.native_tools
    global PRIMER
    PRIMER = a.primer
    global GUIDE
    GUIDE = a.guide
    global ALLOC_CHECK
    ALLOC_CHECK = a.alloc_check
    global SYSTEM_GUIDE, TAGGED, EXAMPLES3, DIVERSE_SAMPLES
    SYSTEM_GUIDE, TAGGED, EXAMPLES3 = a.system_guide, a.tagged, a.examples3
    DIVERSE_SAMPLES = a.diverse_samples
    if a.ladder_fix:
        from nkiknow import ladder_fix
        ladder_fix.apply()

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
