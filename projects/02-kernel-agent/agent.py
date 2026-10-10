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
from failure_selection import Candidate, select_candidate, classify_failure
from candidate_diversity import build_variants, diversity_metrics
from nki_knowledge import ground_prompt
from shape_repair import constrained_prompt, shape_prompt, failure_input_shapes, failure_input_values
from synthetic_nki.retrieval import example_prompt
from repair_history import RepairHistory
from experiment_metrics import (CASE_RESULTS, GRADE_DIRECTORY, ModelReply,
                                enabled as instrumentation_enabled, record_case,
                                response_metadata)

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
    path = os.path.join(GRADE_DIRECTORY.get() or "/tmp", f"_agent_level{level}.py")
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
        simulation_started = time.perf_counter() if CASE_RESULTS.get() is not None else None
        try:
            got, counted = nkibench.simulate_and_count(kernel, args)
        except nkibench.NkiMissing as e:
            record_case(nkibench.label(case, level), simulation_seconds=(time.perf_counter()-simulation_started) if simulation_started is not None else None, ran=False, numerically_correct=None,
                        verified=False, error=str(e))
            return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                    f"CANNOT SIMULATE: {e}")
        except Exception as e:
            record_case(nkibench.label(case, level), simulation_seconds=(time.perf_counter()-simulation_started) if simulation_started is not None else None, ran=False, numerically_correct=None,
                        verified=False, error=f"{type(e).__name__}: {e}")
            failures.append((nkibench.label(case, level),
                             enrich(f"raised {type(e).__name__}: {e}")))
            continue
        simulation_seconds = (time.perf_counter()-simulation_started) if simulation_started is not None else None
        parts["runs"] = True
        if CASE_RESULTS.get() is not None:
            input_error = nkibench.check_inputs_untouched(before, args)
            numerical_error = nkibench.describe_mismatch(got, want) if not input_error else None
            traffic_error = (nkibench.check_traffic_bar(level, counted, args, want)
                             if not input_error and not numerical_error else None)
            m = input_error or numerical_error or traffic_error
        else:
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
        if CASE_RESULTS.get() is not None:
            record_case(nkibench.label(case, level), simulation_seconds=simulation_seconds, ran=True,
                        inputs_untouched=not bool(input_error),
                        numerically_correct=(not bool(numerical_error)) if not input_error else None,
                        traffic_passed=(not bool(traffic_error)) if not input_error and not numerical_error else None,
                        hardware_hazard_passed=not bool(hazards), verified=not bool(m), error=m)
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
        f"Reply with ONE python code block containing the imports and the function. No prose.")


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
        f"ONE python code block.")


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
    measured = instrumentation_enabled(a)
    started = time.perf_counter() if measured else None
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
    if getattr(a,"adapter_mode",None) is not None:body["adapter_mode"]=a.adapter_mode
    r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body,
                   timeout=getattr(a, "request_timeout", 900), verify=False)
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
    if measured:
        metadata = response_metadata(payload, time.perf_counter() - started)
        metadata['requested_max_tokens'] = budget
        return ModelReply(content, metadata)
    return content


def ask_parallel(a, prompt, n):
    import concurrent.futures as cf
    prompts = [prompt] * n if isinstance(prompt, str) else list(prompt)
    if len(prompts) != n:
        raise ValueError("one prompt is required per candidate")
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
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

def solve(a, level, log):
    print(f"\n=========== level {level}: {nkibench.LEVELS[level]['op']} ===========")
    terse = a.terse
    prompt = first_prompt(level, terse)
    best = (0.0, None, "")
    tried, streak, seen = [], 0, {}
    latest = ("", "")
    selection_policy = getattr(a, "selection_policy", "reward")
    candidate_policy = getattr(a, "candidate_policy", "standard")
    repair_policy = getattr(a, "repair_policy", "standard")
    generation_policy = getattr(a, "generation_policy", "standard")
    shape_analysis = getattr(a, "shape_analysis", "off")
    feedback_policy = getattr(a, "feedback_policy", "legacy")
    example_policy = getattr(a, "example_policy", "off")
    adaptive = getattr(a, "adaptive_repair", False)
    repair_history = RepairHistory()
    measured = instrumentation_enabled(a)
    extended = measured or selection_policy == "diagnostic"
    selection_history = []
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        generation_prompt = prompt
        grounding = None
        shape_plan = None
        symbolic_plan = None
        generation_constraints = None
        kernel_plan = None
        synthetic_context = None
        history_context = None
        if getattr(a,"planner_policy","off") == "hardware" and not latest[0]:
            from kernel_planner import generation_prompt as planned_prompt
            generation_prompt,kernel_plan=planned_prompt(generation_prompt,level,
                context=getattr(a,"context",8192),answer_budget=getattr(a,"max_tokens",2500),model=getattr(a,"model",MODEL))
        if getattr(a,"planner_policy","off") == "hardware" and latest[0]:
            from kernel_planner import semantic_prompt
            generation_prompt,kernel_plan=semantic_prompt(generation_prompt,latest[0],level,
                context=getattr(a,"context",8192),answer_budget=getattr(a,"max_tokens",2500),model=getattr(a,"model",MODEL))
        if generation_policy == "constrained" and not latest[0]:
            generation_prompt, generation_constraints = constrained_prompt(
                generation_prompt, nkibench.LEVELS[level]["op"], model=getattr(a,"model",MODEL),
                context=getattr(a,"context",8192), answer_budget=getattr(a,"max_tokens",2500))
        if (repair_policy == "shape-aware" or feedback_policy == "targeted") and latest[0]:
            generation_prompt, shape_plan = shape_prompt(
                generation_prompt, latest[0], latest[1], streak, input_shapes=failure_input_shapes(latest[0],latest[1],level),input_values=failure_input_values(latest[0],latest[1],level), model=getattr(a,"model",MODEL),
                context=getattr(a,"context",8192), answer_budget=getattr(a,"max_tokens",2500))
        if repair_policy == "grounded" and latest[0] and feedback_policy == "legacy":
            generation_prompt, grounding = ground_prompt(
                prompt, classify_failure(latest[1]), latest[0],
                model=getattr(a, 'model', MODEL), context=getattr(a, 'context', 8192),
                answer_budget=getattr(a, 'max_tokens', MIN_ANSWER_TOKENS))
        if shape_analysis == "sympy" and latest[0]:
            from symbolic_shapes import repair_prompt as symbolic_prompt
            generation_prompt, symbolic_plan = symbolic_prompt(
                generation_prompt, latest[0], latest[1],
                input_shapes=failure_input_shapes(latest[0],latest[1],level),input_values=failure_input_values(latest[0],latest[1],level),
                model=getattr(a,"model",MODEL),context=getattr(a,"context",8192),
                answer_budget=getattr(a,"max_tokens",2500))
        if example_policy == "synthetic" and latest[0]:
            generation_prompt, synthetic_context = example_prompt(
                generation_prompt, classify_failure(latest[1]), latest[0],
                model=getattr(a,"model",MODEL), context=getattr(a,"context",8192),
                answer_budget=getattr(a,"max_tokens",2500))
        if adaptive and latest[0]:
            history_context = repair_history.guidance()
            if history_context:
                from nki_knowledge import local_token_counter
                counter,_=local_token_counter(getattr(a,"model",MODEL))
                if counter(generation_prompt+history_context)<=getattr(a,"context",8192)-getattr(a,"max_tokens",2500)-128:
                    generation_prompt += "\n\n" + history_context
                else:history_context=None
        variants = build_variants(generation_prompt, a.samples, candidate_policy,
                                  repair=bool(latest[0]), repeated=streak, repair_scope=shape_plan["scope"] if shape_plan else None)
        request_prompts = generation_prompt if candidate_policy == "standard" else [v.prompt for v in variants]
        replies = (offline_answers(level, a.samples, rnd) if a.offline
                   else ask_parallel(a, request_prompts, a.samples))
        generation_seconds = time.perf_counter() - t0 if measured else None
        graded = []
        grade_metrics = []
        for reply in replies:
            src = extract_code(reply)
            generated_src = src
            primitive_changes = None
            if getattr(a, "primitive_policy", "off") == "legalize":
                from primitive_legalizer import legalize
                src, primitive_changes = legalize(src)
            semantic_analysis = None
            if getattr(a, "planner_policy", "off") == "hardware":
                from kernel_planner import semantic_gate
                semantic_analysis = semantic_gate(src, level)
            details = []
            case_token = CASE_RESULTS.set(details) if measured else None
            grade_dir = getattr(a, 'grade_dir', None)
            directory_token = GRADE_DIRECTORY.set(grade_dir) if grade_dir else None
            grade_started = time.perf_counter() if measured else None
            try:
                reward, parts, feedback = grade(src, level)
            finally:
                if case_token is not None: CASE_RESULTS.reset(case_token)
                if directory_token is not None: GRADE_DIRECTORY.reset(directory_token)
            grade_metrics.append(dict(checker_seconds=time.perf_counter() - grade_started,
                                      shape_results=details,semantic_analysis=semantic_analysis,primitive_changes=primitive_changes,generated_source=generated_src) if measured else {})
            graded.append((reward, src, feedback, parts))
            if not extended:
                log.write(json.dumps(dict(level=level, round=rnd, reward=reward, parts=parts,
                                          prompt_chars=len(prompt), reply_chars=len(reply),
                                          code=src, feedback=feedback)) + "\n")
        if extended:
            candidates = [Candidate(g[0], g[1], g[2]) for g in graded]
            decision = select_candidate(candidates, selection_policy, selection_history)
            diversity = diversity_metrics([c.code for c in candidates])
            round_seconds = time.perf_counter() - t0 if measured else None
            for index, (reward, src, feedback, parts) in enumerate(graded):
                record = dict(level=level, round=rnd, reward=reward, parts=parts,
                              prompt_chars=len(variants[index].prompt), reply_chars=len(replies[index]),
                              code=src, feedback=feedback)
                record.update(decision.log_metadata(index))
                if candidate_policy == "diverse" or repair_policy == "grounded":
                    record.update(candidate_policy=candidate_policy,
                                  repair_policy=repair_policy,
                                  prompt_strategy=variants[index].strategy,
                                  prompt=variants[index].prompt,
                                  diversity=diversity,
                                  exact_source_hash=diversity['exact_source_hashes'][index],
                                  ast_source_hash=diversity['ast_source_hashes'][index])
                if repair_policy in ("grounded", "shape-aware"):
                    record['grounding'] = grounding
                if shape_plan is not None or generation_policy != "standard":
                    record.update(shape_plan=shape_plan, generation_policy=generation_policy, generation_constraints=generation_constraints)
                if feedback_policy != "legacy" or example_policy != "off" or adaptive:
                    record.update(feedback_policy=feedback_policy,example_policy=example_policy,
                                  adaptive_repair=adaptive,shape_plan=shape_plan,
                                  synthetic_context=synthetic_context,repair_history_context=history_context)
                if getattr(a,"primitive_policy","off") != "off":record.update(primitive_policy=a.primitive_policy,primitive_changes=grade_metrics[index].get("primitive_changes"),generated_source=grade_metrics[index].get("generated_source"))
                if getattr(a,"planner_policy","off") != "off":record.update(planner_policy=a.planner_policy,kernel_plan=kernel_plan,semantic_gate=grade_metrics[index].get("semantic_analysis"))
                if shape_analysis != "off":record.update(shape_analysis=shape_analysis,symbolic_analysis=symbolic_plan)
                if getattr(a,"adapter_mode",None) is not None:record["adapter_mode"]=a.adapter_mode
                if measured:
                    evaluated = {case['case'] for case in grade_metrics[index]['shape_results']}
                    shape_results = grade_metrics[index]['shape_results'] + [
                        dict(case=nkibench.label(case, level), evaluated=False, ran=None,
                             numerically_correct=None, verified=None)
                        for case in nkibench.LEVELS[level]['shapes']
                        if nkibench.label(case, level) not in evaluated]
                    record.update(response_metadata({}))
                    record.update(getattr(replies[index], 'metadata', {}))
                    record.update(run_id=getattr(a, 'run_id', None), repeat_index=getattr(a, 'repeat_index', None),
                                  candidate_policy=candidate_policy, repair_policy=repair_policy,
                                  prompt_strategy=variants[index].strategy,
                                  prompt=variants[index].prompt,
                                  diversity=diversity, exact_source_hash=diversity['exact_source_hashes'][index],
                                  ast_source_hash=diversity['ast_source_hashes'][index],
                                  checker_seconds=grade_metrics[index]['checker_seconds'],
                                  shape_results=shape_results, round_seconds=round_seconds,
                                  round_generation_seconds=generation_seconds,
                                  round_checker_seconds=sum(m['checker_seconds'] for m in grade_metrics),
                                  repair_parent_hash=__import__('failure_selection').code_fingerprint(latest[0]) if latest[0] else None,
                                  repair_parent_feedback=latest[1] if latest[0] else None,
                                  timing_scope='generation and grading; excludes JSONL serialization')
                log.write(json.dumps(record) + "\n")
            top = graded[decision.selected_index]
            selection_history.append(tuple(c for c in candidates
                                           if c.reward < sum(WEIGHTS.values()) - 1e-9))
        else:
            # Preserve the baseline's stable reward-only ordering and legacy logs.
            graded.sort(key=lambda g: g[0], reverse=True)
            top = graded[0]
        log.flush()
        if adaptive:
            chosen_index=decision.selected_index if extended else 0
            verified_shapes=sum(case.get("verified") is True for case in grade_metrics[chosen_index].get("shape_results",[]))
            repair_history.observe(top[1],top[2],top[0],verified_shapes)
        if top[0] > best[0]:
            best = (top[0], top[1], top[2])
        # Repair the LATEST attempt, not the best one. Rebuilding from the best attempt with the
        # best attempt's feedback is a fixed point: once a round scores worse, the prompt stops
        # changing, and a greedy model then returns the same answer forever. Measured: level 2
        # stuck at 0.10 for four rounds while the prompt still carried the 0.50 code.
        if (top[1] or "").strip():
            latest = (top[1], top[2])
            if adaptive and repair_history.best and (verified_shapes,top[0]) < (repair_history.best["verified_shapes"],repair_history.best["reward"]):
                latest=(repair_history.best["source"],repair_history.best["feedback"])
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
    ap.add_argument("--selection-policy", choices=("reward", "diagnostic"), default="reward",
                    help="equal-reward tie-breaking; diagnostic uses unverified repairability "
                         "heuristics while reward preserves baseline ordering")
    ap.add_argument("--candidate-policy", choices=("standard", "diverse"), default="standard")
    ap.add_argument("--repair-policy", choices=("standard", "grounded", "shape-aware"), default="standard")
    ap.add_argument("--generation-policy", choices=("standard", "constrained"), default="standard")
    ap.add_argument("--primitive-policy", choices=("off","legalize"), default="off")
    ap.add_argument("--planner-policy", choices=("off","hardware"), default="off")
    ap.add_argument("--shape-analysis", choices=("off","sympy"), default="off")
    ap.add_argument("--feedback-policy", choices=("legacy","targeted"), default="legacy")
    ap.add_argument("--example-policy", choices=("off","synthetic"), default="off")
    ap.add_argument("--adaptive-repair", action="store_true")
    ap.add_argument("--instrument", action="store_true", help="add usage, timing and per-shape experimental metadata")
    ap.add_argument("--grade-dir", help="private candidate-code directory for an isolated experiment")
    ap.add_argument("--adapter-mode", choices=("base","lora"), default=None, help="adapter switch for the isolated CPU evaluation endpoint only")
    ap.add_argument("--request-timeout", type=float, default=900, help="HTTP timeout in seconds; CPU adapter evaluation may need longer")
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
    if a.samples < 1 or a.rounds < 1 or a.repeat < 1:
        ap.error("samples, rounds and repeat must be positive")
    if a.grade_dir:
        os.makedirs(a.grade_dir, exist_ok=True)
        a.grade_dir = os.path.abspath(a.grade_dir)
    if instrumentation_enabled(a):
        import uuid
        a.run_id = uuid.uuid4().hex

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
            a.repeat_index = rep
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
