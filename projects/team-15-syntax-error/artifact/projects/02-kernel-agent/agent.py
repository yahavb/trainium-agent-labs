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

import heldout
import levers
import nkibench
import verdicts

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
_GRADE_SEQ = __import__('itertools').count()


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

    if verdicts.V9:
        # v9: report EVERY API-signature misuse at once instead of one per simulated round
        lint = verdicts.signature_lint(source)
        if lint:
            return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                    "API misuse -- fix ALL of these calls in this one round, change nothing else:\n"
                    + "\n".join(f"- {m}" for _, m in lint[:8]))

    spec = nkibench.LEVELS[level]
    # per-process path: runq runs several jobs of the SAME level at once (race found 12:5x)
    # A FRESH path per call: Python's bytecode cache keys a module on (path, mtime-in-seconds, size),
    # so two different kernels of the same size graded within one second at the same path load the
    # STALE first one. Found 13:1x when regrade.py (many grades per second) disagreed with the logs.
    path = f"/tmp/_agent_level{level}_{os.getpid()}_{next(_GRADE_SEQ)}.py"
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
            msg = enrich(f"raised {type(e).__name__}: {e}", source, level)
            msg += verdicts.l8_hint(msg, source, level)   # '' unless --v8-verdicts finds a cause
            msg += verdicts.mm_hint(msg, source, level)   # '' unless --v9-verdicts finds a cause
            msg += verdicts.softmax_hint(source, level)   # '' unless --v10-verdicts
            failures.append((nkibench.label(case, level), msg))
            continue
        parts["runs"] = True
        m = (nkibench.check_inputs_untouched(before, args)
             or nkibench.describe_mismatch(got, want)
             or nkibench.check_traffic_bar(level, counted, args, want))
        if m and m.startswith("NUMERICAL MISMATCH"):
            m += verdicts.code_hint(source, level)       # '' unless --v2-verdicts finds a cause
            m += verdicts.softmax_hint(source, level)    # '' unless --v10-verdicts
        if m and m.startswith("NON-FINITE OUTPUT"):
            m += verdicts.nonfinite_hint(source)          # '' unless --v6-verdicts finds a cause
            m += verdicts.unloaded_hint(source, level)    # '' unless --v7-verdicts finds a cause
        if m and (m.startswith("NUMERICAL MISMATCH") or m.startswith("NON-FINITE OUTPUT")):
            m += verdicts.kloop_hint(source, level)       # '' unless --v8-verdicts finds a cause
        if m and m.startswith("CORRECT, BUT TOO MUCH HBM TRAFFIC"):
            m += verdicts.traffic_hint(source, level)    # '' unless --v4-verdicts finds a cause
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
        if level >= 3 and counted["bytes"] and "M" in case:   # matmul shapes only (L8 has seq/dim)
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


# The verdict -> instruction table lives in verdicts.py (same wording as the old enrich() chain,
# checked byte-identical on every case in its selftest), so the taxonomy and the doc retrieval
# can key off the same failure buckets.
enrich = verdicts.enrich


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
    meta = dict(finish=finish, usage=payload.get("usage") or {}, reasoning_chars=len(reasoning),
                budget=budget)
    return content, meta


def ask_parallel(a, prompt, n):
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        return [f.result() for f in [ex.submit(ask, a, prompt) for _ in range(n)]]


def run_samples(a, level, prompt, rnd):
    """[(content, meta)] for one round, from the model or the offline replay."""
    if a.offline:
        return [(r, dict(finish="offline", usage={})) for r in offline_answers(level, a.samples, rnd)]
    return ask_parallel(a, prompt, a.samples)


def offline_answers(level, n, rnd):
    """No model. Replays the shipped reference, preceded by a deliberately broken version, so the
    loop and the feedback path can be exercised with no endpoint. Never report a number."""
    ref = open(f"reference_level{level}.py").read()
    if rnd == 0:
        broken = ref.replace("@nki.jit", "", 1)
        return [f"```python\n{broken}\n```"] * n
    return [f"```python\n{ref}\n```"] * n


# ---------------------------------------------------------------- the loop

def build_first(a, level, terse):
    """The first prompt, and the parts inserted into it (for token accounting)."""
    prompt = first_prompt(level, terse)
    if a.v10_verdicts and level == 8:
        # the API card's only reduction is "nl.sum(view, axis=...)"; all 115 L8 attempts then wrote
        # tensor_reduce(op=nl.sum) and none used reciprocal/maximum (LOG 16:4x)
        prompt = prompt.replace(
            "  nl.sum(view, axis=[i, j])                 reduce; axis is a list\n",
            "  nisa.tensor_reduce(dst=, op=nl.add|nl.maximum, data=, axis=(1,))   row sum / row max into (P,1)\n"
            "  nisa.reciprocal(dst=, data=)              1/x\n"
            "  nisa.activation(dst=, op=nl.exp, data=)   exp\n")
    if a.prompt_fixes and level == 8:
        # the API card's matmul paragraph says "The left operand arrives already transposed"; true
        # for levels 3-7 only -- at L8 the model took it to mean k needs no transpose (LOG 15:2x).
        # Scoped to level 8 ONLY: removing it at L1/L2 too changed their first prompt and 2/5 C9 L1
        # runs degenerated into truncated answers (LOG 15:5x).
        prompt = prompt.replace("The left operand arrives already transposed, with K on the "
                                "partition axis.", "")
    parts = dict(api=API_CARD if API_CARD in prompt else "")
    if a.skeleton:
        sk = levers.skeleton(level)
        prompt = prompt + "\n\n" + sk
        parts["skeleton"] = sk
    return prompt, parts


def build_repair(a, level, latest, ledger=""):
    """Repair prompt for the latest attempt: one named change. --retrieve appends the AWS doc
    excerpt keyed by the failure bucket. Returns (prompt, parts, extra log fields)."""
    code, feedback = latest
    prompt = repair_prompt(level, code, feedback)
    parts = dict(code=code, error=feedback)
    info = dict(bucket=verdicts.classify(feedback))
    if a.retrieve:
        text, src = levers.doc_slice(info["bucket"])
        if text:
            docs = f"\n\nFrom the NKI documentation:\n{text}"
            prompt += docs
            parts["docs"] = docs
            info["doc_source"] = src
    if ledger:
        led = f"\n\nThese approaches have already failed, so do something different:\n{ledger}"
        prompt += led
        parts["ledger"] = led
    return prompt, parts, info


def solve(a, level, log):
    print(f"\n=========== level {level}: {nkibench.LEVELS[level]['op']} ===========")
    terse = a.terse
    prompt, parts = build_first(a, level, terse)
    pinfo = dict(kind="first")
    best = (0.0, None, "")
    tried, streak, seen = [], 0, {}
    latest = ("", "")
    grade_cache = {}
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        replies = run_samples(a, level, prompt, rnd)
        sec = levers.sections(prompt, **parts)
        graded = []
        for si, (reply, meta) in enumerate(replies):
            src = extract_code(reply)
            fixes = []
            if a.mech:
                src, fixes = levers.mechanical_fix(src, level)
            # Measured on seat 73: the local server decodes deterministically, so the samples of a
            # round are usually byte-identical. Grading is deterministic too, so grade each text once.
            if src not in grade_cache:
                grade_cache[src] = grade(src, level)
            reward, gparts, feedback = grade_cache[src]
            graded.append((reward, src, feedback, gparts))
            log.write(json.dumps(dict(tag=a.tag, run=a.run_index, level=level, round=rnd,
                                      sample=si, ts=time.time(), reward=reward, parts=gparts,
                                      bucket=verdicts.classify(feedback), mech_fixes=fixes,
                                      prompt_kind=pinfo, prompt_sections=sec,
                                      prompt_chars=len(prompt), reply_chars=len(reply),
                                      finish=meta.get("finish"), usage=meta.get("usage"),
                                      reasoning_chars=meta.get("reasoning_chars"),
                                      prompt=prompt, code=src, feedback=feedback)) + "\n")
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
            # Calibration: say how sure, from the code and the public result only. heldout.py later
            # scores this number against hostile cases the agent never saw.
            conf, why = heldout.confidence(top[1], level)
            log.write(json.dumps(dict(tag=a.tag, run=a.run_index, level=level, round=rnd,
                                      claim="solved", confidence=conf, why=why,
                                      ts=time.time())) + "\n")
            log.flush()
            print(f"  SOLVED on round {rnd}, confidence {conf:.2f} ({'; '.join(why)}). {top[2]}")
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
            log.write(json.dumps(dict(tag=a.tag, run=a.run_index, level=level, claim="not_verified",
                                      confidence=0.0, best=best[0], ts=time.time())) + "\n")
            print(f"  COULD NOT VERIFY level {level} (confidence 0.00).")
            print(f"  STOPPING this level: {how}. The agent is cycling between a fixed set of "
                  f"mistakes rather than converging, so more rounds will not help. Failures seen:")
            for f, n in sorted(seen.items(), key=lambda kv: -kv[1]):
                print(f"    {n}x  {f[:110]}")
            return best[0], rnd + 1
        tried.append(top[2])
        repeats = streak
        empty = not (top[1] or "").strip()
        if a.empty_terse and empty:
            # STATE.md's untried fix: escalate terseness on EVERY empty answer, not only while no
            # code has ever come back -- once a round succeeds the prompt grows again otherwise.
            terse = min(terse + 1, 2)
            prompt, parts = build_first(a, level, terse)
            pinfo = dict(kind="first", terse=terse)
            print(f"  empty answer, so re-asking with a shorter prompt (terseness {terse})")
            continue
        if repeats >= 2 and (best[1] or "").strip():
            # Sampling on this endpoint is greedy, so an unchanged prompt returns an unchanged
            # answer. Measured: the same TypeError 19 rounds running. Changing the prompt is the
            # only thing that can change the answer, so say what has already been tried.
            ledger = "\n".join(f"- {t[:160]}" for t in dict.fromkeys(tried))
            prompt, parts, info = build_repair(a, level, latest, ledger)
            pinfo = dict(kind="repair+ledger", **info)
            print(f"  same failure {repeats}x — adding a ledger of {len(set(tried))} failed "
                  f"attempts to break the repeat")
            continue
        if not (latest[0] or "").strip():
            # Nothing came back to repair. Asking it to "fix" an empty code block produced a
            # 202-character prompt and, under greedy sampling, the identical non-answer six
            # rounds running. Shorten and re-ask instead.
            terse = min(terse + 1, 2)
            prompt, parts = build_first(a, level, terse)
            pinfo = dict(kind="first", terse=terse)
            print(f"  no code yet, so re-asking with a shorter prompt (terseness {terse})")
        else:
            prompt, parts, info = build_repair(a, level, latest)
            pinfo = dict(kind="repair", **info)
    print(f"  not solved in {a.rounds} rounds; best reward {best[0]:.2f}. COULD NOT VERIFY a "
          f"kernel for level {level}: confidence 0.00")
    log.write(json.dumps(dict(tag=a.tag, run=a.run_index, level=level, claim="not_verified",
                              confidence=0.0, best=best[0], ts=time.time())) + "\n")
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
    ap.add_argument("--tag", default="", help="label written into every logged attempt")
    ap.add_argument("--mech", action="store_true",
                    help="LEVER: fix missing @nki.jit / wrong entry name / invented imports by AST "
                         "before grading, without a model call")
    ap.add_argument("--skeleton", action="store_true",
                    help="LEVER: append a rank-aware structural template with TODO slots to the "
                         "first prompt (not a complete kernel)")
    ap.add_argument("--retrieve", action="store_true",
                    help="LEVER: append the AWS doc excerpt keyed by the failure bucket to repairs")
    ap.add_argument("--v2-verdicts", action="store_true",
                    help="code-aware translations found in E1 transcripts (swapped axes, "
                         "collapsed partition axis)")
    ap.add_argument("--v3-verdicts", action="store_true",
                    help="level-1 window-reduction instruction (scalar nl.sum + element assignment "
                         "-> tensor_reduce into a (C,1) tile), from E1/C1 transcripts")
    ap.add_argument("--v4-verdicts", action="store_true",
                    help="level-5 traffic instruction: name the lhsT load that must leave the n loop")
    ap.add_argument("--v5-verdicts", action="store_true",
                    help="levels 5-7 traffic instruction: load every operand tile exactly once")
    ap.add_argument("--v6-verdicts", action="store_true",
                    help="NaN from exp without max-subtraction is an overflow, say so")
    ap.add_argument("--v7-verdicts", action="store_true",
                    help="NaN because lhsT/rhs tiles were allocated but never dma_copy'd: say which")
    ap.add_argument("--v8-verdicts", action="store_true",
                    help="L5-7: name the k loop placed outside m/n; L8: name the PSUM/SBUF one-name "
                         "conflation and the assumed-pretransposed q/k/v")
    ap.add_argument("--prompt-fixes", action="store_true",
                    help="L6 traffic hint = L7's (accumulate in one PSUM tile across the contraction); "
                         "API card's 'already transposed' only for levels 3-7; L8 skeleton names "
                         "the separate psum and sbuf tiles")
    ap.add_argument("--device-rules", action="store_true",
                    help="static scan also rejects constructs the device compiler rejects but the "
                         "simulator accepts (list comprehensions), measured on Trainium")
    ap.add_argument("--v9-verdicts", action="store_true",
                    help="report every nisa/nl signature misuse at once (static lint), translate missing "
                         "positional / undefined / used-before-assigned, L8 scores-matmul hint")
    ap.add_argument("--v10-verdicts", action="store_true",
                    help="L8: API card names tensor_reduce/reciprocal/exp; lint catches op=nl.sum / nisa.* ops "
                         "/ inline data=; softmax op-sequence hint; skeleton TODO lists the op order")
    ap.add_argument("--directional", action="store_true",
                    help="numerical feedback names the direction of the error, never the target "
                         "value (nkibench.DIRECTIONAL)")
    ap.add_argument("--empty-terse", action="store_true",
                    help="LEVER: escalate terseness on every empty answer (STATE.md's untried fix)")
    a = ap.parse_args()
    a.run_index = 0
    nkibench.DIRECTIONAL = a.directional
    verdicts.V2 = a.v2_verdicts
    verdicts.V3 = a.v3_verdicts
    verdicts.V4 = a.v4_verdicts
    verdicts.V5 = a.v5_verdicts
    verdicts.V6 = a.v6_verdicts
    verdicts.V7 = a.v7_verdicts
    verdicts.V8 = a.v8_verdicts
    verdicts.V9 = a.v9_verdicts
    verdicts.V10 = a.v10_verdicts
    levers.V10 = a.v10_verdicts
    nkibench.DEVICE_RULES = a.device_rules
    nkibench.PROMPT_FIXES = levers.PROMPT_FIXES = a.prompt_fixes

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
            a.run_index = rep
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
