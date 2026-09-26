#!/usr/bin/env python3
"""
agent.py — the loop: a model writes an NKI kernel, the harness grades it, the reason goes back,
it tries again.

    controller  picks a rung                          nkibench.py
    generator   the model writes a kernel             the shared gpt-oss endpoint
    checker     rules, then simulate, then roofline   nkibench.py
    loop        the failure becomes the next prompt   this file

Everything here runs on the CPU, so it does NOT need the Trainium device -- which is the point,
because the device is what Project 2 needs later for real timings, and a Neuron device cannot be
shared by two processes. The model therefore comes from the shared endpoint, not from a server on
your own chip.

    export GPTOSS_BASE_URL="https://..."
    python agent.py --rung 1
    python agent.py --rung 4 --rounds 6 --samples 4
    python agent.py --all
    python agent.py --offline --rung 4        # no model; replays the reference kernel

Every attempt is appended to a JSONL file with its reward, so the log is the deliverable.
"""

import argparse
import json
import os
import re
import sys
import time

import nkibench

MODEL = os.environ.get("KERNEL_AGENT_MODEL", "gpt-oss-20b")

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


def grade(source, rung):
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

    violations = nkibench.check_rules(source, rung)
    if violations:
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                "Rule violations, which score zero however fast the kernel is. Fix exactly "
                "these: " + " ".join(violations))
    parts["rules"] = True

    spec = nkibench.LADDER[rung]
    path = f"/tmp/_agent_rung{rung}.py"
    with open(path, "w") as f:
        f.write(source)
    try:
        kernel = nkibench.load_kernel(path, spec["entry"])
    except ModuleNotFoundError as e:
        # An environment problem, not a kernel problem. Feeding this back as kernel feedback
        # would burn every round on a bug the model cannot fix.
        raise SystemExit(
            f"cannot import {e.name!r}, so no NKI kernel can be loaded here. Run this where the "
            f"Neuron SDK exists -- k8s/kernel-agent-job.yaml does that -- or use --offline to "
            f"exercise the loop without a kernel ever running.") from e
    except Exception as e:
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                f"The file imports but {spec['entry']} could not be loaded: "
                f"{type(e).__name__}: {e}")

    failures, passed, intensity = [], 0, None
    for case in spec["shapes"]:
        args, _ = nkibench.make_inputs(case, rung)
        want = spec["ref"](*args)
        try:
            got, counted = nkibench.simulate_and_count(kernel, args)
        except nkibench.NkiMissing as e:
            return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                    f"CANNOT SIMULATE: {e}")
        except Exception as e:
            failures.append((nkibench.label(case, rung),
                             f"raised {type(e).__name__}: {e}"))
            continue
        parts["runs"] = True
        m = nkibench.describe_mismatch(got, want)
        if m:
            failures.append((nkibench.label(case, rung), m))
            continue
        passed += 1
        if rung >= 3 and counted["bytes"]:
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

def first_prompt(rung):
    """Deliberately short, and it does NOT list the rules.

    Measured twice in this repo: hand a model an enumerated list of prohibitions and it audits
    itself against each one and returns nothing, while a bigger budget only buys more thinking.
    So the rules live in the checker. Generate freely, let the checker object, then send back one
    named change.
    """
    s = nkibench.LADDER[rung]
    import inspect
    return (
        f"Write an AWS Neuron NKI kernel.\n\n"
        f"Operation: {s['op']}\n"
        f"Entry point: a function named `{s['entry']}`, decorated with `@nki.jit`.\n"
        f"It must compute exactly what this NumPy reference computes:\n\n"
        f"{inspect.getsource(s['ref'])}\n"
        f"Hardware limits: a tile's partition dimension is at most {nkibench.PMAX}. For matmul, "
        f"the stationary free dimension is at most {nkibench.GEMM_STATIONARY_FMAX} and the "
        f"moving free dimension at most {nkibench.GEMM_MOVING_FMAX}.\n\n"
        f"Use nki, nki.language as nl, and nki.isa as nisa. Allocate with "
        f"`nl.ndarray(shape, dtype=..., buffer=nl.sbuf | nl.psum | nl.shared_hbm)`, move data "
        f"with `nisa.dma_copy(dst=, src=)`, and loop with `nl.affine_range(n)`.\n\n"
        f"Reply with ONE python code block containing the imports and the function. No prose.")


def repair_prompt(rung, source, feedback):
    """One named change, and the previous code. No rules list, no reference re-sent.

    The lesson this whole repo keeps re-learning: feeding a verifier's report back verbatim
    reproduces the same mistake, because a report says what is wrong and never what to do.
    """
    return (
        f"This NKI kernel for {nkibench.LADDER[rung]['op']} is not right yet.\n\n"
        f"```python\n{source}\n```\n\n"
        f"A checker reports:\n{feedback}\n\n"
        f"Change exactly what the checker names and keep everything else identical. Reply with "
        f"ONE python code block.")


CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_code(text):
    blocks = CODE_BLOCK.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip()
    # No fence. If it looks like a kernel anyway, take it.
    return (text or "").strip() if "def " in (text or "") else ""


# ---------------------------------------------------------------- the model

def ask(a, prompt):
    import httpx
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=a.max_tokens)
    r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body,
                   timeout=900, verify=False)
    if r.status_code != 200:
        raise SystemExit(f"the endpoint returned HTTP {r.status_code}:\n{r.text[:600]}")
    payload = r.json()
    ch = payload["choices"][0]
    msg = ch.get("message", {})
    reasoning = next((msg[k] for k in REASONING_KEYS if msg.get(k)), "")
    content = msg.get("content") or ""
    if not content.strip() and reasoning:
        # The single most common surprise on this endpoint, so name it rather than reporting
        # an empty answer as a model failure.
        print(f"    (empty answer, {len(reasoning)} chars of hidden reasoning, "
              f"finish={ch.get('finish_reason')} — shorten the prompt rather than raising the "
              f"budget)")
    return content


def ask_parallel(a, prompt, n):
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        return [f.result() for f in [ex.submit(ask, a, prompt) for _ in range(n)]]


def offline_answers(rung, n, rnd):
    """No model. Replays the shipped reference, preceded by a deliberately broken version, so the
    loop and the feedback path can be exercised with no endpoint. Never report a number."""
    ref = open(f"reference_rung{rung}.py").read()
    if rnd == 0:
        broken = ref.replace("@nki.jit", "", 1)
        return [f"```python\n{broken}\n```"] * n
    return [f"```python\n{ref}\n```"] * n


# ---------------------------------------------------------------- the loop

def solve(a, rung, log):
    print(f"\n=========== rung {rung}: {nkibench.LADDER[rung]['op']} ===========")
    prompt = first_prompt(rung)
    best = (0.0, None, "")
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        replies = (offline_answers(rung, a.samples, rnd) if a.offline
                   else ask_parallel(a, prompt, a.samples))
        graded = []
        for reply in replies:
            src = extract_code(reply)
            reward, parts, feedback = grade(src, rung)
            graded.append((reward, src, feedback, parts))
            log.write(json.dumps(dict(rung=rung, round=rnd, reward=reward, parts=parts,
                                      prompt_chars=len(prompt), reply_chars=len(reply),
                                      code=src, feedback=feedback)) + "\n")
        log.flush()
        graded.sort(key=lambda g: g[0], reverse=True)
        top = graded[0]
        if top[0] > best[0]:
            best = (top[0], top[1], top[2])
        print(f"round {rnd}: rewards {[round(g[0], 2) for g in graded]}  "
              f"best {top[0]:.2f}  ({time.perf_counter() - t0:.1f}s)")
        print(f"  {top[2][:300]}")
        if top[0] >= sum(WEIGHTS.values()) - 1e-9:
            print(f"  SOLVED on round {rnd}")
            return top[0], rnd + 1
        prompt = repair_prompt(rung, best[1] or "", best[2])
    print(f"  not solved in {a.rounds} rounds; best reward {best[0]:.2f}")
    return best[0], a.rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, choices=sorted(nkibench.LADDER))
    ap.add_argument("--all", action="store_true", help="rungs 1 to 4")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=MIN_ANSWER_TOKENS)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base", default=os.environ.get("GPTOSS_BASE_URL"))
    ap.add_argument("--path", default="/agg/v1", help="endpoint path under the base URL")
    ap.add_argument("--log", default="attempts.jsonl")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()

    if not a.offline:
        if not a.base:
            sys.exit("set GPTOSS_BASE_URL, or pass --offline to exercise the loop with no model")
        a.base = a.base.rstrip("/") + a.path
    else:
        print("*** OFFLINE: replaying the reference kernel. Numbers are meaningless. ***")

    rungs = sorted(nkibench.LADDER)[:4] if a.all else [a.rung or 1]
    results = []
    with open(a.log, "a") as log:
        for rung in rungs:
            results.append((rung,) + solve(a, rung, log))

    print("\n=========== summary ===========")
    for rung, reward, rounds in results:
        print(f"  rung {rung}  reward {reward:.2f} after {rounds} round(s)"
              + ("  SOLVED" if reward >= sum(WEIGHTS.values()) - 1e-9 else ""))
    print(f"  solved {sum(1 for _, r, _ in results if r >= sum(WEIGHTS.values()) - 1e-9)}"
          f"/{len(results)}")
    print(f"\nattempts logged to {a.log}")


if __name__ == "__main__":
    main()
