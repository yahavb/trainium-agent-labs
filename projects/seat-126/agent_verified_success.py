#!/usr/bin/env python3
"""
agent.py — the loop: a small model on your trn2 works a heat-equation problem until it solves
it, or runs out of rounds.

    controller  picks a problem            (level0_heatrod.py / level1_heatrod.py)
    generator   the model proposes N answers, sampling ON
    checker     grades each one; the grade is the reward   (pdecheck.py)
    loop        the best answer's feedback goes into the next round's prompt

Every attempt is appended to a JSONL file as prompt, answer and reward. That file is what an
RL trainer consumes; the trainer replaces this file's last few lines and nothing else.

    export HEATROD_BASE_URL="http://qwen3-8b:8000/v1"
    python agent.py --level 1 --sub 3
    python agent.py --level 1 --all --samples 4 --rounds 4
    python agent.py --offline --level 0 --all

Sampling must be ON, or all N answers are identical and there is nothing to compare.
"""

import argparse
import json
import os
import random
import sys
import time

import sympy as sp

import pdecheck
import tool_calc
import symbolic_verifier
import level0_heatrod
import level1_heatrod

LEVELS = {0: level0_heatrod, 1: level1_heatrod}


def ask_once(a, prompt):
    import httpx
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=a.max_tokens, temperature=0.6, top_p=0.95,
                chat_template_kwargs={"enable_thinking": a.think})
    r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body,
                   timeout=900, verify=False)
    if r.status_code != 200:
        raise SystemExit(f"the server returned HTTP {r.status_code}:\n{r.text[:800]}\n\n"
                         f"request was: {json.dumps(body)[:400]}")
    return r.json()["choices"][0]["message"].get("content") or ""


def one_attempt(a, problem, prompt, rnd):
    """One sample, including its tool exchanges.

    The model may ask for exact values with COMPUTE: lines; we evaluate them and hand the
    numbers back, then ask for the final answer. The model chooses which integral to set up,
    which is the part worth measuring, and sympy does the arithmetic it cannot do reliably.
    """
    if a.offline:
        return fake_model(problem, 1, rnd)[0], 0
    convo = prompt if a.no_tools else f"{prompt}\n\n{tool_calc.INSTRUCTIONS}"
    used = 0
    for step in range(a.tool_steps + 1):
        reply = ask_once(a, convo)
        asks = [] if a.no_tools else tool_calc.requests_in(reply)
        if not asks or step == a.tool_steps:
            if a.offline or a.no_tools:
                return reply, used

            expression = pdecheck.extract(reply)
            try:
                if not expression:
                    raise ValueError("No expression found")
                pdecheck.parse(expression)
            except Exception:
                repair_prompt = (
                    convo
                    + "\n\nYour previous response was:\n"
                    + reply
                    + "\n\nFORMAT ERROR: The answer could not be parsed. "
                    + "Write an explicit finite sum with numerical coefficients. "
                    + "Use only Python syntax with sin, exp, pi, x, and t. "
                    + "No LaTeX, no summation notation, no unknown constants. "
                    + "Return exactly one complete line beginning u(x, t) = "
                )
                repaired = ask_once(a, repair_prompt)
                if pdecheck.check(problem, repaired)["reward"] > 0:
                    return repaired, used
                return reply, used

            checks = [
                symbolic_verifier.verify_expression(expression, problem),
                symbolic_verifier.verify_initial_condition(expression, problem),
            ]

            if (problem["left"] == "dirichlet"
                    and problem["right"] == "dirichlet"):
                checks.append(
                    symbolic_verifier.verify_fourier_coefficients(
                        expression, problem
                    )
                )

            if all(check.get("valid", False) for check in checks):
                return reply, used

            feedback = " ".join(
                check["feedback"] for check in checks
                if not check.get("valid", False)
            )

            revision_prompt = (
                convo + "\n\nYour proposed answer was:\n"
                + reply
                + "\n\nSYMBOLIC VERIFICATION FEEDBACK:\n"
                + feedback
                + "\n\nREVISION CHECKLIST:\n"
                + "1. For each sine mode n, use spatial frequency "
                  "n*pi/L and decay exp(-k*(n*pi/L)**2*t). "
                  "Read L and k from the problem.\n"
                + "2. Recalculate each Fourier coefficient from "
                  "(2/L)*Integral(f(x)*sin(n*pi*x/L),(x,0,L)).\n"
                + "3. Remove modes whose coefficients are zero. "
                  "Do not alternate coefficient signs unless the "
                  "integrals require it.\n"
                + "4. Include enough nonzero modes to satisfy the "
                  "specified initial-condition tolerance.\n"
                + "5. Return ONE complete Python expression. "
                  "Do not end with an operator such as +, -, *, or **.\n"
                + "OUTPUT FORMAT IS MANDATORY:\n"
                + "Return exactly ONE line beginning with u(x, t) = .\n"
                + "Use Python syntax only: sin(...), exp(...), pi, *, /, **.\n"
                + "Expand the Fourier series into individual terms.\n"
                + "Never use LaTeX, summation notation, c_n, ellipses, "
                  "or explanations.\n"
                + "Write a complete expression with no trailing operator.\n"
                + "Final answer: u(x, t) = <expression>."
            )

            revised = ask_once(a, revision_prompt)

            original_score = pdecheck.check(problem, reply)
            revised_score = pdecheck.check(problem, revised)

            if revised_score["reward"] > original_score["reward"]:
                return revised, used

            if revised_score["reward"] == original_score["reward"]:
                original_error = original_score["start_error"]
                revised_error = revised_score["start_error"]

                if (revised_error is not None
                        and (original_error is None
                             or revised_error < original_error)):
                    return revised, used

            return reply, used
        used += len(asks)
        convo = (f"{convo}\n\n{reply}\n\n"
                 f"{tool_calc.answer_block(asks)}\n\n"
                 f"Use those values and give the final answer now, as one line "
                 f"u(x, t) = <expression> with every number filled in.")
    return reply, used


def ask_round(a, problem, prompt, rnd):
    """a.samples independent attempts, run at the same time."""
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=a.samples) as ex:
        futures = [ex.submit(one_attempt, a, problem, prompt, rnd)
                   for _ in range(a.samples)]
        return [f.result() for f in futures]


def fake_model(problem, n, rnd):
    """Offline stand-in, so the loop can be exercised with no model. It improves each round.
    Never report a number that came from here."""
    rng = random.Random(rnd)
    mod = LEVELS[problem["level"]]
    if problem["exact"] is not None:
        good = sp.sstr(problem["exact"])
        wrong = [sp.sstr(problem["f"]), sp.sstr(problem["exact"]).replace("sin", "cos", 1)]
    else:
        good = sp.sstr(mod.series_answer(problem, 4))
        wrong = [sp.sstr(mod.series_answer(problem, 1)), sp.sstr(problem["f"])]
    p_good = min(0.9, 0.15 + 0.25 * rnd)
    return [f"u(x, t) = {good if rng.random() < p_good else rng.choice(wrong)}"
            for _ in range(n)]


def solve(problem, a, log):
    base_prompt = pdecheck.prompt_of(problem)
    print(f"\n=========== {problem['name']} ===========")
    print(base_prompt)
    prompt, best_ever = base_prompt, 0.0
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        attempts = ask_round(a, problem, prompt, rnd)
        answers = [ans for ans, _ in attempts]
        tool_calls = sum(used for _, used in attempts)
        graded = [pdecheck.check(problem, ans) for ans in answers]
        for (ans, used), g in zip(attempts, graded):
            log.write(json.dumps(dict(problem=problem["name"], seed=problem["seed"],
                                      round=rnd, prompt=prompt, answer=ans,
                                      tool_calls=used, reward=g["reward"],
                                      parts=g["parts"],
                                      start_error=g["start_error"])) + "\n")
        log.flush()
        rewards = [g["reward"] for g in graded]
        best = max(graded, key=lambda g: g["reward"])
        best_ever = max(best_ever, best["reward"])
        print(f"\nround {rnd}: rewards {rewards}  mean {sum(rewards) / len(rewards):.2f}  "
              f"best {best['reward']:.1f}  {tool_calls} tool call(s)  "
              f"({time.perf_counter() - t0:.1f}s)")
        if best["reward"] == 1.0:
            print(f"SOLVED: u(x, t) = {best['expr']}")
            return 1.0, rnd + 1
        print(f"  best: {best['expr']}")
        print(f"  checker: {best['feedback']}")
        prompt = (f"{base_prompt}\n\nA previous attempt was:\n  u(x, t) = {best['expr']}\n"
                  f"A checker found this problem with it: {best['feedback']}\n"
                  f"Fix it.")
    print(f"not solved in {a.rounds} rounds; best reward {best_ever:.1f}")
    return best_ever, a.rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=0, choices=sorted(LEVELS))
    ap.add_argument("--sub", type=int)
    ap.add_argument("--all", action="store_true", help="every sub-problem in this level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=4)
    # gpt-oss reasons before it answers and needs ~6000 on level 1.3.
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("HEATROD_MAX_TOKENS", 1200)))
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--tool-steps", type=int, default=1,
                    help="rounds of COMPUTE: exchanges allowed per attempt")
    ap.add_argument("--no-tools", action="store_true",
                    help="withhold the calculator, to measure what it buys")
    ap.add_argument("--model", default=os.environ.get("HEATROD_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--base", default=os.environ.get("HEATROD_BASE_URL"))
    ap.add_argument("--log", default="attempts.jsonl")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()

    if not a.base and not a.offline:
        sys.exit("set HEATROD_BASE_URL to your vLLM endpoint, or pass --offline")
    if a.offline:
        print("*** OFFLINE: fake generator, numbers are meaningless ***")

    mod = LEVELS[a.level]
    subs = mod.SUBS if (a.all or not a.sub) else (a.sub,)
    results = []
    with open(a.log, "a") as log:
        for sub in subs:
            reward, rounds = solve(mod.make(sub, a.seed), a, log)
            results.append((f"level{a.level}.{sub}", reward, rounds))

    print(f"\n=========== summary: level {a.level} ===========")
    for name, reward, rounds in results:
        print(f"  {name:<12} reward {reward:.1f} after {rounds} round(s)"
              + ("  SOLVED" if reward == 1.0 else ""))
    solved = sum(1 for _, r, _ in results if r == 1.0)
    print(f"  solved {solved}/{len(results)}")
    print(f"\nattempts logged to {a.log}")


if __name__ == "__main__":
    main()
