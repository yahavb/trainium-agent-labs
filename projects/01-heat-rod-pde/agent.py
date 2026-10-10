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
import subprocess
import sys
import time

import sympy as sp

import pdecheck
import solver_tool
import tool_calc
import level0_heatrod
import level1_heatrod

LEVELS = {0: level0_heatrod, 1: level1_heatrod}


def ask_once(a, prompt, max_tokens=None, temperature=None):
    """One chat call. Never raises: after 3 failed tries it returns finish='error' and no text,
    so one bad request costs one sample instead of the whole run."""
    import httpx
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens or a.max_tokens,
                temperature=a.temperature if temperature is None else temperature, top_p=0.95,
                chat_template_kwargs={"enable_thinking": a.think})
    err = ""
    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body,
                           timeout=900, verify=False)
            if r.status_code == 200:
                j = r.json()
                ch = j["choices"][0]
                return dict(text=ch["message"].get("content") or "", finish=ch.get("finish_reason"),
                            usage=j.get("usage") or {}, latency=round(time.perf_counter() - t0, 2))
            err = f"HTTP {r.status_code}: {r.text[:300]}"
        except httpx.HTTPError as e:
            err = f"{type(e).__name__}: {e}"
        time.sleep(2 * (attempt + 1))
    print(f"  request failed 3 times: {err}", file=sys.stderr)
    return dict(text="", finish="error", usage={}, latency=0.0, error=err)


def one_attempt(a, problem, prompt, rnd, temperature=None):
    """One sample, including its tool exchanges. Returns a dict: answer, calls, and for the
    solver modes the spec and the tool's report.

    freeform: the model may ask for exact values with COMPUTE: lines; we evaluate them and hand
    the numbers back, then ask for the final answer.
    ansatz/full: the model writes a spec (method, parameters, and in ansatz mode the wave family);
    solver_tool derives the rates and coefficients and writes u(x, t). One model call.
    """
    if a.mode in ("ansatz", "full"):
        if a.offline:
            text = fake_spec(problem, rnd)
            call = dict(text=text, finish="stop", usage={}, latency=0.0)
        else:
            call = ask_once(a, f"{prompt}\n\n{solver_tool.instructions(a.mode)}", max_tokens=a.spec_tokens,
                            temperature=temperature)
        spec, err = solver_tool.parse_spec(call["text"])
        if spec is None:          # maybe it wrote u(x, t) = ... anyway; the checker will look
            return dict(answer=call["text"], calls=[call], spec=None, spec_error=err, tool_calls=0)
        line, report = solver_tool.run(spec, a.mode)
        return dict(answer=line or "", calls=[call], spec=spec,
                    spec_error=None if line else report, tool_report=report, tool_calls=1)

    if a.offline:
        text = fake_model(problem, 1, rnd)[0]
        return dict(answer=text, calls=[dict(text=text, finish="stop", usage={}, latency=0.0)],
                    tool_calls=0)
    convo = prompt if a.no_tools else f"{prompt}\n\n{tool_calc.INSTRUCTIONS}"
    used, calls = 0, []
    for step in range(a.tool_steps + 1):
        call = ask_once(a, convo, temperature=temperature)
        calls.append(call)
        reply = call["text"]
        asks = [] if a.no_tools else tool_calc.requests_in(reply)
        if not asks or step == a.tool_steps:
            break
        used += len(asks)
        block = tool_calc.answer_block(asks)
        call["tool_answers"] = block
        convo = (f"{convo}\n\n{reply}\n\n"
                 f"{block}\n\n"
                 f"Use those values and give the final answer now, as one line "
                 f"u(x, t) = <expression> with every number filled in.")
    return dict(answer=reply, calls=calls, tool_calls=used)


def ask_round(a, problem, plan, rnd):
    """One attempt per (variant, prompt, temperature) in plan, all at the same time."""
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=len(plan)) as ex:
        futures = [ex.submit(one_attempt, a, problem, prompt, rnd, temp)
                   for _, prompt, temp in plan]
        return [f.result() for f in futures]


def fake_spec(problem, rnd):
    """Offline stand-in for the spec modes. Picks the right wave family more often each round."""
    rng = random.Random(rnd * 7 + 1 + random.random())
    Ls = sp.sstr(problem["L"])
    good = (f"sin((2*n - 1)*pi*x/(2*{Ls}))" if problem["right"] == "neumann" else f"sin(n*pi*x/{Ls})")
    fam = good if rng.random() < min(0.9, 0.3 + 0.3 * rnd) else f"sin(n*pi*x/{Ls})"
    return json.dumps(dict(pde="heat", method="eigenfunction_expansion", k=sp.sstr(problem["k"]),
                           L=Ls, initial=sp.sstr(problem["f"]), left=problem["left"],
                           right=problem["right"], eigenfunction=fam))


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


RUN_ID = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
try:
    GIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
except OSError:
    GIT = ""


def rank_key(g):
    """Parts-aware ranking: reward, then checks passed, then the starting-shape error. A plain
    max(reward) put a 0.8 answer 4.7% off ahead of a 0.6 answer 0.17% off."""
    se = g["start_error"] if g.get("start_error") is not None else 9e9
    return (g["reward"], sum((g.get("parts") or {}).values()), -se)


CHECK_NAMES = dict(equation="PDE", left_bc="Left boundary", right_bc="Right boundary",
                   start_shape="Initial condition")
_LEADS = {"equation": "The equation", "left_bc": "At the left end", "right_bc": "At the right end",
          "start_shape": "At t=0"}


def compact_feedback(g):
    """For the spec modes: one PASS/FAIL line per check, the checker's first sentence for each
    failure, and the solver's note on the spec. Drops the long assembled expression and the
    term-by-term lists, which describe the answer rather than the spec the model can change."""
    parts = g.get("parts") or {}
    if not parts:
        return g["feedback"]
    lines = [f"{CHECK_NAMES[k]}: {'PASS' if v else 'FAIL'}" for k, v in parts.items()]
    fb = g["feedback"]
    for k, v in parts.items():
        if not v and _LEADS[k] in fb:
            sent = fb[fb.index(_LEADS[k]):]
            sent = sent[:sent.index(".", sent.index("instead of 0") if "instead of 0" in sent else 0) + 1] \
                if k == "equation" else sent[:sent.index(". ") + 1] if ". " in sent else sent
            lines.append(f"  {sent.strip()}")
    if "Solver note: " in fb:
        lines.append("Root cause: " + fb[fb.index("Solver note: ") + len("Solver note: "):].strip())
    return "\n".join(lines)


def repair_prompt(a, statement, shown):
    """Best-only repair. shown: [(label, graded)], best-ever first."""
    if a.mode == "freeform":
        body = "".join(f"\n\n{label} (reward {g['reward']:.1f}):\n  u(x, t) = {g['expr']}\n"
                       f"A checker found this problem with it: {g['feedback']}" for label, g in shown)
        return f"{statement}{body}\nFix it."
    body = "".join(f"\n\n{label} (reward {g['reward']:.1f}) chose:\n  {json.dumps(g.get('spec'))}\n"
                   f"The checker's verdict on the solver's answer:\n{compact_feedback(g)}" for label, g in shown)
    return f"{statement}{body}\nFix the spec."


def solve(problem, a, log):
    style = "expression" if a.mode == "freeform" else "spec"
    base_prompt = pdecheck.prompt_of(problem, style=style)
    print(f"\n=========== {problem['name']} ===========")
    print(base_prompt)
    # Short spec replies at one temperature came back byte-identical four times over; spread them.
    temps = ([float(v) for v in a.temps.split(",")] * a.samples)[:a.samples] if a.temps else [None] * a.samples
    plan = [("V0", base_prompt, temp) for temp in temps]
    best_ever, t_start, history = None, time.time(), []
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        attempts = ask_round(a, problem, plan, rnd)
        graded = []
        for att in attempts:
            g = pdecheck.check(problem, att["answer"])
            if att.get("spec_error") and g["reward"] == 0.0:
                g["feedback"] = att["spec_error"]       # say what was wrong with the spec
            elif att.get("tool_report") and "You stated" in att["tool_report"] and g["reward"] < 1.0:
                g["feedback"] += " Solver note: " + att["tool_report"][att["tool_report"].index("You stated"):]
            g["spec"] = att.get("spec")
            graded.append(g)
        extra = []
        if a.aggregate == "all" and a.splice and a.mode == "freeform":
            import aggregate
            sp_g = aggregate.try_splice(problem, graded, best_ever)
            if sp_g is not None:
                extra.append(sp_g)
        if a.verify_rates and a.mode == "freeform":
            import aggregate
            for g in graded:
                g["verify"] = aggregate.verify_rates(problem, g)
        for i, (att, g, (variant, prompt_i, temp)) in enumerate(zip(attempts, graded, plan)):
            calls = att["calls"]
            log.write(json.dumps(dict(
                kind="cand", run_id=RUN_ID, arm=a.arm, git=GIT, ts=time.time(), model=a.model,
                mode=a.mode, aggregate=a.aggregate, temperature=temp or a.temperature,
                max_tokens=a.spec_tokens if a.mode != "freeform" else a.max_tokens,
                problem=problem["name"], seed=problem["seed"], round=rnd, sample=i,
                variant=variant, source="model", prompt=prompt_i, answer=att["answer"],
                transcript=calls, tool_calls=att.get("tool_calls", 0),
                finish_reason=calls[-1]["finish"] if calls else None,
                prompt_tokens=sum(c["usage"].get("prompt_tokens", 0) for c in calls),
                completion_tokens=sum(c["usage"].get("completion_tokens", 0) for c in calls),
                latency_s=round(sum(c["latency"] for c in calls), 2),
                spec=att.get("spec"), spec_error=att.get("spec_error"),
                tool_report=att.get("tool_report"), reward=g["reward"], parts=g["parts"],
                start_error=g["start_error"], expr=g["expr"], feedback=g["feedback"],
                verify=g.get("verify"))) + "\n")
        for g in extra:
            log.write(json.dumps(dict(kind="cand", run_id=RUN_ID, arm=a.arm, git=GIT, ts=time.time(),
                                      mode=a.mode, aggregate=a.aggregate, problem=problem["name"],
                                      seed=problem["seed"], round=rnd, sample=-1, variant="splice",
                                      source="splice", answer=g["expr"], reward=g["reward"],
                                      parts=g["parts"], start_error=g["start_error"],
                                      feedback=g["feedback"], completion_tokens=0,
                                      latency_s=0.0)) + "\n")
        rewards = [g["reward"] for g in graded]
        wall = time.perf_counter() - t0
        best = max(graded, key=rank_key)
        if best_ever is None or rank_key(best) > rank_key(best_ever):
            best_ever = best
        log.write(json.dumps(dict(kind="round", run_id=RUN_ID, arm=a.arm, problem=problem["name"],
                                  seed=problem["seed"], round=rnd, wall_s=round(wall, 1),
                                  rewards=rewards, variants=[v for v, _, _ in plan],
                                  distinct=len({g["expr"] for g in graded}),
                                  splice=[g["reward"] for g in extra])) + "\n")
        log.flush()
        toks = sum(sum(c["usage"].get("completion_tokens", 0) for c in att["calls"]) for att in attempts)
        print(f"\nround {rnd}: rewards {rewards}  mean {sum(rewards) / len(rewards):.2f}  "
              f"best {best['reward']:.1f}  {sum(att.get('tool_calls', 0) for att in attempts)} tool call(s)  "
              f"{toks} tokens  ({wall:.1f}s)")
        if best["reward"] == 1.0:
            print(f"SOLVED: u(x, t) = {best['expr']}")
            return 1.0, rnd + 1, time.time() - t_start
        if extra and extra[0]["reward"] == 1.0:      # logged only: the run measures the model
            print(f"  a splice of this round's candidates would solve it (not counted): {extra[0]['expr']}")
        print(f"  best: {best['expr'] if a.mode == 'freeform' else json.dumps(best.get('spec'))}")
        print(f"  checker: {best['feedback']}")
        history.extend(graded)
        if a.aggregate == "all" and a.mode != "freeform":
            import aggregate
            plan = aggregate.build_spec_prompts(a, problem, base_prompt, history, best_ever)
        elif a.aggregate == "all":
            import aggregate
            plan = aggregate.build_round_prompts(a, problem, base_prompt, graded, best_ever)
        else:
            shown = [("Your best attempt so far", best_ever)]
            if best is not best_ever and (best["expr"], best.get("spec")) != (best_ever["expr"], best_ever.get("spec")):
                shown.append(("Your latest attempt", best))
            plan = [("best", repair_prompt(a, base_prompt, shown), temp) for temp in temps]
    print(f"not solved in {a.rounds} rounds; best reward {best_ever['reward']:.1f}")
    return best_ever["reward"], a.rounds, time.time() - t_start


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
    ap.add_argument("--mode", default="freeform", choices=("freeform", "ansatz", "full"),
                    help="freeform: the model writes u(x, t). ansatz: it writes a spec with the wave "
                         "family and solver_tool does the maths. full: control, the tool picks the family too")
    ap.add_argument("--aggregate", default="best", choices=("best", "all"),
                    help="next round's prompt from the best (plus best-ever), or from all candidates")
    ap.add_argument("--splice", action="store_true",
                    help="with --aggregate all: also try a candidate spliced from the per-term verdicts")
    ap.add_argument("--verify-rates", action="store_true",
                    help="with --aggregate all: add a per-term decay-rate check (rate == k*freq**2) of every "
                         "candidate to the next prompts (change A)")
    ap.add_argument("--v4-fix", action="store_true",
                    help="with --aggregate all: V4 always uses a worked example other than the original (change B)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--spec-tokens", type=int, default=400)
    ap.add_argument("--temps", default=None,
                    help="per-sample temperatures, e.g. 0.6,0.8,1.0,1.0 (default: --temperature for all)")
    ap.add_argument("--arm", default="", help="label for this run in the log")
    ap.add_argument("--model", default=os.environ.get("HEATROD_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--base", default=os.environ.get("HEATROD_BASE_URL"))
    ap.add_argument("--log", default=None, help="default results/<arm>-<run id>.jsonl")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()

    if a.v4_fix:
        import aggregate
        aggregate.V4_FIX = True
    if not a.base and not a.offline:
        sys.exit("set HEATROD_BASE_URL to your vLLM endpoint, or pass --offline")
    if a.offline:
        print("*** OFFLINE: fake generator, numbers are meaningless ***")

    if a.log is None:
        os.makedirs("results", exist_ok=True)
        a.log = f"results/{a.arm + '-' if a.arm else ''}{RUN_ID}.jsonl"
    a.arm = a.arm or a.mode
    mod = LEVELS[a.level]
    subs = mod.SUBS if (a.all or not a.sub) else (a.sub,)
    results = []
    with open(a.log, "a") as log:
        for sub in subs:
            reward, rounds, secs = solve(mod.make(sub, a.seed), a, log)
            results.append((f"level{a.level}.{sub}", reward, rounds, secs))
            log.write(json.dumps(dict(kind="problem", run_id=RUN_ID, arm=a.arm, mode=a.mode,
                                      aggregate=a.aggregate, problem=f"level{a.level}.{sub}",
                                      seed=a.seed, reward=reward, rounds=rounds,
                                      wall_s=round(secs, 1))) + "\n")
            log.flush()

    print(f"\n=========== summary: level {a.level}, seed {a.seed}, arm {a.arm} ===========")
    for name, reward, rounds, secs in results:
        print(f"  {name:<12} reward {reward:.1f} after {rounds} round(s)  {secs:.0f}s"
              + ("  SOLVED" if reward == 1.0 else ""))
    solved = sum(1 for _, r, _, _ in results if r == 1.0)
    print(f"  solved {solved}/{len(results)}")
    print(f"\nattempts logged to {a.log}")
    solver_tool.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    # The solver's process pool can deadlock at interpreter exit (forked workers + threads);
    # the log is closed and flushed by now, so leave without waiting for it.
    os._exit(0)


if __name__ == "__main__":
    main()
