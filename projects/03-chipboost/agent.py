#!/usr/bin/env python3
"""
agent.py -- CHIPBOOST's loop: Qwen3 rewrites a kernel to run faster, the referee grades it and names ONE
change, and that change is the next prompt. Owner: P3.

    controller  start kernel + budget                    this file
    generator   Qwen3-8B on this seat                    KERNEL_AGENT_BASE_URL (the pod sets it)
    referee     speedcheck.check (P1) if it exists,      dev shapes, then held-out shapes
                else redteam/stage12.check (simulator only, TEMPORARY)
    log         one schema.py line per attempt           attempts.jsonl

    python agent.py --rounds 4 --samples 2 --context 8192          # in the seat pod
    python agent.py --offline                                       # no model; simulator grades (pod)
    python agent.py --dry                                           # no model, rules only (any machine)
    python schema.py --check attempts.jsonl

Reused from projects/02-kernel-agent/agent.py: extract_code, enrich, API_CARD, and the loop's mechanics
(samples per round, repair the LATEST attempt, a ledger of failed approaches when a failure repeats,
give up after N identical failures, --repeat). The prompts are new because the task is new: make a
correct kernel faster, rather than write one from a reference.

BUDGET: one referee evaluation = one attempt = one unit of --budget, whatever the verdict. Phase 4's
arms share this counter, so every arm gets the same number of evaluations.

Until P1's speedcheck.py merges there is NO TIMING: a correct kernel gets verdict None ("passed,
untimed"), never faster/slower, and its next instruction comes from the simulator's byte count
(source "sim"). The loop runs; it cannot produce a speedup until then.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT02_DIR = os.path.join(os.path.dirname(HERE), "02-kernel-agent")
sys.path.insert(0, AGENT02_DIR)
sys.path.insert(0, os.path.join(HERE, "redteam"))
sys.path.insert(0, HERE)

import nkibench  # noqa: E402
import schema    # noqa: E402
import stage12   # noqa: E402


def _load_agent02():
    """02's agent.py, under another name: `import agent` here would import this file."""
    spec = importlib.util.spec_from_file_location("agent02", os.path.join(AGENT02_DIR, "agent.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


agent02 = _load_agent02()

OP = "matmul"
ENTRY = nkibench.LEVELS[stage12.LEVEL]["entry"]   # nki_matmul_tiled_
P2_START = os.path.join(HERE, "kernels", "matmul_start.py")
FALLBACK_START = os.path.join(AGENT02_DIR, "reference_level4.py")

# Best first. None = correct on every shape the referee ran, but untimed (stage12 has no timer).
RANK = {"faster": 6, "no_gain": 5, None: 4, "slower": 3, "heldout_fail": 2, "wrong": 1, "rules": 0}
REJECTED = ("rules", "wrong", "heldout_fail")


# ---------------------------------------------------------------- the referee

def pick_referee():
    if os.path.exists(os.path.join(HERE, "speedcheck.py")):
        import speedcheck
        return "speedcheck", speedcheck.check
    return "stage12 (TEMPORARY, simulator only)", stage12.check


def sim_instruction(path):
    """ONE named change for a correct kernel when there is no timer: the byte analysis nkibench already
    has. The traffic bars of levels 5, 6, 7 are a ladder of byte targets; the first one this kernel misses
    names the next change (hoist loads, then block M/N, then block K). Measured on the largest dev shape,
    where redundant traffic shows."""
    kernel = nkibench.load_kernel(path, ENTRY)
    case = max(stage12.SHAPES["dev"], key=lambda c: c["M"] * c["K"] * c["N"])
    args, _ = nkibench.make_inputs(case, stage12.LEVEL)
    want = nkibench.LEVELS[stage12.LEVEL]["ref"](*args)
    _, counted = nkibench.simulate_and_count(kernel, args)
    for level in (5, 6, 7):
        m = nkibench.check_traffic_bar(level, counted, args, want)
        if m:
            return m
    return ("Correct, and its HBM traffic is already at the byte floor in the simulator, so no byte "
            "reduction is left. Any further speedup has to come from the engine schedule.")


def grade(src, referee, dry, workdir, n):
    """Dev shapes, then held-out shapes if dev passed. Returns the referee's schema fields."""
    path = os.path.join(workdir, f"attempt_{n:04d}.py")
    with open(path, "w") as f:
        f.write(src)
    if dry:
        return stage12.run(path, "dev", rules_only=True)[0]
    name, check = referee
    r = check(path, op=OP, shapes="dev")
    if r.get("verdict") in ("rules", "wrong"):
        return r
    h = check(path, op=OP, shapes="heldout")
    if h.get("verdict") in REJECTED:
        return dict(h, verdict="heldout_fail")
    if name.startswith("stage12") and r.get("verdict") is None:
        r = dict(r, instruction_given=sim_instruction(path))
    return r  # the dev result carries the timing


# ---------------------------------------------------------------- the model

def ask(a, prompt):
    """02's ask(), plus the token count it throws away: prompt_tokens is logged, so it must be the
    endpoint's number, not an estimate. Same request: thinking off, answer budget capped to the context."""
    import httpx
    budget = min(a.max_tokens, max(256, a.context - len(prompt) // 4 - 64))
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=budget, temperature=0.6, top_p=0.95,
                chat_template_kwargs={"enable_thinking": a.think})
    r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body, timeout=900, verify=False)
    if r.status_code != 200:
        raise SystemExit(f"the endpoint returned HTTP {r.status_code}:\n{r.text[:600]}")
    payload = r.json()
    ch = payload["choices"][0]
    if ch.get("finish_reason") == "length":
        print("    (TRUNCATED: finish_reason=length, so a parse error below is the budget, not the model)")
    return (ch.get("message", {}).get("content") or "",
            (payload.get("usage") or {}).get("prompt_tokens"))


def ask_parallel(a, prompt, n):
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        return [f.result() for f in [ex.submit(ask, a, prompt) for _ in range(n)]]


def offline_answers(start_src, n, rnd):
    """No model: a broken start kernel first (no @nki.jit), then the start kernel itself, so the loop,
    the referee path and the log can be exercised. Never report these."""
    src = start_src.replace("@nki.jit", "", 1) if rnd == 0 else start_src
    return [(f"```python\n{src}\n```", None)] * n


# ---------------------------------------------------------------- prompts

TASK = ("This AWS Neuron NKI kernel computes a matrix multiplication: result = lhsT.T @ rhs, with lhsT of "
        "shape [K, M] and rhs of shape [K, N]. It must stay correct for every K and M that are multiples "
        "of 128 and every N that is a multiple of 512, including sizes you are not shown.")
KEEP = (f"Keep the function name {ENTRY}(lhsT, rhs), the @nki.jit decorator, and the imports nki, "
        f"nki.language as nl and nki.isa as nisa. Reply with ONE python code block.")


def first_prompt(src, status):
    return (f"{TASK} Make it run faster on the Trainium chip.\n\n```python\n{src}\n```\n\n"
            f"The referee's report on this kernel: {status}\n\n{agent02.API_CARD}\n{KEEP}")


def repair_prompt(src, instruction):
    """The code plus ONE instruction from the referee. Never a corrected kernel."""
    return (f"{TASK}\n\n```python\n{src}\n```\n\nThe referee says: {instruction}\n\n"
            f"Make exactly that change and keep everything else identical. {KEEP}")


# ---------------------------------------------------------------- the loop

def run_once(a, referee, start_path, rep, log, workdir):
    tag = "offline-" if (a.offline or a.dry) else ""
    run_id = f"{OP}-{a.arm}-{tag}{time.strftime('%H%M%S')}-{rep}"
    start_src = open(start_path).read()
    start = grade(start_src, referee, a.dry, workdir, 0)
    status = start.get("instruction_given") or start.get("referee_message") or ""
    print(f"\n=========== run {run_id} ===========")
    print(f"start kernel {os.path.relpath(start_path, HERE)}: verdict {start.get('verdict') or 'PASS (untimed)'}")
    if start.get("verdict") in REJECTED:
        print(f"  THE START KERNEL IS REJECTED BY THE REFEREE, so there is nothing to speed up:\n  {status}")
        return dict(run_id=run_id, attempts=0, verdicts={}, best=None)

    prompt = first_prompt(start_src, status)
    latest = (start_src, status)
    tried, seen, streak = [], {}, 0
    attempt_no, verdicts, best = 0, {}, None
    for rnd in range(a.rounds):
        n = min(a.samples, a.budget - attempt_no)
        if n <= 0:
            print(f"  budget of {a.budget} referee evaluations used up")
            break
        t0 = time.perf_counter()
        # Generation finishes for the whole round BEFORE the referee runs, so vLLM is idle while timing.
        replies = offline_answers(start_src, n, rnd) if (a.offline or a.dry) else ask_parallel(a, prompt, n)
        graded = []
        for reply, prompt_tokens in replies:
            src = agent02.extract_code(reply)
            attempt_no += 1
            r = grade(src, referee, a.dry, workdir, attempt_no)
            instruction = agent02.enrich(r.get("instruction_given") or r.get("referee_message") or "")
            rec = {k: None for k in schema.ATTEMPT_FIELDS}
            rec.update({k: r.get(k) for k in stage12.REFEREE_FIELDS if k in r})
            rec.update(seat=a.seat, kernel=OP, arm=a.arm, run_id=run_id, attempt_no=attempt_no,
                       round=rnd, prompt_tokens=prompt_tokens,
                       code_hash=hashlib.sha1(src.encode()).hexdigest(), code=src,
                       prompt=prompt, response=reply, instruction_given=instruction,
                       timestamp=time.time())
            problems = schema.validate(rec)
            if problems:
                raise SystemExit(f"BUG: this log line breaks the shared schema: {problems}")
            log.write(json.dumps(rec) + "\n")
            verdicts[rec["verdict"]] = verdicts.get(rec["verdict"], 0) + 1
            graded.append((RANK[rec["verdict"]], rec.get("speedup") or 0.0, src, instruction, rec))
        log.flush()

        graded.sort(key=lambda g: (g[0], g[1]), reverse=True)
        _, speedup, src, instruction, rec = graded[0]
        if rec["verdict"] == "faster" and (best is None or speedup > best[0]):
            best = (speedup, rec["source"], attempt_no)
        if src.strip():
            latest = (src, instruction)
        same = instruction == (tried[-1] if tried else None)
        print(f"round {rnd}: best this round {rec['verdict'] or 'PASS (untimed)'}"
              + (f" speedup {speedup:.3f} ({rec['source']})" if rec.get("speedup") else "")
              + f"  [{attempt_no}/{a.budget} evaluations, {time.perf_counter() - t0:.1f}s]")
        if not same:
            print(f"  {instruction[:300]}")

        # 02's anti-cycling: give up on a fixed set of repeating failures; break a repeat with a ledger.
        seen[instruction] = seen.get(instruction, 0) + 1
        streak = streak + 1 if same else 1
        if seen[instruction] >= a.give_up_after:
            print(f"  STOPPING: the same instruction {seen[instruction]} times; more rounds will not help.")
            break
        tried.append(instruction)
        prompt = repair_prompt(*latest)
        if streak >= 2:
            ledger = "\n".join(f"- {t[:160]}" for t in dict.fromkeys(tried))
            prompt += f"\n\nThese have already been tried and did not work, so do something different:\n{ledger}"
            print(f"  same instruction {streak}x -- adding a ledger of {len(set(tried))} earlier ones")
    return dict(run_id=run_id, attempts=attempt_no, verdicts=verdicts, best=best)


def seat_from_hostname():
    h = socket.gethostname()
    return int(h.split("-")[1]) if h.startswith("seat-") and h.split("-")[1].isdigit() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="referee", choices=["referee"],
                    help="model_alone and random_search arrive in Phase 4")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--budget", type=int, default=24, help="referee evaluations per run, every arm")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--give-up-after", type=int, default=4)
    ap.add_argument("--start", default=None, help="start kernel; default P2's kernels/matmul_start.py "
                                                  "if it exists, else reference_level4.py")
    ap.add_argument("--seat", type=int, default=seat_from_hostname())
    ap.add_argument("--max-tokens", type=int, default=agent02.MIN_ANSWER_TOKENS)
    ap.add_argument("--context", type=int, default=8192, help="the server's max-model-len")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--model", default=agent02.MODEL)
    ap.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL"))
    ap.add_argument("--log", default=os.path.join(HERE, "attempts.jsonl"))
    ap.add_argument("--offline", action="store_true", help="no model; the simulator grades (needs nki)")
    ap.add_argument("--dry", action="store_true", help="no model, rules stage only (no nki needed)")
    a = ap.parse_args()

    start_path = a.start or (P2_START if os.path.exists(P2_START) else FALLBACK_START)
    referee = pick_referee()
    if a.offline or a.dry:
        print("*** OFFLINE: replaying the start kernel, no model. Numbers are meaningless. ***")
    else:
        if not (a.base or "").strip():
            sys.exit("KERNEL_AGENT_BASE_URL is unset. The seat pods set it; or pass --offline / --dry.")
        print(f"endpoint {a.base}  model {a.model}")
    print(f"referee: {'stage12 rules only (--dry)' if a.dry else referee[0]}")
    print(f"start:   {os.path.relpath(start_path, HERE)}   arm: {a.arm}   budget: {a.budget}   log: {a.log}")

    results = []
    workdir = tempfile.mkdtemp(prefix="chipboost_")
    try:
        with open(a.log, "a") as log:
            for rep in range(a.repeat):
                results.append(run_once(a, referee, start_path, rep, log, workdir))
    except nkibench.NkiMissing as e:
        sys.exit(f"{e}\nThe simulator needs the seat pod. Use --dry here.")

    print("\n=========== summary ===========")
    for r in results:
        best = (f"best speedup {r['best'][0]:.3f} ({r['best'][1]}) at attempt {r['best'][2]}"
                if r["best"] else "no verified speedup")
        counts = ", ".join(f"{k or 'pass-untimed'}={v}" for k, v in sorted(r["verdicts"].items(),
                                                                           key=lambda kv: str(kv[0])))
        print(f"  {r['run_id']}: {r['attempts']} evaluations; {counts}; {best}")
    if a.repeat > 1:
        got = [r["best"][0] if r["best"] else 1.0 for r in results]
        print(f"\n  over {a.repeat} runs: best speedup min {min(got):.3f} / max {max(got):.3f}; "
              f"{sum(1 for r in results if r['best'])}/{a.repeat} runs found one. Report the spread.")
    print(f"\nattempts logged to {a.log}")


if __name__ == "__main__":
    main()
