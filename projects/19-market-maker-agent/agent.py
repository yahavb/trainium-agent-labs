"""The market-maker agent. Qwen3-8B writes quote(state); checker.py grades it on 16 seeded dev
episodes; the grade and the reason go into the next attempt. When a level is solved, or the budget
runs out, the chosen strategy is scored once on the frozen held-out seeds, and the agent's claim
("verified on dev" or "not verified") is logged next to that held-out result, so the claim itself
can be graded.

    python agent.py --levels 1 2 3 --feedback located --repeat 2 --tag C      # the real run
    python agent.py --offline --levels 2                                       # no model needed

Every attempt is one line in runs/<tag>-<rep>.jsonl. report.py turns those into the tables.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import re
import sys
import time

import checker
import mmsim

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("MM_AGENT_MODEL") or os.environ.get("KERNEL_AGENT_MODEL") or "Qwen/Qwen3-8B"
BASE_URL = os.environ.get("MM_AGENT_BASE_URL") or os.environ.get("KERNEL_AGENT_BASE_URL")
REASONING_KEYS = ("reasoning_content", "reasoning")

SPEC = """You write the quoting logic of a market maker in a simulated stock. Every step you post one bid and one ask; incoming market orders trade against your quotes. You earn the spread, and you lose when the price moves against the inventory you hold.

Write a short Python module (under 60 lines) with one top-level function quote(state) that returns a dict. No classes needed.

state (all prices are integers, in ticks):
  t, T                    current step, steps in the episode (1500)
  best_bid, best_ask      the market's best prices now
  bid_sizes, ask_sizes    displayed size at the 5 best price levels per side; index 0 = best
  mid_history             list of mid prices, oldest first, last = now (at most 100; only 1 at t=0)
  inventory, cash         your position in shares, your cash in ticks
  max_inventory           10, the hard limit on abs(inventory)
  max_order_size          5
  memory                  a dict kept between your calls in one episode; store anything you like

return {"bid": int or None, "bid_size": int, "ask": int or None, "ask_size": int}

Rules. Any violation scores zero:
  - prices are whole ticks (int); bid < ask; bid < best_ask and ask > best_bid (you post, never take)
  - 0 <= bid_size, ask_size <= max_order_size
  - inventory + bid_size <= max_inventory and inventory - ask_size >= -max_inventory
  - only `import math` and `import numpy as np`; no randomness, no files, no network

Fills: a market sell fills your bid first if bid > best_bid; at bid == best_bid you get it half the time; below best_bid never. Asks mirror this. A quote the price moves through is taken at your price.

Goal: positive profit with 95% confidence over 16 episodes, while quoting both sides within 2 ticks of best_bid/best_ask on at least half the steps."""

# Run C6 only (MM_HINT=1): published ideas, given as background. Ideas and citations, no code and no
# coefficients. Tests whether the model lacked the concept rather than the evidence.
HINT = """

Background from market-microstructure research:
  - Volume imbalance at the best quotes, (bid_sizes[0] - ask_sizes[0]) / (bid_sizes[0] + ask_sizes[0]), predicts the side of the next market order and the next price move. Makers that always post both sides at the best quotes suffer from adverse selection (Cartea, Donnelly & Jaimungal 2018).
  - The weighted mid (micro-price), (best_ask * bid_sizes[0] + best_bid * ask_sizes[0]) / (bid_sizes[0] + ask_sizes[0]), estimates the next price better than the mid; when bids are heavier it sits near the ask (Stoikov 2018).
  - Quote around a fair-value estimate, not the mid, and lean it against your inventory (Avellaneda & Stoikov 2008)."""
if os.environ.get("MM_HINT", "0") == "1":
    SPEC = SPEC + HINT

# The seats return identical text for identical prompts sent together, so parallel samples differ
# only if their prompts do. These change the framing, not the content.
FRAMINGS = [
    "",
    "Keep the module short: under 40 lines.",
    "Start from the simplest version that obeys every rule, then add one idea.",
    "Put a one-line comment on each decision.",
]

CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text):
    blocks = CODE_BLOCK.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip()
    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*(import |from |def )", line):
            return "\n".join(lines[i:]).strip()
    return ""


def build_prompt(spec, code, fb, ledger, framing):
    """Spec, then the latest code and ONE instruction about it. Older rounds survive only as a
    one-line ledger: long repair histories cost tokens and stopped helping in published runs."""
    parts = {"spec": spec, "code": "", "feedback": "", "ledger": "", "framing": framing}
    if code:
        parts["code"] = f"\n\nYour current code:\n```python\n{code}\n```"
        parts["feedback"] = f"\n\n{fb}\nReturn the full corrected module."
    if ledger:
        parts["ledger"] = "\n\nEarlier attempts already failed with: " + "; ".join(ledger) + "."
    tail = (f"\n\n{framing}" if framing else "") + "\nAnswer with one ```python block only."
    text = parts["spec"] + parts["code"] + parts["feedback"] + parts["ledger"] + tail
    return text, {k: len(v) for k, v in parts.items()}


# ---------------------------------------------------------------- the model

def ask(a, prompt):
    import httpx
    est = len(prompt) // 3
    budget = max(256, min(a.max_tokens, a.context - est - 64))
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=budget, temperature=a.temperature, top_p=0.8, top_k=20,
                chat_template_kwargs={"enable_thinking": False})
    assert "seed" not in body, "a seed in the request kills the engine on the seats"
    t0 = time.perf_counter()
    try:
        r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body, timeout=900)
    except httpx.HTTPError as e:
        return dict(content="", finish="error", error=str(e)[:200], seconds=time.perf_counter() - t0)
    if r.status_code != 200:
        return dict(content="", finish="error", error=f"HTTP {r.status_code}: {r.text[:200]}",
                    seconds=time.perf_counter() - t0)
    js = r.json()
    ch = js["choices"][0]
    msg = ch.get("message", {})
    usage = js.get("usage") or {}
    return dict(content=msg.get("content") or "", finish=ch.get("finish_reason"),
                reasoning_chars=len(next((msg[k] for k in REASONING_KEYS if msg.get(k)), "")),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                seconds=time.perf_counter() - t0, budget=budget)


def offline_reply(rnd):
    """No model: a crash, then the naive quoter, then the reference. Exercises every path of the
    loop with no endpoint. Never report a number from it."""
    if rnd == 0:
        code = checker.PLANTED["crash"][0]
    elif rnd == 1:
        code = open(os.path.join(HERE, "baseline.py")).read()
    else:
        code = open(os.path.join(HERE, "reference.py")).read()
    return dict(content=f"```python\n{code}\n```", finish="stop", seconds=0.0,
                prompt_tokens=None, completion_tokens=None)


# ---------------------------------------------------------------- the loop

def short(res):
    if res["solved"]:
        return "SOLVED"
    lcb = res["metrics"].get("pnl_lcb")
    extra = f" lcb {lcb:+.0f}" if lcb is not None else ""
    return f"{res['score']:.2f} {res['stage']} {','.join(res['tags'][:2])}{extra}"


def solve(a, level, rep, log):
    name = mmsim.LEVELS[level]["name"]
    print(f"\n=========== level {level}: {name} ({a.feedback} feedback, rep {rep}) ===========",
          flush=True)
    ledger, latest_code, latest_fb = [], None, None
    best = dict(score=-1.0, code=None, res=None)
    stall, attempts, tokens, solved_at = 0, 0, 0, None
    t_level = time.perf_counter()
    for rnd in range(a.rounds):
        restart = stall >= 2
        if restart:
            stall = 0
        mode = "first" if latest_code is None else ("restart" if restart else "repair")
        prompts = []
        for i in range(a.samples):
            framing = FRAMINGS[(i + rnd) % len(FRAMINGS)]
            code_in = None if mode in ("first", "restart") else latest_code
            prompts.append((framing,) + build_prompt(SPEC, code_in, latest_fb, ledger, framing))
        t0 = time.perf_counter()
        if a.offline:
            replies = [offline_reply(rnd) for _ in prompts]
        else:
            with cf.ThreadPoolExecutor(max_workers=a.samples) as ex:
                replies = list(ex.map(lambda p: ask(a, p[1]), prompts))
        round_best, seen = None, []
        for i, ((framing, text, chars), rep_) in enumerate(zip(prompts, replies)):
            attempts += 1
            tokens += (rep_.get("prompt_tokens") or len(text) // 4) + (rep_.get("completion_tokens") or 0)
            code = extract_code(rep_["content"])
            if not code or rep_["finish"] == "length":
                # A cut-off answer fails to parse, but that is the budget, not the model: say so.
                tag = ("ENDPOINT_ERROR" if rep_["finish"] == "error"
                       else "TRUNCATED" if rep_["finish"] == "length" else "NO_CODE")
                msg = {"TRUNCATED": f"Your answer was cut off at {rep_.get('budget')} tokens "
                                    f"before the code ended ({len(code.splitlines())} lines). Write "
                                    f"a much shorter module: one function, under 60 lines.",
                       "NO_CODE": "No ```python block came back. Answer with the module only.",
                       "ENDPOINT_ERROR": rep_.get("error", "")}[tag]
                res = dict(level=level, score=0.0, solved=False, stage="no_code", tags=[tag],
                           metrics={}, error=dict(type=tag, message=msg),
                           violation=None, violation_counts={}, drivers={})
                fb = (f"Checker: score 0.00 of 1.00, stage reached: no_code. "
                      + {"none": "Not solved.", "raw": f"finish_reason={rep_['finish']}",
                         "located": msg}[a.feedback])
                code = code if rep_["finish"] != "length" else ""
            else:
                res = checker.check(code, level)
                fb = checker.feedback(res, a.feedback)
            seen.append(res)
            log.write(json.dumps(dict(
                run=a.tag, rep=rep, level=level, round=rnd, sample=i, feedback_mode=a.feedback,
                prompt_mode=mode, framing=framing, prompt_chars=len(text), chars=chars,
                prompt_tokens=rep_.get("prompt_tokens"), completion_tokens=rep_.get("completion_tokens"),
                finish_reason=rep_["finish"], seconds=round(rep_["seconds"], 1),
                reasoning_chars=rep_.get("reasoning_chars"), model=a.model if not a.offline else "offline",
                code_sha=hashlib.sha1(code.encode()).hexdigest()[:10] if code else None,
                score=res["score"], solved=res["solved"], stage=res["stage"], tags=res["tags"],
                pnl_lcb=res["metrics"].get("pnl_lcb"), metrics=res["metrics"],
                violation=res["violation"], error=res["error"], feedback=fb, code=code)) + "\n")
            log.flush()
            if round_best is None or res["score"] > round_best[0]["score"]:
                round_best = (res, code, fb)
            if not res["solved"] and res["tags"]:
                entry = f"{res['tags'][0]} ({res['stage']})"
                if entry not in ledger:
                    ledger = (ledger + [entry])[-4:]
        res, code, fb = round_best
        print(f"round {rnd} [{mode}] " + " | ".join(f"s{i} {short(r)}" for i, r in enumerate(seen))
              + f"  ({time.perf_counter() - t0:.0f}s)", flush=True)
        if res["solved"]:
            best = dict(score=1.0, code=code, res=res)
            solved_at = attempts
            break
        print(f"   feedback: {fb.splitlines()[-1][-420:]}", flush=True)
        if code:
            latest_code, latest_fb = code, fb
        if res["score"] > best["score"]:
            best = dict(score=res["score"], code=code, res=res)
            stall = 0
        else:
            stall += 1
    # the claim, then the held-out check it will be graded against
    claim = "VERIFIED on dev" if best["res"] and best["res"]["solved"] else "NOT VERIFIED"
    held = checker.check(best["code"], level, mmsim.heldout_seeds()) if best["code"] else None
    held_ok = bool(held and held["solved"])
    print(f"=> level {level}: {claim}"
          + (f" after {solved_at} attempts" if solved_at else f", best score {best['score']:.2f}")
          + f"; held-out {'PASS' if held_ok else 'fail'}"
          + (f" (lcb {held['metrics'].get('pnl_lcb', float('nan')):+.0f})" if held and held['metrics'] else "")
          + f"; {tokens} tokens, {time.perf_counter() - t_level:.0f}s", flush=True)
    # confidence that the chosen strategy is profitable, from its dev mean and standard error
    bm = (best["res"] or {}).get("metrics") or {}
    confidence = (0.5 * (1 + math.erf(bm["pnl_mean"] / bm["pnl_se"] / math.sqrt(2)))
                  if bm.get("pnl_se") else 0.0)
    log.write(json.dumps(dict(
        run=a.tag, rep=rep, level=level, final=True, feedback_mode=a.feedback, claim=claim,
        confidence=round(confidence, 4), flags={k: v for k, v in os.environ.items() if k.startswith("MM_")},
        dev_score=best["score"], dev_solved=claim.startswith("VERIFIED"), attempts=attempts,
        attempts_to_solve=solved_at, tokens=tokens, seconds=round(time.perf_counter() - t_level, 1),
        heldout_solved=held_ok, heldout_score=held["score"] if held else 0.0,
        heldout_lcb=held["metrics"].get("pnl_lcb") if held else None,
        heldout_tags=held["tags"] if held else [], code=best["code"])) + "\n")
    log.flush()
    return claim, held_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=list(mmsim.CALIBRATED))
    ap.add_argument("--feedback", default="located", choices=("none", "raw", "located"))
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--tag", default=None, help="run name, e.g. B or C; default = feedback mode")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base", default=BASE_URL)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    a.tag = a.tag or a.feedback
    if not a.offline and not a.base:
        sys.exit("No endpoint. In a seat pod KERNEL_AGENT_BASE_URL is preset; otherwise\n"
                 "  export MM_AGENT_BASE_URL=http://localhost:8000/v1")
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    summary = []
    for rep in range(1, a.repeat + 1):
        path = os.path.join(HERE, "runs", f"{a.tag}-{rep}{'-offline' if a.offline else ''}.jsonl")
        with open(path, "a") as log:
            for lv in a.levels:
                summary.append((rep, lv) + solve(a, lv, rep, log))
        print(f"\nlog: {path}")
    print("\nrep level claim              held-out")
    for rep, lv, claim, held in summary:
        print(f"{rep:3d} {lv:5d} {claim:18s} {'PASS' if held else 'fail'}")


if __name__ == "__main__":
    main()
