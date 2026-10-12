"""The checker: what counts as a working market maker, and what the next attempt is told.

Layers, cheapest first. A layer that fails stops the check, and its message is the feedback:

  L0 static      parses; defines quote(state); imports only math and numpy; no I/O, no randomness
  L1 runs        a child process plays every dev episode, with a timeout; any exception is reported
                 with the line, the step and the state that caused it
  L2 rules       every quote on every step: whole ticks, bid < ask, never takes liquidity, sizes in
                 range, cannot breach the inventory limit. One violation zeroes the performance score
  L3 obligation  both sides within 2 ticks of the market on at least half the steps (a market maker
                 that never quotes cannot lose, so this stops "do nothing" from passing)
  L4 profit      mean PnL minus two standard errors, over all dev episodes, is above zero

Score out of 1.0: 0.1 static, 0.1 runs, 0.2 rules, 0.2 obligation, 0.4 profit. Solved = all five.
The profit bar is a 95% one-sided bound across 16 seeded episodes, so a lucky run cannot pass it and
"solved" means the same thing every time it is checked. Held-out seeds are scored by the same code.
"""
import ast
import json
import os
import subprocess
import sys
import tempfile

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

import mmsim

HERE = os.path.dirname(os.path.abspath(__file__))
ALLOWED_IMPORTS = {"math", "numpy"}
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
                   "globals", "locals", "vars", "getattr", "setattr", "delattr", "exit", "quit"}
NO_FILLS_RULE = os.environ.get("MM_NO_FILLS_RULE", "0") == "1"   # off in the frozen runs A, B, C
EXC_RULE = os.environ.get("MM_EXC_RULE", "0") == "1"              # C3: exceptions name the cause
ROBUST = os.environ.get("MM_ROBUST", "0") == "1"                  # C4: must also survive spread 1
TABLE = os.environ.get("MM_TABLE", "0") == "1"                    # C5: show adverse fills as data
ROBUST_DIAG = os.environ.get("MM_ROBUST_DIAG", "0") == "1"        # C7: diagnose the 1-tick failure
MIN_FILLS = 5                                                     # per episode
# numpy can touch files; a strategy has no reason to. Added after the runs; no logged strategy uses
# any of these (audit.py checks), so no result changes.
NUMPY_IO = {"load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt",
            "fromfile", "tofile", "memmap", "fromregex", "DataSource", "lib", "ctypeslib"}
RULE_TAGS = ("FORMAT", "OFF_TICK", "BAD_SIZE", "CROSSED", "TAKES", "INVENTORY_LIMIT")

RULE_FIX = {
    "INVENTORY_LIMIT": "cap bid_size at max_inventory - inventory and ask_size at "
                       "max_inventory + inventory (0 means that side is not quoted)",
    "TAKES": "keep bid <= best_ask - 1 and ask >= best_bid + 1 after every adjustment",
    "CROSSED": "after all adjustments, if bid >= ask then move one of them so bid < ask",
    "OFF_TICK": "prices are whole ticks: wrap them in int(...) after math.floor or math.ceil",
    "BAD_SIZE": "sizes must be int from 0 to max_order_size",
    "FORMAT": "return exactly {'bid': int or None, 'bid_size': int, 'ask': int or None, "
              "'ask_size': int}",
}


# ---------------------------------------------------------------- L0

def static_check(code):
    """Return (tag, message) or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return "SYNTAX", f"SyntaxError on line {e.lineno}: {e.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return "FORBIDDEN_IMPORT", (f"line {node.lineno}: import {a.name} is not allowed;"
                                                f" only math and numpy")
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return "FORBIDDEN_IMPORT", (f"line {node.lineno}: from {node.module} import ... is "
                                            f"not allowed; only math and numpy")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS:
            return "FORBIDDEN_CALL", f"line {node.lineno}: {node.id} is not allowed"
        if isinstance(node, ast.Attribute):
            if node.attr == "random":
                return "FORBIDDEN_CALL", (f"line {node.lineno}: randomness is not allowed; the "
                                          f"strategy must be deterministic")
            if node.attr.startswith("__"):
                return "FORBIDDEN_CALL", f"line {node.lineno}: dunder attribute {node.attr}"
            if node.attr in NUMPY_IO:
                return "FORBIDDEN_CALL", f"line {node.lineno}: {node.attr} touches files; not allowed"
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "quote"]
    if not fns:
        return "NO_QUOTE_FN", "the module must define a top-level function quote(state)"
    args = fns[0].args
    if len(args.args) - len(args.defaults) != 1:
        return "NO_QUOTE_FN", ("quote must be callable as quote(state): one required argument "
                               "(extra ones need defaults)")
    return None


def unpacks_state(code):
    """Line numbers where the code does `a, b, ... = state`. On a dict that assigns the KEYS, so
    best_bid becomes the string 'best_bid' and the crash appears lines later, somewhere else."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name) and n.value.id == "state"
            and any(isinstance(t, (ast.Tuple, ast.List)) for t in n.targets)]


# ---------------------------------------------------------------- L1, in a child process

def run_child(code, level, seeds, timeout=180, robust=False):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    cmd = ([sys.executable, os.path.join(HERE, "robust_worker.py"), path] if robust else
           [sys.executable, os.path.join(HERE, "mmsim.py"), "--worker", path])
    try:
        p = subprocess.run(cmd + [str(level), ",".join(map(str, seeds))],
                           capture_output=True, text=True, timeout=timeout, cwd=HERE)
    except subprocess.TimeoutExpired:
        return dict(error=dict(type="TIMEOUT", message=f"no result within {timeout}s",
                               step=None, line=None, seed=None), episodes=[], violations=[])
    finally:
        os.unlink(path)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return dict(error=dict(type="CRASH", message=(p.stderr or p.stdout)[-300:], step=None,
                               line=None, seed=None), episodes=[], violations=[])


def probe(code, timeout=60):
    """Hostile-state probes (probe_worker.py): quote() on 252 crafted corner states. Returns
    dict(probes, findings) or dict(error). Reported by hostile.py and `checker.py FILE --probe`."""
    bad = static_check(code)
    if bad:
        return dict(error=f"static check: {bad[1]}")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        p = subprocess.run([sys.executable, os.path.join(HERE, "probe_worker.py"), path],
                           capture_output=True, text=True, timeout=timeout, cwd=HERE)
        return json.loads(p.stdout) if p.stdout else dict(error=(p.stderr or "no output")[-200:])
    except subprocess.TimeoutExpired:
        return dict(error=f"no result within {timeout}s")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------- L2-L4

def check(code, level, seeds=None):
    seeds = list(seeds if seeds is not None else mmsim.DEV_SEEDS)
    r = dict(level=level, n=len(seeds), score=0.0, solved=False, stage="static", tags=[],
             metrics={}, error=None, violation=None, violation_counts={}, drivers={})
    bad = static_check(code)
    if bad:
        r["tags"], r["error"] = [bad[0]], dict(type=bad[0], message=bad[1])
        return r
    r["score"], r["stage"] = 0.1, "runs"
    out = run_child(code, level, seeds)
    if out["error"]:
        e = out["error"]
        tag = e["type"] if e["type"] in ("TIMEOUT", "CRASH") else "EXCEPTION"
        r["tags"], r["error"] = [tag], e
        r["unpack_lines"] = unpacks_state(code)
        return r
    r["score"], r["stage"] = 0.2, "rules"
    eps = out["episodes"]
    m = aggregate(eps)
    r["metrics"] = m
    if out["violations"]:
        counts = {}
        for v in out["violations"]:
            counts[v[2]] = counts.get(v[2], 0) + 1
        s, t, rule, detail = out["violations"][0]
        r["violation_counts"] = counts
        r["violation"] = dict(seed=s, step=t, rule=rule, detail=detail)
        r["tags"] = sorted(counts, key=counts.get, reverse=True)
        return r
    r["score"], r["stage"] = 0.4, "obligation"
    if m["obligation"] < mmsim.OBLIGATION:
        r["tags"] = ["OBLIGATION"]
        r["score"] += 0.2 * m["obligation"] / mmsim.OBLIGATION
        return r
    r["score"], r["stage"] = 0.6, "profit"
    if m["pnl_lcb"] > 0:
        if ROBUST:
            # L5 (run C4 only): the same seeds at a 1-tick spread must also pass every layer
            rb = run_child(code, level, seeds, robust=True)
            fail = None
            if rb["error"]:
                fail = dict(kind="exception", detail=f"{rb['error']['type']}: {rb['error']['message']}"
                            f" at step {rb['error'].get('step')}")
            elif rb["violations"]:
                s_, t_, rule, detail = rb["violations"][0]
                fail = dict(kind="rule", rule=rule, count=len(rb["violations"]),
                            detail=f"{rule} first at step {t_}: {detail}")
            else:
                rm = aggregate(rb["episodes"])
                if rm["obligation"] < mmsim.OBLIGATION:
                    fail = dict(kind="obligation", detail=f"obligation {rm['obligation']:.0%}")
                elif rm["pnl_lcb"] <= 0:
                    fail = dict(kind="profit", detail=f"profit lower bound {rm['pnl_lcb']:+.0f}")
            if fail:
                r["score"], r["stage"], r["tags"], r["robust_fail"] = 0.9, "robust", ["SPREAD1"], fail
                r["robust_sub"] = sub_result(rb, code)
                return r
        r["score"], r["solved"], r["stage"] = 1.0, True, "solved"
        return r
    r["score"] += 0.2 * m["win_rate"]
    r["drivers"] = drivers(m)
    if NO_FILLS_RULE and m["fills"] < MIN_FILLS:
        # Added after the frozen runs (C2): a strategy parked one tick behind the touch meets the
        # obligation, never trades, earns exactly 0 and was told only "no single mechanism".
        r["tags"] = ["NO_FILLS"]
    else:
        r["tags"] = [max(r["drivers"], key=lambda k: -r["drivers"][k])] if r["drivers"] else ["LOSS"]
    return r


def sub_result(out, code):
    """The 1-tick run graded like a normal check, so located() can diagnose it (run C7)."""
    sub = dict(score=0.0, solved=False, tags=[], metrics={}, error=None, violation=None,
               violation_counts={}, drivers={})
    if out["error"]:
        sub.update(stage="runs", tags=["EXCEPTION"], error=out["error"],
                   unpack_lines=unpacks_state(code))
        return sub
    m = aggregate(out["episodes"])
    sub["metrics"] = m
    if out["violations"]:
        counts = {}
        for v in out["violations"]:
            counts[v[2]] = counts.get(v[2], 0) + 1
        s_, t_, rule, detail = out["violations"][0]
        sub.update(stage="rules", violation=dict(seed=s_, step=t_, rule=rule, detail=detail),
                   violation_counts=counts, tags=sorted(counts, key=counts.get, reverse=True))
        return sub
    if m["obligation"] < mmsim.OBLIGATION:
        sub.update(stage="obligation", tags=["OBLIGATION"])
        return sub
    sub["stage"], sub["drivers"] = "profit", drivers(m)
    if m["fills"] < MIN_FILLS:
        sub["tags"] = ["NO_FILLS"]
    else:
        sub["tags"] = [max(sub["drivers"], key=lambda k: -sub["drivers"][k])] if sub["drivers"] else ["LOSS"]
    return sub


def aggregate(eps):
    p = np.array([e["pnl"] for e in eps])
    n = len(p)
    se = float(p.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    mean = lambda k: float(np.nanmean([e[k] for e in eps]))  # noqa: E731
    total = lambda k: float(np.sum([e[k] for e in eps]))     # noqa: E731
    fills_against = total("against_n")
    fills = total("fills")
    return dict(
        pnl_mean=float(p.mean()), pnl_se=se, pnl_lcb=float(p.mean() - 2 * se),
        win_rate=float(np.mean(p > 0)), worst=float(p.min()),
        fills=mean("fills"), swept=mean("swept"), edge_pnl=mean("edge_pnl"),
        hold_pnl=mean("hold_pnl"), markout=mean("markout_mean"),
        against_n=fills_against / n, against_markout=(
            float(np.sum([e["against_markout"] * e["against_n"] for e in eps]) / fills_against)
            if fills_against else 0.0),
        with_markout=(float(np.sum([e["with_markout"] * (e["fills"] - e["against_n"]) for e in eps])
                            / (fills - fills_against)) if fills > fills_against else 0.0),
        heavy_pnl=mean("heavy_pnl"), at_limit=mean("at_limit"), mean_abs_inv=mean("mean_abs_inv"),
        hot_steps=mean("hot_steps"), hot_pnl=mean("hot_pnl"), calm_pnl=mean("calm_pnl"),
        hot_half=mean("hot_half") if any(np.isfinite(e["hot_half"]) for e in eps) else float("nan"),
        calm_half=mean("calm_half"),
        obligation=mean("obligation"), both_sides=mean("both_sides"),
        skew_inv=mean("skew_inv"), skew_imb=mean("skew_imb"), widen_vol=mean("widen_vol"),
        max_drawdown=mean("max_drawdown"),
        table_n=[[int(sum(e["table_n"][i][b] for e in eps)) for b in range(3)] for i in range(2)]
        if all("table_n" in e for e in eps) else None,
        table_sum=[[float(sum(e["table_sum"][i][b] for e in eps)) for b in range(3)] for i in range(2)]
        if all("table_sum" in e for e in eps) else None,
        worst_fills=next((e["worst"] for e in eps if e.get("worst")), []),
        worst_seed=next((e["seed"] for e in eps if e.get("worst")), None),
    )


def drivers(m):
    """Where the losses come from, per episode, in ticks. Negative is a loss. Each driver is the
    loss a specific mechanism explains, so the largest one is the instruction worth giving."""
    d = {}
    d["LOSS_INVENTORY"] = m["heavy_pnl"]
    d["LOSS_ADVERSE"] = m["against_n"] * (m["against_markout"] - m["with_markout"])
    if m["hot_steps"] > 0:
        d["LOSS_VOLATILITY"] = m["hot_pnl"]
    return {k: round(v, 1) for k, v in d.items() if v < 0}


# ---------------------------------------------------------------- feedback

def feedback(r, mode="located"):
    """What the next attempt is told. 'none' is the score only, 'raw' is the measurements with no
    interpretation (the tool output), 'located' names the mechanism and a direction, never a
    coefficient: revealing target numbers made the heat-rod model copy instead of derive."""
    head = f"Checker: score {r['score']:.2f} of 1.00, stage reached: {r['stage']}."
    if r["solved"]:
        return head + " Solved."
    if mode == "none":
        return head + " Not solved."
    if mode == "raw":
        raw = dict(error=r["error"], violation=r["violation"], violation_counts=r["violation_counts"],
                   metrics={k: (round(v, 3) if isinstance(v, float) else v)
                            for k, v in r["metrics"].items()})
        return head + "\n" + json.dumps(raw)
    return head + "\n" + located(r)


def located(r):
    e, m = r["error"], r["metrics"]
    tag = r["tags"][0] if r["tags"] else ""
    if r["stage"] == "static":
        return f"{e['message']}. Fix that line; nothing else was run."
    if r["stage"] == "runs":
        if tag == "TIMEOUT":
            return ("Your code took too long: 16 episodes of 1,500 steps must finish in 3 minutes. "
                    "Avoid loops over the whole history on every step.")
        if tag == "CRASH":
            return f"The run crashed: {e['message']}"
        where = f" on line {e['line']} of your code" if e.get("line") else ""
        hint = ""
        if EXC_RULE and r.get("unpack_lines"):
            ln = r["unpack_lines"][0]
            return (f"{e['type']}: {e['message']}{where}, at step {e.get('step')}. The cause is "
                    f"line {ln}: state is a dict, and `a, b, ... = state` assigns its KEYS (the "
                    f"strings 't', 'T', 'best_bid', ...), not its values. Read each field by name: "
                    f"best_bid = state['best_bid'].")
        if EXC_RULE and e["type"] != "IndexError":
            pass
        elif e.get("history_len") is not None and e["history_len"] < 25:
            hint = (f" At that step mid_history had only {e['history_len']} entries: the first "
                    f"steps of an episode have a short history, so guard any window you take.")
        return (f"{e['type']}: {e['message']}{where}, at step {e.get('step')} (inventory "
                f"{e.get('inventory')}, best_bid {e.get('best_bid')}, best_ask {e.get('best_ask')})."
                f"{hint}")
    if r["stage"] == "rules":
        v = r["violation"]
        n = sum(r["violation_counts"].values())
        others = ", ".join(f"{k} x{c}" for k, c in r["violation_counts"].items() if k != v["rule"])
        return (f"Rule broken: {v['rule']}, {r['violation_counts'][v['rule']]} times"
                f"{' (also ' + others + ')' if others else ''}; {n} in total. First at step "
                f"{v['step']}: {v['detail']}. Any violation zeroes the profit score. "
                f"Fix: {RULE_FIX[v['rule']]}.")
    if r["stage"] == "robust" and ROBUST_DIAG and r.get("robust_sub"):
        sub = r["robust_sub"]
        msg = located(sub)
        if sub["tags"] and sub["tags"][0] == "NO_FILLS":
            msg = msg.replace("A market sell fills a bid only at best_bid (half the time) or above it "
                              "(first in line); a bid below best_bid never fills, and asks mirror this.",
                              "At a 1-tick spread there is no room inside: a bid fills only at exactly "
                              "best_bid (half the time) and never below it; asks mirror this.")
        return ("Passes at a 2-tick spread. The same markets at a 1-tick spread (where real stocks "
                "trade most of the day) fail, and this is the diagnosis at 1 tick: " + msg)
    if r["stage"] == "robust":
        f = r["robust_fail"]
        return ("Passes at a 2-tick spread, but the same markets at a 1-tick spread (where real "
                f"stocks trade most of the day) fail: {f['detail']}. Make every quote valid for any "
                "spread of 1 tick or more: clamp bid <= best_ask - 1 and ask >= best_bid + 1, and "
                "if bid >= ask after your adjustments, fall back to bid = best_bid, "
                "ask = best_ask. Prices must be whole ticks even when the mid is a half tick.")
    if r["stage"] == "obligation":
        return (f"You quoted both sides within {mmsim.NEAR} ticks of the market on "
                f"{m['obligation']:.0%} of steps; the obligation is {mmsim.OBLIGATION:.0%}. "
                f"(Both sides posted at all: {m['both_sides']:.0%} of steps.) Quote closer to "
                f"best_bid/best_ask, and only withdraw a side when the inventory limit forces it.")
    # profit
    lead = (f"Profit per episode {m['pnl_mean']:+.0f} ± {m['pnl_se']:.0f} (standard error), so the "
            f"95% lower bound is {m['pnl_lcb']:+.0f}; it must be above 0. "
            f"{m['win_rate']:.0%} of episodes made money. You earned {m['edge_pnl']:+.0f} at the "
            f"moment of your fills and {m['hold_pnl']:+.0f} from price moves while holding inventory.")
    if tag == "NO_FILLS":
        return lead + (
            f" You traded only {m['fills']:.1f} times per episode: both quotes sit where nobody "
            f"trades. A market sell fills a bid only at best_bid (half the time) or above it "
            f"(first in line); a bid below best_bid never fills, and asks mirror this. Quote at "
            f"best_bid/best_ask and manage risk from there.")
    if tag == "LOSS_INVENTORY":
        return lead + (
            f" The holding loss is the problem: {m['heavy_pnl']:+.0f} per episode came while you "
            f"held 7 or more shares either way, and you sat at the limit on {m['at_limit']:.0%} of "
            f"steps. The price drifts in one direction for long stretches and the order flow leans "
            f"the same way, so you keep selling into a rise (or buying into a fall). Your quote "
            f"centre barely moves with your inventory (correlation {m['skew_inv']:+.2f}). Lean "
            f"against your position: when long, move both quotes down; when short, move both up; "
            f"and take into account which way the mid has been moving.")
    if tag == "LOSS_ADVERSE" and TABLE and m.get("table_n"):
        n, sm = m["table_n"], m["table_sum"]
        cell = lambda i, b: (f"n={n[i][b]:<5d} {sm[i][b] / n[i][b]:+.2f}" if n[i][b] else  # noqa: E731
                             f"n=0{'':9s}")
        lines = [lead + " The fills are the problem. Your fills, split by which way you traded "
                 "and by the top-of-book imbalance imb = (bid_sizes[0] - ask_sizes[0]) / "
                 "(bid_sizes[0] + ask_sizes[0]) at that moment, with the mid move over the next "
                 "10 steps per fill (+ = in your favour), all 16 episodes:",
                 "                 imb < -0.3        -0.3..0.3        imb > 0.3",
                 f"  you bought     {cell(0, 0)}   {cell(0, 1)}   {cell(0, 2)}",
                 f"  you sold       {cell(1, 0)}   {cell(1, 1)}   {cell(1, 2)}"]
        if m["worst_fills"]:
            ex = "; ".join(f"step {w['step']}: you {w['side']} at {w['price']} with bid_sizes[0]="
                           f"{w['bid0']}, ask_sizes[0]={w['ask0']}, mid {w['mid']:g} then "
                           f"{w['mid_later']:g} 10 steps later" for w in m["worst_fills"])
            lines.append(f"Your two worst fills (episode seed {m['worst_seed']}): {ex}.")
        if m["skew_imb"] < -0.1:
            lines.append(f"Your quotes move the WRONG way: when bids outweigh asks your quote centre "
                         f"goes down (correlation {m['skew_imb']:+.2f}); it should go up.")
        lines.append("Stop trading in the cells that lose: when imb is high do not post your ask "
                     "at the touch (move it up or skip it); when imb is low, do the same with your bid.")
        return "\n".join(lines)
    if tag == "LOSS_ADVERSE":
        return lead + (
            f" The fills are the problem. Split by the top-of-book sizes at the moment of the "
            f"fill: when you sold while bid_sizes[0] outweighed ask_sizes[0] (or bought with the "
            f"reverse), {m['against_n']:.0f} fills per episode, the mid then moved "
            f"{m['against_markout']:+.2f} ticks per fill in your favour within 10 steps "
            f"(negative = against you); your other fills made {m['with_markout']:+.2f}. The "
            f"displayed sizes predict the next move and your quotes ignore them (correlation of "
            f"your quote centre with the imbalance {m['skew_imb']:+.2f}). Shift both quotes toward "
            f"the heavier side, and be slow to sell into heavy bids or buy into heavy asks.")
    if tag == "LOSS_VOLATILITY":
        return lead + (
            f" {m['hot_pnl']:+.0f} per episode came in the {m['hot_steps']:.0f} steps where recent "
            f"mid moves were large (std of the last 20 moves above 1 tick). There your quotes were "
            f"{m['hot_half']:.1f} ticks from mid, against {m['calm_half']:.1f} in calm steps. When "
            f"the market moves fast, quote wider or step back from the touch.")
    return lead + " No single mechanism explains the loss; reduce risk before chasing edge."


# ---------------------------------------------------------------- self-test

PLANTED = {
    "syntax":      ("def quote(state)\n    return {}", "SYNTAX"),
    "import_os":   ("import os\ndef quote(state):\n    return None", "FORBIDDEN_IMPORT"),
    "random":      ("import numpy as np\ndef quote(state):\n    x = np.random.rand()\n"
                    "    return None", "FORBIDDEN_CALL"),
    "no_fn":       ("def quotes(state):\n    return None", "NO_QUOTE_FN"),
    "crash":       ("def quote(state):\n    m = state['mid_history']\n    x = m[-30]\n"
                    "    return {'bid': state['best_bid'], 'bid_size': 1, 'ask': state['best_ask'],"
                    " 'ask_size': 1}", "EXCEPTION"),
    "crossed":     ("def quote(state):\n    return {'bid': state['best_ask'] - 1, 'bid_size': 1,"
                    " 'ask': state['best_ask'] - 1, 'ask_size': 1}", "CROSSED"),
    "off_tick":    ("def quote(state):\n    return {'bid': state['best_bid'] - 0.5, 'bid_size': 1,"
                    " 'ask': state['best_ask'], 'ask_size': 1}", "OFF_TICK"),
    "takes":       ("def quote(state):\n    return {'bid': state['best_ask'], 'bid_size': 1,"
                    " 'ask': state['best_ask'] + 1, 'ask_size': 1}", "TAKES"),
    "over_limit":  ("def quote(state):\n    return {'bid': state['best_bid'], 'bid_size': 5,"
                    " 'ask': state['best_ask'], 'ask_size': 5}", "INVENTORY_LIMIT"),
    "numpy_io":    ("import numpy as np\ndef quote(state):\n    x = np.load('f.npy')\n"
                    "    return None", "FORBIDDEN_CALL"),
    "absent":      ("def quote(state):\n    return {'bid': None, 'bid_size': 0, 'ask': None,"
                    " 'ask_size': 0}", "OBLIGATION"),
}


def selftest():
    ok = True
    for name, (code, want) in PLANTED.items():
        r = check(code, 1)
        good = want in r["tags"] and not r["solved"]
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} {name:11s} -> {r['tags']} score {r['score']:.2f}")
        if not good:
            print("      ", located(r)[:200])
    expect = {("baseline.py", 1): True, ("baseline.py", 2): False, ("baseline.py", 3): False,
              ("reference.py", 1): True, ("reference.py", 2): True, ("reference.py", 3): True}
    for (f, lv), want in expect.items():
        code = open(os.path.join(HERE, f)).read()
        for split, seeds in (("dev", mmsim.DEV_SEEDS), ("held-out", mmsim.heldout_seeds())):
            r = check(code, lv, seeds)
            good = r["solved"] == want
            ok &= good
            print(f"  {'ok ' if good else 'BAD'} {f:12s} level {lv} {split:8s} solved={r['solved']}"
                  f" score {r['score']:.2f} lcb {r['metrics'].get('pnl_lcb', float('nan')):+.0f}"
                  f" {r['tags']}")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("file", nargs="?")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--feedback", default="located", choices=("none", "raw", "located"))
    ap.add_argument("--seeds", default=None,
                    help="judges: grade on your own held-back seeds, e.g. --seeds 5000,5001,5002")
    ap.add_argument("--probe", action="store_true", help="hostile-state probes instead of episodes")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    code = open(a.file).read()
    if a.probe:
        out = probe(code)
        if out.get("error"):
            sys.exit(out["error"])
        print(f"{out['probes']} hostile states, {len(out['findings'])} findings")
        for f in out["findings"][:10]:
            print("  ", f)
        sys.exit(0 if not out["findings"] else 1)
    seeds = ([int(x) for x in a.seeds.split(",")] if a.seeds else
             mmsim.heldout_seeds() if a.heldout else None)
    res = check(code, a.level, seeds)
    print(feedback(res, a.feedback))
    print(f"solved={res['solved']} seeds={len(seeds) if seeds else len(mmsim.DEV_SEEDS)}")
