#!/usr/bin/env python3
"""
agent2.py -- the kernel agent with six roles: manager, planner, retriever, coder, debugger, reviewer.
See DESIGN.md for why, and agents2/ for each role. agent.py is unchanged and stays the baseline.

    python agent2.py --level 1                      # one level, 2 threads, up to 4 approaches
    python agent2.py --level 1 --repeat 3           # a solve rate, not an anecdote
    python agent2.py --all --hint                   # levels 1-4, with nkibench's level hint
    python agent2.py --offline --all                # no model: exercises every role (numbers meaningless)
    python agent2.py --dry-run --level 1            # print the prompts and their token counts, no calls
    python agent2.py --classify runs/*/attempts.jsonl   # error kinds and lint over old agent.py logs
    python agent2.py --level 1 --role planner='http://seat-198.seat:8000/v1|Qwen/Qwen3-32B'

Each run writes runs/<tag>/: config.json, events.jsonl (every call, plan, check, change, verdict),
attempts.jsonl (agent.py's schema, for the comparison scripts) and summary.json.
"""

import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import agent                                    # noqa: E402  (reused: extract_code, enrich, fragments)
import nkibench                                 # noqa: E402
from agents2 import config, errors, lint        # noqa: E402
from agents2.events import Events               # noqa: E402
from agents2.retriever import Retriever         # noqa: E402

FULL = sum(agent.WEIGHTS.values())


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", type=int, choices=sorted(nkibench.LEVELS))
    ap.add_argument("--all", action="store_true", help="levels 1 to 4")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2, help="threads per level (2 = the server's peak)")
    ap.add_argument("--approaches", type=int, default=4, help="plans per level, across threads")
    ap.add_argument("--attempts", type=int, default=10, help="checks per thread")
    ap.add_argument("--no-gain", type=int, default=6, help="checks without a better reward ending a thread")
    ap.add_argument("--max-calls", type=int, default=80, help="model calls per level")
    ap.add_argument("--check-workers", type=int, default=2, help="0 = check in this process")
    ap.add_argument("--coder-max-tokens", type=int, default=None)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--hint", action="store_true", help="add nkibench's level hint (organisers' text)")
    ap.add_argument("--no-skeleton", action="store_true",
                    help="leave the organisers' example kernel (the end of agent.API_CARD) out of the "
                         "coder's first prompt, as in the pull design's first runs")
    ap.add_argument("--cards", choices=("checked", "introspect"), default="checked",
                    help="checked: the cheat-sheet and agents2/cards.md cards first (default); "
                         "introspect: signatures and docstrings only, as in the first runs")
    ap.add_argument("--no-lookup", action="store_true",
                    help="roles may not LOOKUP documentation: the problem statement and the index only")
    ap.add_argument("--aws-docs", action="store_true",
                    help="add prose from third_party/ AWS docs (nki 0.4.0), filtered by withhold.json")
    ap.add_argument("--model", default=config.default_model())
    ap.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--path", default="")
    ap.add_argument("--role", action="append", metavar="ROLE=BASE|MODEL",
                    help="send one role elsewhere, e.g. planner='http://seat-198.seat:8000/v1|Qwen/Qwen3-32B'")
    ap.add_argument("--out", default=None, help="run directory (default runs/agent2-MMDD-HHMM)")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the prompts and token counts; no calls")
    ap.add_argument("--classify", nargs="+", metavar="ATTEMPTS_JSONL",
                    help="analyse agent.py attempt logs: error kinds, rule coverage, lint")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- --classify

def classify(paths):
    """How much of an old run would the debugger's rules and the lint have handled?

    Lint must never flag a kernel that ran: a lint finding on an attempt whose `runs` part was true is
    a false positive, and is listed."""
    retr = Retriever()
    kinds, rule_ok, lint_hits, false_pos, lint_kinds = (collections.Counter() for _ in range(5))
    n = 0
    for p in paths:
        for line in open(p):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "feedback" not in d or "level" not in d:
                continue
            n += 1
            k = errors.classify_feedback(d["feedback"])
            kinds[k] += 1
            body = d["feedback"].split(": ", 1)[1] if "shapes passed. On" in d["feedback"] else d["feedback"]
            body = body[len("raised "):] if body.startswith("raised ") else body
            if k not in ("CORRECT",):
                rule = errors.rule_change(dict(kind=k, error=body), retr, d["level"], agent.enrich,
                                          agent.fragment_note)
                rule_ok[(k, bool(rule))] += 1
            spec = nkibench.LEVELS.get(d["level"], {})
            found = lint.lint(d.get("code") or "", traffic_level=bool(spec.get("max_waste")))
            if found:
                lint_hits[k] += 1
                lint_kinds[found[0]["kind"]] += 1
                if (d.get("parts") or {}).get("runs"):
                    false_pos[(d["level"], found[0]["msg"][:80])] += 1
    print(f"{n} attempts from {len(paths)} file(s)\n")
    print("error kinds (recorded feedback):")
    for k, c in kinds.most_common():
        handled = rule_ok[(k, True)]
        print(f"  {k:<14} {c:>5}   rule names a change: {handled}/{c}" if k != "CORRECT" else f"  {k:<14} {c:>5}")
    other = kinds["OTHER"]
    fails = n - kinds["CORRECT"]
    print(f"\nclassified: {fails - other}/{fails} failures ({(fails - other) / max(fails, 1):.0%})")
    rule_total = sum(c for (k, ok), c in rule_ok.items() if ok)
    print(f"a rule names the change for {rule_total}/{fails} ({rule_total / max(fails, 1):.0%}); "
          f"the rest go to the debugger's model path")
    print(f"\nlint flags {sum(lint_hits.values())} attempts before the simulator: "
          + ", ".join(f"{k} {c}" for k, c in lint_kinds.most_common()))
    print("  by recorded kind: " + ", ".join(f"{k} {c}" for k, c in lint_hits.most_common()))
    if false_pos:
        print(f"\nLINT FALSE POSITIVES ({sum(false_pos.values())}): flagged kernels that ran")
        for (lv, msg), c in false_pos.most_common(10):
            print(f"  level {lv}  x{c}  {msg}")
    else:
        print("lint false positives (flagged, yet the kernel ran): 0")
    return 0


# ---------------------------------------------------------------- --dry-run

def dry_run(cfg, levels, a):
    from agents2.coder import Coder
    from agents2.ledger import Ledger, Plan
    from agents2.llm import Tokens, pack
    from agents2.planner import Planner
    tokens = Tokens(cfg.roles["planner"].model)
    retr = Retriever(aws_docs=cfg.aws_docs, cards=a.cards == "checked")
    shown = {}

    class Capture:
        context = {}

        def chat(self, role, sections, tags=None, temperature=None, max_tokens=None):
            text, report = pack(sections, cfg.roles[role].prompt_cap, tokens.count)
            shown[role] = (text, report)
            return "", dict(role=role, error="dry run")

    for level in levels:
        led = Ledger(level, 0)
        Planner(Capture(), retr, cfg).plan(level, led)
        plan = Plan(approach="(example plan)", calls=["nl.sum", "nisa.tensor_scalar", "tile.permute"])
        Coder(Capture(), retr, cfg, agent.extract_code).write(level, plan, led)
        for role, (text, report) in shown.items():
            print(f"\n======== level {level} {role}: ~{report['prompt_tokens_est']} tokens "
                  f"({tokens.how}); sections {report['sections']}; dropped {report['dropped']}")
            print(text)
    return 0


# ---------------------------------------------------------------- a run

def main(argv=None):
    a = parse_args(argv)
    if a.classify:
        return classify(a.classify)
    cfg = config.build(a)
    levels = sorted(nkibench.LEVELS)[:4] if a.all else [a.level or 1]
    if a.dry_run:
        return dry_run(cfg, levels, a)

    from agents2.checks import CheckPool, InlineChecks
    from agents2.llm import LLM, OfflineLLM, Tokens
    from agents2.manager import Manager

    tag = a.out or os.path.join("runs", time.strftime("agent2-%m%d-%H%M"))
    os.makedirs(tag, exist_ok=True)
    tokens = Tokens(a.model)
    llm = (OfflineLLM if a.offline else LLM)(cfg, tokens)
    contexts = llm.connect()
    retr = Retriever(aws_docs=cfg.aws_docs, cards=a.cards == "checked")
    with open(os.path.join(tag, "config.json"), "w") as f:
        json.dump(dict(argv=sys.argv, config=cfg.as_dict(), contexts={f"{b}|{m}": c for (b, m), c
                  in contexts.items()}, tokenizer=tokens.how, nki=retr.version, cards=a.cards,
                  card_names=sorted(retr.card_text)), f, indent=1)
    events = Events(os.path.join(tag, "events.jsonl"), os.path.join(tag, "attempts.jsonl"))
    llm.events = events
    checks = InlineChecks() if a.check_workers == 0 else CheckPool(a.check_workers, cfg.check_timeout)

    if a.offline:
        print("*** OFFLINE: canned plans, the reference kernel replayed. Numbers are meaningless. ***")
    else:
        for name, r in cfg.roles.items():
            print(f"{name:<9} {r.model} @ {r.base}  context {contexts.get((r.base, r.model))}  "
                  f"T={r.temperature} top_p={r.top_p} max_out={r.max_tokens}")
    print(f"lookup rounds {cfg.lookup}")
    print(f"threads {cfg.threads}, approaches {cfg.max_approaches}, hint {cfg.hint}, "
          f"skeleton {cfg.skeleton}, cards {a.cards} "
          f"({len(retr.card_text)}), aws-docs {cfg.aws_docs}, tokens: {tokens.how}, nki {retr.version}"
          f"\nlog: {tag}/")

    mgr = Manager(cfg, llm, checks, retr, events, agent)
    history = {lv: [] for lv in levels}
    results = []
    try:
        for rep in range(a.repeat):
            if a.repeat > 1:
                print(f"\n################ run {rep + 1} of {a.repeat} ################")
            for level in levels:
                print(f"\n=========== level {level}: {nkibench.LEVELS[level]['op']} ===========")
                res = mgr.run_level(level, rep)
                results.append(res)
                history[level].append(res["reward"])
                print(f"level {level}: {'SOLVED' if res['solved'] else 'not solved'}  best "
                      f"{res['reward']:.2f}  stop: {res['stop_reason']}  checks {res['checks']}  "
                      f"calls {res['calls']}  {res['wall_seconds']:.0f}s")
                if res["solved"]:
                    print("---------------- the kernel ----------------\n" + res["best_code"]
                          + "\n--------------------------------------------")
                if res["stop_reason"] == "environment":
                    sys.exit("stopping: the environment cannot run checks (see events.jsonl)")
    finally:
        checks.close()
        events.close()

    print(f"\n=========== over {a.repeat} run(s) ===========")
    summary = {}
    for lv in levels:
        got = history[lv]
        solves = sum(1 for r in got if r >= FULL - 1e-9)
        summary[lv] = dict(solved=solves, runs=len(got), rewards=got)
        print(f"  level {lv}: solved {solves}/{len(got)}  best {max(got):.2f}  worst {min(got):.2f}  "
              f"mean {sum(got) / len(got):.2f}  all={[round(r, 2) for r in got]}")
    with open(os.path.join(tag, "summary.json"), "w") as f:
        json.dump(dict(levels=summary, results=[{k: v for k, v in r.items() if k != "best_code"}
                                                for r in results]), f, indent=1)
    print(f"\nlog: {tag}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
