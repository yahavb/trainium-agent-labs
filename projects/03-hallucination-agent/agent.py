#!/usr/bin/env python3
"""
agent.py — the loop: the model answers from a passage, a checker grades the answer, and the
grade plus the REASON go into the next attempt. Same shape as project 1. Inference only: the
model's weights never change. Everything the agent does, it does with prompts and checks.

    controller  poses an item                    halworld.py
    generator   the model proposes N replies     Qwen3-8B on your chip, sampling ON
    checker     grades each reply                halcheck.py
    loop        the best reply's feedback becomes the next prompt

Two modes, and comparing them is the point:

    oracle (default)   feedback comes from check(), which knows the gold label. This is the dev
                       loop and the source of the example bank. It is NOT a deployable agent.
    --label-free       feedback comes only from selfcheck(), which knows only the passage. The
                       agent stops when its own check passes and reports "verified". The oracle
                       then grades it, so you can see how often "verified" was true.

Three inference-time levers for the label-free agent, each off by default so you can measure it:

    --agree K            accept only when K samples pass selfcheck AND give the same answer
                         (sample agreement, after SelfCheckGPT). Disagreement is a signal.
    --challenge-abstain  when the agent is about to accept NOT_IN_CONTEXT, make it list every
                         sentence about the subject first. selfcheck cannot see a lazy abstention;
                         this is one way to look for one without the answer key.
    --fail-closed        if nothing is verified within --rounds, the agent answers NOT_IN_CONTEXT
                         (NOT_SURE closed-book) instead of presenting an unverified answer.
    --workers N          answer N questions at the same time. The server runs MAX_NUM_SEQS (4 in
                         serve.sh) requests at once, and --samples 1 alone uses one of them.
                         Keep workers x samples <= 4, or the extra requests just queue.
    --examples FILE      prepend K worked examples (--k) from a bank built by build_examples.py.
                         The loop's successes, reused as in-context examples on unseen items.
    --verify [basic|checklist]
                         before accepting a real answer, ask the model in a FRESH call whether the
                         quoted evidence states every detail of the question (Chain-of-Verification,
                         factored). Catches a real quote that misses one detail, which selfcheck
                         cannot. Costs one call per accepted answer. Off by default; measure it.
                         Measured on the same 12 items, checklist caught 3 of 3 on the first
                         check, but the model re-answered until a later check passed. Pair it
                         with --fail-closed.
                         basic:     one yes/no judgement. Measured on 12 dev items: caught 1 of 3
                                    hallucinations, wrongly rejected 1 of 5 correct answers.
                         checklist: the verifier first lists every detail the question requires
                                    (who, what, to whom, when, where, how), marks each STATED or
                                    NOT STATED, and paraphrase counts as stated. Any NOT STATED
                                    line overrides a SUPPORTED verdict. A rejection asks for a
                                    better quote before suggesting NOT_IN_CONTEXT.

    export HALLU_BASE_URL=http://localhost:8000/v1        # falls back to HEATROD_BASE_URL
    python agent.py --offline --level 1                    # no model; proves the plumbing
    python agent.py --level 2                              # one level, 4 items
    python agent.py --all --rounds 1                       # zero-shot baseline, every level
    python agent.py --data faith-unans+squad2 --n 100      # a real dataset (see halsets.py)
    nohup python agent.py --all --repeat 5 > run.log 2>&1 < /dev/null &
"""

import argparse
import collections
import concurrent.futures as cf
import json
import os
import random
import sys
import time

import halcheck
import halsets
import halworld

HALLUCINATED = {"fabricated_quote", "wrong_entity", "wrong_answer", "answered_unanswerable",
                "accepted_false_premise", "parametric_override", "picked_one_side"}


def ask_once(a, prompt):
    import httpx
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=a.max_tokens, temperature=a.temperature, top_p=0.95,
                chat_template_kwargs={"enable_thinking": a.think})
    r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body, timeout=900, verify=False)
    if r.status_code != 200:
        raise SystemExit(f"the server returned HTTP {r.status_code}:\n{r.text[:800]}")
    choice = r.json()["choices"][0]
    if choice.get("finish_reason") == "length":
        print("  [warning] finish_reason=length: the reply was cut off. Shorter prompt, not more tokens.")
    return choice["message"].get("content") or ""


def fake_model(item, rnd, rng):
    """Offline stand-in that improves each round. Never report a number that came from here."""
    if rng.random() < min(0.9, 0.2 + 0.3 * rnd):
        return "Let me read it.\n" + item["ideal"]
    return rng.choice([
        'ANSWER: 1987\nQUOTE: "It was established in 1987."',
        "ANSWER: NOT_IN_CONTEXT\nQUOTE: NONE",
        "I believe the answer is in the second sentence.",
    ])


def ask_many(a, item, prompt, rnd, rng, n):
    if a.offline:
        return [fake_model(item, rnd, rng) for _ in range(n)]
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(lambda _: ask_once(a, prompt), range(n)))


def short(reply):
    p = halcheck.parse(reply)
    q = " ".join(f'"{x}"' for x in p["quotes"]) or ("NONE" if p["quote_none"] else "(none)")
    return f"ANSWER: {p['answer']}\nQUOTE: {q}"

# ---------------------------------------------------------------- worked examples


def load_bank(path):
    bank = [json.loads(l) for l in open(path)]
    if any(isinstance(e["level"], int) and e["seed"] >= 1000 for e in bank):
        raise SystemExit(f"{path} holds judging seeds; rebuild it from seeds < 1000")
    return bank


def pick_examples(bank, item, k, rng):
    """k examples, never from the same item and seed, spread across answer kinds so the examples
    do not teach 'always abstain' or 'always answer'."""
    pool = [e for e in bank if not (e["item"] == item["id"] and e["seed"] == item["seed"])]
    by_kind = collections.defaultdict(list)
    for e in pool:
        by_kind[e["kind"]].append(e)
    kinds = sorted(by_kind)
    rng.shuffle(kinds)
    out = []
    while len(out) < k and any(by_kind.values()):
        for kd in kinds:
            if by_kind[kd] and len(out) < k:
                out.append(by_kind[kd].pop(rng.randrange(len(by_kind[kd]))))
    return out


def with_examples(base, examples):
    if not examples:
        return base
    blocks = []
    for i, e in enumerate(examples, 1):
        it = (halworld.make(e["level"], e["sub"], e["seed"]) if isinstance(e["level"], int)
              else halsets.get(e["item"]))
        if it.get("closed_book"):
            blocks.append(f"EXAMPLE {i}\nQUESTION: {it['question']}\n"
                          f"ANSWER: {halcheck.parse(e['reply'])['answer']}")
        else:
            blocks.append(f"EXAMPLE {i}\nPASSAGE:\n{it['passage']}\nQUESTION: {it['question']}\n"
                          f"{short(e['reply'])}")
    return "Worked examples:\n\n" + "\n\n".join(blocks) + "\n\nNow your turn.\n\n" + base

# ---------------------------------------------------------------- label-free acceptance


def agreed(replies, graded, k):
    """The largest group of selfcheck-passing replies that give the same answer, if it has >= k."""
    groups = collections.defaultdict(list)
    for r, g in zip(replies, graded):
        if g["selfcheck"][0]:
            groups[halcheck._norm(g["parsed"]["answer"] or "")].append((r, g))
    if not groups:
        return None
    best = max(groups.values(), key=len)
    return best[0] if len(best) >= k else None


def challenge_abstention(a, item, prompt, rng):
    """The agent is about to accept NOT_IN_CONTEXT. Ask the model to show its search first.
    Uses only the question and passage: no answer key."""
    subj = halcheck._entities_in(item["question"], item["passage"])
    who = subj[0] if subj else "the subject of the question"
    follow = (f"{prompt}\n\nBefore you answer: copy, word for word, every sentence in the passage "
              f"that mentions {who}. Then check whether one of them answers the question, and end "
              f"with the ANSWER and QUOTE lines.")
    reply = ask_many(a, item, follow, 1, rng, 1)[0]
    return reply, halcheck.check(item, reply)

# ---------------------------------------------------------------- verification (label-free)

VERIFY_PROMPT = """You are checking someone else's answer. Read only the evidence below.

QUESTION: {question}
PROPOSED ANSWER: {answer}
EVIDENCE: {evidence}

Does the evidence, read literally, state that the proposed answer is the answer to this exact
question? Check every detail the question asks: who did what to whom, when, where, and how.
A related fact is not enough.

Reply with exactly two lines:
VERDICT: SUPPORTED or VERDICT: NOT_SUPPORTED
MISSING: <the detail of the question the evidence does not state, or NONE>"""


CHECKLIST_PROMPT = """You are checking someone else's answer. Read only the evidence below.

QUESTION: {question}
PROPOSED ANSWER: {answer}
EVIDENCE: {evidence}

Step 1. List every detail the question requires, one per line:
DETAIL: <detail>
Include each who, what, to whom, when (dates, years, decades, centuries), where and how that the
question mentions. A question that names a time period has a "when" detail.

Step 2. Check each detail against the evidence, one per line:
CHECK: <detail> -> STATED
CHECK: <detail> -> NOT STATED
The same meaning in different words counts as STATED. A different time, place, person, or
direction (who did what to whom) counts as NOT STATED. Compare dates and periods exactly,
converting centuries to years where needed.

Step 3. End with exactly two lines:
VERDICT: SUPPORTED (only if every detail is STATED) or VERDICT: NOT_SUPPORTED
MISSING: <the first detail that is NOT STATED, or NONE>"""


def parse_verdict(text, mode="basic"):
    """-> (supported, missing). For checklist mode, any 'NOT STATED' check overrides a SUPPORTED
    verdict: the model's own itemised check is trusted over its summary line."""
    import re
    m = re.search(r"VERDICT\s*:\s*\**\s*(NOT[_ ]SUPPORTED|SUPPORTED)", text, re.I)
    miss = re.search(r"MISSING\s*:\s*(.+)", text, re.I)
    supported = bool(m) and not m.group(1).upper().startswith("NOT")
    missing = miss.group(1).strip() if miss else "no verdict given"
    if mode == "checklist":
        bad = re.findall(r"CHECK\s*:\s*(.+?)\s*-+>\s*NOT[_ ]STATED", text, re.I)
        if bad:
            supported = False
            if missing.upper().startswith("NONE") or missing == "no verdict given":
                missing = bad[0].strip()
    if supported:
        missing = "NONE"
    return supported, missing


def verify(a, item, reply, rnd, rng):
    """A fresh call that sees only the question, the answer and the quotes: not the passage, not
    the model's own reasoning. Returns (supported, missing_detail, raw_text)."""
    p = halcheck.parse(reply)
    evidence = " ".join(f'"{q}"' for q in p["quotes"]) or "(none)"
    if a.offline:   # plumbing only: the fake verifier peeks at the answer key
        ok = halcheck.matches(item, p["answer"])
        return ok, ("NONE" if ok else "(offline fake)"), "VERDICT: " + ("SUPPORTED" if ok else "NOT_SUPPORTED")
    tmpl = CHECKLIST_PROMPT if a.verify == "checklist" else VERIFY_PROMPT
    text = ask_once(a, tmpl.format(question=item["question"], answer=p["answer"], evidence=evidence))
    ok, missing = parse_verdict(text, a.verify)
    return ok, missing, text

# ---------------------------------------------------------------- the loop


def solve(item, a, log, run, rng, bank):
    base = halworld.prompt_of(item)
    examples = pick_examples(bank, item, a.k, random.Random(f"{item['id']}-{item['seed']}-{run}")) \
        if bank else []
    first = with_examples(base, examples)
    prompt, ledger = first, []
    zero_shot, final, claimed, challenged = None, None, None, False
    if a.verbose:
        print(f"\n--- {item['id']} seed {item['seed']} ({item['kind']}; trap: {item['trap']}) ---")
        print(f"Q: {item['question']}")

    def record(rnd, reply, g, note=""):
        log.write(json.dumps(dict(
            run=run, item=item["id"], level=item["level"], sub=item["sub"], seed=item["seed"],
            kind=item["kind"], round=rnd, mode="label_free" if a.label_free else "oracle",
            examples=len(examples), note=note, base_prompt=base, prompt=prompt, reply=reply,
            reward=g["reward"], label=g["label"], parts=g["parts"],
            selfcheck_ok=g["selfcheck"][0])) + "\n")

    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        replies = ask_many(a, item, prompt, rnd, rng, a.samples)
        graded = [halcheck.check(item, r) for r in replies]
        for r, g in zip(replies, graded):
            record(rnd, r, g)
        log.flush()
        if rnd == 0:
            zero_shot = [g["label"] for g in graded]
        best = max(graded, key=lambda g: g["reward"])
        if a.verbose:
            print(f"  round {rnd}: rewards {[g['reward'] for g in graded]}  "
                  f"labels {[g['label'] for g in graded]}  ({time.perf_counter() - t0:.1f}s)")

        if a.label_free:
            hit = agreed(replies, graded, a.agree)
            if hit:
                r, g = hit
                if (a.challenge_abstain and g["parsed"]["answer"] == "NOT_IN_CONTEXT"
                        and not challenged):
                    challenged = True
                    r2, g2 = challenge_abstention(a, item, prompt, rng)
                    record(rnd, r2, g2, note="abstention challenge")
                    if a.verbose:
                        print(f"  challenged the abstention -> {g2['label']}")
                    if g2["selfcheck"][0] and g2["parsed"]["answer"] not in halcheck.ABSTAIN:
                        r, g = r2, g2
                rejected = None
                if a.verify and g["parsed"]["answer"] not in halcheck.ABSTAIN:
                    ok, missing, vtext = verify(a, item, r, rnd, rng)
                    log.write(json.dumps(dict(
                        run=run, item=item["id"], level=item["level"], sub=item["sub"],
                        seed=item["seed"], kind=item["kind"], round=rnd, mode="label_free",
                        note="verify", verify_mode=a.verify,
                        verdict="SUPPORTED" if ok else "NOT_SUPPORTED",
                        missing=missing, reply=vtext, label="verifier_call", reward=None,
                        candidate_label=g["label"])) + "\n")
                    if a.verbose:
                        print(f"  verify: {'SUPPORTED' if ok else 'NOT_SUPPORTED'} "
                              f"(missing: {missing})  candidate was {g['label']}")
                    if not ok:
                        rejected = missing
                if rejected is None:
                    final, claimed = g, True
                    break
                if a.verify == "checklist":
                    feedback = (f"A separate check found that your quoted evidence does not state "
                                f"this detail of the question: {rejected}. Look for another "
                                f"sentence in the passage that does state it, and quote that one. "
                                f"Only if no sentence states it, answer NOT_IN_CONTEXT.")
                else:
                    feedback = (f"A separate check of your answer found that the quoted evidence does "
                                f"not state this detail of the question: {rejected}. Find a sentence "
                                f"that states every detail. If there is none, answer NOT_IN_CONTEXT.")
                shown = r
                final, claimed = g, False          # the answer the agent holds, not the oracle's pick
            elif any(g["selfcheck"][0] for g in graded):
                passing = [g for g in graded if g["selfcheck"][0]]
                feedback = (f"Your samples disagreed: {sorted({g['parsed']['answer'] for g in passing})}. "
                            f"At most one of these is right. Reread the sentences and answer again.")
                shown = replies[graded.index(passing[0])]
                final, claimed = passing[0], False
            else:
                feedback, shown = graded[0]["selfcheck"][1], replies[0]
                final, claimed = graded[0], False
        else:
            final = best
            if best["reward"] == 1.0:
                break
            shown = replies[graded.index(best)]
            feedback = best["feedback"]

        if a.verbose:
            print(f"  feedback: {feedback}")
        ledger.append(short(shown).replace("\n", " | "))
        tried = "\n".join(f"  - {x}" for x in ledger[-3:][:-1])
        prompt = (f"{first}\n\nA previous reply was:\n{short(shown)}\n"
                  f"A checker found this problem with it: {feedback}\n"
                  + (f"These earlier replies were also wrong; do not repeat them:\n{tried}\n" if tried else "")
                  + "Fix it.")
    closed = False
    if a.label_free and a.fail_closed and not claimed:
        # could not verify anything: say so, rather than present an unverified answer
        abst = "ANSWER: NOT_SURE" if item.get("closed_book") else "ANSWER: NOT_IN_CONTEXT\nQUOTE: NONE"
        final, closed = halcheck.check(item, abst), True
        log.write(json.dumps(dict(
            run=run, item=item["id"], level=item["level"], sub=item["sub"], seed=item["seed"],
            kind=item["kind"], round=rnd, mode="label_free", note="fail_closed", reply=abst,
            reward=final["reward"], label=final["label"])) + "\n")
        if a.verbose:
            print(f"  could not verify any answer: fail closed -> {final['label']}")
    log.flush()
    return dict(item=item["id"], level=item["level"], zero_shot=zero_shot, final=final["label"],
                reward=final["reward"], rounds=rnd + 1, claimed=claimed, fail_closed=closed)


class LockedLog:
    """One log shared by worker threads. Every write is a whole JSON line, so a lock per write
    keeps lines from interleaving."""
    def __init__(self, f):
        import threading
        self.f, self.lock = f, threading.Lock()

    def write(self, s):
        with self.lock:
            self.f.write(s)
            self.f.flush()

    def flush(self):
        pass


def run_items(items, a, log, run, bank):
    """Every item once. With --workers > 1 they run concurrently; results keep item order."""
    if a.workers <= 1:
        rng = random.Random(run)
        return [solve(it, a, log, run, rng, bank) for it in items]
    import threading
    done, lock = [0], threading.Lock()

    def one(it):
        res = solve(it, a, log, run, random.Random(f"{run}-{it['id']}"), bank)
        with lock:
            done[0] += 1
            print(f"  [{done[0]}/{len(items)}] {it['id']:<18} -> {res['final']:<22} "
                  f"({res['rounds']} round{'s' if res['rounds'] > 1 else ''})", flush=True)
        return res
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        return list(ex.map(one, items))


def rate(labels, which):
    return sum(l in which for l in labels) / max(1, len(labels))


def report(results, a):
    by = collections.defaultdict(list)
    for r in results:
        by[r["level"]].append(r)
    print("\n=========== summary ===========")
    w = max(5, *(len(str(lv)) for lv in by))
    print(f"{'level':<{w}}  solved   round-0 halluc  round-0 over-abstain  final halluc  "
          + ("verified-and-right" if a.label_free else "mean rounds"))
    for lv in sorted(by, key=str):
        rs = by[lv]
        zs = [l for r in rs for l in r["zero_shot"]]
        fin = [r["final"] for r in rs]
        solved = sum(r["final"] == "correct" for r in rs)
        if a.label_free:
            cl = [r for r in rs if r["claimed"]]
            tail = f"{sum(r['final'] == 'correct' for r in cl)}/{len(cl)}"
        else:
            tail = f"{sum(r['rounds'] for r in rs) / len(rs):.1f}"
        print(f"{str(lv):<{w}}  {f'{solved}/{len(rs)}':<7}  {rate(zs, HALLUCINATED):>13.0%}  "
              f"{rate(zs, {'over_abstain'}):>19.0%}  {rate(fin, HALLUCINATED):>12.0%}  {tail}")
    fc = [r for r in results if r.get("fail_closed")]
    if fc:
        print(f"\nfail-closed: {len(fc)} question(s) had no verified answer and abstained -> "
              f"{dict(collections.Counter(r['final'] for r in fc))}")
    allzs = collections.Counter(l for r in results for l in r["zero_shot"])
    print("\nround-0 failure taxonomy (every round-0 sample):")
    for lab, n in allzs.most_common():
        print(f"  {lab:<24} {n}")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, choices=sorted(halworld.LEVELS))
    ap.add_argument("--all", action="store_true", help="every level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1, help="seeds per item, starting at --seed")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=4, help="1 = zero-shot, no loop")
    ap.add_argument("--repeat", type=int, default=1, help="identical runs; report a rate, not a best")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("HALLU_MAX_TOKENS", 600)))
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--label-free", action="store_true", help="feedback from selfcheck only")
    ap.add_argument("--agree", type=int, default=1,
                    help="label-free: samples that must pass selfcheck with the same answer")
    ap.add_argument("--challenge-abstain", action="store_true",
                    help="label-free: make the model show its search before accepting NOT_IN_CONTEXT")
    ap.add_argument("--verify", nargs="?", const="basic", choices=["basic", "checklist"],
                    help="label-free: a fresh call checks the quote states every detail before "
                         "accepting. --verify alone = basic; --verify checklist = itemised")
    ap.add_argument("--selftest", action="store_true", help="test the verdict parser; no model")
    ap.add_argument("--fail-closed", action="store_true",
                    help="label-free: if no answer is verified within --rounds, answer NOT_IN_CONTEXT")
    ap.add_argument("--workers", type=int, default=1,
                    help="questions answered at the same time (server slots: workers x samples <= 4)")
    ap.add_argument("--examples", help="example bank from build_examples.py")
    ap.add_argument("--k", type=int, default=3, help="worked examples per prompt")
    ap.add_argument("--model", default=os.environ.get("HALLU_MODEL",
                                                       os.environ.get("HEATROD_MODEL", "Qwen/Qwen3-8B")))
    ap.add_argument("--base", default=os.environ.get("HALLU_BASE_URL",
                                                      os.environ.get("HEATROD_BASE_URL")))
    ap.add_argument("--log", default="attempts.jsonl")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("-q", "--quiet", dest="verbose", action="store_false")
    ap.add_argument("--data", help="a real dataset from halsets.py, e.g. squad2 or faith-unans+squad2")
    ap.add_argument("--n", type=int, default=50, help="--data: how many items")
    ap.add_argument("--offset", type=int, default=0, help="--data: skip this many items first")
    ap.add_argument("--max-chars", type=int, default=12000,
                    help="--data: skip longer prompts (about 3.4k tokens; fits MAX_MODEL_LEN=4096)")
    return ap


def items_for(a):
    if a.data:
        return halsets.load(a.data, n=a.n, offset=a.offset, max_chars=a.max_chars)
    levels = sorted(halworld.LEVELS) if (a.all or not a.level) else [a.level]
    return [halworld.make(lv, sub, seed) for lv in levels
            for seed in range(a.seed, a.seed + a.seeds) for sub in halworld.SUBS]


def selftest():
    """The verdict parser, on replies shaped like the ones Qwen3-8B wrote on seat-17."""
    cases = [
        ("basic", "VERDICT: SUPPORTED\nMISSING: NONE", True, "NONE"),
        ("basic", "VERDICT: NOT_SUPPORTED\nMISSING: the year", False, "the year"),
        ("basic", "**VERDICT:** NOT SUPPORTED\nMISSING: who", False, "who"),
        ("basic", "I think it is fine.", False, "no verdict given"),
        ("checklist", "DETAIL: region\nDETAIL: in the 2000s\nCHECK: region -> STATED\n"
                      "CHECK: in the 2000s -> NOT STATED\nVERDICT: SUPPORTED\nMISSING: NONE",
         False, "in the 2000s"),                       # itemised check overrides the summary
        ("checklist", "CHECK: who -> STATED\nCHECK: when -> STATED\nVERDICT: SUPPORTED\nMISSING: NONE",
         True, "NONE"),
        ("checklist", "CHECK: originated from -> STATED\nVERDICT: NOT_SUPPORTED\nMISSING: origin",
         False, "origin"),
        ("checklist", "CHECK: drug -> STATED\nCHECK: signal transduction --> NOT STATED\n"
                      "VERDICT: NOT_SUPPORTED\nMISSING: signal transduction pathways", False,
         "signal transduction pathways"),
    ]
    fails = 0
    for mode, text, want_ok, want_miss in cases:
        ok, miss = parse_verdict(text, mode)
        good = ok == want_ok and miss == want_miss
        fails += not good
        print(f"  {'ok  ' if good else 'FAIL'} {mode:<9} -> supported={ok!s:<5} missing={miss!r}")
    print("\nSELFTEST " + ("PASSED" if not fails else f"FAILED ({fails})"))
    return 1 if fails else 0


def main():
    a = build_parser().parse_args()
    if a.selftest:
        sys.exit(selftest())

    if not a.base and not a.offline:
        sys.exit("set HALLU_BASE_URL (or HEATROD_BASE_URL) to your vLLM endpoint, or pass --offline")
    if a.offline:
        print("*** OFFLINE: fake generator, numbers are meaningless ***")
    if a.seed >= 1000 and not a.data:
        print("*** seeds >= 1000 are reserved for judging ***")
    if a.agree > a.samples:
        sys.exit("--agree cannot exceed --samples")
    bank = load_bank(a.examples) if a.examples else None
    items = items_for(a)
    if a.offline and a.data:
        print("*** --offline with --data: the fake model answers from the gold label ***")
    levels = list(dict.fromkeys(it["level"] for it in items))

    slots = int(os.environ.get("HALLU_SERVER_SLOTS", 4))
    if a.workers > 1:
        a.verbose = False   # per-question traces would interleave; one line per question instead
        if a.workers * a.samples > slots:
            print(f"*** {a.workers} workers x {a.samples} samples = {a.workers * a.samples} requests "
                  f"at once, but the server runs {slots}; the rest will queue ***")
        print(f"answering {len(items)} questions, {a.workers} at a time")
    per_run = []
    with open(a.log, "a") as raw:
        log = LockedLog(raw)
        for run in range(a.repeat):
            results = run_items(items, a, log, run, bank)
            if a.repeat > 1:
                print(f"\n##### run {run + 1}/{a.repeat}")
            report(results, a)
            per_run.append(results)

    if a.repeat > 1:
        print(f"\n=========== {a.repeat} runs: solved per level, one number per run ===========")
        for lv in levels:
            counts = [sum(r["final"] == "correct" for r in rs if r["level"] == lv) for rs in per_run]
            n = sum(1 for r in per_run[0] if r["level"] == lv)
            print(f"  level {lv}: {counts}  (of {n})")
    print(f"\nattempts logged to {a.log}")


if __name__ == "__main__":
    main()
