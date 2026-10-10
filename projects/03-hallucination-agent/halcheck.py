#!/usr/bin/env python3
"""
halcheck.py — the checker. Grades one reply against one item. No LLM judge anywhere: every verdict
is string matching against a passage we generated, so it is exact, cheap and reproducible.

Reward, out of 1.0:
    0.1  format     an ANSWER: line exists
    0.3  grounded   every quote is copied word for word from the passage AND holds the claim
                    (an abstention is grounded only when abstaining was right)
    0.6  correct    the answer matches the gold label

So "always abstain" scores 0.1 on every answerable item, and "always answer" scores at most 0.4
on every unanswerable one. Neither shortcut survives a level.

Two checks, and the difference between them is the most important idea in this project:

    check(item, reply)        ORACLE. Knows the gold label. For the dev loop, for scoring, and for
                              building the example bank. You will not have it in deployment.
    selfcheck(passage, reply) LABEL-FREE. Knows only the passage. Is every quote real? Does the
                              quote hold the answer? An agent can run this on a question nobody
                              has answered yet. That is what "the agent knows when it failed" means
                              here.

    python halcheck.py --selftest
"""

import argparse
import re
import sys

# NOT_IN_CONTEXT, FALSE_PREMISE: halworld and passage datasets. CONFLICT: FaithEval-inconsistent,
# where the passage contradicts itself. NOT_SURE: closed-book datasets (PopQA), where there is no passage.
ABSTAIN = {"NOT_IN_CONTEXT", "FALSE_PREMISE", "CONFLICT", "NOT_SURE"}

# ---------------------------------------------------------------- parsing


def _norm(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _words(s):
    s = _norm(s)
    s = re.sub(r"[^\w\s]", " ", s)
    return [w for w in s.split() if w not in ("the", "a", "an", "its", "of")]


def parse(reply):
    """-> dict(answer=str|None, quotes=[str], quote_none=bool). Uses the LAST ANSWER/QUOTE lines,
    so reasoning above them is ignored. <think> blocks are stripped first."""
    text = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S)
    answer, quote_line = None, None
    for line in text.splitlines():
        m = re.match(r"\s*\**\s*ANSWER\s*\**\s*:\s*(.*)", line, re.I)
        if m:
            answer = m.group(1).strip().strip("*").strip()
        m = re.match(r"\s*\**\s*QUOTE\s*\**\s*:\s*(.*)", line, re.I)
        if m:
            quote_line = m.group(1).strip()
    if answer is not None:
        answer = answer.strip().strip('"').strip()
        up = answer.upper().replace(" ", "_").strip(".")
        if up in ABSTAIN:
            answer = up
    quotes, quote_none = [], False
    if quote_line is not None:
        if quote_line.strip().strip('"').upper().startswith("NONE"):
            quote_none = True
        else:
            quotes = re.findall(r'["“](.+?)["”]', quote_line) or [quote_line.strip()]
    return dict(answer=answer, quotes=[q.strip() for q in quotes if q.strip()],
                quote_none=quote_none)


def _numbers(s):
    return [int(n.replace(",", "")) for n in re.findall(r"-?\d[\d,]*", s or "")]


def same_answer(given, gold):
    """Exact after normalisation; numbers compared as numbers; or the gold words all present in a
    short answer ("the Atlantic Ocean" for "Atlantic"). Long answers that merely contain the gold
    somewhere do not pass."""
    if given is None or given in ABSTAIN:
        return False
    gn, an = _numbers(gold), _numbers(given)
    if gn and len(gn) == 1 and re.fullmatch(r"\d+", gold.strip()):
        return len(an) == 1 and an[0] == gn[0]
    gw, aw = _words(gold), _words(given)
    if gw == aw:
        return True
    return all(w in aw for w in gw) and len(aw) <= len(gw) + 3


def matches(item, given):
    """Is `given` the right answer for this item? Handles the three shapes of gold:
    a string (halworld), a list of accepted aliases (SQuAD, MuSiQue, PopQA, HotpotQA), and
    multiple choice (FaithEval-counterfactual), where "B", "(B)", "B) granite" and "granite" all count."""
    if given is None or given in ABSTAIN:
        return False
    if item.get("choices"):
        m = re.match(r"\s*\(?([A-Ha-h])(?:[\).:]|\s*$|\s+\S)", given)
        letters = {l for l, _ in item["choices"]}
        if m and m.group(1).upper() in letters:
            return m.group(1).upper() == item["gold_letter"]
        return same_answer(given, item["gold"])
    golds = item["gold"] if isinstance(item["gold"], list) else [item["gold"]]
    return any(same_answer(given, g) for g in golds if g)


def _canon(s):
    """_norm, plus spacing around punctuation ignored. HotpotQA (and other tokenised corpora) store
    "New York 's", "( born", "2017 ." and "do n't"; a model quoting it naturally writes "New York's".
    Words and their order still have to match exactly."""
    s = _norm(s)
    s = re.sub(r"\s+([.,;:!?%)\]}'])", r"\1", s)     # no space before closing punctuation / 's
    s = re.sub(r"([(\[{])\s+", r"\1", s)               # no space after opening brackets
    s = re.sub(r"\s+(n't)\b", r"\1", s)                 # do n't -> don't
    s = re.sub(r"\s*([-\u2013\u2014/])\s*", r"\1", s)    # hyphens and dashes with or without spaces
    return re.sub(r"\s+", " ", s).strip()


def quote_is_verbatim(q, passage):
    """Word for word. A quote that skips words with "..." passes if every piece is verbatim and the
    pieces appear in order: "The Normans ... gave their name to Normandy" is an honest quote."""
    pieces = [p.strip(" .,;") for p in re.split(r"\.\.\.|\u2026", q)]
    pieces = [p for p in pieces if p]
    if not pieces:
        return False
    text, at = _canon(passage), 0
    for p in pieces:
        cp = _canon(p).strip(" .,;")
        i = text.find(cp, at)
        if i < 0:
            return False
        at = i + len(cp)
    return True

# ---------------------------------------------------------------- the label-free check


def selfcheck(passage, reply, answer_in_quote=True):
    """What can be verified WITHOUT knowing the answer. Returns (ok, reason).

    ok=True does not mean correct. It means nothing in the reply is provably invented. An
    abstention always passes here, which is exactly why abstention needs the oracle to grade it
    and why the levels mix answerable items in."""
    p = parse(reply)
    if p["answer"] is None:
        return False, "Your reply has no line starting with ANSWER:. End with ANSWER: and QUOTE: lines."
    if p["answer"] in ABSTAIN:
        if p["answer"] == "FALSE_PREMISE":
            if not p["quotes"]:
                return False, ("You answered FALSE_PREMISE without quoting the sentence that "
                               "contradicts the question. Quote it word for word.")
            bad = [q for q in p["quotes"] if not quote_is_verbatim(q, passage)]
            if bad:
                return False, (f'The text you quoted, "{bad[0]}", is not in the passage. Copy the '
                               f"contradicting sentence word for word.")
        return True, "abstained"
    if not p["quotes"]:
        return False, ("You gave an answer but no quote. Copy the sentence that states it word for "
                       "word, or answer NOT_IN_CONTEXT if no sentence does.")
    bad = [q for q in p["quotes"] if not quote_is_verbatim(q, passage)]
    if bad:
        return False, (f'The text you quoted, "{bad[0]}", does not appear in the passage. Copy a '
                       f"sentence word for word from the passage, or answer NOT_IN_CONTEXT if no "
                       f"sentence states it.")
    joined = " ".join(p["quotes"])
    nums = _numbers(p["answer"])
    if nums and not all(str(n) in joined.replace(",", "") for n in nums):
        # a computed figure: the operands must be quoted, the total need not be
        if len(_numbers(joined)) < 2:
            return False, (f"Your answer {p['answer']} does not appear in your quote, and the quote "
                           f"does not hold the figures it was computed from. Quote the sentences "
                           f"holding each figure you used.")
        return True, "computed from quoted figures"
    if not answer_in_quote:
        return True, "every quote is in the passage"
    if not nums and not all(w in _words(joined) for w in _words(p["answer"])):
        return False, (f'Your quote is real but does not contain your answer "{p["answer"]}". '
                       f"Quote the sentence that actually states it.")
    return True, "every quote is in the passage and holds the answer"

# ---------------------------------------------------------------- the oracle check


def _entities_in(text, passage):
    """Company names from the passage that occur in text. Used to say WHICH company a quote is about."""
    names = set(re.findall(r"\b([A-Z][a-z]+ (?:Systems|Mills|Works|Labs|Freight|Instruments|"
                           r"Foundry|Textiles))\b", passage))
    return sorted(n for n in names if n in text)


def _asked_entity(item):
    m = _entities_in(item["question"], item["passage"])
    return m[0] if m else None


def check(item, reply):
    """-> dict(reward, parts, label, feedback, parsed, selfcheck).

    label is the failure taxonomy, one of:
      correct, correct_ungrounded, fabricated_quote, wrong_entity, wrong_answer,
      answered_unanswerable, accepted_false_premise, parametric_override,
      over_abstain, wrong_abstain_type, no_format,
      picked_one_side (passage contradicts itself), abstained (closed book, NOT_SURE)

    feedback is DIRECTIONAL: it says where to look and what kind of mistake it was, never the gold
    value. Project 1 measured why: shown the target, the model copies it and learns nothing."""
    p = parse(reply)
    parts = dict(format=0.0, grounded=0.0, correct=0.0)
    closed = item.get("closed_book", False)
    sc_ok, sc_reason = ((True, "closed book: nothing to verify against") if closed else
                        selfcheck(item["passage"], reply, item.get("answer_in_quote", True)))
    out = dict(parsed=p, selfcheck=(sc_ok, sc_reason))

    def done(label, feedback):
        out.update(parts=parts, reward=round(sum(parts.values()), 2), label=label, feedback=feedback)
        return out

    if p["answer"] is None or p["answer"] == "":
        return done("no_format", "Your reply has no ANSWER: line. End with exactly two lines, "
                                 "ANSWER: ... and QUOTE: ...")
    parts["format"] = 0.1
    ans, kind = p["answer"], item["kind"]
    ent = _asked_entity(item)

    # ---- closed book (PopQA): no passage, so no quote to check. The only honest alternative to a
    # right answer is NOT_SURE. A wrong answer is the hallucination; NOT_SURE is not.
    if closed:
        if matches(item, ans):
            parts["grounded"], parts["correct"] = 0.3, 0.6
            return done("correct", "Correct.")
        if ans in ABSTAIN:
            parts["grounded"] = 0.3
            return done("abstained", "You said you are not sure. That is honest, but scores less "
                                     "than a right answer.")
        return done("wrong_answer", "That answer is not right. If you are not certain, answer "
                                    "NOT_SURE rather than guessing.")

    # ---- gold says: the passage contradicts itself (FaithEval-inconsistent)
    if kind == "conflict":
        if ans == "CONFLICT":
            parts["grounded"], parts["correct"] = 0.3, 0.6
            return done("correct", "Correct: the passage gives conflicting answers.")
        if ans in ABSTAIN:
            parts["correct"] = 0.2
            return done("wrong_abstain_type", "The passage does address this, but in more than one "
                                              "way. Look for two statements that disagree: that is CONFLICT.")
        return done("picked_one_side", "Your answer is supported by one part of the passage, but "
                                       "another part says something different. Check whether the "
                                       "passage agrees with itself.")

    # ---- gold says: answer it
    if kind == "answer":
        if ans in ABSTAIN:
            where = f" about {ent}" if ent else ""
            return done("over_abstain",
                        f"You answered {ans}, but the passage does state this{where}. Read every "
                        f"sentence{where} again; the answer is there.")
        correct = matches(item, ans)
        grounded = sc_ok
        if item["level"] == 7 and grounded:   # operands, not just any two numbers
            grounded = all(k in " ".join(p["quotes"]) for k in item["key"])
        if grounded:
            parts["grounded"] = 0.3
        if correct:
            parts["correct"] = 0.6
            if grounded:
                return done("correct", "Correct and grounded.")
            return done("correct_ungrounded", sc_reason if not sc_ok else
                        "Your answer is right, but the quotes do not hold the figures it comes "
                        "from. Quote each sentence you used.")
        if not sc_ok and any(not quote_is_verbatim(q, item["passage"]) for q in p["quotes"]):
            if item["level"] == 5 and _norm(ans) in _norm(" ".join(r[2] for r in _real())):
                return done("parametric_override",
                            f"{sc_reason} Your answer comes from what you already know, not from "
                            f"the briefing. Answer with what the passage says, even where it "
                            f"differs from the real world.")
            return done("fabricated_quote", sc_reason)
        quoted = _entities_in(" ".join(p["quotes"]), item["passage"])
        if ent and quoted and ent not in quoted:
            return done("wrong_entity",
                        f"Your quote is about {', '.join(quoted)}, but the question asks about "
                        f"{ent}. Find the sentence about {ent}.")
        if item["level"] == 7:
            gn, an = int(item["gold"]), (_numbers(ans) or [None])[0]
            hint = ("too high" if an is not None and an > gn else "too low")
            return done("wrong_answer", f"The figures you quoted are the right ones, but your result "
                                        f"is {hint}. Redo the arithmetic on the quoted figures.")
        return done("wrong_answer", "Your quote does not support your answer. Read the quoted "
                                    "sentence again and answer with what it says.")

    # ---- gold says: the passage is silent
    if kind == "not_in_context":
        if ans == "NOT_IN_CONTEXT":
            parts["grounded"], parts["correct"] = 0.3, 0.6
            return done("correct", "Correct: the passage does not say.")
        if ans in ABSTAIN:
            parts["correct"] = 0.2
            return done("wrong_abstain_type",
                        f"Nothing in the passage contradicts the question; it is simply silent. "
                        f"That is NOT_IN_CONTEXT, not {ans}.")
        if item["level"] == 5:
            return done("parametric_override",
                        f"{sc_reason if not sc_ok else 'Your quote does not state this.'} Your "
                        f"answer comes from what you already know. The question asks what the "
                        f"briefing says, and only the briefing.")
        if sc_ok:   # a real quote, about something else
            quoted = _entities_in(" ".join(p["quotes"]), item["passage"])
            if ent and quoted and ent not in quoted:
                return done("answered_unanswerable",
                            f"Your quote is about {', '.join(quoted)}, not {ent}. Check whether any "
                            f"sentence states this for {ent} itself.")
        return done("answered_unanswerable", sc_reason if not sc_ok else
                    "Your quote is real, but compare it with the question detail by detail: who did "
                    "what to whom, when, and how. The quote leaves at least one of those out or "
                    "says something different. If no sentence states every detail, the answer is "
                    "NOT_IN_CONTEXT.")

    # ---- gold says: the question's premise is contradicted
    if ans == "FALSE_PREMISE":
        if sc_ok and any(item["key"] in q for q in p["quotes"]):
            parts["grounded"], parts["correct"] = 0.3, 0.6
            return done("correct", "Correct: the passage contradicts the question.")
        parts["correct"] = 0.6
        return done("correct_ungrounded", sc_reason if not sc_ok else
                    "Right that the premise is wrong, but your quote is not the sentence that "
                    "contradicts it. Quote that one.")
    if ans in ABSTAIN:
        parts["correct"] = 0.2
        return done("wrong_abstain_type",
                    "Something the question takes for granted is contradicted by a sentence in the "
                    "passage. Find it: that is FALSE_PREMISE, and quote it.")
    return done("accepted_false_premise",
                "Before answering, check that the passage agrees with what the question assumes "
                "about who and where. It does not.")


def _real():
    import halworld
    return halworld.REAL

# ---------------------------------------------------------------- selftest


def selftest():
    """Prove the checker before trusting a score. Every gold reply must score 1.0 on every
    generated item; every planted mistake must be caught AND land in the right taxonomy bucket."""
    import halworld
    fails = 0

    def expect(name, got, want_reward=None, want_label=None):
        nonlocal fails
        ok = (want_reward is None or abs(got["reward"] - want_reward) < 1e-9) and \
             (want_label is None or got["label"] == want_label)
        if not ok:
            fails += 1
            print(f"  FAIL {name}: reward {got['reward']} label {got['label']} "
                  f"(wanted {want_reward} {want_label})\n       {got['feedback']}")

    n = 0
    for lv in halworld.LEVELS:
        for sub in halworld.SUBS:
            for seed in range(25):
                it = halworld.make(lv, sub, seed)
                n += 1
                # determinism
                assert it == halworld.make(lv, sub, seed), "generator is not deterministic"
                # the gold reply is perfect, and passes the label-free check
                expect(f"{it['id']}s{seed} gold", check(it, "Some reasoning.\n" + it["ideal"]), 1.0, "correct")
                assert selfcheck(it["passage"], it["ideal"])[0], f"{it['id']} gold fails selfcheck"
                # labels are honest
                for s in it["support"]:
                    assert s in it["passage"], f"{it['id']}s{seed}: support sentence missing"
                if it["kind"] == "answer":
                    expect(f"{it['id']}s{seed} over-abstain",
                           check(it, "ANSWER: NOT_IN_CONTEXT\nQUOTE: NONE"), 0.1, "over_abstain")
                    expect(f"{it['id']}s{seed} invented quote",
                           check(it, f'ANSWER: {it["gold"]}\nQUOTE: "The ledger says {it["gold"]}."'),
                           0.7, "correct_ungrounded")
                else:
                    expect(f"{it['id']}s{seed} answers anyway",
                           check(it, 'ANSWER: 1987\nQUOTE: "It was established in 1987."'), 0.1)
    print(f"  gold replies score 1.0 on all {n} generated items; shortcuts are caught")

    # hand-written replies, as a model writes them
    it = halworld.make(3, 1, 0)
    other = [c for c in _entities_in(it["passage"], it["passage"]) if c != _asked_entity(it)][0]
    wrong_sent = [s for s in re.split(r"(?<=\.) ", it["passage"])
                  if s.startswith(other + " was founded by")][0]
    wrong_val = wrong_sent.split(" was founded by ")[1].rstrip(".")
    cases = [
        ("markdown bold", halworld.make(1, 1, 0),
         "**ANSWER:** " + halworld.make(1, 1, 0)["gold"] + "\n**QUOTE:** \""
         + halworld.make(1, 1, 0)["support"][0] + "\"", 1.0, "correct"),
        ("quote about the look-alike company", it,
         f'ANSWER: {wrong_val}\nQUOTE: "{wrong_sent}"', 0.4, "wrong_entity"),
        ("no answer line", it, "I think it was founded long ago.", 0.0, "no_format"),
        ("silent vs contradicted", halworld.make(2, 2, 0),
         "ANSWER: FALSE_PREMISE\nQUOTE: NONE", 0.3, "wrong_abstain_type"),
        ("false premise accepted", halworld.make(4, 1, 0),
         f'ANSWER: 1950\nQUOTE: "{halworld.make(4,1,0)["support"][0]}"', 0.1, "accepted_false_premise"),
        ("false premise called silent", halworld.make(4, 4, 0),
         "ANSWER: NOT_IN_CONTEXT\nQUOTE: NONE", 0.3, "wrong_abstain_type"),
    ]
    lv5 = halworld.make(5, 1, 0)
    true_val = [r[2] for r in halworld.REAL if r[0] == lv5["question"]][0]
    tmpl = [r[1] for r in halworld.REAL if r[0] == lv5["question"]][0]
    cases.append(("answers from memory", lv5,
                  f'ANSWER: {true_val}\nQUOTE: "{tmpl.format(v=true_val)}"', 0.1, "parametric_override"))
    lv7 = halworld.make(7, 1, 0)
    q7 = " ".join(f'"{s}"' for s in lv7["support"])
    cases.append(("sum, off by one", lv7, f"ANSWER: {int(lv7['gold']) + 1}\nQUOTE: {q7}",
                  0.4, "wrong_answer"))
    cases.append(("sum, right, figures quoted", lv7, f"ANSWER: {lv7['gold']}\nQUOTE: {q7}",
                  1.0, "correct"))
    for name, item, reply, rw, lb in cases:
        g = check(item, reply)
        expect(name, g, rw, lb)
        print(f"  {'ok  ' if g['reward'] == rw and g['label'] == lb else 'FAIL'} {name:<36} "
              f"{g['reward']:.1f}  {g['label']:<24} {g['feedback'][:70]}")

    # tokenised corpora (HotpotQA): spacing around punctuation must not matter; words still must
    T = ("John Faso : John James Faso Jr. ( born August 25 , 1952 ) is the Representative for "
         "New York 's 19th congressional district . He do n't run .")
    for q, want in [("John James Faso Jr. (born August 25, 1952) is the Representative for New York's "
                     "19th congressional district.", True),
                    ("He don't run.", True),
                    ("is the Representative for New York's 19 congressional district", False),
                    ("is the Senator for New York's 19th congressional district", False)]:
        got = quote_is_verbatim(q, T)
        fails += got != want
        print(f"  {'ok  ' if got == want else 'FAIL'} tokenised text {q[:44]!r:48} -> {got}")

    # quotes that skip words with "..."
    P = "The Normans (Norman: Nourmands) were the people who gave their name to Normandy, a region in France."
    for q, want in [("The Normans ... gave their name to Normandy, a region in France.", True),
                    ("The Normans \u2026 gave their name to Normandy", True),
                    ("gave their name to Normandy ... The Normans", False),
                    ("The Normans ... gave their name to Brittany", False)]:
        got = quote_is_verbatim(q, P)
        fails += got != want
        print(f"  {'ok  ' if got == want else 'FAIL'} ellipsis quote {q[:44]!r:48} -> {got}")

    print("\nSELFTEST " + ("PASSED" if fails == 0 else f"FAILED ({fails})"))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else 0)
