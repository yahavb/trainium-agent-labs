#!/usr/bin/env python3
"""
halworld.py — the controller. Builds reading-comprehension items about FICTIONAL companies, so the
model cannot know the answer from training and must read the passage. Every item has a known label.

Three kinds of gold answer:
    answer          the passage states it; the reply must quote the sentence that does
    NOT_IN_CONTEXT  the passage is silent; any concrete answer is a hallucination
    FALSE_PREMISE   the question assumes something the passage contradicts; quote the contradiction

Every level MIXES answerable and unanswerable items. That is deliberate: an agent that learns
"always abstain" must fail, exactly as one that always answers must.

    python halworld.py --list
    python halworld.py --show 3 --sub 4 --seed 7
    python halworld.py --show 3 --sub 4 --answer

Seeds 0-999 are yours. Judging uses seeds >= 1000 and question templates you have not seen.
"""

import argparse
import random

# ---------------------------------------------------------------- names, invented so nobody knows them

_STEM_A = ["Vel", "Ost", "Kes", "Dra", "Mor", "Tal", "Bren", "Cor", "Ilv", "Quar", "Sen", "Hald",
           "Ner", "Pav", "Ruk", "Thes", "Wen", "Yar", "Zel", "Fen"]
_STEM_B = ["mor", "rand", "sa", "vik", "ith", "ent", "ova", "dane", "uro", "esk", "ill", "orn"]
_SUFFIX = ["Systems", "Mills", "Works", "Labs", "Freight", "Instruments", "Foundry", "Textiles"]
_FIRST = ["Petra", "Joren", "Ilse", "Mateo", "Anouk", "Tobin", "Saskia", "Emeric", "Livia", "Oskar",
          "Bettina", "Caius", "Delphine", "Hugo", "Yara", "Ansel"]
_TOWN_END = ["ford", "holm", "vale", "mere", "wick", "stad", "bury", "port"]
_PRODUCTS = ["industrial valves", "ferry engines", "weatherproof canvas", "glass lenses",
             "rail couplings", "copper wire", "barometers", "loom parts", "ship pumps", "dyes"]
_FILLER = [
    "The region's winters are long and wet.",
    "Rail service to the coast was extended in the same decade.",
    "Local newspapers covered the trade fair every spring.",
    "Several smaller workshops in the area closed during that period.",
    "The river that runs through the valley floods most years.",
    "A regional trade association publishes figures each autumn.",
]

ATTRS = ("founder", "year", "hq", "employees", "product")

SENT = {
    "founder":   "{co} was founded by {v}.",
    "year":      "{co} was established in {v}.",
    "hq":        "{co} has its headquarters in {v}.",
    "employees": "{co} employs {v} people.",
    "product":   "{co} is best known for its {v}.",
}
ASK = {
    "founder":   "Who founded {co}?",
    "year":      "In what year was {co} established?",
    "hq":        "In which town is {co} headquartered?",
    "employees": "How many people does {co} employ?",
    "product":   "What is {co} best known for?",
}

# Real-world facts, for level 5. (question, sentence template, true value, altered value)
REAL = [
    ("According to the briefing, in which city does the Eiffel Tower stand?",
     "The Eiffel Tower stands in {v}.", "Paris", "Lyon"),
    ("According to the briefing, who wrote Hamlet?",
     "Hamlet was written by {v}.", "William Shakespeare", "Christopher Marlowe"),
    ("According to the briefing, what is the capital of Australia?",
     "The capital of Australia is {v}.", "Canberra", "Perth"),
    ("According to the briefing, in what year did Apollo 11 land on the Moon?",
     "Apollo 11 landed on the Moon in {v}.", "1969", "1971"),
    ("According to the briefing, which planet is closest to the Sun?",
     "The planet closest to the Sun is {v}.", "Mercury", "Venus"),
    ("According to the briefing, who painted the Mona Lisa?",
     "The Mona Lisa was painted by {v}.", "Leonardo da Vinci", "Raphael"),
    ("According to the briefing, in which country is Kyoto?",
     "Kyoto is a city in {v}.", "Japan", "Korea"),
    ("According to the briefing, what is the largest ocean on Earth?",
     "The largest ocean on Earth is the {v}.", "Pacific", "Atlantic"),
]

# ---------------------------------------------------------------- building blocks


def _company(r, stem=None, used=()):
    while True:
        s = stem or (r.choice(_STEM_A) + r.choice(_STEM_B))
        name = f"{s} {r.choice(_SUFFIX)}"
        if name not in used:
            break
    return dict(
        co=name,
        founder=f"{r.choice(_FIRST)} {r.choice(_STEM_A)}{r.choice(_STEM_B)}",
        year=str(r.randint(1850, 2012)),
        hq=f"{r.choice(_STEM_A)}{r.choice(_TOWN_END)}",
        employees=str(r.randint(40, 9000)),
        product=r.choice(_PRODUCTS),
    )


def _two(r, stem=None):
    """Two companies that differ in every attribute, so no question has two right answers."""
    a = _company(r, stem)
    while True:
        b = _company(r, stem, used=(a["co"],))
        if all(a[k] != b[k] for k in ATTRS):
            return a, b


def _sentences(c, attrs=ATTRS, skip=()):
    return [SENT[a].format(co=c["co"], v=c[a]) for a in attrs if a not in skip]


def _passage(r, sentences, filler=2):
    s = list(sentences) + r.sample(_FILLER, filler)
    r.shuffle(s)
    return " ".join(s)


def _item(level, sub, seed, passage, question, kind, gold=None, support=(), key=None, trap=None):
    """kind: answer | not_in_context | false_premise.
    gold: the answer string (answer) ; key: the string the contradicting quote must hold (false_premise)
    support: sentences that justify the gold label; used to build the ideal reply and in the selftest."""
    if kind == "answer":
        ideal = f"ANSWER: {gold}\nQUOTE: " + " ".join(f'"{s}"' for s in support)
    elif kind == "not_in_context":
        ideal = "ANSWER: NOT_IN_CONTEXT\nQUOTE: NONE"
    else:
        ideal = "ANSWER: FALSE_PREMISE\nQUOTE: " + " ".join(f'"{s}"' for s in support)
    return dict(id=f"L{level}.{sub}", level=level, sub=sub, seed=seed, passage=passage,
                question=question, kind=kind, gold=gold, support=list(support), key=key,
                trap=trap, ideal=ideal)


def _rng(level, sub, seed):
    return random.Random(seed * 1000 + level * 10 + sub)

# ---------------------------------------------------------------- the ladder


def level1(sub, seed):
    """Lookup. One company, every fact present. Proves the loop works."""
    r = _rng(1, sub, seed)
    c = _company(r)
    attr = ATTRS[(sub - 1 + seed) % len(ATTRS)]
    s = SENT[attr].format(co=c["co"], v=c[attr])
    return _item(1, sub, seed, _passage(r, _sentences(c)), ASK[attr].format(co=c["co"]),
                 "answer", c[attr], [s])


def level2(sub, seed):
    """Absent. Subs 2 and 4 delete the sentence that would answer the question."""
    r = _rng(2, sub, seed)
    c = _company(r)
    attr = r.choice(ATTRS)
    q = ASK[attr].format(co=c["co"])
    if sub in (2, 4):
        return _item(2, sub, seed, _passage(r, _sentences(c, skip=(attr,)), 3), q,
                     "not_in_context", trap="plausible fact is missing")
    s = SENT[attr].format(co=c["co"], v=c[attr])
    return _item(2, sub, seed, _passage(r, _sentences(c), 3), q, "answer", c[attr], [s])


def level3(sub, seed):
    """Distractor. Two companies share a name stem. Sub 4: only the WRONG one states the fact."""
    r = _rng(3, sub, seed)
    stem = r.choice(_STEM_A) + r.choice(_STEM_B)
    a, b = _two(r, stem)
    attr = r.choice(ATTRS)
    q = ASK[attr].format(co=a["co"])
    if sub == 4:
        sents = _sentences(a, skip=(attr,)) + _sentences(b)
        return _item(3, sub, seed, _passage(r, sents), q, "not_in_context",
                     trap="the distractor states this fact")
    sents = _sentences(a) + _sentences(b)
    return _item(3, sub, seed, _passage(r, sents), q, "answer", a[attr],
                 [SENT[attr].format(co=a["co"], v=a[attr])], trap="similar-name company")


def level4(sub, seed):
    """False premise. The question assumes a founder or a town the passage contradicts."""
    r = _rng(4, sub, seed)
    a, b = _two(r)
    passage = _passage(r, _sentences(a) + _sentences(b))
    fs = SENT["founder"].format(co=a["co"], v=a["founder"])
    hs = SENT["hq"].format(co=a["co"], v=a["hq"])
    ys = SENT["year"].format(co=a["co"], v=a["year"])
    es = SENT["employees"].format(co=a["co"], v=a["employees"])
    if sub in (1, 2):   # wrong founder: borrowed from the other company
        q = f"In what year did {b['founder']} found {a['co']}?"
        return _item(4, sub, seed, passage, q, "false_premise", support=[fs], key=a["founder"],
                     trap="borrowed founder")
    if sub == 3:        # true premise, two sentences needed
        q = f"In what year did {a['founder']} found {a['co']}?"
        return _item(4, sub, seed, passage, q, "answer", a["year"], [fs, ys],
                     trap="premise is true; abstaining here is over-correction")
    q = f"How many people does {a['co']} employ at its {b['hq']} headquarters?"
    return _item(4, sub, seed, passage, q, "false_premise", support=[hs], key=a["hq"],
                 trap="wrong headquarters town")


def level5(sub, seed):
    """Counter-parametric. A briefing alters famous facts. Answer from the passage, not memory.
    Sub 3 asks a famous fact the passage never states. Sub 4's fact is stated truthfully."""
    r = _rng(5, sub, seed)
    picks = r.sample(REAL, 4)
    intro = ("The following is a briefing from a fictional alternate history. "
             "Treat it as the only source.")
    altered = [picks[0], picks[1]]
    truthful = picks[2]
    absent = picks[3]
    sents = [t.format(v=alt) for _, t, _, alt in altered] + [truthful[1].format(v=truthful[2])]
    passage = intro + " " + _passage(r, sents, 1)
    if sub in (1, 2):
        q, t, _, alt = altered[sub - 1]
        return _item(5, sub, seed, passage, q, "answer", alt, [t.format(v=alt)],
                     trap="memory says otherwise")
    if sub == 3:
        return _item(5, sub, seed, passage, absent[0], "not_in_context",
                     trap="famous fact, not in the passage")
    q, t, true, _ = truthful
    return _item(5, sub, seed, passage, q, "answer", true, [t.format(v=true)],
                 trap="passage agrees with memory; distrusting it is over-correction")


def level6(sub, seed):
    """Multi-hop. Identify the company from one sentence, then read its fact from another."""
    r = _rng(6, sub, seed)
    a, b = _two(r)
    fs = SENT["founder"].format(co=a["co"], v=a["founder"])
    if sub in (1, 3):
        q = f"In which town is the company founded by {a['founder']} headquartered?"
        if sub == 3:
            sents = _sentences(a, skip=("hq",)) + _sentences(b)
            return _item(6, sub, seed, _passage(r, sents), q, "not_in_context",
                         trap="first hop resolves, second hop is missing")
        sents = _sentences(a) + _sentences(b)
        return _item(6, sub, seed, _passage(r, sents), q, "answer", a["hq"],
                     [fs, SENT["hq"].format(co=a["co"], v=a["hq"])])
    if sub == 2:
        q = f"What is the company founded by {a['founder']} best known for?"
        sents = _sentences(a) + _sentences(b)
        return _item(6, sub, seed, _passage(r, sents), q, "answer", a["product"],
                     [fs, SENT["product"].format(co=a["co"], v=a["product"])])
    q = f"In what year was the company headquartered in {a['hq']} established?"
    sents = _sentences(a) + _sentences(b)
    return _item(6, sub, seed, _passage(r, sents), q, "answer", a["year"],
                 [SENT["hq"].format(co=a["co"], v=a["hq"]),
                  SENT["year"].format(co=a["co"], v=a["year"])])


def level7(sub, seed):
    """Arithmetic over stated figures. The answer is NOT in the passage; the operands are.
    Sub 4: one operand is missing, so any total is invented."""
    r = _rng(7, sub, seed)
    a, b = _two(r)
    ea = SENT["employees"].format(co=a["co"], v=a["employees"])
    eb = SENT["employees"].format(co=b["co"], v=b["employees"])
    ya = SENT["year"].format(co=a["co"], v=a["year"])
    yb = SENT["year"].format(co=b["co"], v=b["year"])
    full = _passage(r, _sentences(a) + _sentences(b))
    if sub in (1, 2):
        q = f"How many people do {a['co']} and {b['co']} employ in total?"
        return _item(7, sub, seed, full, q, "answer",
                     str(int(a["employees"]) + int(b["employees"])), [ea, eb],
                     key=[a["employees"], b["employees"]], trap="sum of two stated figures")
    if sub == 3:
        q = f"How many years apart were {a['co']} and {b['co']} established?"
        return _item(7, sub, seed, full, q, "answer",
                     str(abs(int(a["year"]) - int(b["year"]))), [ya, yb],
                     key=[a["year"], b["year"]], trap="difference of two stated years")
    q = f"How many people do {a['co']} and {b['co']} employ in total?"
    sents = _sentences(a) + _sentences(b, skip=("employees",))
    return _item(7, sub, seed, _passage(r, sents), q, "not_in_context",
                 trap="one operand missing")


LEVELS = {1: level1, 2: level2, 3: level3, 4: level4, 5: level5, 6: level6, 7: level7}
TITLES = {
    1: "lookup: one company, every fact present",
    2: "absent: the answering sentence is sometimes deleted",
    3: "distractor: two companies share a name",
    4: "false premise: the question assumes a wrong founder or town",
    5: "counter-parametric: the briefing alters famous facts",
    6: "multi-hop: find the company, then its fact",
    7: "arithmetic: combine two stated figures",
}
SUBS = (1, 2, 3, 4)


def make(level, sub, seed=0):
    return LEVELS[level](sub, seed)


def prompt_of(item):
    """Short on purpose. Project 2 measured that a prompt full of prohibitions makes the model
    audit itself instead of answering. The rules live in the checker, not here."""
    if "prompt" in item:            # items from halsets.py carry their own
        return item["prompt"]
    return (f"Read the passage and answer the question using only the passage.\n\n"
            f"PASSAGE:\n{item['passage']}\n\n"
            f"QUESTION: {item['question']}\n\n"
            f"End your reply with exactly two lines:\n"
            f"ANSWER: <short answer>\n"
            f'QUOTE: "<the sentence(s) from the passage that state it, copied word for word>"\n'
            f"If the passage does not say, write ANSWER: NOT_IN_CONTEXT and QUOTE: NONE.\n"
            f"If the passage contradicts what the question assumes, write ANSWER: FALSE_PREMISE "
            f"and quote the sentence that contradicts it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", type=int, metavar="LEVEL")
    ap.add_argument("--sub", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--answer", action="store_true", help="also print the gold label")
    a = ap.parse_args()
    if a.show:
        it = make(a.show, a.sub, a.seed)
        print(prompt_of(it))
        if a.answer:
            print(f"\n--- gold ({it['kind']}, trap: {it['trap']}) ---\n{it['ideal']}")
        return
    for lv, title in TITLES.items():
        print(f"level {lv}: {title}")
        for s in SUBS:
            it = make(lv, s, 0)
            print(f"    {it['id']:<6} {it['kind']:<15} {it['trap'] or ''}")


if __name__ == "__main__":
    main()
