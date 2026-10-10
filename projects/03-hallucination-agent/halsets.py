#!/usr/bin/env python3
"""
halsets.py — real datasets, turned into the same items halworld.py makes, so the checker and the
loop run on them unchanged.

    name          source (Hugging Face unless noted)              setting       gold kinds
    squad2        rajpurkar/squad_v2, validation                   passage       answer / not_in_context
    hotpot        hotpotqa/hotpot_qa (distractor), validation      10 paragraphs answer (2 support sentences)
    musique       MuSiQue full, local JSONL (see below)            20 paragraphs answer / not_in_context
    faith-unans   Salesforce/FaithEval-unanswerable-v1.0           passage       not_in_context (all)
    faith-incon   Salesforce/FaithEval-inconsistent-v1.0           passage       conflict (all)
    faith-cf      Salesforce/FaithEval-counterfactual-v1.0         passage, MC   answer (against world knowledge)
    popqa         akariasai/PopQA, test                            CLOSED BOOK   answer, bucketed by popularity

Combine with "+": `faith-unans+squad2` interleaves the two. Datasets where every item has the same
gold kind (faith-unans, faith-incon) MUST be mixed with an answerable set, or "always abstain"
scores perfectly. The prompt for a mix offers every abstention token any member needs, so the
prompt never reveals which kind an item is.

    python halsets.py --fetch squad2 --n 300          # download into data/, spread across the split
    python halsets.py --inspect squad2                # one raw row next to the item made from it
    python halsets.py --list                          # what is cached
    python halsets.py --selftest                      # loaders vs. fixture rows; no network
    python agent.py --data faith-unans+squad2 --n 100 --rounds 1

Downloads use the Hugging Face dataset viewer API (no `datasets` package needed), falling back to
`datasets.load_dataset` if it is installed. Set HF_TOKEN if a dataset asks for a login.
MuSiQue is not reliably on the Hub: download musique_full_v1.0_dev.jsonl from
https://github.com/StonyBrookNLP/musique and run  python halsets.py --import musique FILE

The FaithEval and PopQA field names below were written from their documentation, not checked
against a live download. Run --inspect on each before trusting a number from it.
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# ---------------------------------------------------------------- helpers


def _f(row, *names, default=None):
    """First field present in row. Tolerates schema differences between dataset versions."""
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return default


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("["):
            try:
                return [str(v) for v in json.loads(x)]
            except json.JSONDecodeError:
                pass
        return [x]
    if isinstance(x, dict):            # SQuAD-style {"text": [...], "answer_start": [...]}
        return [str(v) for v in x.get("text", [])]
    return [str(v) for v in x]


def _dedupe(xs):
    return list(dict.fromkeys(x.strip() for x in xs if x and x.strip()))


def _sentences(text):
    """(start, end, sentence) spans, each a verbatim substring of text."""
    out = []
    for m in re.finditer(r"[^.!?\n]+(?:[.!?]+[\"')\]]*|\n|$)", text):
        s = m.group(0).strip()
        if s:
            st = text.index(s, m.start())
            out.append((st, st + len(s), s))
    return out


def _sentence_with(text, needle=None, at=None):
    """The sentence containing character offset `at`, or else the first one containing `needle`."""
    sents = _sentences(text)
    if at is not None:
        for st, en, s in sents:
            if st <= at < en:
                return s
    if needle:
        for _, _, s in sents:
            if needle.lower() in s.lower():
                return s
    return None


def _best_overlap(text, target):
    """The sentence sharing the most words with target (for MC items with no answer span)."""
    tw = set(re.findall(r"\w+", (target or "").lower()))
    sents = _sentences(text)
    if not sents:
        return None
    return max(sents, key=lambda s: len(tw & set(re.findall(r"\w+", s[2].lower()))))[2]


def _item(ds, idx, passage, question, kind, gold=None, support=(), **extra):
    it = dict(id=f"{ds}:{idx}", dataset=ds, level=f"{ds}:{kind}", sub=0, seed=idx,
              passage=passage, question=question, kind=kind, gold=gold, support=list(support),
              key=None, trap=None)
    it.update(extra)
    return it

# ---------------------------------------------------------------- converters: raw row -> item


def conv_squad2(row, idx):
    ctx, q = row["context"], row["question"]
    golds = _dedupe(_as_list(row.get("answers")))
    if not golds:
        return _item("squad2", idx, ctx, q, "not_in_context")
    starts = (row.get("answers") or {}).get("answer_start") or [None]
    sup = _sentence_with(ctx, golds[0], starts[0])
    return _item("squad2", idx, ctx, q, "answer", golds, [sup] if sup else [])


def conv_hotpot(row, idx):
    c = row["context"]
    titles, sents = c["title"], c["sentences"]
    paras = {t: s for t, s in zip(titles, sents)}
    passage = "\n\n".join(f"{t}: {''.join(s).strip()}" for t, s in zip(titles, sents))
    sf = row.get("supporting_facts") or {}
    support = []
    for t, i in zip(sf.get("title", []), sf.get("sent_id", [])):
        if t in paras and i < len(paras[t]) and paras[t][i].strip():
            support.append(paras[t][i].strip())
    ans = row["answer"]
    yesno = ans.strip().lower() in ("yes", "no")
    return _item("hotpot", idx, passage, row["question"], "answer", [ans], support,
                 answer_in_quote=not yesno)


def conv_musique(row, idx):
    paras = row["paragraphs"]
    passage = "\n\n".join(f"{p['title']}: {p['paragraph_text']}" for p in paras)
    q = row["question"]
    if row.get("answerable") is False:
        return _item("musique", idx, passage, q, "not_in_context")
    golds = _dedupe([row.get("answer", "")] + list(row.get("answer_aliases") or []))
    support = []
    for p in paras:
        if p.get("is_supporting"):
            s = _sentence_with(p["paragraph_text"], golds[0]) if golds else None
            support.append(s or p["paragraph_text"].strip())
    return _item("musique", idx, passage, q, "answer", golds, support)


def _faith_ctx(row):
    return _f(row, "context", "passage", "ctx", default=""), _f(row, "question", "query", default="")


def conv_faith_unans(row, idx):
    ctx, q = _faith_ctx(row)
    return _item("faith-unans", idx, ctx, q, "not_in_context")


def conv_faith_incon(row, idx):
    ctx, q = _faith_ctx(row)
    return _item("faith-incon", idx, ctx, q, "conflict")


def conv_faith_cf(row, idx):
    ctx, q = _faith_ctx(row)
    ch = row.get("choices") or {}
    if isinstance(ch, dict):
        choices = list(zip(ch.get("label", []), ch.get("text", [])))
    else:   # list of {"label","text"} or of strings
        choices = [(c.get("label"), c.get("text")) if isinstance(c, dict) else (chr(65 + i), c)
                   for i, c in enumerate(ch)]
    choices = [(str(l).strip().upper(), str(t)) for l, t in choices]
    key = str(_f(row, "answerKey", "answer_key", "answer", default="")).strip().upper()
    gold_text = dict(choices).get(key)
    if not choices or gold_text is None:
        return None
    opts = "\n".join(f"  {l}) {t}" for l, t in choices)
    sup = _best_overlap(ctx, gold_text)
    return _item("faith-cf", idx, ctx, f"{q}\nOptions:\n{opts}", "answer", gold_text,
                 [sup] if sup else [], choices=choices, gold_letter=key, answer_in_quote=False)


def _pop_bucket(p):
    p = int(p or 0)
    return "<100" if p < 100 else "100-1k" if p < 1000 else "1k-10k" if p < 10000 else ">10k"


def conv_popqa(row, idx):
    golds = _dedupe(_as_list(_f(row, "possible_answers", "answers", "obj")))
    if not golds:
        return None
    pop = _f(row, "s_pop", default=0)
    it = _item("popqa", idx, "", row["question"], "answer", golds, closed_book=True, pop=pop)
    it["level"] = f"popqa:pop{_pop_bucket(pop)}"
    return it


DATASETS = {
    "squad2":      dict(hf="rajpurkar/squad_v2", config="squad_v2", split="validation", conv=conv_squad2),
    "hotpot":      dict(hf="hotpotqa/hotpot_qa", config="distractor", split="validation", conv=conv_hotpot),
    "musique":     dict(hf=None, conv=conv_musique),
    "faith-unans": dict(hf="Salesforce/FaithEval-unanswerable-v1.0", config=None, split=None,
                        conv=conv_faith_unans),
    "faith-incon": dict(hf="Salesforce/FaithEval-inconsistent-v1.0", config=None, split=None,
                        conv=conv_faith_incon),
    "faith-cf":    dict(hf="Salesforce/FaithEval-counterfactual-v1.0", config=None, split=None,
                        conv=conv_faith_cf),
    "popqa":       dict(hf="akariasai/PopQA", config=None, split="test", conv=conv_popqa),
}

# ---------------------------------------------------------------- prompts


def _tokens(kinds):
    return {"conflict": "CONFLICT"} if "conflict" in kinds else {}


def prompt_for(item, conflict_line):
    if item.get("closed_book"):
        return (f"Answer the question from your own knowledge.\n\n"
                f"QUESTION: {item['question']}\n\n"
                f"End your reply with exactly one line:\n"
                f"ANSWER: <short answer>\n"
                f"If you are not sure of the answer, write ANSWER: NOT_SURE instead of guessing.")
    p = (f"Read the passage and answer the question using only the passage.\n\n"
         f"PASSAGE:\n{item['passage']}\n\n"
         f"QUESTION: {item['question']}\n\n"
         f"End your reply with exactly two lines:\n"
         f"ANSWER: <short answer>\n"
         f'QUOTE: "<the sentence(s) from the passage that support it, copied word for word>"\n'
         f"If the passage does not say, write ANSWER: NOT_IN_CONTEXT and QUOTE: NONE.")
    if conflict_line:
        p += ("\nIf the passage gives conflicting answers, write ANSWER: CONFLICT and QUOTE: NONE.")
    return p


def ideal_for(item):
    if item.get("closed_book"):
        return f"ANSWER: {item['gold'][0]}"
    k = item["kind"]
    if k == "not_in_context":
        return "ANSWER: NOT_IN_CONTEXT\nQUOTE: NONE"
    if k == "conflict":
        return "ANSWER: CONFLICT\nQUOTE: NONE"
    g = item["gold"][0] if isinstance(item["gold"], list) else item["gold"]
    if item.get("choices"):
        g = f"{item['gold_letter']}) {item['gold']}"
    q = " ".join(f'"{s}"' for s in item["support"]) or "NONE"
    return f"ANSWER: {g}\nQUOTE: {q}"

# ---------------------------------------------------------------- download and cache


def _cache(ds):
    return os.path.join(DATA, f"{ds}.jsonl")


def _http_json(url):
    import httpx
    h = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    r = httpx.get(url, headers=h, timeout=60, follow_redirects=True)
    r.raise_for_status()
    return r.json()


def _fetch_viewer(spec, n, chunks):
    from urllib.parse import quote
    api = "https://datasets-server.huggingface.co"
    ds = quote(spec["hf"], safe="")
    config, split = spec.get("config"), spec.get("split")
    if not config or not split:
        splits = _http_json(f"{api}/splits?dataset={ds}")["splits"]
        pick = next((s for s in splits if (not config or s["config"] == config)
                     and (not split or s["split"] == split)), splits[0])
        config, split = pick["config"], pick["split"]
        print(f"  using config={config} split={split}")
    first = _http_json(f"{api}/rows?dataset={ds}&config={config}&split={split}&offset=0&length=1")
    total = first.get("num_rows_total") or n
    per = max(1, min(100, -(-n // chunks)))
    starts = sorted({int(i * max(0, total - per) / max(1, chunks - 1)) for i in range(chunks)})
    rows = []
    for off in starts:
        got = _http_json(f"{api}/rows?dataset={ds}&config={config}&split={split}"
                         f"&offset={off}&length={per}")["rows"]
        rows += [dict(r["row"], _idx=r["row_idx"]) for r in got]
        if len(rows) >= n:
            break
    return rows[:n], total


def _fetch_datasets_lib(spec, n, chunks):
    from datasets import load_dataset
    d = load_dataset(spec["hf"], spec.get("config"), split=spec.get("split") or "test")
    total = len(d)
    step = max(1, total // n)
    idx = list(range(0, total, step))[:n]
    return [dict(d[i], _idx=i) for i in idx], total


def fetch(ds, n=300, chunks=6):
    spec = DATASETS[ds]
    if not spec.get("hf"):
        sys.exit(f"{ds} is not downloaded from the Hub; use --import {ds} FILE")
    print(f"fetching {n} rows of {spec['hf']} in {chunks} spread-out chunks ...")
    try:
        rows, total = _fetch_viewer(spec, n, chunks)
    except Exception as e:
        print(f"  dataset viewer API failed ({e}); trying the datasets library")
        rows, total = _fetch_datasets_lib(spec, n, chunks)
    os.makedirs(DATA, exist_ok=True)
    with open(_cache(ds), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(rows)} of {total} rows to {_cache(ds)}")


def import_file(ds, path, n=None):
    os.makedirs(DATA, exist_ok=True)
    k = 0
    with open(path) as src, open(_cache(ds), "w") as out:
        for i, line in enumerate(src):
            if n and k >= n:
                break
            r = json.loads(line)
            r["_idx"] = i
            out.write(json.dumps(r) + "\n")
            k += 1
    print(f"imported {k} rows into {_cache(ds)}")


def _raw(ds):
    path = _cache(ds)
    if not os.path.exists(path):
        sys.exit(f"no cached data for {ds}. Run: python halsets.py --fetch {ds}")
    return [json.loads(l) for l in open(path)]


def _convert(ds, rows):
    out = []
    for r in rows:
        it = DATASETS[ds]["conv"](r, r.get("_idx", len(out)))
        if it is not None:
            out.append(it)
    return out


def load(spec, n=None, offset=0, max_chars=None):
    """spec: 'squad2' or 'faith-unans+squad2'. Returns finished items, interleaved, with prompts."""
    names = spec.split("+")
    for nm in names:
        if nm not in DATASETS:
            sys.exit(f"unknown dataset {nm}; choose from {', '.join(DATASETS)}")
    per = [_convert(nm, _raw(nm))[offset:] for nm in names]
    items, i = [], 0
    while any(i < len(p) for p in per):
        for p in per:
            if i < len(p):
                items.append(p[i])
        i += 1
    conflict = any(it["kind"] == "conflict" for it in items)
    for it in items:
        it["prompt"] = prompt_for(it, conflict)
        it["ideal"] = ideal_for(it)
    skipped = 0
    if max_chars:
        keep = [it for it in items if len(it["prompt"]) <= max_chars]
        skipped = len(items) - len(keep)
        items = keep
    if skipped:
        print(f"  skipped {skipped} items whose prompt exceeds {max_chars} chars "
              f"(raise --max-chars and serve with MAX_MODEL_LEN=8192 to include them)")
    return items[:n] if n else items


_by_id = {}


def get(item_id, spec=None):
    """One item by id ('squad2:1234'), for rebuilding example-bank prompts."""
    if not _by_id:
        for nm in DATASETS:
            if os.path.exists(_cache(nm)):
                for it in load(spec or nm):
                    _by_id[it["id"]] = it
    return _by_id[item_id]

# ---------------------------------------------------------------- selftest on fixture rows


FIXTURES = {
    "squad2": [
        dict(id="a", title="T", question="Where were the Normans based?", _idx=0,
             context="The Normans were a people. They gave their name to Normandy, a region in "
                     "France. Their leader was Rollo.",
             answers=dict(text=["Normandy", "Normandy, a region in France"], answer_start=[47, 47])),
        dict(id="b", title="T", question="Who led the Vikings in 1066?", _idx=1,
             context="The Normans were a people. Their leader was Rollo.",
             answers=dict(text=[], answer_start=[])),
    ],
    "hotpot": [dict(id="h", question="Which magazine was started first?", answer="Arthur's Magazine",
                    _idx=0, type="comparison", level="medium",
                    supporting_facts=dict(title=["Arthur's Magazine", "First for Women"], sent_id=[0, 0]),
                    context=dict(title=["Arthur's Magazine", "First for Women", "Radio City"],
                                 sentences=[["Arthur's Magazine was an American literary periodical "
                                             "published in Philadelphia, first issued in 1844.",
                                             " It was edited by T.S. Arthur."],
                                            ["First for Women is a woman's magazine started in 1989."],
                                            ["Radio City is an Indian FM station."]]))],
    "musique": [
        dict(id="m1", question="Where was the founder of Acme born?", answer="Lyon",
             answer_aliases=["Lyon, France"], answerable=True, _idx=0,
             paragraphs=[dict(idx=0, title="Acme", paragraph_text="Acme was founded by Jean Roux.",
                              is_supporting=True),
                         dict(idx=1, title="Jean Roux", paragraph_text="Jean Roux was born in Lyon. "
                              "He studied law.", is_supporting=True),
                         dict(idx=2, title="Other", paragraph_text="Paris is large.", is_supporting=False)]),
        dict(id="m2", question="Where was the founder of Acme born?", answer="Lyon", answerable=False,
             _idx=1, paragraphs=[dict(idx=0, title="Acme", paragraph_text="Acme was founded by Jean Roux.",
                                      is_supporting=True)]),
    ],
    "faith-unans": [dict(qid="u", question="What year did the bridge open?", _idx=0,
                         context="The bridge spans the river. It is made of steel.", answers=["unknown"])],
    "faith-incon": [dict(qid="c", question="What year did the bridge open?", _idx=0,
                         context="The bridge opened in 1932. The bridge first opened to traffic in 1951.",
                         answers=["conflict"])],
    "faith-cf": [dict(id="f", question="What is the Moon made of?", answerKey="B", _idx=0,
                      context="Recent studies confirm the Moon is made of marshmallows. It is soft.",
                      choices=dict(label=["A", "B", "C"], text=["rock", "marshmallows", "cheese"]))],
    "popqa": [dict(id=1, question="What is George Rankin's occupation?", s_pop=142, _idx=0,
                   possible_answers='["politician", "political leader"]')],
}


def selftest():
    import halcheck
    fails = 0
    for ds, rows in FIXTURES.items():
        items = _convert(ds, rows)
        conflict = any(i["kind"] == "conflict" for i in items)
        for it in items:
            it["prompt"], it["ideal"] = prompt_for(it, conflict), ideal_for(it)
            g = halcheck.check(it, "Reasoning here.\n" + it["ideal"])
            ok = g["reward"] == 1.0 and g["label"] == "correct"
            for s in it["support"]:
                ok &= s in it["passage"]
            # the shortcut that must fail
            if it.get("closed_book"):
                bad = halcheck.check(it, "ANSWER: farmer")
                ok &= bad["label"] == "wrong_answer" and bad["reward"] == 0.1
                ns = halcheck.check(it, "ANSWER: NOT_SURE")
                ok &= ns["label"] == "abstained" and ns["reward"] == 0.4
            elif it["kind"] == "answer":
                ok &= halcheck.check(it, "ANSWER: NOT_IN_CONTEXT\nQUOTE: NONE")["label"] == "over_abstain"
            elif it["kind"] == "conflict":
                first = _sentences(it["passage"])[0][2]
                ok &= halcheck.check(it, f'ANSWER: 1932\nQUOTE: "{first}"')["label"] == "picked_one_side"
            else:
                ok &= halcheck.check(it, 'ANSWER: 1950\nQUOTE: "It is made of steel."')["reward"] < 0.5
            fails += not ok
            print(f"  {'ok  ' if ok else 'FAIL'} {it['id']:<14} {it['kind']:<15} ideal={it['ideal']!r:.70}")
    cf = _convert("faith-cf", FIXTURES["faith-cf"])[0]
    for reply, want in [("ANSWER: B", True), ("ANSWER: (B)", True), ("ANSWER: B) marshmallows", True),
                        ("ANSWER: marshmallows", True), ("ANSWER: A", False), ("ANSWER: rock", False)]:
        got = halcheck.matches(cf, halcheck.parse(reply)["answer"])
        fails += got != want
        print(f"  {'ok  ' if got == want else 'FAIL'} multiple choice: {reply!r:28} -> {got}")
    mix = load_fixture_mix()
    fails += mix
    print("\nSELFTEST " + ("PASSED" if fails == 0 else f"FAILED ({fails})"))
    return 0 if fails == 0 else 1


def load_fixture_mix():
    """A mix that contains a conflict set must offer CONFLICT to every item, or the prompt leaks."""
    items = _convert("faith-incon", FIXTURES["faith-incon"]) + _convert("squad2", FIXTURES["squad2"])
    conflict = any(i["kind"] == "conflict" for i in items)
    ps = [prompt_for(i, conflict) for i in items]
    ok = all("CONFLICT" in p for p in ps)
    print(f"  {'ok  ' if ok else 'FAIL'} mixed prompt offers CONFLICT to every item")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", choices=[k for k, v in DATASETS.items() if v.get("hf")])
    ap.add_argument("--import", dest="imp", nargs=2, metavar=("NAME", "FILE"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--chunks", type=int, default=6, help="spread the download over this many offsets")
    ap.add_argument("--inspect", metavar="NAME")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.fetch:
        fetch(a.fetch, a.n, a.chunks)
    if a.imp:
        import_file(a.imp[0], a.imp[1], a.n)
    if a.inspect:
        raw = _raw(a.inspect)[0]
        it = load(a.inspect)[0]
        print("RAW ROW:\n" + json.dumps(raw, indent=1)[:3000])
        print("\nITEM:\n" + json.dumps({k: v for k, v in it.items() if k != "prompt"}, indent=1)[:3000])
        print("\nPROMPT:\n" + it["prompt"][:3000])
    if a.list:
        for nm in DATASETS:
            p = _cache(nm)
            if os.path.exists(p):
                items = load(nm)
                kinds = {}
                for i in items:
                    kinds[i["kind"]] = kinds.get(i["kind"], 0) + 1
                print(f"  {nm:<12} {len(items):>5} items  {kinds}")
            else:
                print(f"  {nm:<12} not fetched")


if __name__ == "__main__":
    main()
