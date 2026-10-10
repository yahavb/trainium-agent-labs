#!/usr/bin/env python3
"""
nki_fix_examples.py — load, look up and check the examples in nki_fix_examples.md.

    from nki_fix_examples import example_for
    note = example_for(error_text, level)   # "" when no example matches

example_for() returns the same text agent.fragment_note() does, so it can be passed to the new
agent's debugger in its place.

    PYTHONDONTWRITEBYTECODE=1 python nki_fix_examples.py                      # check every example
    PYTHONDONTWRITEBYTECODE=1 python nki_fix_examples.py --coverage LOG...    # which failures each catches

The check needs nki, so run it in a seat pod. For each example it confirms three things:
- the `when` regex matches the example's own recorded `error`;
- that error, at the example's level, is routed to this example and not an earlier one;
- the kernel matches NumPy in nki.simulate.

The exit status is the number of failures.
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

MD = Path(__file__).with_name("nki_fix_examples.md")
IMPORTS = "import nki\nimport nki.isa as nisa\nimport nki.language as nl\n\n"
SECTION = re.compile(r"^## (\S+)\n(.*?)(?=^## |\Z)", re.S | re.M)
FIELD = re.compile(r"^- (\w+): (`+) ?(.*?) ?\2$", re.M)


def load_examples(path=MD):
    """The examples in file order, which is match order. Sections without `when` and code are prose."""
    examples = []
    for name, body in SECTION.findall(path.read_text()):
        fields = {k: v for k, _, v in FIELD.findall(body)}
        code = re.search(r"```python\n(.*?)```", body, re.S)
        if "when" not in fields or not code:
            continue
        prose = body[:code.start()].splitlines()
        note = " ".join(l.strip() for l in prose if l.strip() and not l.startswith("- "))
        levels = None if fields["levels"] == "all" else {int(n) for n in fields["levels"].split(",")}
        examples.append(dict(name=name, levels=levels, when=fields["when"], error=fields["error"],
                             input=fields["input"], expect=fields["expect"], note=note,
                             code=code.group(1).rstrip(),
                             entry=re.search(r"^def (\w+)", code.group(1), re.M).group(1)))
    return examples


def match(error_text, level, examples=None):
    """The first example allowed at this level whose regex matches, or None."""
    for e in examples if examples is not None else load_examples():
        if (e["levels"] is None or level in e["levels"]) and re.search(e["when"], error_text or ""):
            return e
    return None


def example_for(error_text, level, examples=None):
    """The note and checked example for this error at this level, formatted like agent.fragment_note."""
    e = match(error_text, level, examples)
    return f"\n{e['note']}\n```python\n{e['code']}\n```" if e else ""


def check(examples):
    import nki
    failed = 0
    for e in examples:
        level = min(e["levels"]) if e["levels"] else 1
        routed = match(e["error"], level, examples)
        problems = []
        if not re.search(e["when"], e["error"]):
            problems.append("its regex does not match its own error")
        elif routed is not e:
            problems.append(f"its error is routed to `{routed['name'] if routed else None}` first")
        ns = {}
        try:
            exec(compile(IMPORTS + e["code"], f"<example {e['name']}>", "exec"), ns)
            a = np.random.default_rng(0).standard_normal(eval(e["input"])).astype(np.float32)
            got = np.asarray(nki.simulate(ns[e["entry"]])(a.copy()))
            want = eval(e["expect"], {"a": a, "np": np})
            if got.shape != want.shape or not np.allclose(got, want, rtol=1e-4, atol=1e-4):
                problems.append(f"wrong result: shape {got.shape} vs {want.shape}")
        except Exception as ex:
            problems.append(f"raised {type(ex).__name__}: {str(ex).splitlines()[0][:120]}")
        failed += bool(problems)
        print(f"  {e['name']:<10} {'ok' if not problems else 'FAIL ' + '; '.join(problems)}")
    print(f"{len(examples)} examples, {failed} failed")
    return failed


def coverage(examples, logs):
    """For every failed attempt in the logs: which example its error and level would get."""
    hit, missed, total = collections.Counter(), collections.Counter(), 0
    for p in logs:
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("reward", 0) >= 0.999 or not r.get("feedback"):
                continue
            total += 1
            e = match(r["feedback"], r["level"], examples)
            if e:
                hit[(e["name"], r["level"])] += 1
            else:
                err = re.search(r"raised (\w+: [^.]*)", r["feedback"])
                missed[re.sub(r"\d+", "N", (err.group(1) if err else r["feedback"])[:90])] += 1
    print(f"{total} failed attempts in {len(logs)} logs; {sum(hit.values())} get an example\n")
    for (name, level), n in sorted(hit.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5}  {name} (level {level})")
    print("\n  most common errors with no example:")
    for k, n in missed.most_common(8):
        print(f"  {n:5}  {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", nargs="+", metavar="LOG", help="attempts.jsonl files to replay")
    a = ap.parse_args()
    examples = load_examples()
    if a.coverage:
        coverage(examples, a.coverage)
        return 0
    return check(examples)


if __name__ == "__main__":
    sys.exit(main())
