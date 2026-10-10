"""Check task 08's code in the nki 0.6.0 simulator. Run from their projects/02-kernel-agent directory,
with that directory and this file's directory on PYTHONPATH:

    PROMPT1=v2 CARD=category MESSAGES=v5 PYTHONDONTWRITEBYTECODE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
        python test_v7.py

1. Levels 12-14 are registered, with categories 12 reduce, 13 reduce, 14 other.
2. Our answers for levels 8-14 score 1.0. A log-sum-exp that skips the max fails the scale-30 shape.
3. PROMPT1=theirs would leave the first prompt byte-identical to task 07's (CARD=category).
4. For levels 1-4 and 8-14, the v2 first prompt has the exact def line, the function block once, the
   category's worked example (and not its old list), and fits the context with room for the answer.
5. Every nl./nisa. name in the function block exists in nki 0.6.0.
"""
import os
import re
import sys

import feedback_v7 as v7

agent, nkibench, v5 = v7.agent, v7.nkibench, v7.v5
assert v7.PROMPT1 == "v2" and v5.CARD == "category", "run with PROMPT1=v2 CARD=category"
HERE = os.path.dirname(os.path.abspath(__file__))
fails = 0


def check(cond, what):
    global fails
    print(("ok   " if cond else "FAIL ") + what)
    fails += not cond


# 1
cats = {n: v5.category(n) for n in (12, 13, 14)}
check(cats == {12: "reduce", 13: "reduce", 14: "other"}, f"1. levels 12-14 registered, categories {cats}")

# 2
for lv in range(8, 15):
    if not os.path.exists(f"{HERE}/answers/ans_level{lv}.py"):    # level 8's answer is not published
        print(f"skip 2. no answer for level {lv}")
        continue
    r, _, fb = agent.grade(open(f"{HERE}/answers/ans_level{lv}.py").read(), lv)
    check(r >= 1 - 1e-9, f"2. our answer for level {lv} scores {r:.2f}" + ("" if r >= 1 - 1e-9 else f": {fb[:160]}"))
nomax = open(f"{HERE}/answers/ans_level13.py").read().replace("bias=negmax)", ")").replace(
    "data1=ls, data2=negmax, op=nl.subtract", "data1=ls, data2=ls, op=nl.maximum")
r, _, fb = agent.grade(nomax, 13)
check(r < 1 and "scale=30" in fb, f"2. log-sum-exp without the max fails the scale-30 shape ({r:.2f})")

# 3, 4
LEVELS = [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14]
EXAMPLE = {"matmul": "Mean over the free axis", "reduce": "Mean of each channel", "other": None}
for lv in LEVELS:
    theirs = v5.first_prompt_category(lv)
    mine = v7.first_prompt_v2(lv)
    agent.API_CARD = v5.BASE_CARD   # reset, so the next call can't inherit
    line = v7.def_line(lv)
    cat = v5.category(lv)
    ok = (f"    {line}\n" in mine and mine.count("More real functions. Each writes into dst") == 1
          and "More real functions, and one worked example" not in mine
          and (EXAMPLE[cat] is None or EXAMPLE[cat] in mine)
          and (EXAMPLE[cat] is not None or "Mean " not in mine.split("More real functions")[1])
          and len(mine) // 4 + agent.MIN_ANSWER_TOKENS + 64 <= 8192)
    check(ok, f"4. level {lv:2d} ({cat}): `{line}`, block once, example {EXAMPLE[cat] is not None}, "
              f"~{len(mine) // 4} prompt tokens (theirs ~{len(theirs) // 4})")
    if lv == 11:
        check(line == "def nki_swiglu_(a, b):", f"4. level 11's def line has both inputs: {line}")
# PROMPT1=theirs means agent.first_prompt stays v5's first_prompt_category: check what v7 installs.
check(v7._category_first_prompt is v5.first_prompt_category,
      "3. with PROMPT1=theirs the first prompt is v5's category prompt, unchanged (v7 wraps nothing)")

# 5
names = set(re.findall(r"\b(nl|nisa)\.(\w+)", v7.API_V2))
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402
missing = [f"{m}.{n}" for m, n in names if not hasattr({"nl": nl, "nisa": nisa}[m], n)]
check(not missing, f"5. all {len(names)} names in the function block exist" + (f"; missing: {missing}" if missing else ""))

print("\n----- level 11's v2 first prompt, for reading -----")
print(v7.first_prompt_v2(11))
agent.API_CARD = v5.BASE_CARD
print(f"\n{fails} failed")
sys.exit(1 if fails else 0)
