"""Checker mutation suite for level 8: does the checker name every attention bug we have seen?

Each mutation puts ONE bug observed in the agent's logs into the verified level-8 kernel (solved/level08_attention.py), grades the
result exactly as the agent loop does (lint + simulator + hints + wrong-variant diagnosis), and checks
that the feedback names the expected category. A checker crash is its own outcome. Run on a seat (needs
nki for the simulator; no NeuronCore):

    python mutation_check.py
"""
import re, sys
import agent

agent.LINT, agent.LEVEL_HINTS = True, True
REF = open("solved/level08_attention.py").read()

# (name, old text, new text, a pattern the feedback must contain)
MUTATIONS = [
    ("q not transposed for q k^T", "stationary=qT_sb, moving=kT_sb", "stationary=q_sb, moving=kT_sb",
     r"sums over the FIRST|stationary\^T x moving"),
    ("matmul operand left in psum", "stationary=qT_sb, moving=kT_sb", "stationary=qT_ps, moving=kT_sb",
     r"is in psum, must be sbuf"),
    ("scores stored into the output early", "    # Stage 5: row max",
     "    nisa.dma_copy(dst=out, src=scores_sb)\n    # Stage 5: row max", r"copying scores_sb"),
    ("invented API", "nl.max(scores_sb", "nisa.max(scores_sb", r"nisa\.max does not exist"),
    ("dtype called as a function", "operand0=1.0 / (dim ** 0.5)", "operand0=1.0 / nl.float32(dim ** 0.5)",
     r"is a dtype, not a function"),
    ("read before write (v never loaded)", "    nisa.dma_copy(dst=v_sb, src=v)\n", "",
     r"nothing has written v_sb"),
    ("kernel never returns", "    return out\n", "", r"never returns"),
    ("1-D tile", "inverse_sum = nl.ndarray((seq, 1)", "inverse_sum = nl.ndarray((seq,)", r"is 1-D"),
    ("P^T v instead of P v", "stationary=pT_sb, moving=v_sb", "stationary=probabilities, moving=v_sb",
     r"DIAGNOSIS: .*P\^T v"),
    ("scale inverted", "operand0=1.0 / (dim ** 0.5)", "operand0=(dim ** 0.5)", r"DIAGNOSIS: .*inverted"),
    ("scale missing", "operand0=1.0 / (dim ** 0.5)", "operand0=1.0", r"DIAGNOSIS: .*WITHOUT the 1/sqrt"),
    ("never normalised", "data=exp_scores, op0=nl.multiply, operand0=inverse_sum",
     "data=exp_scores, op0=nl.multiply, operand0=1.0", r"DIAGNOSIS: .*never divided"),
    # Blind spot, recorded honestly: without the max subtraction the maths is identical, and the test
    # inputs are too small for exp() to overflow, so the checker cannot tell.
    ("no max subtraction (expected blind spot)", "op0=nl.subtract, operand0=row_max",
     "op0=nl.subtract, operand0=0.0", r"^PASS$"),
]

rows, ok_all = [], True
for name, old, new, want in MUTATIONS:
    assert old in REF, f"mutation '{name}' does not apply"
    src = REF.replace(old, new, 1)
    try:
        reward, _, feedback = agent.grade(src, 8)
        outcome = "PASS" if reward >= 0.999 else feedback
    except Exception as e:                       # a checker crash is its own outcome
        outcome = f"CHECKER CRASH: {type(e).__name__}: {e}"
    hit = bool(re.search(want, outcome, re.S)) and not outcome.startswith("CHECKER CRASH")
    ok_all &= hit
    rows.append((name, hit, outcome))
    flat = re.sub(r"\s+", " ", outcome)
    print(f"{'ok  ' if hit else 'MISS'} {name:42s} -> {flat[:150]}", flush=True)

base, _, _ = agent.grade(REF, 8)
print(f"\nunmutated reference: reward {base:.2f}")
print(f"{sum(h for _, h, _ in rows)}/{len(rows)} mutations reported in the expected category")
sys.exit(0 if ok_all and base >= 0.999 else 1)
