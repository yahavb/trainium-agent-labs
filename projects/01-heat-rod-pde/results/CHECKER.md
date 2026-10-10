# The checker, and the reasoning behind it (hand-in item 1)

The checker is `pdecheck.py`. It never knows the answer. It plugs a candidate u(x, t) into the
problem and scores four things separately:

| check | weight | passes when |
|---|---|---|
| The equation u_t = k·u_xx | 0.4 | the residual is under 1e-6 (relative) at 24 random points |
| Left end | 0.2 | u = 0 (fixed end) or u_x = 0 (insulated end) at every sampled time |
| Right end | 0.2 | the same, at x = L |
| Starting shape | 0.2 | the L2 error against u(x, 0) is under the problem's tolerance (1e-6, or 0.5% for the parabola) |

A reward of 1.0 means all four hold, so the answer is correct. That is checked, not compared with a
key.

## What we changed, and why

**The reward is unchanged.** Same checks, same weights, same tolerances. Every result in
`RESULTS.md` is graded by the same rule as the as-shipped run.

1. **Markdown bold around the answer line is stripped** (`extract`). `**u(x, t) = ...**` used to fail
   to parse and score 0. One answer in the as-shipped log was lost this way: it passes three of the four
   checks and now scores 0.8 instead of 0.
   Formatting is not maths, so the checker should not punish it.
2. **The prompt can be produced without the answer-format block** (`prompt_of(style="spec")`), for
   the modes where the model writes a spec instead of an answer. The default prompt is byte-identical
   to the shipped one (checked on all 30 level-0/1 problems × seeds 0–4).

## The feedback, and why it is shaped this way

- **Directional only.** The checker says which term is too large, too small, missing, or decays too
  fast. It never prints a target value. The repo found that printing targets made the model copy them
  and derive nothing.
- **Per check, compact, in the solver modes** (`agent.compact_feedback`). PASS/FAIL for each of the four
  checks, the checker's first sentence for each failure, and a root-cause line. The long solver
  expression is no longer sent back: it described the answer, not the spec the model can change.
- **Per term, across all four candidates, in the free-form mode** (`aggregate.py`). For each wave:
  which candidate had the coefficient right, which had the decay rate right, plus a keep/fix line.
  On level 1.3 this turned a loop that repeated the same 0.6 answer for four rounds (0/3 seeds) into
  one that solved in round 2 (2/3 seeds).

## The solver tool, and why it is kept apart from the checker

`solver_tool.py` turns the model's spec (k, L, starting shape, end types) into u(x, t) with SymPy:
the allowed waves from the end types, each decay rate from the waves, each coefficient by projecting
the starting shape. Two rules keep the evaluation honest:

- **The tool never reads the problem.** It uses only what the model wrote. A wrong end type or a
  mis-copied k gives a faithfully computed wrong answer, and the checker catches it.
- **The checker never sees the tool.** It grades the final u(x, t) exactly as it grades a
  hand-written answer.

## Known limits

- The checker samples 24 points; a pathological answer could pass the equation check between them.
  The analytic answers here are smooth, so this did not arise.
- The starting-shape tolerance sets how many series terms the parabola needs (3 at 0.5%).
- On level 0–1 the problem text is a fixed template, so in the solver modes the model mostly copies
  fields. The checker cannot tell how much reasoning the model did, only whether the answer is right.
