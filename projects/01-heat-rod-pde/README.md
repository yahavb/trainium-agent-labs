# Project 1 — The heat-rod agent

**A small model on your chip solves heat-equation problems it cannot solve in one shot. The checker
does the work.**

Status: **worked end to end on real hardware. The solution is in this folder.** Read it, run it, then
break it — a harder level or a different physics is a better use of your day than reproducing this.

---

## The problem

Heat spreads along a rod. Both ends are held at zero, or one end is insulated so no heat escapes
there. Given the starting temperature, what is the temperature at every point for all later time?

Every answer can be **checked without knowing it**. Plug the candidate back into the equation, into
each boundary condition, and into the starting shape, and each of those either holds or does not. That
is what makes this a good agent problem: the checker is exact, cheap, and never has to be told the
answer.

## The loop

```
controller   poses a problem                           level0_heatrod.py / level1_heatrod.py
generator    the model proposes N answers, sampling ON  Qwen3-8B on your chip
checker       grades each one; the grade is the reward   pdecheck.py
calculator   evaluates an integral when asked           tool_calc.py
loop         the best answer's reason becomes the next prompt   agent.py
```

Scoring, out of 1.0: **0.4** the equation holds, **0.2** each boundary condition, **0.2** the starting
shape matches. Partial credit is the point — a wrong answer that satisfies the equation and both ends
scores 0.8 and the agent is told exactly what is missing.

Every attempt is appended to a JSONL file as prompt, answer and reward. **That file is what a
reinforcement-learning trainer consumes**, which is the natural extension of this project: replace the
last step of the loop with a trainer and the model improves instead of just retrying.

## Run it

Start the model first, from the repo root on your instance:

```bash
./serve.sh
```

Then work inside that container, where the model is on `localhost` and this repo is at
`/workspace`:

```bash
docker exec -it vllm bash
cd /workspace/projects/01-heat-rod-pde
pip install sympy
export HEATROD_BASE_URL=http://localhost:8000/v1

python level0_heatrod.py --selftest             # prove the checker before trusting it
python level1_heatrod.py --selftest
python agent.py --level 1 --all
python agent.py --level 1 --sub 3 --no-tools    # the controlled comparison
```

`--offline` needs no model at all and exercises the whole loop against a canned generator, so you can
write code while the server is still compiling. Never report a number from it.

```bash
pip install sympy numpy
python agent.py --offline --level 1 --all
```

There is also a Kubernetes job in [`../../k8s/heatrod-agent.yaml`](../../k8s/heatrod-agent.yaml),
which is how this was developed. You do not need it on the instance.

## The levels

**Level 0** — both ends at zero, starting shape built from sine waves. Exact closed forms. Qwen3-8B
solves all three on the first round. That is what a level 0 is for: it proves the loop works before
anything is hard.

**Level 1** — two things that break the model's habits.

*Insulated end.* One end has zero slope instead of zero temperature. The allowed waves are no longer
the familiar ones; their frequencies are halved. A model reaching for the habitual form satisfies the
equation and the near end, fails the insulated end, and scores 0.6.

*A parabola starting shape.* No finite answer exists — the true answer is an infinite sum whose
coefficients must be computed. The checker accepts a truncation whose starting shape is within 0.5
percent, which measurement shows takes three terms: one term is off by 3.80 percent, two by 0.87,
three by 0.34.

## What we learned, which is the actual content of this project

Three times the fix was **not a better model but a better error message**.

**1. The model answered in notation, not numbers.** Asked plainly, it replied `X(x)T(t)`, then a LaTeX
sum over undetermined coefficients. Both were correct derivations and neither could be evaluated. A
**worked example** of an acceptable answer line fixed it outright. Forbidding the alternatives was not
needed and, on a different model in this repo, actively backfires.

**2. A true number that says nothing.** Told only *"your starting shape is off by 341 percent, add
more terms or correct the coefficients"*, it sat at 0.8 for four rounds, kept alternating signs and
kept including the even terms that must vanish. The checker now projects both shapes onto the
problem's own waves and reports **which term** is wrong.

**3. Revealing the target made it stop thinking.** With the correct coefficients printed in the
feedback, it solved the parabola by copying them verbatim — `1.032` and `0.03822` appear in its answer
and in the message. Fine for a demo, worthless as a reward signal, because nothing was derived. The
report is now **directional only**: too large, too small, missing, wrong sign, should not be there.
`REVEAL_COEFFICIENTS = True` in `pdecheck.py` restores the numbers for a teaching walkthrough.

### Then the calculator, and a failure that moved

Given only direction, the model **guesses**: measured coefficients 1.6, −0.8, 0.4, −0.2, 0.1, each
half the last with alternating signs, then it repeats itself byte for byte until the rounds run out.
It cannot do the integral.

So it gets one tool. It writes `COMPUTE: <expression>` lines, sympy evaluates them, the values come
back, and it answers. **The model decides which integral to set up — that is the reasoning, and it
stays with the model — while the arithmetic moves to sympy.** Nothing is revealed: the tool answers
exactly the question it is asked, so a wrong integral returns a wrong number that has to be noticed.

With the calculator the coefficients came out **exactly right**: `32/pi**3`, `32/(27*pi**3)`,
`32/(125*pi**3)`. And the failure **moved** rather than vanishing — it then paired `exp(-2*pi**2*t)`
with `sin(pi*x/2)`, using frequency `pi` where the wave it multiplies uses `pi/2`. So the checker
learned to measure each term's decay rate on its own and name it, and after that level 1.3 solves in
two rounds while `--no-tools` stalls at 0.8.

**Two things worth stealing for your own project:**

*A single score can move the wrong way while the work gets better.* Reward fell 0.8 → 0.6 at the
moment the model started getting coefficients right, because the equation carries 0.4 and the starting
shape only 0.2. If you are designing a reward, expect this.

*One run is not a result.* At four samples per round these numbers move between runs — problems that
solved in round 1 sometimes take three, and occasionally stall. Report how many runs you did.

## Where to take it

* **Add a trainer.** The attempt log is already in the right shape. Reward climbing across training
  steps, on your own chip, is the strongest demo available here.
* **Harder physics.** A heat source, a moving boundary, two dimensions, or a starting shape that is
  discontinuous at the ends and converges slowly.
* **Fewer hints.** The projection formula is currently given as method. Take it away and see whether
  the agent can find it.
* **Make the checker worse on purpose** and measure how much slower the agent gets. That graph is the
  clearest evidence for the claim this whole project is making.
