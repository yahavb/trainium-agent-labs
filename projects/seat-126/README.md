# Seat 126 — Verifier-Guided PDE Agent

**Annapurna Labs Hackathon | AWS Trainium2 | Qwen3-8B**

Our contribution: SymPy-based PDE verification, verifier-guided candidate revision, and independent candidate selection.

**Benchmark:** Baseline 4/5 solved; verifier-guided 5/5; independent-selection 2/2. Small-sample results only; official checker feedback remains in the revision loop.

See `agent_independent.py`, `symbolic_verifier.py`, and `experiment-results/` for the implementation and evidence.

---

## Original starter-project documentation

# Project 1 — The heat-rod agent

**A small model on your chip solves heat-equation problems it cannot solve in one shot. The checker
does the work.**

> ## STATUS: SOLVED
>
> **All six problems solved**, Qwen3-8B on one Trainium chip. Levels 0.1 to 0.3 on the first round;
> level 1.1 and 1.2 on the first; level 1.3 — the hard one — on the second, after the checker told it
> what was wrong. Transcript below.
>
> **The solution is in this folder.** So read it, run it, then break it: a harder level, different
> physics, or an RL trainer on the attempt log is a better use of your day than reproducing this.
>
> One caveat that is the most interesting thing here: that transcript was produced while the checker
> still printed the target values, and the model **copied them** rather than deriving anything. The
> checker is now directional only, so your run will be harder than the one shown.

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

Your seat is a pod, `seat-<your number>`, and you work inside it — there is
no container to enter. Seat 42 is the example; use your own number.

**Terminal 1 — start the model.** Get a shell in your pod and **wait for the prompt**
(`root@seat-42:/workspace#`) before typing:

```bash
kubectl exec -it seat-42 -- bash
```

The repo is at `/workspace`, and `serve.sh` is at its top:

```bash
cd /workspace
./serve.sh
```

It prints `READY` when the model answers on `http://localhost:8000` — about 4 minutes the first time.
Leave this terminal open.

**Terminal 2 — run the agent.** Open a second terminal, paste the credentials again, and get a second
shell in **the same pod**:

```bash
kubectl exec -it seat-42 -- bash
```

Then:

```bash
cd /workspace/projects/01-heat-rod-pde

python level0_heatrod.py --selftest             # prove the checker before trusting a score
python level1_heatrod.py --selftest
python agent.py --level 0 --all                 # the warm-up
```

The real runs take minutes each, so start them in the background and watch the log:

```bash
nohup python agent.py --level 1 --all > run.log 2>&1 < /dev/null &             # the real one
tail -f run.log
nohup python agent.py --level 1 --sub 3 --no-tools > run-notools.log 2>&1 < /dev/null &   # the controlled comparison
tail -f run-notools.log
```

> **Long runs survive a dropped connection only if you start them like this.** A `kubectl exec` session
> can drop (Wi-Fi, laptop sleep, `connection reset by peer`), and anything running in the foreground of
> that shell dies with it. So start the run in the background, writing to a log, and watch the log:
>
> ```bash
> nohup python agent.py ARGS > run.log 2>&1 < /dev/null &
> tail -f run.log        # Ctrl+C stops watching; the run keeps going
> ```
>
> Disconnected? Reconnect with `kubectl exec -it seat-42 -- bash` and pick up where you were with
> `tail -f /workspace/projects/01-heat-rod-pde/run.log`. `pgrep -af agent.py` shows whether it is still running.

The pod already sets `HEATROD_BASE_URL=http://localhost:8000/v1` and the model name, so there is
nothing to export.

`sympy` and `numpy` are already in the pod.

### Without a model at all

`--offline` exercises the whole loop against a canned generator, so you can write code while the
server is still compiling. Never report a number from it.

```bash
pip install sympy numpy
python agent.py --offline --level 1 --all
```

There is also a Kubernetes job in [`../../k8s/heatrod-agent.yaml`](../../k8s/heatrod-agent.yaml),
which is how this was developed. You do not need it in your pod.

## What it looks like when you run it

Real output, Qwen3-8B on one Trainium chip, four attempts per round. **Read this before running
anything** — it is what a working loop looks like, so you can tell progress from flailing.

The easy levels are solved on the first round, which is what level 0 is for:

```
=========== level0.1 ===========
round 0: rewards [1.0, 1.0, 1.0, 1.0]  mean 1.00  best 1.0  (8.7s)
SOLVED: u(x, t) = exp(-(3*pi)**2*t)*sin(3*pi*x)
```

The interesting one is the parabola, where the first attempt is wrong and the checker's reason is
what fixes it:

```
=========== level1.3 ===========
round 0: rewards [0.0, 0.0, 0.4, 0.4]  mean 0.20  best 0.4  (28.7s)
  checker: The equation u_t = 2*u_xx does not hold: at x=1.087, t=0.0270,
  u_t - 2*u_xx = -14.88 instead of 0. At t=0 the answer differs from the required
  starting shape by 103.71 percent, and it must be under 0.5 percent. Term by term:
  the coefficient of sin(pi*x) should be 0 but yours is -0.8 ... The terms
  sin(pi*x), sin(2*pi*x) should not be there at all.

round 1: rewards [0.0, 1.0, 0.4, 0.0]  mean 0.35  best 1.0  (25.4s)
SOLVED: u(x, t) = 1.032*exp(-2*(pi/2)**2*t)*sin(pi*x/2)
              + 0.03822*exp(-2*(3*pi/2)**2*t)*sin(3*pi*x/2)
              + 0.00656*exp(-2*(5*pi/2)**2*t)*sin(5*pi*x/2)
```

```
=========== summary: level 1 ===========
  level1.1     reward 1.0 after 1 round(s)  SOLVED
  level1.2     reward 1.0 after 1 round(s)  SOLVED
  level1.3     reward 1.0 after 2 round(s)  SOLVED
  solved 3/3
```

**Four things to notice, because they are the whole point:**

* **The rewards in a round differ** — `[0.0, 1.0, 0.4, 0.0]`. Sampling is on, so the attempts are
  genuinely different. If every number in a round is identical, sampling is off and there is
  nothing for the loop to choose between.
* **0.4 means the equation and the boundaries are right and the starting shape is wrong.** Partial
  credit tells you *which part* is broken, which is why the reward is graded rather than pass/fail.
* **Round 0 fails and round 1 succeeds.** That is the loop working. A level solved on round 0 taught
  the agent nothing.
* **That transcript was produced with the target values printed in the feedback**, and the model
  copied `1.032` and `0.03822` straight out of the message. The checker is now directional only, so
  your run will look harder than this one. See "What we learned" below — this is the single most
  important thing in the project.

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
