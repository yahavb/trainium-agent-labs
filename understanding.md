# Understanding this project

A calm, plain-language map of the repo. Jargon is used on purpose, and every term is explained the
first time it shows up (there is also a glossary at the bottom).

Read in this order: **1 → 2 → 3**. Those three sections are enough to start. The rest is reference.

---

## 1. The one-minute version

This is a **one-day hackathon kit** from **NYU × Annapurna Labs**. Annapurna Labs is the Amazon team
that designs **AWS Trainium**, a custom AI chip. For one day they give each participant:

- a **chip** (a Trainium chip, inside a pod you reach from your laptop),
- a **small language model** already able to run on it (Qwen3-8B),
- and a **question** they care about.

The question, in their words:

> *A small model on a chip you control, driven by an agent, can solve problems the model cannot solve
> on its own. Build the loop that does it, and prove it.*

So the job is **not** "train a model" and **not** "make a faster chip program". The job is to build
an **agent loop** around a small model, and then to *show evidence* that the loop works.

---

## 2. What they expect from you (the most important section)

If you only absorb one section, make it this one. Everything else is machinery.

### 2.1 What you hand in

For whichever project you pick (README, Part 4), three things:

| # | deliverable | what it really is |
|---|---|---|
| 1 | **Your checker**, and the reasoning behind what it accepts and rejects | The thing that grades an attempt. *"This is the artifact we keep."* |
| 2 | **Your attempt log** | Every attempt with its score, so someone else can watch the loop working. |
| 3 | **A one-page note** | What you ran, on what, what came out, **how many runs, and the spread**. |

If you take the kernel-writing challenge, the long version (`projects/02-kernel-agent/CHALLENGE-kernel-agent.md`)
asks for six things: the agent, the verification harness (with its tolerance justified), your eval set
(including hostile cases), a **failure taxonomy**, **token instrumentation**, and a one-page
reproduction note.

### 2.2 How you are graded

From `CHALLENGE-kernel-agent.md`, which says it is the "same rubric as every problem":

| weight | what is scored | in plain words |
|---|---|---|
| **30%** | Correctness | Does it pass on **held-back** shapes and hostile values, not only the ones you tested? A broken rule is a zero, not a small deduction. |
| **25%** | Delivered result | Levels cleared, and how many attempts each took. Fewer attempts is better agent design, not luck. |
| **25%** | Method & honesty | **Does the agent know when it failed?** Reporting "verified" on something that fails scores *worse* than honestly reporting failure. Plus token instrumentation and failure taxonomy. |
| **20%** | Demo & write-up | Show a **failure and the recovery**, not just a success. Can they reproduce it? |

### 2.3 The things they say they reward

Pulled straight from the docs, in the order they come up:

1. **A good checker.** The whole repo repeats one claim: *the checker decides whether the loop works,
   not the model.* "Wrong, off by 341 percent" is true and useless. "The term sin(pi\*x) should not be
   there at all" names the fix. Their claim is that the fix was a better error message rather than a
   better model **eight separate times**.
2. **Turning a verdict into an instruction.** *"This is wrong"* → *"change exactly this."* They call
   this "the single most valuable thing your agent does."
3. **Honesty.** "We did not get it onto the chip" is a fine answer *if you say it first*. Say which
   numbers came from the **simulator** and which from the **device**.
4. **Rates, not best runs.** One run is partly luck. Report how many runs and the spread
   (`--repeat N` does it). A reasonable-looking prompt change made every level *worse* here, and it
   would have shipped without repeated runs.
5. **Calibration.** The agent should state a confidence and be right about it. "I could not verify
   level 8" beats a confident "done" that is lying.
6. **A failure taxonomy.** Collect every wrong kernel, group them into a few named failure modes with
   counts. Needs no accelerator, and it is "the artifact we would most want to keep."
7. **Showing a failure and its recovery** in the demo.

### 2.4 What they say they do *not* expect

- **You do not need to be a programmer to matter.** Deciding what counts as correct, and turning a
  verdict into an instruction, are the hard parts, and neither is a coding task.
- **You do not need to climb the whole ladder.** Verbatim: *"A team that clears four levels and brings
  a rigorous taxonomy of the six it failed will beat a team claiming level 9 that cannot show its
  verification."* That is the grading criterion, not a consolation prize.
- **You do not need to beat the model's limits.** Realistic target: levels 1–4 achievable, 5–7 "a good
  day", 8–10 hard, 9 genuinely hard.
- **Project 2 is unsolved on purpose.** It is "a real open problem, not an exercise with a hidden
  answer."

### 2.5 The format of the day

- **Teams of 3–5.** "The teams that stack five programmers tend to lose these."
- **Your own project is allowed:** it must run on the hardware provided and produce the three
  deliverables. Find an Annapurna engineer **before 11:30** and they will say honestly whether it fits
  in a day.
- Annapurna engineers are on the floor all day. Use them.
- **One rule:** your seat is yours. Don't go into anyone else's pod.

### 2.6 A quick self-check

If you can answer these in a sentence each, you understand what they want:

1. What does my checker accept, what does it reject, and *why*?
2. When it rejects something, what does the message tell the model to **do**?
3. How many times did I run it, and what was the spread?
4. Where does my agent say "I failed" instead of pretending?
5. Which numbers came from the simulator, and which from the chip?

---

## 3. The core idea: the loop and the checker

```
   ┌──────────────┐   attempt    ┌──────────────┐
   │    MODEL     │ ───────────▶ │   CHECKER    │
   │ (Qwen3-8B)   │              │ (plain code) │
   └──────────────┘              └──────┬───────┘
          ▲                             │  score + the REASON
          └─────────────────────────────┘
              next prompt = problem + that reason
```

- The **model** guesses an answer.
- The **checker** is ordinary code (not AI) that grades the guess. It is cheap and exact because it
  never needs to know the answer, only to *verify* one (for the heat rod: plug it back into the
  equation and see if it holds).
- The score **plus the reason** go back into the next prompt. Loop until solved or out of rounds.

**Why the checker is the interesting part:** the same model scores wildly differently depending on how
the checker phrases its complaint. That is the lesson the repo spends all its pages on.

Other terms you will meet:

- **Reward**: the score (0.0–1.0) an attempt gets. Graded, not pass/fail, so a near-miss can be told
  apart from nonsense and the loop has something to climb.
- **Round**: one pass of the loop. **Sample**: one model answer within a round (several per round).
- **Greedy vs sampling**: greedy decoding always picks the likeliest token, so the same prompt gives
  the same answer. Sampling adds randomness so several samples can differ.

---

## 4. The two sample projects

Both live in `projects/` and both are the same loop with a different checker.

### Project 1 — heat-rod agent (`projects/01-heat-rod-pde/`) — **solved, 6 of 6**

- **Problem:** heat spreads along a rod; given the starting temperature, what is the temperature at
  every point later? (A **PDE**: a partial differential equation, an equation about how something
  changes over space and time.)
- **Checker (`pdecheck.py`):** uses **SymPy** (a symbolic maths library) to plug the model's answer
  back into the equation, each boundary condition, and the starting shape.
- **Reward out of 1.0:** 0.4 equation holds, 0.2 per boundary, 0.2 starting shape.
- **Needs no chip cores at all.** It still uses the chip to *serve the model*.
- **The four findings, each changed the code:**
  1. It answered in notation, not numbers → a **worked example** fixed it (prohibitions did not).
  2. "Off by 341 percent" said nothing → the checker now names *which term* is wrong.
  3. Printing the right coefficients made it copy them instead of thinking → feedback is now
     **directional only** (too big, too small, missing, should not be there).
  4. It can't do the integral → it gets a **calculator it aims itself** (`COMPUTE: <expr>`, SymPy
     evaluates). The model chooses *what* to compute; SymPy does the arithmetic.
- **Headline result:** level 1.3 solves in two rounds with the calculator and stalls at 0.8 without it
  (`--no-tools`).

### Project 2 — kernel agent (`projects/02-kernel-agent/`) — **runs, unsolved**

- **Problem:** have the agent write a **kernel** (a small program that runs directly on the
  accelerator, moving tiles of data into on-chip memory and doing the maths there). In the language
  **NKI** (Neuron Kernel Interface, Annapurna's kernel language for Trainium).
- **Why it's hard:** a wrong kernel does not crash. It returns plausible-looking wrong numbers.
- **Checker (`nkibench.py`)**, three layers by cost:
  1. **Static rules** (milliseconds): did it cheat or break a rule?
  2. **`nki.simulate` on the CPU** (seconds): are the numbers right on awkward shapes?
  3. **On-device timing and profile** — *not built yet.*
- **Reward:** 0.1 parses, 0.2 rules clean, 0.2 runs, 0.5 correct on every shape (prorated).
- **Where it stands** (5 runs, Qwen3-8B):

  ```
  level 1: solved 0/5   always 0.30     level 3: solved 0/5   always 0.30
  level 2: solved 4/5   0.5 – 1.0       level 4: solved 0/5   always 0.62
  ```

  Levels 1, 3 and 4 hit the same wall: **tiling** (the model builds one oversized tile instead of
  slicing). Zero spread there means it's a capability wall, not dice, so changes are cleanly
  attributable.
- **Ladder** in `nkibench.py`: 1 avg-pool, 2 transpose, 3 matmul single tile, 4 matmul tiled,
  5–7 matmul optimised (no reference kernels yet), 8 single-head attention (the worked "add your own
  op" example).

### The hidden constraint behind the kernel challenge

The kernel challenge doc frames the real problem as **context budget**: the model sees only **8192
tokens** per call, and the docs, spec, reference, broken kernel, error trace and rules do not all fit.
So the agent has to decide *which 8000 tokens matter* on each attempt. Ideas offered: tiered docs,
error distillation, progressive disclosure, a one-line-per-attempt "already tried" ledger, retrieval.
They want you to **instrument** it: input tokens per attempt, and where they went.

---

## 5. The hardware and setup, in plain words

| thing | what it is |
|---|---|
| **Trainium / trn2** | AWS's custom AI chip family. You are using `trn2`. |
| **NeuronCore (NC)** | A compute unit on the chip. Your chip has 4 logical ones. A core can't be shared by two processes. |
| **Seat / pod** | Your seat is a Kubernetes **pod** named `seat-<number>`: your own box with one chip and this repo at `/workspace`. 150 seats total. |
| **EKS / kubectl** | EKS is AWS's managed Kubernetes. `kubectl` is the command line you use to reach the pod (`kubectl exec -it seat-42 -- bash` = open a shell inside it). |
| **vLLM** | The server that runs the model and speaks an OpenAI-style HTTP API on `localhost:8000`. |
| **Qwen3-8B** | The 8-billion-parameter model your pod serves. Sampling on. |
| **`serve.sh`** | Starts that server (about 4 min the first time) and prints `READY`. |
| **Tensor parallel (TP) = 2** | The model is split over 2 cores. NC 2–3 hold it; NC 0–1 stay free. TP=4 gave no benefit (the loop is limited by generation length, not compute). |
| **`gpt-oss-20b`** | A *shared*, separate 20B model endpoint run by the organisers. **Greedy**, `tools=` doesn't work, ~4 requests/sec shared by everyone. |

### The path, nine steps (full detail in `README.md` Part 1)

On your **laptop:** install `aws` + `kubectl` → paste the workshop AWS credentials into that terminal →
`aws eks update-kubeconfig --name hack-hyd --region ap-south-2` → `kubectl get pod seat-N` →
`kubectl exec -it seat-N -- bash`.

Inside the **pod:** `./serve.sh` (leave it) → open a *second* shell in the same pod →
`git config --global --add safe.directory /workspace` → run a project.

Practical notes that save time:

- The credentials are per-terminal and **expire** (`ExpiredToken`). Re-paste; your pod is untouched.
- **Wait for the new prompt** before typing after `kubectl exec`, or your keystrokes go to your laptop.
- Start long runs with `nohup python agent.py ... > run.log 2>&1 < /dev/null &` then `tail -f run.log`.
  (`nohup` = "don't die when the shell does.") A dropped connection would otherwise kill the run.
- If the pod is replaced, `/workspace` is **gone**. `git push` anything you want to keep.

---

## 6. Things that will bite you (all measured in the repo)

1. **Thinking mode hurts.** With reasoning on, rounds went from ~8 s to 446 s and scores to 0.00.
   Keep it off. A bigger token budget does **not** fix it; a *shorter prompt* does.
2. **A bigger model is not a better agent.** The 8B model beat the 20B on level 4 (0.62 vs 0.30)
   because the 20B spent its budget on hidden reasoning and returned no code.
3. **Greedy means retrying is pointless.** Same prompt → same answer. Change the *prompt*. That is
   why the agent keeps a ledger of failed attempts.
4. **Constraints belong in the checker, not the prompt.** List ten rules and the model audits itself
   forever and answers nothing; list none and it writes confident, illegal code. Generate freely,
   catch the violation, send back **one named change**.
5. **Don't print the answer in your feedback.** The model copies it and derives nothing.
6. **The failure moves, it doesn't vanish.** Fix one thing and the next wall appears. A score can even
   drop (0.8 → 0.6) while understanding improves, because of how the reward is weighted.
7. **Check `finish_reason`.** `length` with no error means your answer was silently truncated.
8. **One run is not a result.** Level 2 swings 0.50 ↔ 1.00 on luck alone.

---

## 7. What's in the repo

```
README.md                  The main guide: get onto your chip, what was measured, the challenge
STATE.md                   Status notes written for resuming (partly stale, see section 8)
understanding.md           This file
serve.sh                   Starts Qwen3-8B inside your pod
projects/
  01-heat-rod-pde/         Solved. agent.py = the loop, pdecheck.py = the checker, tool_calc.py = calculator
  02-kernel-agent/         Unsolved. agent.py = the loop, nkibench.py = NKI checker (CPU simulator),
                           kernelbench.py = NumPy "laptop" checker, reference_level*.py = known-good kernels,
                           CHALLENGE-kernel-agent.md = the full brief, grading and expectations
gptoss/                    Client, probe and load-test tools for the shared gpt-oss-20b endpoint
k8s/                       Kubernetes manifests (cluster side; participants don't need these)
workshop/FACILITATOR.md    Organiser-only: running 150 seats, credentials, RBAC
```

Reading order I'd suggest: this file → `README.md` Part 4 → `CHALLENGE-kernel-agent.md` (scoring and
honest expectations) → the README of the project you choose → `README.md` Part 2 (so you don't
re-learn what they already measured).

---

## 8. Things that are slightly confusing (so they don't trip you)

- **Two kernel harnesses.** `kernelbench.py` + `CHALLENGE-kernel-agent.md` describe the **laptop
  version**: NumPy under kernel-shaped rules, a 10-level ladder, no accelerator needed ("Stage A").
  `nkibench.py` + the project README describe the **NKI version**, run on the CPU simulator, with
  levels 1–8. Same idea, different checker. The challenge doc's *grading and expectations* apply to both.
- **`STATE.md` is partly out of date.** It talks about a standalone `trn2.3xlarge` and docker. The
  README now uses per-participant **seat pods** with no docker; README itself says those notes "are
  history".
- **Solve rates differ between docs** (level 2 appears as ~2/3, 2/5 and 4/5). `STATE.md` explains the
  gap isn't attributable. Treat **4/5** as the latest baseline and don't claim an improvement from it.
- **Which model project 2 talks to:** the README says the local Qwen3-8B; the docstring at the top of
  `projects/02-kernel-agent/agent.py` still describes the shared gpt-oss endpoint. Both work; the
  README is the current plan.
- **Still open per `STATE.md`:** on-device timing (layers 2–3), reference kernels for levels 5–7, and
  which levels are held back for judging. The held-back answers live in a gitignored `RUN-OF-DAY.md`,
  which isn't in this checkout.

---

## 9. Glossary

| term | meaning |
|---|---|
| **Agent** | Here: a model plus a loop that tries, checks and retries. |
| **Checker / verifier / harness** | The non-AI code that grades an attempt. |
| **Reward** | The checker's score for an attempt. |
| **Ledger** | A running list of failed attempts, so the prompt changes instead of repeating. |
| **Kernel** | A small program that runs directly on the accelerator. |
| **NKI** | Neuron Kernel Interface: the language for writing Trainium kernels. |
| **Tile / tiling** | Cutting a big tensor into small chunks that fit on-chip memory. Level 4's wall. |
| **Partition dimension** | The first axis of an on-chip tile; max 128. |
| **SBUF / PSUM** | On-chip memory (SBUF) and the accumulator area for matmul results (PSUM). |
| **HBM** | The chip's large main memory. Moving data to and from it is the costly part. |
| **DMA** | Direct memory access: the data mover between HBM and on-chip memory. |
| **Arithmetic intensity** | Flops per byte moved. Low means waiting on data ("memory bound"). |
| **Roofline / ridge** | The model that says whether you're limited by memory or compute; the ridge is the crossover point (222 Flops/Byte for bf16 here). |
| **Memory bound / compute bound** | Limited by moving data vs limited by arithmetic. |
| **Simulator (`nki.simulate`)** | Runs a kernel on the CPU, with no chip, in seconds. Where the agent should spend its attempts. |
| **Context window** | How many tokens the model sees per call (8192 here). |
| **`max_tokens` / `finish_reason`** | Answer budget, and why generation stopped (`stop` vs `length`). |
| **Greedy decoding** | Always choose the likeliest token; deterministic. |
| **Hostile values** | Test inputs chosen to break things: huge numbers, zeros, all-identical rows, awkward shapes. |
| **Ragged edge** | The last, partial tile when a size doesn't divide evenly. The most common silent bug. |
| **RL (reinforcement learning)** | Training a model from reward signals. The attempt log is already in the right shape to feed one. |
| **StatefulSet / pod** | Kubernetes terms: the controller that keeps the 150 seat pods alive with stable names / one running container group. |
| **EFA** | AWS's high-speed networking. `serve.sh` turns off an EFA-related check since one chip has none. |

---

## 10. A sensible first hour

1. Get to `READY` on your seat (README Part 1, steps 1–6).
2. Run `python level0_heatrod.py --selftest`, then `python agent.py --level 0 --all` in project 1.
   Watch what a working loop prints.
3. Run `python nkibench.py --selftest` in project 2 and **read a failure message** from a real run.
   Ask: does it name a fix, or just a verdict?
4. Decide: take project 2 as given, extend it, or propose your own to an engineer before 11:30.
5. Whatever you pick: build the **checker first**, break it on purpose to confirm it catches your own
   bug, and only then bring in the model.
6. From the first attempt, **log every attempt with its score**, and decide how many runs you'll do.
   That is deliverable 2 and 3 produced as you go instead of at 5 pm.
