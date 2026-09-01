# Challenge — The kernel writing agent

**Build an agent that writes chip-level kernels using a model that can only see 8192 tokens at a
time. The kernel writing is the visible problem. Deciding what the model gets to look at is the
actual problem.**

---

## Why this is hard, and why it isn't the reason you think

A *kernel* is the small program that runs directly on the accelerator. Instead of `y = softmax(x)`,
you write the tiles of data you move into on-chip memory, the work you issue, and the order you do it
in. It is specialist work, it is slow, and it is exactly the kind of thing people now try to automate
with a language model.

So you point a model at it. And immediately hit the real wall:

| what the model would need to see | tokens |
|---|---|
| the API documentation for the kernel language | thousands |
| the design spec for the op you're implementing | ~11,600 for one real example |
| the reference implementation you must match | hundreds |
| your current broken kernel | hundreds |
| the error trace and the numerical diff from the last attempt | hundreds to thousands |
| the tiling rules and hardware constraints | thousands |

**Your budget is 8192 tokens of input, total, for all of it.** It does not fit. It is not close to
fitting.

That is the challenge. Not "can a model write a kernel" — it can, sometimes — but **can you build the
thing that decides, on each attempt, which 8000 tokens matter?** Every iteration you spend a budget.
Spend it on API docs and the model can't see its own error. Spend it on the error trace and it forgets
the tiling rules and reinvents an illegal one.

An agent that solves this well is doing something genuinely useful and genuinely underexplored. An
agent that ignores it will hit a wall in about twenty minutes.

---

## The task

Given a reference implementation of an operation, your agent produces a **tiled, explicit-memory
kernel** that computes the same thing, then **runs it, checks it numerically, reads its own failure,
and tries again** — all within the token budget, on every loop.

### Stage A — runs on your laptop, no accelerator needed

Write kernels in **NumPy under kernel-shaped rules**. This is not busywork: the rules are what make
kernel writing hard, and they are all checkable on a CPU in milliseconds.

Your generated kernel must obey:

1. **Fixed tiles.** Work in tiles of at most `128` rows × `512` columns. One tile at a time.
2. **Explicit loops over tiles.** No whole-array operations that hide the tiling.
3. **No fancy indexing, no boolean masks as indexing, no `np.einsum`, no broadcasting tricks.**
   Slices and arithmetic on slices only.
4. **Handle shapes that don't divide evenly.** The last tile is partial. This is where most
   generated kernels quietly break.
5. **No calling the thing you're implementing.** `np.mean` inside your layernorm kernel is cheating
   and the checker looks for it.

These constraints are deliberately close to what real accelerator kernels must satisfy — fixed tile
shapes, explicit data movement, no free lunch on ragged edges — so what you learn transfers.

### Stage B — on the accelerator (stretch, only if you have one)

Same agent, emitting real kernel code for the chip, verified on device. **Do not start here.** Get
the loop working in Stage A first; the agent architecture is identical and the feedback is a thousand
times faster.

---

## The operation ladder

Climb it. Stopping partway with honest results beats claiming the top.

| # | operation | the trap in it |
|---|---|---|
| 1 | `y = relu(a*x + b)`, elementwise | none — this is your "the loop works" checkpoint |
| 2 | row-wise sum over a 2D array | partial final tile |
| 3 | row-wise max | partial tile + the identity for max is `-inf`, not `0` |
| 4 | RMSNorm over the last axis | accumulation order; float32 vs float64 |
| 5 | softmax over the last axis | **naive `exp` overflows.** Needs max-subtraction. A wrong one still returns numbers |
| 6 | tiled transpose | tile boundaries on non-square, non-divisible shapes |
| 7 | tiled matmul with accumulation | accumulator dtype; K not divisible by the tile |
| 8 | layernorm, mean and variance | `E[x²] − E[x]²` catastrophically cancels on large means. Looks fine on test data, wrong in practice |
| 9 | windowed (banded) attention, window `w` | edges of the band; `w` larger than the sequence |
| 10 | 1D convolution with stride and dilation | output-size formula off-by-one; dilation at the tail |

Three more are held back for judging, drawn from the same distribution.

**Every one of these fails silently.** A wrong softmax returns plausible numbers. A wrong layernorm
returns plausible numbers. Nothing crashes. That property is the whole reason this problem is
interesting, and it is why the next section is not optional.

---

## Non-negotiable: the verification harness

**Build this before you build the agent.** Teams that build it second spend the afternoon debugging
their debugger.

It must:

* compare against the reference on **shapes that don't divide evenly** by your tile size — include a
  prime-numbered dimension, and a dimension of exactly 1
* test **hostile values**, not friendly ones: large magnitudes (softmax overflow), large means
  (layernorm cancellation), zeros, negatives, and a row that is entirely identical values
* state a **numerical tolerance and justify it.** "`rtol=1e-5`" is an answer; "it looked close" is
  not
* **statically reject rule violations** — fancy indexing, banned functions, tiles over the limit.
  A fast kernel that broke the rules scores zero, so catch it yourself before a judge does
* return a **failure description your agent can actually use.** "assert failed" teaches the model
  nothing. "row 47 of 128, expected 0.0031, got 0.0034, relative error 0.09, worst at the last
  partial tile" teaches it exactly where to look

That last point is where this challenge is won. **The quality of your error message is the quality of
your agent.**

---

## The context problem, concretely

You have ~8000 input tokens per call. On attempt 4, your agent could include: the API rules, the
reference, the current kernel, the last error, the previous three kernels and their errors, and a
list of things already tried. That's far too much. So it has to choose.

Approaches worth trying — and this is the interesting design space:

* **Tiered docs.** A 300-token rules card always included; the full docs only on demand for a
  specific construct.
* **Error distillation.** Turn a 2000-token diff into 50 tokens: which tile, which row, which
  direction, how large. Ask the model, or write a function.
* **Progressive disclosure.** Attempt 1 sees only the reference. The error decides what attempt 2
  additionally sees.
* **A "what I have already tried" ledger**, compressed to one line per attempt so the agent stops
  re-proposing the same broken idea.
* **Retrieval over your rules document** rather than pasting it.

Whatever you choose, **instrument it.** Report input tokens per attempt, and where they went. A graph
of "tokens spent on docs vs error context vs code, per attempt" is the single most interesting artifact
you can bring to the demo.

---

## Facts about this endpoint you will otherwise learn the hard way

All measured; see the README for how.

* **8192 tokens on INPUT.** Exceed it and you get a clean 400. Come *close* to it and your answer is
  silently truncated instead — `finish_reason: "length"` and no error. **Check `finish_reason` on
  every call** or your agent will parse half a kernel.
* **Ask for at least 2500 `max_tokens`.** The model writes to a hidden reasoning channel before it
  writes any answer. A coding task burned 900 tokens thinking and returned *empty content*. That
  looks exactly like refusal and isn't.
* **Sampling is greedy. Retrying is pointless.** Identical input gives byte-identical output. Your
  retry-on-failure loop must change the *prompt* — more context, different framing, a narrower
  question. Temperature does nothing. There is no best-of-n, no self-consistency.
* **The `tools=` parameter does not work.** No function calling. Hand-roll JSON in the prompt, and
  demand a **final answer** explicitly or the model resolves the whole task in its reasoning channel
  and returns nothing.
* **Long prompts are cheap.** A 37× longer prompt costs ~15–20% more latency. Fill the window — just
  don't overflow it.
* **~4 req/s shared across everyone.** An iterate-and-retry loop is 4–5 chat users' worth of load.
  Cache aggressively; don't re-ask what you already know.

---

## A worked example, and the trap that eats the day

We ran rung 5 (softmax) against the endpoint before writing this, to check the challenge is fair.
Here is what actually happened, because you will hit the same wall within the hour.

**Attempt 1 — the obvious prompt.** Reference implementation, the tiling rules, the banned-function
list, "your FINAL ANSWER must be a python block." Result:

```
max_tokens=2500 → finish=length, reasoning=10,194 chars, content=0 chars
max_tokens=7000 → finish=length, reasoning=26,401 chars, content=0 chars
```

**Nothing came back. Twice.** Not a refusal, not an error — the model spent the entire budget in its
hidden reasoning channel and never began the answer. Raising the budget nearly threefold did not
help; it just thought for longer. Reading the reasoning shows why:

> *"We need to ensure that we don't use np.sum or np.max. We used no banned functions. We used no
> fancy indexing. We used no whole-array ops. We used tile loops..."*

It was **auditing itself against every rule we gave it, one at a time.** Our carefully specified
prompt was the cause. Rephrasing the rules positively instead of as prohibitions did not help either:
13,700 chars of reasoning, no answer.

**Why it never recovers: sampling is greedy.** There is no randomness in the decoding, so once the
model enters a repeating state nothing perturbs it out. Caught mid-loop:

```
We used loops over columns for max. Good.
We used loops over columns for sum. Good.
We used loops over columns for max. Good.
We used loops over columns for sum. Good.
```

That is not slow progress, it is a closed cycle. **You cannot buy your way out of it with a bigger
budget** — which is exactly why 7000 tokens failed identically to 2500. Across eight attempts every
one-sentence prompt produced code in 300–700 tokens and every prohibition-carrying prompt spiralled.

**Attempt 2 — drop the rules entirely.** One sentence: "compute a numerically-stable softmax over the
last axis, at most 128 rows at a time." Clean code in 617 tokens. And it used `np.max` and `np.sum` —
**a rule violation, instant zero.**

So there's your dilemma, and it is the real content of this challenge:

> **Specify the constraints and the model produces nothing. Omit them and it produces something
> confident and illegal.**

**What worked.** Generate naively, let the *verifier* find the violation, then send back one
surgical repair request naming only what to fix — no rules list, no reference, just the code and the
one change:

> *"Replace the np.max and np.sum calls with explicit python loops over the columns that compute the
> row maximum and the row sum. Keep everything else identical."*

897 output tokens. **32/32 cases passed, verified.** Two calls, ~1,900 tokens total, for the rung
with the overflow trap in it.

**And the near-miss that proves the point.** Feeding the verifier's report back *verbatim* — the
literal text `line 16: calls banned max` — produced **the same violation again**. The report says what
is wrong; it never says what to do. Rewriting it as the instruction above fixed it in one round. That
translation, from verdict to instruction, is the single most valuable thing your agent does.

The lesson generalises: **constraints belong in your verifier, not in your generation prompt.** The
model is bad at holding ten rules in mind and good at making one named change. Your agent's job is to
convert "this is wrong" into "change exactly this."

That is one rung. Nine more, and the held-back three, are yours — and the harder ones (8, 9, 10) will
not fall to a single repair round. But you now know the shape of the problem, which is more than we
did an hour ago.

## Scoring

Same rubric as every problem.

| weight | | |
|---|---|---|
| **30%** | Correctness | How many ladder rungs pass **on the held-back shapes and hostile values** — not just yours. A rule violation is a zero, not a deduction. |
| **25%** | Delivered result | Rungs cleared, and attempts needed per rung. Fewer attempts to green is better agent design, not luck. |
| **25%** | Method & honesty | **Does the agent know when it failed?** An agent reporting "verified" on a kernel that fails the harness scores worse than one reporting honest failure. Plus: your token-budget instrumentation, and your failure taxonomy. |
| **20%** | Demo & write-up | Show a failure and the recovery, not just a success. Can we reproduce it? |

### Two things that separate the top teams

**Calibration.** Your agent must output a confidence, and be right about it. Confidently wrong is the
worst possible behaviour on a problem where wrong kernels don't crash — in production that is a silent
numerical bug shipped to users. An agent that says "I could not verify rung 8" is more valuable than
one that says "done" and is lying.

**The failure taxonomy.** Run your agent across the whole ladder, collect every wrong kernel it wrote,
and classify them into a handful of named failure modes. Which rungs did it fail, and *why* — did it
misread the tiling rules, forget the partial tile, reinvent an illegal indexing pattern, fix the error
it was shown while breaking something else? This is analysis work, it needs no accelerator, and it is
the artifact we would most want to keep.

---

## What to hand in

1. **The agent.**
2. **The verification harness** — with its tolerance and its reasoning stated.
3. **Your eval set** — the shapes and values you tested, including the hostile ones. Mandatory.
4. **The failure taxonomy** — what it got wrong, grouped, with counts.
5. **Token instrumentation** — input tokens per attempt and what they were spent on.
6. **A one-page reproduction note.**

## Where to start, in order

1. Write the reference for rungs 1–3 and the harness. **Do not call the model yet.**
2. Hand-write a correct rung-1 kernel yourself. Confirm the harness passes it, then break it on
   purpose and confirm the harness catches it *and* produces a useful message. If the harness can't
   catch your own deliberate bug, it will not catch the model's.
3. Now add the model. One call, one attempt, rung 1.
4. Add the retry loop, and only then start caring about the token budget.
5. Climb.

## Honest expectations

Rungs 1–4 are achievable. Rungs 5–7 are a good day. Rungs 8–10 are hard, and 9 is genuinely hard.

**A team that clears four rungs and brings a rigorous taxonomy of the six it failed will beat a team
claiming rung 9 that cannot show its verification.** That is not a consolation prize — it is the
grading criterion.
