# Project 2 — The kernel agent

**An agent writes small programs that run directly on the chip, and keeps verifying its own output
as it goes.**

> ## STATUS: RUNNABLE, NOT YET SOLVED
>
> **Works today, verified on a trn2 node:** the ladder, the checker (`nkibench.py`), four reference
> kernels that pass it, and the agent loop (`agent.py`) writing kernels against a live model.
>
> **Not there yet:** the agent has not solved a single rung. Its best score is 0.30 of 1.0 — code
> that parses, obeys the rules and runs, but computes the wrong numbers. The transcripts below show
> exactly where it stalls, and that is the problem you are being handed.
>
> **Also missing:** reference kernels for rungs 5 to 7, so the optimization half of the ladder is
> unmarked; and layers 2 and 3 of the checker, so **latency cannot be measured at all yet** — every
> number here is throughput reasoning from the simulator.
>
> This is a genuinely open problem, not a tidied-up exercise with a hidden answer. If you get a rung
> to 1.0, you have done something nobody here has.

---

## The idea

Linear algebra is where the time goes on an AI chip. A matrix multiply, a normalisation, a softmax —
each is one line in PyTorch and underneath it someone wrote a **kernel**: the explicit tiles of data
moved into on-chip memory, the work issued to the right engine, the reuse that avoids paying for the
same byte twice.

Writing one is specialist work, and getting it subtly wrong is worse than getting it obviously wrong,
because **nothing crashes**. A wrong kernel returns plausible numbers.

So: build the agent that writes them, runs them, checks the numbers against a reference, reads its own
failure, and tries again. Same loop as Project 1, different checker — and here the checker has to run
on the hardware.

## The constraint that shapes this project

**A Neuron device cannot be shared by two processes.**

Your agent needs the chip, to compile and run each candidate kernel and time it. So **vLLM cannot be
running on your chip at the same time** — the two cannot coexist.

That means the model driving your agent has to come from somewhere else:

> **Use the shared `gpt-oss-20b` endpoint** ([`../../gptoss/`](../../gptoss/)). It runs on separate
> hardware, so your own chip stays free for kernels.

This is the opposite arrangement from Project 1, where the chip serves the model and the agent needs no
hardware at all. Getting this the wrong way round costs you an afternoon, so settle it before you write
code.

Read [`../../gptoss/README.md`](../../gptoss/README.md) first. That endpoint is shared by the whole
room, sampling is greedy so **retrying an identical prompt is pointless**, the `tools=` parameter does
nothing, and the input limit is 8192 tokens. All measured.

## The ladder

Three operations from the [NKI tutorials](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/tutorials/index.html),
ordered so that both the **difficulty** and the **amount of optimization available** go up together.
Early rungs have almost nothing to optimize, which is the point: the agent has to learn to write
*legal* NKI before it can write *fast* NKI.

### Tier A — can the agent write a correct kernel at all?

**Rung 1. Average pooling 2D.** Reduce `C × [H, W]` down over both spatial axes. Pure reduction, no
data reuse to exploit, and its arithmetic intensity is inherently tiny — there is essentially nothing
to optimize, so a rung the agent either passes or fails. What it teaches is the programming model: the
**partition axis is not like the others**, and multi-dimensional access patterns have to be written
explicitly. This is the "the loop works" checkpoint.

**Rung 2. 2D transpose.** Still no arithmetic — only data movement. Teaches layout: a transpose has to
cross partitions, and the partition axis is the constrained one. **The first optimization appears
here**, and it is not about compute at all: a naive transpose moves tiny pieces and pays a
per-transfer issue cost, so the cost is the **number of transfers, not the number of bytes**. An agent
that reports "it is slow because it is moving a lot of data" has misdiagnosed it.

### Tier B — the roofline, and a number the agent can compute

**Rung 3. Matrix multiplication, single tile.** `64(M) × 128(K) × 512(N)` on the Tensor Engine.
Teaches the machine: results land in **PSUM** and have to be copied to **SBUF**; the left operand
arrives **already transposed** (`lhsT`) because of how the engine consumes partitions; tiles have hard
shape limits. Mostly a correctness rung, but one where the layout rules bite.

#### The roofline, in one paragraph

A chip can only do two things: move bytes and do arithmetic. Each has a ceiling — a peak memory
bandwidth and a peak Flops rate — and **whichever ceiling you hit first is the one that limits you**.
So for any kernel, ask how much arithmetic it does per byte it reads from memory. That ratio is its
**arithmetic intensity**, in Flops per byte. Low intensity means the kernel is starved: the engine sits
idle waiting for data, and it is **memory bound**. High intensity means the data keeps up and the
engine is the limit, so it is **compute bound**. The crossover is a property of the hardware, not of
your code: divide peak Flops by peak bandwidth and you get the intensity at which the two ceilings
meet. Plot achievable performance against arithmetic intensity and you get a line rising with slope
equal to bandwidth, then flattening at peak Flops — the shape is why it is called a roofline.

Why it matters for an agent: **it tells you which half of the kernel to fix, before you change
anything.** Memory bound means find reuse, so the same bytes do more work. Compute bound means the
loads are already keeping up and tuning them is wasted effort. Guessing this wrong is the most common
way to spend a day optimizing the wrong thing.

**Rung 4. Tiled matrix multiplication.** Now it works for matrices larger than one tile — and it is
**measurably memory bound**. This is the most important rung in the ladder, because the diagnosis is
*derivable rather than guessed*:

> To saturate the Tensor Engine on NeuronCore-v2 in `bfloat16`, a kernel needs an arithmetic intensity
> of **222 Flops/Byte**. The tiled kernel's inner loop reads 160 KB from HBM per 16 MFlops of work,
> which is **102** — well under the threshold. So it is memory bound, and no amount of tuning the
> compute will help.

The agent should compute that ratio **from its own code**, classify the kernel, and only then act. A
team whose agent arrives at "memory bound, here is the arithmetic intensity, here is why" has built
something genuinely useful.

### Tier C — the optimization search

Each of these raises arithmetic intensity by finding more reuse. They are the four kernels the tutorial
itself ships, in order.

**Rung 5. Hoist the redundant loads.** The same tiles are re-read on every pass of the inner loop.
Move the loads out. Cheap, mechanical, and the first measurable win.

**Rung 6. Block the M and N dimensions.** Hoisting reuses one row of tiles; SBUF holds far more than
that. Spend the capacity on reuse. Now there is a **search space** — block sizes, bounded by SBUF
capacity — and the agent has to explore it rather than derive it.

**Rung 7. Block M, N and K.** The fully optimized version. Hard, and a fine place to stop short.

---

## Latency and throughput are different problems

Everything above is a **throughput** story. Arithmetic intensity and the roofline describe sustained
Flops per second on a large matrix, and they say **nothing about how long one call takes**.

That distinction is the most interesting thing in this project, because the two can pull in opposite
directions:

| | what it wants | what it measures |
|---|---|---|
| **Throughput** | reuse — bigger blocks, more data resident in SBUF | Flops/second sustained over a large matrix |
| **Latency** | overlap — loads happening while compute happens, short dependency chains | wall time of a single invocation |

Blocking raises throughput and **adds setup cost and a larger working set**. On a large square matmul
rung 7 should win. On a **small or skinny** shape — a single short sequence, `M` of 8 or 32 — the
blocked kernel may well be *slower* than rung 5, because it pays for reuse it never gets to exploit.

**So run the ladder at two shapes and ask which metric you are optimizing:**

* a **large** matmul, where throughput dominates and the roofline applies;
* a **small or skinny** matmul, where latency dominates and the roofline is irrelevant.

Two consequences worth stating plainly, because they are what a judge will push on:

* **A kernel can be roofline-optimal and have bad latency.** If loads and compute take turns instead
  of overlapping, the engine idles between tiles. Arithmetic intensity cannot see that; only a profile
  can, by showing DMA busy and the Tensor Engine idle at the same moment.
* **A faster kernel need not make the model faster.** If the operation you sped up was 3 percent of
  the whole, you have bought 3 percent at best. Say what fraction you were working on.

**The honest deliverable is a table**, not a single number: each rung, at each shape, with its latency,
its throughput, and its arithmetic intensity — and a sentence on which one you would ship and why.
Reporting that rung 7 lost on the small shape is a *result*, not a failure.

---

## The checker, in three layers

Same principle as Project 1: **the quality of the error message is the quality of the agent.** Here it
has three levels, and they cost wildly different amounts of time.

| layer | what it catches | cost | needs the chip |
|---|---|---|---|
| **1. Simulator** — `nki.simulate_kernel` | wrong logic, illegal shapes, bad access patterns | seconds | no |
| **2. Device** — `nki.baremetal` | real latency, and anything the simulator models loosely | a compile | yes |
| **3. Profile** | *why* it is slow — which engine owns the time, whether DMA and compute overlap | a profiled run | yes |

Plus a numerical comparison against the PyTorch reference, which is the gate: **a fast kernel with
wrong output scores zero.**

> **Put the simulator in the agent's inner loop.** Get the kernel *correct* in the simulator, then
> compile and run it on the chip to find out how *fast* it is. Correctness questions go to layer 1,
> performance questions to layers 2 and 3. An agent that compiles on every attempt will manage a
> handful of iterations all day; one that simulates will manage hundreds. **This is the single biggest
> design decision in the project.**

## Layer 1 is built: `nkibench.py`

Everything in it runs on a CPU and needs **no Trainium device**, which is the point — it is where
the agent should spend its iterations.

```bash
python nkibench.py --selftest                              # prove the harness first
python nkibench.py --list                                  # the ladder
python nkibench.py --rung 4 --show                          # what this rung wants
python nkibench.py --rung 4 --check reference_rung4.py      # verify a kernel
python nkibench.py --roofline 4096 4096 4096                # the verdict before you write code
```

It answers three questions in order, and stops at the first failure:

1. **Does it break the rules?** A static scan: framework calls that hand over the whole
   operation (`np.mean`, `torch.matmul`, the `@` operator, `.T` on an argument), a partition
   dimension over 128, a missing `@nki.jit`, the wrong entry-point name. **NKI's own primitives
   are never flagged** — `nl.sum` over a strided view is how the pooling tutorial does it, and
   `nisa.nc_matmul` is the whole point of the matmul rungs. Rejecting a correct kernel is worse
   than missing a cheat.
2. **Does it compute the right thing?** `nki.simulate_kernel` against a NumPy reference, on
   hostile shapes including ones that do not divide evenly by the tile size. The failure message
   names the element, says what fraction of the output is wrong, and says whether the error sits
   in a **ragged edge tile** or in the core arithmetic — those have different causes.
3. **Is it memory bound or compute bound?** HBM bytes are counted by wrapping `nisa.dma_copy`
   for the duration of the simulation, flops come from the shapes, and the ratio is compared
   against the ridge. So **an arithmetic intensity is measurable without a device**, which is
   what makes rung 4's diagnosis available in the inner loop rather than after a compile.

`reference_rung1.py` … `reference_rung4.py` are the tutorial's own kernels, shipped deliberately.
The tutorials are public, so hiding them buys nothing, and a harness whose reference nobody can read
is a harness nobody should trust. Use them to confirm the harness works, then write your own.

### What it looks like when you run the agent

Real output from `agent.py` against Qwen3-8B, two attempts per round. **Read this before running
anything.** Nothing here is solved yet, and that is the honest state of the project — but the wall
moved on every fix, and watching *which* wall you are at is the skill.

**Run 1 — nothing loadable came back.**

```
round 0: rewards [0.0, 0.0]  best 0.00  (54.8s)
  The code does not parse: invalid decimal literal on line 2.
round 3: rewards [0.1, 0.1]  best 0.10  (13.3s)
  Rule violations: `tensor_avgpool_kernel` is not decorated with `@nki.jit`
```

Two bugs, both in the harness rather than the model. Thinking mode was on, so the model spent its
whole token budget reasoning and returned a fragment — note the 54.8s rounds. And the code extractor
handed prose to the compiler, so a numbered list became "invalid decimal literal".

**Run 2 — code runs, but every NKI function is invented.**

```
round 0: rewards [0.3, 0.3]  best 0.30  (9.0s)
  raised AttributeError: module 'nki.language' has no attribute 'dot'
round 1: raised AttributeError: module 'nki.language' has no attribute 'value'
round 2: raised AttributeError: module 'nki.language' has no attribute 'sbuf_scalar'
```

Rounds dropped to 5–12 seconds, so the truncation was fixed. `nl.dot`, `nl.value`,
`nl.sbuf_scalar`, `tile.mean` — none exist. **And it guessed a different fake name every round**,
because the feedback named the mistake and never the fix. That is this repo's recurring lesson,
arriving for the fourth time.

**Run 3 — the names are real, one specific misuse remains.**

```
round 0: rewards [0.3, 0.3]  best 0.30  (9.0s)
  raised TypeError: 'MemoryRegion' object is not callable
round 3: rewards [0.3, 0.3]  best 0.30  (7.8s)
  raised TypeError: 'MemoryRegion' object is not callable
```

The API card fixed the invented names. Now it writes `nl.sbuf(shape, dtype)` — calling a memory
region as if it were a function — instead of `nl.ndarray(..., buffer=nl.sbuf)`. The prompt described
the right call and the model read past it, so the prompt now carries a **complete worked kernel**
instead, which is what fixed the same class of problem in Project 1.

**How to read your own run:**

| you see | it means |
|---|---|
| `0.0` and rounds near 55s | the answer was truncated — check for the TRUNCATED notice |
| `0.1` | it parses but breaks a rule; read exactly which |
| `0.3` | rules clean and it ran, but the numbers are wrong — now you are doing real work |
| `1.0` | correct on every shape, and the roofline verdict prints |
| the **same** error three rounds running | your feedback is a verdict, not an instruction. Fix the message, not the prompt. |
| identical rewards within a round | sampling is off, so there is nothing to choose between |

### How suboptimality shows up in the log

This is what a `--check` prints, from a real run on a trn2 node:

```
rung 4: matmul, tiled
  rules      clean
  numerics   4/4 shapes passed

  K=128 M=128 N=512: 294,912 HBM bytes in 3 transfers
    MEMORY BOUND: 56.9 Flops/Byte against a ridge of 222, so 26% of what the engine
    could sustain. The engine is idle waiting for data. Find reuse -- make the same
    bytes do more work -- rather than tuning the arithmetic. 3.9x more reuse needed.

  K=256 M=256 N=1024: 1,835,008 HBM bytes in 20 transfers
    MEMORY BOUND: 73.1 Flops/Byte against a ridge of 222, so 33% of what the engine
    could sustain. ... 3.0x more reuse needed.
```

**Four numbers per shape carry the diagnosis, and none of them is a timing:**

* **Bytes moved, and the transfer count** — `294,912 HBM bytes in 3 transfers`. Bytes catch a
  kernel re-reading the same data. The count catches the opposite failure: many tiny moves, where
  the bytes are trivial and the per-transfer issue cost is everything. Rung 2 exists to teach that
  those are different problems. The count is also a correctness check on the instrumentation
  itself — 20 transfers for `K=256 M=256 N=1024` is exactly 8 inner iterations × 2 loads + 4
  result stores, so if that number is wrong, nothing below it means anything.
* **Arithmetic intensity against the ridge** — `56.9 Flops/Byte against a ridge of 222`. The
  suboptimality as a ratio rather than an opinion.
* **Fraction of the ceiling** — `26% of what the engine could sustain`. The line that reads as a
  score: three quarters of the machine is idle.
* **The remaining gap as a factor** — `3.9x more reuse needed`. This turns a verdict into a
  target, so an agent knows **when to stop** instead of optimizing forever.

**The instrumentation is the trend across rungs, not any single line.** In that run rung 3 sits at
18% of the ceiling and rung 4's shapes at 26%, 33% and 38%. Tiling is visibly buying reuse, and
every rung is still memory bound, so the ladder has not finished paying out. *That progression is
the measurement* — hand in the table, not one number.

Two caveats that run also taught us, both worth repeating to a team:

* **A silent factor of two in the byte count is worse than no measurement.** The first run
  undercounted bytes by exactly 2x (it assumed 2 bytes per element on float32 inputs), which
  *doubled* every intensity. Rung 3 read 39.4 Flops/Byte when the float32 figure is 14.2. That
  error runs in the dangerous direction: it makes a starved kernel look closer to compute bound
  than it is, and sends you to optimize the wrong half. Check your instrumentation against a
  number you worked out by hand before you trust it.
* **Match the dtype to the ridge.** 222 Flops/Byte is the published *bfloat16* figure. The shapes
  here are float32, so the tool labels its own verdict indicative rather than quoting it. A
  measurement compared against a ceiling for a different dtype is not a like-for-like number.

**What is deliberately absent: latency.** Everything above is throughput reasoning and needs no
device. Wall time, and whether loads overlap compute, are layers 2 and 3 — and no amount of
arithmetic intensity can see them.

### Run it on the cluster

Layer 1 claims **no Neuron device**, so it can run beside a model server without fighting it:

```bash
kubectl create configmap nkibench-code \
  --from-file=nkibench.py --from-file=reference_rung1.py --from-file=reference_rung2.py \
  --from-file=reference_rung3.py --from-file=reference_rung4.py \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f ../../k8s/nkibench-job.yaml
kubectl logs -f job/nkibench
```

That job prints the SDK's own tile-size constants, runs the selftest, and puts all four reference
kernels through the harness.

### What is verified, and what is not

The rule checker, the references, the shape generators, the failure messages and the roofline
arithmetic are all exercised by `--selftest` on any machine. **The roofline model reproduces the
tutorial's published figures exactly** — 160 KB per 16.8 MFlops, an arithmetic intensity of 102.4
against the ridge of 222 — which is the check that matters, since the whole diagnosis rests on it.

Not yet verified: `simulate_and_count()` imports `nki`, so it has never run outside a Neuron
environment. Its byte counting patches `nisa.dma_copy`, which is an assumption about how the
kernels call it. **The cluster job above is what confirms it**, and its selftest says so out loud.

## The suboptimalities, in the order the agent meets them

Each one has a **signal** the harness prints, a **fix**, and a **next** suboptimality that the fix
exposes. That last column is the point: fixing one does not finish the job, it moves the bottleneck.

> **Measured vs expected.** Rows 1 to 4 are measured on a trn2 node. Rows 5 onward are **expected**
> from the tutorial's structure — nobody has run an agent on this ladder yet, and there are no
> reference kernels for rungs 5 to 7 in this repo. Treat them as the map, not the territory.

### 1. It does not compile or run at all — *measured*

**Signal:** the rule scan fires, or the simulator raises. `line 5: partition dimension 256 exceeds
the maximum of 128`, or `no function named nki_matmul_tiled_ is defined`.
**Fix:** tile to 128 partitions, name the entry point, decorate with `@nki.jit`.
**Exposes:** whether it is correct, which is a different question.

### 2. It is correct in the middle and wrong at the edge — *expected, and the harness is built for it*

**Signal:** `NUMERICAL MISMATCH ... this is in the final partial PARTITION tile — the ragged edge is
the likely cause, not the core arithmetic`.
**Fix:** handle the last, partial tile separately.
**Exposes:** performance, once the thing is finally right.

### 3. One tile only — it does not scale — *measured, rung 3*

**Signal:** correct on `K=128 M=64 N=512` and nothing larger, at **19.7 Flops/Byte, 9% of the
ridge**.
**Fix:** loop over tiles in all three dimensions.
**Exposes:** redundant loads, which tiling introduces.

### 4. The same bytes crossing the bus repeatedly — *measured, rung 4*

**Signal:** intensity rises with shape but stays far under the ridge. Measured: **28.4, 36.6, 42.7,
28.4 Flops/Byte — 13% to 19% of the ridge**, every shape still `MEMORY BOUND`, with
`3.0x to 3.9x more reuse needed`.
**Fix:** hoist the loads out of the innermost loop — the same tiles are being re-read every pass.
**Exposes:** that hoisting only reuses one row of tiles while SBUF holds far more.

### 5. SBUF underused — reuse limited to one row of tiles — *expected, rung 5 → 6*

**Signal:** intensity up from row 4 but still `MEMORY BOUND`, and the byte count still far above
what the operands themselves weigh.
**Fix:** block the M and N dimensions to keep more tiles resident.
**Exposes:** a **search space** rather than a derivation — block sizes bounded by SBUF capacity. The
agent has to explore now, and the log's `Nx more reuse needed` is what tells it when to stop.

### 6. K still streamed — *expected, rung 6 → 7*

**Signal:** intensity close to but under the ridge, and shape-dependent.
**Fix:** block K as well.
**Exposes:** the ridge, and with it the end of the throughput story.

### 7. At the ridge and still slow on one call — *expected, and invisible to layer 1*

**Signal:** nothing in this harness. Arithmetic intensity is at the ridge and the kernel is still
slow per invocation. **Only a profile shows it**: DMA busy and the Tensor Engine idle at the same
moment, meaning loads and compute take turns instead of overlapping.
**Fix:** double-buffer, so a load for the next tile is in flight during the current compute.
**Exposes:** that the blocked kernel may now be *slower on small shapes* than rung 5 was, because it
pays for reuse it never gets to exploit. That is the latency-versus-throughput tension, and it is a
finding to report, not a regression to hide.

### 8. Fast and wrong — *the gate, at every rung*

**Signal:** numerics fail while the intensity looks good.
**Fix:** none — it scores zero. Re-verify every candidate automatically, before its timing counts.

---

### And the suboptimalities of the *agent*, which are the ones that cost the day

Every one of these was measured in [Project 1](../01-heat-rod-pde/) or in the earlier kernel work
in this repo. They are not hypothetical.

| the agent's failure | what the log looks like | the fix that worked |
|---|---|---|
| Answers in notation, not code | `Could not read the expression 'X(x)T(t)'` | a **worked example** of an acceptable answer, not a list of prohibitions |
| Audits itself against your rules and never answers | empty content, more thinking at a bigger budget | move the constraints **out of the prompt and into the verifier** |
| Retries identically and expects a different result | byte-identical answers | change the **prompt**, not the dice — sampling may be greedy |
| Guesses instead of computing | plausible patterns that do not converge; the same answer twice | give it a **tool** it aims itself |
| Copies the answer out of your feedback | it "solves" it, having derived nothing | report **direction only**, never the target value |
| Reports "verified" on a kernel that fails | — | scored as worse than an honest failure |
| Fixes one thing and breaks another | the score **drops** while understanding rises | expect it; weight your reward knowing it happens |

## Hints, in the order they will save you time

1. **Put the simulator in the inner loop.** Correctness to layer 1, performance to layers 2 and 3.
   An agent that compiles on every attempt gets a handful of iterations all day; one that simulates
   gets hundreds. This is the single biggest design decision here.
2. **Run `--selftest` and read what it proves before you trust a score.** It caught a mislabelled
   reference, a rule that rejected correct kernels, and a 2x byte error — all in this harness,
   before it graded anything of yours.
3. **Compute the arithmetic intensity before you change code.** Memory bound means find reuse.
   Compute bound means the loads already keep up and tuning them buys nothing. Guessing this is the
   most common way to lose a day.
4. **Diagnose the transfer count and the byte count separately.** They are different failures with
   different fixes, and rung 2's cost is the count.
5. **Constraints belong in your verifier, not in your generation prompt.** Measured, twice, in this
   repo: a model given a list of rules audits itself and returns nothing; a model given none writes
   something confident and illegal. Generate freely, let the checker catch it, then send back **one
   named change**.
6. **A verdict is not an instruction.** "It is slow" and "off by 341 percent" are both true and both
   useless. "The sin(pi\*x) term should not be there at all" names the fix. Project 1 hit this three
   separate times, and each time the fix was a better error message rather than a better model.
7. **Do not print the answer in your feedback.** Project 1 measured this: told the target
   coefficients, the model copied them verbatim and derived nothing. Say *which* thing is wrong and
   in *which direction*, not what it should be.
8. **Give the model a tool instead of a hint.** It could not do the integrals, and directional
   feedback alone just made it guess. A calculator it aims itself fixed that — the model decides
   *what* to compute, which is the reasoning worth measuring, and sympy does the arithmetic.
9. **Expect the failure to move rather than vanish.** When the coefficients came right, the decay
   rates broke. A score can even go **down** while understanding goes up, if the newly-broken part
   carries more weight than the newly-fixed one.
10. **Check the ragged edge first.** Most generated kernels are correct in the interior and wrong in
    the final partial tile. The harness tells you which of the two you are looking at — believe it.
11. **One run is not a result.** At four samples these numbers move between runs. Report how many
    runs, and what the spread was.
12. **Say which numbers came from the simulator and which from the device.** A judge will ask, and
    "we did not get it onto the chip" is a fine answer *if you say it first*.

## What still has to be built

* The harness: reference implementations, the three-layer checker, and failure messages an agent can
  act on rather than a verdict it cannot.
* Which rungs are held back for judging.
* Whether the arithmetic-intensity calculation is given to the agent or expected from it. Giving it is
  the difference between a tool and a hint — Project 1 measured exactly this trade-off and found that
  handing over computed values stops the model from deriving anything.

## What is ready now: the laptop version

[`CHALLENGE-kernel-agent.md`](CHALLENGE-kernel-agent.md) is the full write-up, and
[`kernelbench.py`](kernelbench.py) is a working harness for it. It asks for kernels written in **NumPy
under kernel-shaped rules** — fixed tiles, explicit loops over them, no whole-array shortcuts, and
ragged final tiles that must be handled. Those rules are what make kernel writing hard and they are
all checkable on a CPU in milliseconds, so the lesson transfers.

```bash
pip install numpy
python kernelbench.py --selftest        # proves it catches planted bugs
python kernelbench.py --list            # the ten-rung ladder
python kernelbench.py --rung 1 --show
python kernelbench.py --rung 1 --check my_kernel.py
python try_rung.py --rung 5             # drive the model at one rung
```

**Start here even when the on-device harness exists.** The agent architecture is identical and the
feedback loop is a thousand times faster.

The challenge document also records the single most useful thing we learned driving a model at this:
specify the constraints in the prompt and it produces nothing; omit them and it produces something
confident and illegal. What worked was generating naively, letting the **verifier** find the violation,
and sending back one surgical instruction naming only the change. **Constraints belong in your
verifier, not in your generation prompt.** Project 1 hit the same wall three more times.
