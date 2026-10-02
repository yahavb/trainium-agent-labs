# Project 2 — The kernel agent

**An agent writes small programs that run directly on the chip, and keeps verifying its own output
as it goes.**

> ## STATUS: RUNNABLE, NOT YET SOLVED
>
> **Works today, verified on a trn2 node:** the ladder, the checker (`nkibench.py`), four reference
> kernels that pass it, and the agent loop (`agent.py`) writing kernels against a live model.
>
> **Not there yet:** the agent has not solved a single level. Its best score is 0.30 of 1.0 — code
> that parses, obeys the rules and runs, but computes the wrong numbers. The transcripts below show
> exactly where it stalls, and that is the problem you are being handed.
>
> **Also missing:** reference kernels for levels 5 to 7, so the optimization half of the ladder is
> unmarked; and layers 2 and 3 of the checker, so **latency cannot be measured at all yet** — every
> number here is throughput reasoning from the simulator.
>
> This is a genuinely open problem, not a tidied-up exercise with a hidden answer. If you get a level
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

## Which model, and which cores

**A NeuronCore cannot be shared by two processes** — but your chip has four logical cores and the model
server only uses two, so two are free. Check with `neuron-top` while the server runs: NC 2 and NC 3 hold
the model, NC 0 and NC 1 are idle.

**Layer 1 of this harness needs no cores at all.** It checks kernels with `nki.simulate`, on the CPU, in
seconds — which is the point, because that is where an agent should spend its hundreds of attempts
rather than waiting on compiles. So the agent runs against **the Qwen3-8B already serving on your own
instance**, and nothing has to be remote.

```bash
docker exec -it vllm bash          # wait for the prompt
git config --global --add safe.directory /workspace    # needed before any git command here
cd /workspace/projects/02-kernel-agent
export KERNEL_AGENT_BASE_URL=http://localhost:8000/v1
export KERNEL_AGENT_MODEL=Qwen/Qwen3-8B

python nkibench.py --selftest                       # prove the harness first
python nkibench.py --level 4 --check reference_level4.py
python agent.py --all --rounds 6 --samples 2 --context 4096
```

`--context 4096` matches what the server was started with; the agent caps its answer budget to fit.

### When the small model stalls, try the big one

Measured on the local Qwen3-8B: level 2 solved, the rest stalling at 0.30 to 0.62. That is the point at
which a bigger model is worth trying:

```bash
export KERNEL_AGENT_BASE_URL="«the URL the organisers give you»/agg/v1"
export KERNEL_AGENT_MODEL=gpt-oss-20b
python agent.py --all --rounds 6 --samples 1 --context 8192 --terse 1
```

`--samples 1` because sampling there is greedy, so extra samples are identical; `--terse 1` because a
long prompt makes it reason instead of answering. Report which model cleared which level and in how many
rounds — that comparison is a result in itself.

Only when you add **real on-device timing** — layers 2 and 3, which are not built — do you need cores,
and then you pin to the two the server is not using:

```bash
export NEURON_RT_VISIBLE_CORES=0,1
```

That is the documented mechanism and we have not exercised it here, so expect to debug it. If you would
rather have the whole chip for your own kernels, use the shared `gpt-oss-20b` endpoint
([`../../gptoss/`](../../gptoss/)) for the model instead and stop the local server — it runs on separate
hardware, so your chip stays entirely free. Read
[`../../gptoss/README.md`](../../gptoss/README.md) first: sampling there is greedy so retrying an
identical prompt is pointless, the `tools=` parameter does nothing, and the input limit is 8192 tokens.

## Layer 1 is built: `nkibench.py`

Everything in it runs on a CPU and needs **no Trainium device**, which is the point — it is where
the agent should spend its iterations.

```bash
python nkibench.py --selftest                              # prove the harness first
python nkibench.py --list                                  # the ladder
python nkibench.py --level 4 --show                          # what this level wants
python nkibench.py --level 4 --check reference_level4.py      # verify a kernel
python nkibench.py --roofline 4096 4096 4096                # the verdict before you write code
```

It answers three questions in order, and stops at the first failure:

1. **Does it break the rules?** A static scan: framework calls that hand over the whole
   operation (`np.mean`, `torch.matmul`, the `@` operator, `.T` on an argument), a partition
   dimension over 128, a missing `@nki.jit`, the wrong entry-point name. **NKI's own primitives
   are never flagged** — `nl.sum` over a strided view is how the pooling tutorial does it, and
   `nisa.nc_matmul` is the whole point of the matmul levels. Rejecting a correct kernel is worse
   than missing a cheat.
2. **Does it compute the right thing?** `nki.simulate_kernel` against a NumPy reference, on
   hostile shapes including ones that do not divide evenly by the tile size. The failure message
   names the element, says what fraction of the output is wrong, and says whether the error sits
   in a **ragged edge tile** or in the core arithmetic — those have different causes.
3. **Is it memory bound or compute bound?** HBM bytes are counted by wrapping `nisa.dma_copy`
   for the duration of the simulation, flops come from the shapes, and the ratio is compared
   against the ridge. So **an arithmetic intensity is measurable without a device**, which is
   what makes level 4's diagnosis available in the inner loop rather than after a compile.

`reference_level1.py` … `reference_level4.py` are the tutorial's own kernels, shipped deliberately.
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
level 4: matmul, tiled
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
  the bytes are trivial and the per-transfer issue cost is everything. Level 2 exists to teach that
  those are different problems. The count is also a correctness check on the instrumentation
  itself — 20 transfers for `K=256 M=256 N=1024` is exactly 8 inner iterations × 2 loads + 4
  result stores, so if that number is wrong, nothing below it means anything.
* **Arithmetic intensity against the ridge** — `56.9 Flops/Byte against a ridge of 222`. The
  suboptimality as a ratio rather than an opinion.
* **Fraction of the ceiling** — `26% of what the engine could sustain`. The line that reads as a
  score: three quarters of the machine is idle.
* **The remaining gap as a factor** — `3.9x more reuse needed`. This turns a verdict into a
  target, so an agent knows **when to stop** instead of optimizing forever.

**The instrumentation is the trend across levels, not any single line.** In that run level 3 sits at
18% of the ceiling and level 4's shapes at 26%, 33% and 38%. Tiling is visibly buying reuse, and
every level is still memory bound, so the ladder has not finished paying out. *That progression is
the measurement* — hand in the table, not one number.

Two caveats that run also taught us, both worth repeating to a team:

* **A silent factor of two in the byte count is worse than no measurement.** The first run
  undercounted bytes by exactly 2x (it assumed 2 bytes per element on float32 inputs), which
  *doubled* every intensity. Level 3 read 39.4 Flops/Byte when the float32 figure is 14.2. That
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
  --from-file=nkibench.py --from-file=reference_level1.py --from-file=reference_level2.py \
  --from-file=reference_level3.py --from-file=reference_level4.py \
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
> reference kernels for levels 5 to 7 in this repo. Treat them as the map, not the territory.

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

### 3. One tile only — it does not scale — *measured, level 3*

**Signal:** correct on `K=128 M=64 N=512` and nothing larger, at **19.7 Flops/Byte, 9% of the
ridge**.
**Fix:** loop over tiles in all three dimensions.
**Exposes:** redundant loads, which tiling introduces.

### 4. The same bytes crossing the bus repeatedly — *measured, level 4*

**Signal:** intensity rises with shape but stays far under the ridge. Measured: **28.4, 36.6, 42.7,
28.4 Flops/Byte — 13% to 19% of the ridge**, every shape still `MEMORY BOUND`, with
`3.0x to 3.9x more reuse needed`.
**Fix:** hoist the loads out of the innermost loop — the same tiles are being re-read every pass.
**Exposes:** that hoisting only reuses one row of tiles while SBUF holds far more.

### 5. SBUF underused — reuse limited to one row of tiles — *expected, level 5 → 6*

**Signal:** intensity up from row 4 but still `MEMORY BOUND`, and the byte count still far above
what the operands themselves weigh.
**Fix:** block the M and N dimensions to keep more tiles resident.
**Exposes:** a **search space** rather than a derivation — block sizes bounded by SBUF capacity. The
agent has to explore now, and the log's `Nx more reuse needed` is what tells it when to stop.

### 6. K still streamed — *expected, level 6 → 7*

**Signal:** intensity close to but under the ridge, and shape-dependent.
**Fix:** block K as well.
**Exposes:** the ridge, and with it the end of the throughput story.

### 7. At the ridge and still slow on one call — *expected, and invisible to layer 1*

**Signal:** nothing in this harness. Arithmetic intensity is at the ridge and the kernel is still
slow per invocation. **Only a profile shows it**: DMA busy and the Tensor Engine idle at the same
moment, meaning loads and compute take turns instead of overlapping.
**Fix:** double-buffer, so a load for the next tile is in flight during the current compute.
**Exposes:** that the blocked kernel may now be *slower on small shapes* than level 5 was, because it
pays for reuse it never gets to exploit. That is the latency-versus-throughput tension, and it is a
finding to report, not a regression to hide.

### 8. Fast and wrong — *the gate, at every level*

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

## Adding your own operation

**The ladder is a starting point, not the assignment.** The matmul levels exist because the tutorial
supplies reference answers for them, which makes the harness verifiable. Once you trust it, point it at
whatever you actually care about.

Adding an operation is **one reference, one input builder, and one `level(...)` call.** Nothing else in
the harness needs to know. Level 8 in `nkibench.py` is that worked example — single-head attention — and
it was added by writing exactly these three things:

```python
def ref_attention(q, k, v):                    # 1. the truth, in NumPy
    d = q.shape[1]
    scores = (q @ k.T) / np.sqrt(d)
    scores = scores - scores.max(axis=-1, keepdims=True)   # or exp() overflows
    e = np.exp(scores)
    return (e / e.sum(axis=-1, keepdims=True)) @ v

def _args_attention(spec, r):                  # 2. how to build inputs
    n, d = spec["seq"], spec["dim"]
    return tuple(r.standard_normal((n, d)).astype(np.float32) for _ in range(3))

level(8, "single-head attention", "nki_attention_",       # 3. register it
      "what it teaches", "what there is to optimize",
      ref_attention,
      [dict(seq=128, dim=64), dict(seq=64, dim=128), dict(seq=96, dim=32)],
      {"softmax", "attention", "matmul", "einsum"},        # framework calls that would cheat
      make_args=_args_attention,
      label=lambda sp: f"seq={sp['seq']} dim={sp['dim']}")
```

You immediately get, for free: the static rule scan, CPU simulation against your reference, hostile
shapes, the input-mutation check, HBM bytes and transfer counts, traffic against the byte floor, the
intensity ceiling, and the agent loop. `python agent.py --level 8` works with no further changes.

**Three things to get right when you write one:**

* **Ban the framework call that does the whole job**, or the model will just call it. Ban `softmax` and
  `matmul`, not the NKI primitives — banning `nl.sum` would reject correct kernels.
* **Include a shape that does not divide evenly** by 128. Most generated kernels are right in the
  interior and wrong on the last partial tile.
* **Make the reference obviously correct, and say where the trap is.** Attention's is that a naive
  `exp()` overflows: a kernel that skips the max subtraction looks fine on small test data and returns
  NaN on real data. That property — wrong but plausible — is what makes an operation worth putting in
  this harness at all.

**Why attention is the natural next one.** It is a matmul, then a numerically stable softmax, then a
second matmul, and the intermediate scores matrix is `seq × seq`. Writing that intermediate out to HBM
and reading it back is the mistake that dominates everything else, which is exactly why fused attention
kernels exist. The traffic measurement already in the harness will show it: compare bytes moved against
the floor and the fused version separates from the naive one immediately.

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
   different fixes, and level 2's cost is the count.
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
* Which levels are held back for judging.
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
python kernelbench.py --list            # the ten-level ladder
python kernelbench.py --level 1 --show
python kernelbench.py --level 1 --check my_kernel.py
python try_level.py --level 5             # drive the model at one level
```

**Start here even when the on-device harness exists.** The agent architecture is identical and the
feedback loop is a thousand times faster.

The challenge document also records the single most useful thing we learned driving a model at this:
specify the constraints in the prompt and it produces nothing; omit them and it produces something
confident and illegal. What worked was generating naively, letting the **verifier** find the violation,
and sending back one surgical instruction naming only the change. **Constraints belong in your
verifier, not in your generation prompt.** Project 1 hit the same wall three more times.
