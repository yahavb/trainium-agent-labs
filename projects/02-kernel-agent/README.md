# Project 2 — The kernel agent

**An agent writes small programs that run directly on the chip, and keeps verifying its own output
as it goes.**

> ## STATUS: NOT FINALISED
>
> The idea and the constraint below are settled. The on-device harness is not written yet, so treat
> this as a direction rather than a specification, and expect the details to change before the event.
> What *is* ready today is the laptop version described at the bottom — `kernelbench.py` works and
> proves itself.

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
