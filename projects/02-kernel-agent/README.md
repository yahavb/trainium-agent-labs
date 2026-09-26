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

## What still has to be decided

* **Which operations.** A ladder from an elementwise fused op up to a tiled matmul, with the harder
  rungs optional. Held-back operations for judging.
* **The on-device harness** — compile, run, compare against a PyTorch reference, and report a failure
  an agent can act on. This is the missing piece.
* **Whether timing counts, or only correctness.** Correctness is a gate either way; a speedup with
  wrong output is worth nothing.
* **The simulator path.** Kernel correctness is debuggable on the host CPU in seconds, without a
  device and without a compile. If that works here, the intended loop is write → simulate → fix the
  logic → *then* compile and run on the chip for real timings. A team that uses the chip as its
  debugger will spend the day watching a progress bar.

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
