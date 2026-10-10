# NOTE — one page: what we ran, on what, and what came out

**Team seat 73 (+ teammates' seats 70, 71, 72, 74), Hack the Chip, NYU × Annapurna Labs.**
Every number below is a rate over *n* runs, tagged **[sim]** (nki 0.6 CPU simulator) or **[device]** (Trainium2).

## The claim, and the evidence

> A small model on a chip we control, plus a checker that turns every verdict into **one named change**, solves
> problems the model cannot solve alone. The model never changed during the day; only the checker's messages did.

**Model:** Qwen3-8B, vLLM-Neuron 0.24, TP=2, context 8192, one Trainium2 chip per seat (`serve.sh` defaults).
**Stage B ladder (`nkibench.py`, real NKI kernels):**

| | L1 pool | L2 transpose | L3 matmul | L4 tiled | L5 hoist | L6 block MN | L7 block MNK | L8 attention |
|---|---|---|---|---|---|---|---|---|
| upstream (STATE.md) | 0/5 | 4/5 | 0/5 | 0/5 (0.62) | no reference | no reference | no reference | no reference |
| **our baseline** (upstream agent) [sim] | 0/5 | 0/5 | 0/5 | 0/5 | — | — | — | — |
| **final agent C10** [sim] | **5/5** (pooled 17/20) | **5/5** | **5/5** | **5/5** | **5/5** | **5/5** | **5/5** | 0/5 |
| rounds to solve | 2 | 1 | 2 | 2 | 4 | 3–4 | 3 | — |

**What moved each level was one sentence of feedback**, and each sentence was A/B tested against a control:

| level | the verdict upstream gave | what our checker says instead | control → treatment |
|---|---|---|---|
| L3/L4 | out of bounds / "most elements are wrong" | "the slice `lhsT[m0:…, k0:…]` lists the axes in the wrong order; write `lhsT[k0:k0+k_sz, m0:m0+m_sz]`" | 0/5 → 5/5 |
| L1 | 1-D tile / partition dim collapsed | "reduce each window into a (C,1) tile with tensor_reduce, then tensor_copy it into a column" | 0/5 → 5/5 |
| L5 | "hoist the operand loads out of the innermost loop" | "the lhsT load is inside the n loop but does not depend on n; move it out" | **0/5 → 5/5** (same seat, interleaved) |
| L7 | "block K as well" | "load every operand tile exactly once and keep it in SBUF" | **0/5 → 5/5** |
| L6 | (after load-once) upstream's L6 sentence pulled the model to a k-outer loop | L6 gets L7's sentence (or: "the k loop is outside m and n…") | 0/5 → 5/5 (both fixes) |

A **misleading** verdict was worse than a vague one: told "usually an uninitialised tile" when it had simply not
copied its inputs, the model invented `tile.fill()` and `nisa.psum_zero` chasing the wrong cause (LOG 13:58).

## Verification, calibration, honesty

* **Held-out hostile cases** (`heldout.py`: ragged K=96 M=200 N=700, M=1, prime K=131, ×1e4, constant rows) [sim]:
  every agent-written kernel it claimed solved passes all of them; **every official AWS tutorial matmul kernel
  (L4–L7) fails them**, and the verbatim L6 tutorial kernel returns *silently wrong* numbers at K=131.
* **Calibration** (`CALIBRATION.md`): the agent attaches a confidence to every SOLVED claim. All **31** agent claims
  at 0.95 pass held-out (one reference kernel also sits at 0.95 and passes); shape-asserting references at 0.38 pass 1 of 5. **Brier 0.030** (n=38 incl. references).
* **[device]** (seat-70, logical core 2): agent-written **L3, L4, L5** kernels and the AWS reference L4 run on the
  Trainium2 chip and match NumPy (max error/RMS 1e-6). The agent's **first L7 kernel was rejected by the device
  compiler** (a list comprehension the simulator accepts), so we added that as a static rule (`--device-rules`);
  the C10 L7 kernels use plain loops. **No kernel latency is claimed**: per-call wall clock (~1.6 s) is host-bound,
  and device-side timing was not reached within its time box.
* **The 354 attempts behind the main decisions were re-graded from their own code** (`regrade.py`): 0 mismatches. That check found two of our own
  tool bugs (bytecode-cache staleness, missing flags), both fixed and logged.
* **Repeats are not independent**: the server decodes deterministically, so repeats of one prompt often follow
  the same path (Stage A: always byte-identical, effectively **n=1** per level). Stage B baseline repeats did
  differ (batching), so its n=5 carries some, not full, variance.

## What did not work (measured, reported)

* E2 `--retrieve` (AWS docs keyed by failure): 0 effect on any level, reverted.
* E3 `--mech` (AST fixes): dropped on data, 0 of 145 attempts would have changed.
* Upstream's "chunks of 128" style hints, sampling-parameter tuning: moot, the server ignores temperature.
* Global prompt edits: a fix written for L8 changed L1's prompt and cost 2/5 L1 runs; scoped to L8 it recovered.
* **L8 attention: 0/5** after four rounds of root-cause analysis (v8; v9, a static signature lint that lists every
  API misuse at once; v10, which fixed upstream's API card offering only `nl.sum`). Runs now reach the softmax and
  loop there. v10 also cost L1 runs and was reverted.
* **Spread.** L1 is the least robust level: 17/20 pooled over every final-family run, 2/5 in one batch on another
  seat. Other levels were 5/5 in every final-family batch.

## Stage A (NumPy under kernel rules, `kernelbench.py`, 10 levels) [CPU]

Final configuration (`stagea --v2 --v3 --v4-smart`): **9/10** (all but A5, softmax). Repeats are byte-identical
(deterministic decoding), so each level is effectively n=1. v4's accumulator renaming solved A8 but broke A3/A4
when applied everywhere; applied only when a kernel has two or more reductions, it keeps both (RESULTS.md).
We declined to paste a full softmax kernel into the feedback, which would have "solved" A5 for the model. Harness fixes: mixed tolerance derived from dtype and
accumulation length (a correct float32 matmul passed 5/10 under the old bare 1e-4), spatial error localisation.

**Reproduce:** `REPRODUCE.md`. **Everything that happened, in order:** `LOG.md`.
