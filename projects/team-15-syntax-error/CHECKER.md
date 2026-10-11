# CHECKER — what it accepts, what it rejects, and why

The checker is the product. Over the day it took one unchanged model (Qwen3-8B) from 0 of 4 to 7 of 8 levels
solved in 5/5 runs **[sim]**. It has two jobs, and they are different jobs:

1. **Decide** whether a kernel is correct: a verdict that must never pass a wrong kernel, and should not fail a
   right one.
2. **Instruct**: turn that verdict into the *one change* the model should make next. A verdict is not an
   instruction. Every gain in `RESULTS.md` came from improving job 2, and every one was A/B measured.

Code: `projects/02-kernel-agent/nkibench.py` (Stage B judge), `verdicts.py` (verdict → instruction),
`agent.py:grade` (the pipeline), `heldout.py` (held-out cases + confidence), `kernelbench.py` (Stage A judge),
`stagea.py:instruct` (Stage A instructions). Selftests: `python nkibench.py --selftest`, `python verdicts.py`,
`python kernelbench.py --selftest`.

---

## Part 1 — the verdict (what is accepted)

A kernel is **SOLVED** only if every stage below passes, in this order. The reward is graded so that progress is
visible: parses 0.1, rules 0.2, runs 0.2, correct 0.5 × fraction of shapes. Nothing counts until it is correct.

| stage | accepts | rejects | why |
|---|---|---|---|
| parse | one python block | prose, truncated code | `finish=length` is reported as a budget problem, never as the model's |
| **static rules** (`check_rules`) | NKI primitives (`nl.*`, `nisa.*`), loops, slices | framework calls that do the whole op (`np.mean`, `@`, `.T` on an input), a literal partition dim > 128, missing `@nki.jit`, wrong entry name; with `--device-rules`, **comprehensions** | "rejecting a correct kernel is worse than missing a cheat", so NKI's own reductions are never banned. Comprehensions: **measured on the device**: the simulator accepts them, the Trainium compiler rejects them (LOG 15:1x) |
| **signature lint** (`--v9`) | calls whose keywords match the SDK's real signature | unknown/missing kwargs; `tensor_scalar` with a number in `data=` or an op in `operand0=`; `op=nl.sum` in `tensor_reduce` (`--v10`) | the simulator reports one error per round; the lint reports all of them in one. 0 false positives on every correct kernel we have |
| simulate (`nki.simulate`) | runs on every shape | raises | CPU simulation in seconds is where an agent should spend its attempts |
| **numerics** | max error ≤ 2e-2 × output RMS on every shape | NaN/Inf, wrong shape, mostly-zero output, mismatch | the RMS-relative tolerance keeps near-zero entries from false-failing |
| **input untouched** | inputs unchanged | writing into an input | a solved level once did `dma_copy(dst=x)`; inputs are snapshotted and compared |
| hardware hazard | no simulator hazard warning | "incorrect results on hardware" warnings | correct on CPU and wrong on device is wrong |
| **traffic bar** (L5–7) | HBM bytes ≤ 1.90× / 1.30× / 1.05× the floor | above the bar | levels 5–7 compute the same thing as L4; bytes are the level. **The bars were re-derived from AWS's own tutorial kernels** (upstream's 1.6/1.25 rejected them, LOG 11:49), and the byte counter matches a hand calculation exactly |

**Shapes are hostile on purpose**: ragged tiles, single-tile shapes, and the largest shape exists so that
redundant traffic is visible. **Held-out cases the agent never sees** (`heldout.py`): C=1, P=1, M=1, prime K=131,
ragged K=96 M=200 N=700, N=513, H/W not divisible by the pool, ×1e4 magnitudes (×30 for attention: a naive
softmax overflows), constant rows. Every AWS tutorial matmul kernel fails these. The verbatim L6 tutorial kernel
returns **silently wrong** numbers at K=131.

**Confidence** (`heldout.confidence`), computed from the code and public result only, before held-out runs:
0.95 baseline, ×0.4 for shape asserts, ×0.5 for no remainder handling when no public shape was ragged, ×0.7
for hard-coded public dims, ×0.3 for no row-max subtraction before exp (L8). Scored in `CALIBRATION.md`.

### Stage A judge (`kernelbench.py`)
NumPy under kernel rules (tiles ≤128×512, explicit loops, the level's reducing function banned). Tolerance
`|got−want| ≤ atol + rtol·|want|`, `rtol = max(8·√n_acc·eps32, 1e-5)`, `atol = rtol·RMS(ref)`: float32 error grows
~√n over an n-term reduction; 8 is the safety factor; atol judges near-zero outputs against the output's
magnitude. Under the old bare relative 1e-4, a *correct* float32 matmul passed 5/10 cases and layernorm 24/32.

---

## Part 2 — the instruction (what the model is told)

Rules for every message, each one learned from a measurement in LOG.md:

* **Name the change, not the symptom.** "Out of bounds" → "`lhsT[m0:…, k0:…]` lists the axes in the wrong order;
  write `lhsT[k0:k0+k_sz, m0:m0+m_sz]`".
* **Read the code, not just the error.** The same swapped slice raises out-of-bounds on L3 and *silently computes
  the wrong product* on L4's square tiles; only the code shows the cause (`verdicts.code_hint`).
* **Never reveal target values** (`--directional`): direction in bands ("too small, by more than 2×"), a constant
  factor or a transpose pattern. Never "expected X".
* **A misleading verdict is worse than a vague one.** "Usually an uninitialised tile" sent the model inventing
  `nisa.psum_zero` when it had only forgotten a `dma_copy` (v7 names the missing copy).
* **Fix the prompt when the prompt is the cause.** Twice upstream's own text caused the failure: L6's traffic
  hint, and L8's API card offering only `nl.sum`. Changing one sentence beat any amount of feedback.
* **Scope every fix to the level it was measured on.** An L8 prompt fix applied globally cost L1 2/5 runs.
* **Prove an instruction completable before giving it.** Each skeleton and idiom was filled in by hand and passed
  the harness first (`handwritten/`).

| version | instruction it added | measured effect (n=5, [sim]) |
|---|---|---|
| v1 (upstream, as a table) | regex → instruction, byte-identical to upstream `enrich()` | baseline 0/4 levels |
| skeleton | rank-aware template with TODO slots | L2 0→5/5, L4 0.36→0.62 |
| **v2** | swapped-axes slice named; partition axis kept whole | **L3 0→5/5, L4 0→5/5** |
| **v3** | L1: reduce each window into a (C,1) tile | **L1 0→5/5** |
| **v4** | L5: lhsT load is inside the n loop; move it out | **L5 0→5/5** (control 0/5) |
| **v5** | L5–7: load every tile exactly once | **L7 0→5/5** (control 0/5) |
| v7 / v8 / prompt fix | never-loaded inputs; k-loop outside; L6 hint = L7's | **L6 0→5/5** |
| v6, v9 | overflow; static signature lint (all misuses at once) | L1–L7 held at 5/5 (C10); L8 0/5 |
| v10 | L8: API card names tensor_reduce/reciprocal; op-order TODO | **reverted**: L8 unchanged, L1 2/5 |
| Stage A v2→v4-smart | combine column tiles; snippet built from the model's own call; accumulator named after its variable only when 2+ reductions | Stage A 4/10 (v1) → **9/10** |
| `--retrieve` | AWS doc excerpt per failure bucket | **no effect**, reverted |
| `--mech` | AST fixes | dropped: 0 of 145 attempts needed it |

Verification of the checker itself: `regrade.py` re-grades logged attempts from their own code. The 354 attempts
behind the main decisions (E1, C1, C2, the L5 pair, c5 and its controls, C10): 0 mismatches. Two bugs found that way (bytecode-cache staleness, missing flags) are fixed and logged.
