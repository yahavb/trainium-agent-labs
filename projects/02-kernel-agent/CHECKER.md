# The checker: what it accepts, what it rejects, and why

The checker is `nkibench.py` (rules, simulation, numerics) plus `agent.grade()` / `agent.enrich()`
(score and the message the model gets). Everything runs in the NKI 0.6.0 **CPU simulator**; nothing
in the loop runs on the device. The held-out set, the value kinds and the tolerance are in
[EVAL.md](EVAL.md) and are not repeated here.

```bash
python nkibench.py --selftest                       # checks the harness itself
python nkibench.py --level N --check kernel.py      # what the loop sees
python nkibench.py --level N --eval  kernel.py      # the held-out set (EVAL.md)
```

## 1. Order of checks

Each step runs only if the one before passed. The model sees the first failure, prefixed with how
many shapes passed (`agent.grade`).

1. **Parse**: `compile()`. On failure: "does not parse: <msg> on line N".
2. **Static rules** (`nkibench.check_rules`, AST scan, milliseconds; no execution).
3. **Load**: written to a path never used before in this process (`nkibench.candidate_path`, see
   §5), then imported.
4. **Simulate each loop shape** (`nkibench.simulate_and_count`): `nki.simulate`, with HBM bytes
   counted on every `dma_copy` and every `nl.ndarray/zeros/ones/full` allocation recorded.
5. **Per shape, first failure wins**: an exception (made into an instruction by `enrich()`) →
   input modified → **illegal allocation** → numerical mismatch → HBM traffic bar (levels 5–7) →
   simulator's "incorrect results on hardware" warning.
6. **Score** (§3). The level is solved only when every loop shape passes.
7. **After the level ends** (not during the loop): confidence is stated, then the held-out set runs
   once, and its result never reaches the model ([EVAL.md](EVAL.md), `agent.verdict`).

## 2. What it rejects

| rule | triggered by | why it exists | the model is told |
|---|---|---|---|
| framework call | `np.mean`, `torch.matmul`, … whose name is in the level's banned set (`LEVELS[n]["banned"]`) | the level is about computing it in the kernel; only framework modules are banned, so `nl.sum` and `nisa.nc_matmul` stay legal | the line and the call |
| `@` operator, `arg.T` | AST `MatMult` / `.T` on a kernel argument | does the whole job, or transposes on the host | the line |
| entry point | no function with the level's name, or no `@nki.jit` | without `@nki.jit` it runs as Python, not as a kernel | the exact `def` line to use |
| partition > 128 (static) | a literal first dimension > 128 in `nl.ndarray((…))` | `nl.tile_size.pmax` = 128 (read from nki 0.6.0; documented in `nki.language` dims) | "Tile it." |
| **illegal allocation** | an SBUF/PSUM tile with partition dim > 128, PSUM > 16 KiB per partition, or SBUF > 192 KiB per partition | **the simulator checks the 128 limit on copies, not on allocation**: a (256, 256) SBUF tile ran and could have scored 1.0 (26c43ed). PSUM: 128 partitions × 16 KB, 8 banks (NKI *Trainium/Inferentia2 Architecture Guide*); a tile spanning several banks is legal and is not flagged. SBUF: see §4 | ILLEGAL ON HARDWARE + "loop over rows in chunks of at most 128" |
| matmul operand limits | the simulator's own asserts, e.g. `Matmul stationary free dimension 256 exceeds gemm_stationary_fmax=128`, `Matmul moving free dimension 1024 exceeds max 512 for nc_version=nc_version.gen3` (both measured with the L4 reference, tile sizes doubled) | stationary free ≤ 128, moving free ≤ 512 (`nl.tile_size.gemm_stationary_fmax` / `gemm_moving_fmax`, read from nki 0.6.0), contraction ≤ `pmax` | `enrich()`: e.g. for K, split it and accumulate in one PSUM tile |
| modified input | an input array differs after the run | the caller owns it; the first "solve" in this repo wrote into its input and passed by luck | allocate a new `shared_hbm` output |
| numerical mismatch | max \|error\| > 2e-2 × RMS(reference) | tolerance and its measurement: [EVAL.md](EVAL.md), "Tolerance" | where the worst error is, whether it is in a ragged edge tile, or "output is N% zeros" |
| non-finite output | NaN/Inf in the result | usually an uninitialised tile read | count and first index |
| hardware hazard | the simulator warns "incorrect results on hardware" | right on CPU, wrong on the device | CORRECT ON CPU BUT WRONG ON HARDWARE |
| traffic bar (levels 5–7) | HBM bytes > 1.6× / 1.25× / 1.05× the floor | levels 5–7 have level 4's reference and shapes, so bytes are what makes them levels | how far over, and what buys the difference |

### Verdict → instruction

The simulator's message says what is wrong; `enrich()` adds what to change. Two real pairs from the
baseline log (`runs/seat-116/latest/.../attempts.jsonl`, level 1 round 0 and level 4 round 0):

> **Simulator:** `dma_copy requires src and dst to have the same number of elements, got src=4, dst=16384`
> **Sent:** "The tile you allocated holds 16384 elements but you copied 4 into it. nisa.dma_copy does
> not slice or broadcast: allocate the destination with EXACTLY the shape of the slice you are
> moving … `t = nl.ndarray((128, 512), …)` and then `nisa.dma_copy(dst=t, src=a[0:128, 0:512])`."

> **Simulator:** `dma_copy dst partition dimension 256 exceeds maximum 128`
> **Sent:** "A tile may have at most 128 rows, and you asked for 256. Do not allocate one tile for
> the whole tensor: loop over the partition dimension in chunks of at most 128 with
> nl.affine_range … If a dimension is already 128 or smaller, use it whole -- do NOT pad it up."

**Measured, these were not enough.** In the baseline the next round repeated the same failure mode
62% (copy size) and 100% (partition > 128) of the time
([analysis/taxonomy_baseline_seat116.md](../../analysis/taxonomy_baseline_seat116.md), "stuck").
That is what the feedback versions below change.

### Feedback v7

*Placeholder: filled in when the v7 feedback code is in the repository (PLAN.md §3), with its
before/after pairs and the measured change in stuck rate.*

## 3. How a score is computed

`agent.WEIGHTS`: parses 0.1, rules 0.2, runs 0.2, correct 0.5. "Runs" means at least one loop shape
simulated without raising; "correct" is prorated by the shapes that pass.

| score | meaning | where it appears |
|---|---|---|
| 0.0 / 0.1 | no code, or code that does not parse / breaks a rule (a rule violation is never more than 0.1) | |
| **0.30** | parses and rules clean, but **no shape simulates** (0.1 + 0.2) | baseline L1 and L3, every run |
| 0.50 | simulates on some shape, no shape correct | baseline L4 run 3 |
| **0.62** | runs, 1 of 4 shapes correct: 0.5 + 0.5 × 1/4 = 0.625 | baseline L4: the single-tile shape only |
| **1.0** | every loop shape correct | baseline L2, 3 of 5 runs |

A loop score of 1.0 is the loop's claim, not the final one: the verdict after the held-out set is
in `verdicts.jsonl` ([EVAL.md](EVAL.md), "Confidence and calibration").

## 4. What it does not check

- **No device.** Correctness comes from `nki.simulate` on a CPU. On-device latency and profiles
  (layers 2–3 in `nkibench.py`'s docstring) are not built. Kernels are not compiled with
  `neuronx-cc` in the loop.
- **Traffic is a model.** Levels 5–7 count the bytes the kernel asks `dma_copy` to move in
  simulation. The roofline uses 222 Flops/Byte, the published bf16 figure for NeuronCore-v2,
  while the test inputs are float32; treat it as indicative.
- **SBUF limit: a deliberately conservative 192 KiB per partition.** That is NeuronCore-v2's
  24 MiB / 128. Trn2 is NeuronCore-v3, which the *Trainium2 Architecture Guide* (NKI docs,
  `architecture/trainium2_arch.md`) gives 28 MiB, i.e. 224 KiB per partition, and the simulator
  itself reports `nc_version.gen3`. We keep 192 KiB on purpose: being stricter than the chip can
  only reject a legal kernel in the 192–224 KiB range, never accept an illegal one, and level 1–4
  shapes rarely reach that range. Re-grading all 424 baseline attempts with the audit on changed
  no score (b39989f). If a run ever shows an SBUF rejection between 192 and 224 KiB, the limit
  gets raised to 224 KiB.
- **Static rules are a text scan.** They catch framework calls by name; a kernel that hides one
  behind an alias would pass the scan (then still has to pass the simulator's numerics).
- **Shapes**: level 3 is one shape (its reference asserts it); shapes outside each reference's
  contract are not tested; no NaN/Inf inputs ([EVAL.md](EVAL.md), "Value kinds").
- **Dtype** is only checked on the held-out set, not in the loop (kept out to stay comparable with
  the baseline).

## 5. Checker bugs found today, and how earlier numbers were re-verified

| commit | bug | effect | fix |
|---|---|---|---|
| 26c43ed | the simulator does not check tile limits at allocation | a kernel the chip cannot run could score 1.0 | allocation audit (§2) |
| c39c0ce, corrected by ded1ef2 | nki 0.6.0 caches by file path: a same-size candidate written to a reused path was simulated as **the first one, numerics and allocations both** (the L4 reference with swapped matmul operands scored 4/4) | every run before c39c0ce could hold false solves or false failures | a fresh path per candidate (`candidate_path`) |
| 183b384 | the audit's wrappers stayed installed after loading | `neuronx-cc` failed on every kernel in the process (the loop's scores were unaffected; it never compiles) | install the audit only during a simulation |
| eeb13e9 | `dma_copy` casts float16 inputs silently | a kernel hard-coding float32 passed float16 inputs | dtype check in the held-out set |
| 7868c08 (before push) | the verdict ran the held-out set without the rule check | an unsolved kernel missing `@nki.jit` passed held-out 20/20 | rules first; a violation fails held-out |

**Re-verified**: `scripts/reaudit.py` (4350038) re-graded each distinct 1.0 kernel of the baseline
in a fresh process: the level-2 solves pass. `scripts/calibrate.py` (b39989f) then re-graded **all
424** baseline attempts, one fresh path each, with the audit on: every score equals the logged one,
so the baseline numbers are unaffected, unsolved runs included
([analysis/calibration_baseline_seat116.md](../../analysis/calibration_baseline_seat116.md)).
