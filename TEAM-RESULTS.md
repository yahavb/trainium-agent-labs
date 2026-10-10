# Team 44 — Hack the Chip, combined results

One day, one Trainium seat cluster, one NKI kernel-agent project with three workstreams running on
top of the provided harness (`nkibench.py`, `agent.py`, the reference kernels and the seat pods,
built by the organizers). This file is the combined summary: what each teammate built, the measured
numbers, and the honest limits. Each workstream's own document has the full evidence trail.

**Verification status, read first.** All agent results below are **simulator-verified**
(`nki.simulate` on the CPU, HBM traffic counted with complete accounting). No agent-produced kernel
has been executed on Trainium. The only device measurements in this repo are the latency
simulator's inputs: tilings timed on a real trn2 core with `neuronxcc.nki.benchmark`.

## Results at a glance

| # | Workstream | Measured result | Status |
|---|---|---|---|
| 1 | **Minimum-traffic kernel agent** — repair the model's own near-miss with a classifier over a fixed guidance menu | Official checker accepts **1.00x — the exact HBM byte floor — on all four shapes**, at two evaluation seeds, CLI exit 0, for **levels 5, 6 and 7** (one native kernel per level). Two verified **1.857x** intermediates. Unguided baseline: **0 valid improvements in 90 attempts**. | Simulator-verified; final repairs line-guided (disclosed) |
| 1b | Robustness suite, same checker, cases never used for optimization | Gates **24/24**, held-out aligned shapes **9/9**, hostile values **20/20**, ragged non-divisible shapes **0/9** (partial final tiles unimplemented — the honest open failure) | Simulator-verified |
| 2 | **Verifier-grounded reflection/RL trace** — prompt-level controller with restarts, lint and held-out checks on levels 1–4 | Levels 1/3/4 measured (arm `reflect/shape`, 5 episodes): solved **0/5, 4/5, 0/3**; level 4 ceiling **0.62**. Where level-3 solves came from: direct round 0 **5%**, repair rounds **0/20**, after a real restart **25%**. 64 offline tests pass. | Simulator-verified; one arm, no ablations |
| 3 | **Latency simulator** — EnergAIzer-style work counts + fitted unit costs + overlap, from 216 device-timed kernels | Leave-one-shape-out: **MAPE 17.6%**, rank corr **0.80**, top-1 pick **11/18**, average regret **14.3%** (roofline baseline: 89.1%, −0.30, 0/18, 267%). Tile search on 10 unseen shapes: **12.5x cheaper** than brute force for top-1; **4.0x cheaper** for top-3 (9/10 best found, 0.3% average regret) | Fitted to device measurements; agent-facing prediction projection, `nisa` path assumed |

---

## Workstream 1 — the minimum-traffic kernel agent

**Owner:** Keshav Krishna, with the agent loop, menu and formal checks in
`projects/02-kernel-agent/`.

### The problem

The task: make a correct float32 matmul kernel (`lhsT[K,M]`, `rhs[K,N]` → `out[M,N]`) move the
minimum possible HBM bytes — read each input once, write the output once. That ideal is **1.00x**
the byte floor; the shipped seed kernel is correct but wastes up to **2.00x**. Levels 5/6/7 demand
worst-case ratios ≤ 1.60x / 1.25x / 1.05x.

### The working approach (three findings, in order of importance)

1. **The checker carries the constraints.** `nkibench.py` plus the fail-closed gate in
   `traffic_eval.py` reject wrong numbers, mutated inputs, unmeasured traffic and counts below the
   floor. The prompt stays short; the checker objects with a *named change*, never a verdict.
2. **Repair the model's own near-miss — do not invent from the seed.** The unguided loop's 90
   attempts were not wasted: several had already found the byte-saving structure (1.857x) but
   computed wrong numbers. Repairing that is much easier than transforming the clean 2.00x seed.
   This is the project's single biggest empirical finding.
3. **Behavioral diagnosis + a menu, model as classifier.** When numbers are wrong, the checker
   tests hypotheses against the produced values and names the fault (e.g. "your numbers equal
   chunk 0 of the left operand multiplied against every right chunk — your stationary operand
   never advanced"). The repair knowledge then lives in a fixed menu (`guidance_menu.py`; 15
   options in the current revision, 9 in the level-5 run) with a `when` and an `exact` text. Each
   round is two short model calls: a **classifier** picks one id (no code), an **applier** applies
   only that guidance. A back-out control makes a bad choice cost one round, not the run.

### Verified results

| level | kernel (native entry point) | worst waste | formal acceptance |
|---|---|---|---|
| 5 | [`agent_kernel_menu_guided_1p00x.py`](projects/02-kernel-agent/evidence/agent_kernel_menu_guided_1p00x.py) (`nki_matmul_hoist_load_`) | **1.00x** | accepted, seeds 0 + 1; CLI exit 0 both |
| 6 | [`agent_kernel_l6_nearmiss_1p00x.py`](projects/02-kernel-agent/evidence/agent_kernel_l6_nearmiss_1p00x.py) (`nki_matmul_block_free_dimension_`) | **1.00x** | accepted, seeds 0 + 1; CLI exit 0 both |
| 7 | [`agent_kernel_l7_nearmiss_1p00x.py`](projects/02-kernel-agent/evidence/agent_kernel_l7_nearmiss_1p00x.py) (`nki_matmul_fully_optimized_`) | **1.00x** | accepted, seeds 0 + 1; CLI exit 0 both |

| shape (K, M, N) | shipped seed | agent kernel | floor bytes |
|---|---:|---:|---:|
| 128, 128, 512 | 589,824 (1.00x) | 589,824 (1.00x) | 589,824 |
| 256, 256, 1024 | 3,670,016 (1.56x) | 2,359,296 (1.00x) | 2,359,296 |
| 512, 128, 512 | 1,572,864 (1.00x) | 1,572,864 (1.00x) | 1,572,864 |
| 256, 512, 1024 | 7,340,032 (2.00x) | 3,670,016 (1.00x) | 3,670,016 |

![Levels 5–7 results](projects/02-kernel-agent/evidence/levels_results.png)

| robustness family | passed | note |
|---|---|---|
| optimization shapes × gates | **24/24** | 4 shapes × 2 seeds × 3 bars |
| held-out aligned shapes | **9/9** | seeds 17/29/43, never used for optimization |
| hostile value families | **20/20** | zeros, negatives, repeated rows, cancellation, large finite |
| ragged/non-divisible shapes | **0/9** | open failure: partial final tiles are not handled |

![Robust check](projects/02-kernel-agent/evidence/robust_matrix.png)

### Disclosure — read before quoting the 1.00x

- **The unguided loop produced zero valid improvements in 90 attempts** (a pilot plus five frozen
  repeats, `projects/02-kernel-agent/PILOT-LOGS.md`). That rate stands as the honest baseline.
- The two **final repairs were line-guided**: three exact lines each, authored by the team, applied
  by the model. The sequence: the model restacked the operand **by itself** after the behavioral
  diagnosis (two shapes to 1.00x), then repeated the same kernel; the line-precise instructions
  cleared the remaining cache-stride bug. Full lineage in
  [`AGENT-IMPROVEMENT.md`](projects/02-kernel-agent/evidence/AGENT-IMPROVEMENT.md) and
  [`REPAIR-RESULTS.md`](projects/02-kernel-agent/evidence/REPAIR-RESULTS.md).
- The **from-seed route still fails**: starting from the clean 2.00x kernel, the loop has not
  produced even a below-2.00x result. The near-miss start is load-bearing.
- The menu texts and expected-choice rules are team-authored; selection and application are
  model-driven and logged per round.
- Levels 5–7 are **threshold achievements of one method**, not three separate capabilities.
- One run per configuration; the applier is near-deterministic, not deterministic (the level-6
  floor landed on a disclosed retry). Self-reported confidences are uncalibrated (95–100 claimed
  on kernels failing three of four shapes).

### Evidence and reproduction

- Full write-ups: [`REPORT.md`](projects/02-kernel-agent/REPORT.md) ·
  [`evidence/FORMAL-CHECK.md`](projects/02-kernel-agent/evidence/FORMAL-CHECK.md) ·
  [`README.md`](projects/02-kernel-agent/README.md) ("Our run") ·
  [`evidence/GUIDANCE-CLASSIFIER.md`](projects/02-kernel-agent/evidence/GUIDANCE-CLASSIFIER.md) ·
  [`evidence/ERROR-CATALOG.md`](projects/02-kernel-agent/evidence/ERROR-CATALOG.md) (23 curated
  failure families) · [`PAPER.pdf`](projects/02-kernel-agent/PAPER.pdf) (6-page paper).
- Run logs: `evidence/*_rounds.jsonl` (every state, choice, reply, evaluation, tokens).
- Repro (needs the Neuron SDK for the simulator):

```bash
cd projects/02-kernel-agent
python nkibench.py --selftest
python nkibench.py --level 7 --check evidence/agent_kernel_l7_nearmiss_1p00x.py
python robust_check.py --jobs 6        # the robustness matrix
python test_traffic_eval.py            # 37 offline tests
```

---

## Workstream 2 — verifier-grounded reflection / RL trace

**Owner:** Satyapragnya Kar. Additive controller, reusing `agent.py`'s prompts and the
`nkibench.py` verifier **unchanged**. No model weights are trained — this is prompt-level
learning (hint tiers, restarts, lesson bank, bandit), not fine-tuning or GRPO.

### What it adds

- **`rl_trace_agent.py`** — the controller: shape-tier hints (`none < api < shape < algo`),
  per-sample variant hints, duplicate re-asks, a lesson bank, a bandit over prompt variants, and
  **real restarts** (two identical failures = stuck; a restart is a fresh prompt carrying the
  grounded analysis, escalating one hint tier).
- **`kernel_lint.py`** — taint analysis for hard-coded tile sizes: parameters and anything derived
  from `.shape` are derived; a literal dimension ≥ 2 is flagged. Attached only to failures of
  category `tile_limits`, never to a passing kernel.
- **`holdout.py`** — runs a kernel on shapes the checker never showed it; excludes shapes the
  shipped reference itself fails.
- **`run_matrix.py` / `summarize_rl_trace.py`** — arms × levels, interleaved and resumable; pass@1
  and pass@k with bootstrap and Wilson intervals; confound warnings.
- **`test_rl_trace_agent.py`** — 64 offline tests (NumPy stand-in for the SDK), all passing.

### Measured results (v6, `reflect/shape`, seat-224, simulator)

| level | episodes solved | Wilson 95% | notes |
|---|---|---|---|
| 1 average pooling | **0 / 5** | [0.00, 0.43] | hard-coded tiles dominate; reduction over non-trailing axes next |
| 3 single-tile matmul | **4 / 5** | [0.38, 0.96] | one shape only; "verified" means correct on that one shape |
| 4 tiled matmul | **0 / 3 complete** | [0.00, 0.56] | partial 4th episode; 0.62 ceiling (one shape of four) |

Where the level-3 solves came from: round-0 direct prompt **1/20 (5%)**, repair rounds (prompt
carries the failing code) **0/20**, after a `STUCK-RESTART` **5/20 (25%)** — all in the first round
after the restart. **Restarts were the productive mechanism; repair rounds were not.**

An earlier level-2 A/B (`plain` vs `reflect`) read 3/5 vs 5/5, but every reflect episode verified
in round 0, so the repair path was never exercised; per-sample round-0 rates were 15% vs 35%,
Fisher exact **p = 0.27**. Suggestive, not shown — and the arms differed in more than the named
flags, so v6 adds real ablations.

### Limits (stated in the workstream's own README)

- One arm per level, no ablations; episode samples share a prompt, so sample-level independence is
  optimistic.
- `holdout.py` on level 3 scored **0/0 (not a pass)**: the shipped reference fails both held-out
  shapes, so nothing remained to grade; level 3 also has a single checker shape.
- `kernel_lint` is a heuristic; it never gates a passing kernel.
- Prompt-level only; nothing says anything about real Trainium latency.

### Evidence and reproduction

- [`README-RL-TRACE.md`](projects/02-kernel-agent/README-RL-TRACE.md) — the full v6 document (what
  the v5 A/B said and did not, every change, flags, results, limitations; a copy also sits at the
  repo root as `README-RL-TRACE.md`).
- Logs: `matrix_runs/reflect_shape_L{1,3,4}.jsonl` (on the seat).

```bash
cd projects/02-kernel-agent
python test_rl_trace_agent.py                 # 64 offline tests
python run_matrix.py --levels 3 --episodes 20 \
    --arms reflect/shape -- --context 8192 --no-seed --same-temp
python summarize_rl_trace.py matrix_runs/*.jsonl --brief
```

---

## Workstream 3 — the latency simulator

**Owner:** Shubham Ojha. What the byte checker cannot see: **time**. This workstream learns an
analytical latency model for tiled NKI matmul on Trainium2 from measured kernels, then uses it to
rank tilings on the CPU.

### What it does

- **`bench.py`** sweeps 12 tilings × shapes on a real trn2 core (`naive` vs `hoist` variants,
  `tile_k` 64/128, `tile_n` 128/256/512), timed with `neuronxcc.nki.benchmark`, two shards in
  parallel on the free cores, resumable. The measurements are committed as `results_*.csv`.
- **`model.py`** counts every action exactly from the loop nest (DMA loads, PE columns, idle
  contraction rows, accumulation-chain length — no guessing at tilings or hit rates, because on
  Trainium the tiling is in the source). Units are bf16; the model is after EnergAIzer (ISPASS'26).
- **`fit.py`** fits unit costs with non-negative least squares, weighted so the fit minimizes
  *relative* error, then fits an overlap correction (how much of DMA hides under the Tensor
  Engine). Evaluation is leave-one-shape-out: the model never sees the shape it ranks.
- **`search_savings.py`** answers the practical question: does the simulator replace brute-force
  tile search on the device?
- **`latency_hint.py`** wires it into the agent: `nkibench.py` records each `nc_matmul` tile shape
  during simulation, and a correct kernel gets one speed sentence plus tiling advice.

### Measured results (reproduced from the committed CSVs, 2026-10-10)

Leave-one-shape-out, all 216 device-timed kernels over 18 shapes:

| model | MAPE | rank corr | top-1 | top-3 | regret |
|---|---:|---:|---:|---:|---:|
| roofline baseline | 89.1% | −0.30 | 0/18 | 4/18 | 267.4% |
| fitted work-count | **17.6%** | **0.80** | **11/18** | **14/18** | **14.3%** |

The roofline gives every tiling of a shape the same time, so it cannot rank tilings at all; the
fitted model can, and its top-1 pick averages 14.3% slower than the true best.

Tile search, fitted on the 96 original kernels, tested on 10 shapes the model never saw:

| search method | device cost | vs brute force | best found | avg regret | worst |
|---|---:|---:|---:|---:|---:|
| brute force, all 12 tilings | 2,864 s (47.7 min) | 1.0x | 10/10 by definition | 0% | — |
| simulator + benchmark top-1 | 230 s (3.8 min) | **12.5x cheaper** | 6/10 | 20.6% | 112% |
| simulator + benchmark top-3 | 709 s (11.8 min) | **4.0x cheaper** | 9/10 | **0.3%** | 3% |
| simulator ranking alone | **0.1 ms** CPU, all 120 kernels | — | — | — | — |

Prediction error on the unseen shapes: 21.8% mean, 74% worst. The top-1 miss on one shape is the
honest caveat; benchmarking the top-3 recovers 9/10 with 0.3% average regret.

### Limits

- Unit costs were fitted on `nl.load` / `nl.matmul` kernels; `nisa.dma_copy` / `nisa.nc_matmul`
  kernels are assumed to cost the same per unit — **not measured**.
- The agent-facing prediction is projected onto a 2048³ matmul, because at the level-4 shapes the
  fixed launch cost hides tiling differences. The hint quotes an approximate ±12%.
- Speed is secondary in this project: bytes are the scored objective. The simulator is here to
  show that a small fitted model can rank tilings **without** burning device compiles: 12.5x
  cheaper search (47.7 minutes of brute-force compiles down to 3.8 minutes for the top-1 pick).

```bash
cd projects/02-kernel-agent
python bench.py --check                 # simulator correctness of every config
python fit.py                           # fit + leave-one-shape-out + pred_vs_meas.png
python search_savings.py                # brute force vs simulator search on unseen shapes
```

---

## Combined limitations

- **Simulator only for agent results.** Byte counts come from `nki.simulate`; no agent kernel ran
  on the device. The latency model's *inputs* are device measurements, but its predictions for the
  agent are not device-validated against the same kernels.
- **Ragged shapes fail (0/9).** Partial final tiles are unimplemented in the winning kernel; the
  challenge's non-divisible-shape trap is open. The latency simulator supports only
  tile-multiple shapes (its search grid requires divisibility).
- **Single runs per configuration** in both agent workstreams; the honest baseline for the traffic
  project is 0 valid improvements in 90 unguided attempts, and the successful final repairs were
  partly team-guided (three exact lines each). The reflection workstream runs multiple episodes
  and reports intervals, but one arm and no ablations.
- **Team-authored content is disclosed everywhere**: the guidance menu and its `exact` texts, the
  demo/calibration example, the expected-guidance rules, `holdout.py`'s shape choices, and the
  latency model's feature set. Selection, application and code are the model's.
- **No weights were trained anywhere.** All "learning" is prompt-level or table-level
  (bandit/memory/population) test-time adaptation.

## Credits and provenance

- **Team 44** (NYU × Annapurna Labs): Keshav Krishna (minimum-traffic agent, checker hardening,
  formal checks, reports, paper), Satyapragnya Kar (reflection/RL-trace controller, linters,
  held-out and matrix tooling), Shubham Ojha (latency simulator, device sweep, fit and search).
- The base harness — `nkibench.py`'s ladder, references, failure messages, `agent.py`, the
  `serve.sh` model server and the seat pods — was provided by the organizers (Yahav and team).
  Their original README is kept intact at the repo root and in `projects/02-kernel-agent/`.
- This summary was compiled from the committed evidence below; it adds no new claims.

## File map (root → evidence)

| What | Where |
|---|---|
| Traffic short report / paper | `projects/02-kernel-agent/REPORT.md`, `PAPER.pdf`, `PAPER.tex` |
| Traffic formal evidence | `projects/02-kernel-agent/evidence/FORMAL-CHECK.md`, `GUIDANCE-CLASSIFIER.md`, `REPAIR-RESULTS.md`, `AGENT-IMPROVEMENT.md`, `ERROR-CATALOG.md` |
| Traffic raw logs | `projects/02-kernel-agent/evidence/*_rounds.jsonl`, `data/` |
| Unguided pilots | `projects/02-kernel-agent/PILOT-LOGS.md`, `evidence/progress.png` |
| RL/reflection workstream | `projects/02-kernel-agent/README-RL-TRACE.md`, `rl_trace_agent.py`, `run_matrix.py`, `test_rl_trace_agent.py` |
| Latency simulator | `projects/02-kernel-agent/bench.py`, `model.py`, `fit.py`, `search_savings.py`, `latency_hint.py`, `latency_coef.json`, `results_*.csv` |
| Plan and tuning logs | `docs/superpowers/plans/2026-10-10-minimum-traffic-agent.md` |
