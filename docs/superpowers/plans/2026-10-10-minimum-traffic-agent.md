# Minimum-Traffic NKI Agent Implementation Plan

> **For agentic workers:** Use the available `executing-plans` skill to implement this plan task by task. Steps use checkbox syntax. This document is a plan; no kernel improvement or level clear has been measured yet.

**Goal:** Build an actual model → kernel → checker → feedback agent loop that minimizes valid explicit HBM traffic for the supplied NKI matmul tasks, targeting 1.00× the input/output byte floor, and delivers the hackathon's required artifacts within eight hours.

**Architecture:** Start each independent run from the shipped correct level-4 kernel. The provided model proposes and implements memory-reuse changes; a common evaluator checks rules, numerical results, input preservation, transfer coverage, and resource estimates before ranking candidates. A lightweight test-time learning layer — a strategy bandit, an attempt memory, and a small parent population (Section 4) — chooses which strategy to prompt and which kernel to improve next. Preserve the best valid kernel, repair promising failed candidates separately, and validate finalists on the assigned hardware.

**Tech stack:** Existing Python agent, NumPy reference, installed NKI SDK and simulator, provided Qwen3-8B endpoint, JSONL artifacts, Trainium compiler/runtime. Use the SDK already installed on the seat.

## 1. Scope and documented requirements

This is an extension of project 2, using `nkibench.py` and its NKI matmul levels. `kernelbench.py` is a different, older NumPy ladder: its level numbers and Stage-A restrictions must not be substituted for the NKI ladder's rules. Preserve the NKI checker and shape contracts; disclose any additional unsupported shapes.

Local sources:

- `README.md`, Part 4: an agent around a model; checker, complete attempt log, one-page note; teams of 3–5.
- `projects/02-kernel-agent/CHALLENGE-kernel-agent.md`, Scoring / What to hand in: correctness 30%, delivered result 25%, method and honesty 25%, demo/write-up 20%; agent, checker, evaluation set, failure taxonomy, token instrumentation, reproduction note, and confidence reporting.
- `projects/02-kernel-agent/README.md`: supplied tutorial kernels are public starting points; repeated-run reporting; distinguish simulated results from device measurements.
- `projects/02-kernel-agent/nkibench.py`: executable NKI rules and original level thresholds: 5 = 1.60×, 6 = 1.25×, 7 = 1.05×.

**Administrative condition:** Earlier discussion assumed two people, but the README and facilitator guide specify teams of 3–5. Confirm an organizer-approved two-person exception or register a third teammate. The engineering schedule below has two implementation owners; a third teammate can own evaluation, failure classification, and presentation. The repository is the available rules source; any event-day amendments take precedence.

**Submission conditions:**

- Use the provided model and assigned seat. Log the actual endpoint/model identity without credentials.
- The model must generate the optimization edits and consume checker feedback. A hand-written optimized kernel or offline reference replay does not establish an agent success.
- Public kernels, API guidance, and human-written controller/checker code are disclosed. Any hand-written optimized calibration kernel has separate provenance and is excluded from agent success counts.
- Every request must fit the documented 8,192-token input limit and the server's prompt-plus-completion context limit.
- Preserve original correctness tolerances, dtype, mathematical operation, checker bans, and thresholds. Add checks instead of weakening acceptance.
- Evaluate hostile inputs and additional shapes. Include dimensions of 1 and prime/ragged cases for the general correctness assessment; distinguish these from the shipped aligned matmul ladder.
- Report a confidence estimate for passing unseen cases, collected before their evaluation, separately from deterministic verification status. Five runs are insufficient to establish strong calibration claims.
- Run a bounded diagnostic sweep of registered NKI levels and report every failure/unattempted level. Do not claim optimizing matmul solves unrelated operations.

## 2. Objective and stopping rule

For dense float32 matmul with `lhsT[K,M]`, `rhs[K,N]`, and `out[M,N]`:

```python
floor_bytes = 4 * (K*M + K*N + M*N)
waste_ratio = measured_hbm_bytes / floor_bytes
avoidable_bytes = measured_hbm_bytes - floor_bytes
```

The objective is to read each input element once and write each output element once under this interface. On supported cases, a count below the floor indicates incomplete accounting or a contract violation; it is not an optimization win. The floor may be unattainable for larger shapes under capacity constraints.

Predictions from the shipped source, not measured results:

| K,M,N | Baseline bytes | Floor bytes | Baseline ratio | Resident input bytes |
|---|---:|---:|---:|---:|
| 128,128,512 | 589,824 | 589,824 | 1.00 | 327,680 |
| 256,256,1024 | 3,670,016 | 2,359,296 | 1.556 | 1,310,720 |
| 512,128,512 | 1,572,864 | 1,572,864 | 1.00 | 1,310,720 |
| 256,512,1024 | 7,340,032 | 3,670,016 | 2.00 | 1,572,864 |

**Primary target:** all four shipped shapes numerically valid at 1.00× explicit traffic. This would meet the traffic thresholds for all three optimization levels if their other checks pass. Levels 5–7 use the same operation and shapes; report them as threshold achievements, not three independent capabilities.

**Ranking:** reject invalid/unmeasured candidates first; then minimize worst-case waste ratio across the evaluation shapes, then total bytes, then explicit transfer calls. Runtime is reported separately. Never trade correctness for bytes.

**Stop:** once all target shapes reach the floor, stop searching for lower bytes. Use remaining time for held-out correctness, actual compilation/execution, reproducibility, and the demo.

## 3. Optimization sequence the agent should explore

### First choice: retain both inputs

The maximum supplied input footprint is 1.5 MiB. A 128×512 float32 output staging tile adds 256 KiB of SBUF; one accumulator tile adds 256 KiB of PSUM. A conservative initial live-storage estimate is therefore 1.75 MiB SBUF and 256 KiB PSUM on the largest case, before compiler overhead or duplicated buffers.

Verify actual SDK/device capacity, per-partition limits, allocation layout, and compiler behavior. The estimate alone is not a hardware-validity proof.

Proposed algorithm, for the model to implement with installed NKI APIs:

```text
Allocate input caches in SBUF with the K partition dimension tiled at 128.
Load each LHS input tile once into its cache location.
Load each RHS input tile once into its cache location.
For each output tile (m, n):
    Initialize a float32 PSUM accumulator using documented NKI semantics.
    For each k tile:
        Multiply cached LHS and RHS views; accumulate in PSUM.
    Copy the complete accumulator to an SBUF output staging tile.
    Store that output tile to HBM once.
Return the newly allocated output.
```

An allocation layout to investigate is LHS `[128, K_tiles, M]` and RHS `[128, K_tiles, N]`, sliced on both axes as operands are consumed. The operand views must respect the GEMM free-dimension limits (stationary <= `gemm_stationary_fmax`, moving <= `gemm_moving_fmax`), so a resident buffer is sliced on its free axis as well as its K axis. Have the compiler validate this representation before accepting it. Reuse the existing tiled `nc_matmul` computation and avoid writing partial sums to HBM.

### Second choice: retain one full input and stream blocks of the other

If retaining both inputs introduces allocation/compiler problems, retain the smaller complete input. With LHS resident, iterate N blocks; load all K tiles for a RHS block once; compute all M output tiles against that block. This can still reach the byte floor while lowering live input storage.

### Third choice: bounded blocking

For shapes that cannot fit either strategy, have the model adjust M/N/K blocking while checking live SBUF and PSUM. Preserve complete output accumulations on-chip when possible. Accept a ratio above 1.00 when required by the validated implementation; retain the original thresholds.

On the largest supplied shape, an intermediate strategy reusing only RHS across M is predicted to reduce traffic from 7 MiB to 4 MiB (1.143×). Reusing only LHS across N predicts 6.5 MiB (1.857×). Use these as engineering calibration calculations, not claimed outcomes.

### General correctness path

Do not pad input tensors on the host and exclude the padding traffic. For ragged shapes, generate an explicit tiled fallback with valid slice extents and on-chip zero padding where full tensor-engine tiles are required. Count all external transfers. Keep aligned target-shape optimization and ragged-case traffic results separate. If the fallback cannot be verified in time, report those failures; do not claim arbitrary-shape support.

## 4. Test-time learning layer: bandit, memory, population

Three small controller mechanisms steer the search across rounds. They are tables, text, and folders: no model training, no extra model calls, no second agent role. The model still authors every candidate kernel; the learning layer only chooses which strategy to prompt and which kernel to improve. Freeze every parameter of this layer together with the rest of the controller by hour 4.5.

### 4.1 Strategy bandit

- **Arms:** the reuse-strategy menu from Section 3 — `retain_both`, `retain_rhs`, `retain_lhs`, `bounded_blocking`, `tidy_only`. Extend the menu only if a pilot shows a missing strategy, and record the extension.
- **Reward:** `0` for any invalid candidate; otherwise the improvement in the worst-case waste ratio over the parent, clamped to `[0, 1]`. Correctness is a hard gate before any reward.
- **Selection:** UCB1 with one fixed, logged exploration constant; deterministic tie-break on arm name so the same seed reproduces the same choices.
- **Update:** after each candidate is evaluated, before the next round. Log `(round, arm, parent_hash, reward)` every round.
- **Why:** MemCon (arXiv 2607.13591) learns an online policy over memory operations with a lightweight tabular bandit — zero extra model calls, convergence within tens of attempts.

### 4.2 Attempt memory

- One JSONL line per attempt: strategy, parent hash, the model's own strategy label when present, outcome (valid/invalid, per-shape waste, failure kind), and one templated lesson, e.g. `retain_rhs: largest shape 2.00x -> 1.14x`.
- The next prompt carries the best `--memory-k` lines plus the most recent failures. Memory lines count toward the input-token budget; trim the oldest failures first.
- The file is a submitted artifact. No hidden state.

### 4.3 Parent population

- Size `P = 3` by default (`--population-size`). The population starts as the seed kernel and grows to `P` as valid candidates arrive.
- A valid candidate replaces the member with the worst objective value if it improves it. Every member is valid at all times.
- Parent choice: the best member most rounds; a second member with a fixed, small, logged probability to keep diversity.
- **Why:** a slightly worse but different parent can hold the single trick the leader lacks — here, the working residency layout for the largest shape. AlphaEvolve (arXiv 2506.13131) keeps a diverse pool of parents for exactly this reason, and it is how it found a 23% faster matmul tiling.
- The final winner is model-authored. Hand-written calibration kernels (including the probe in Task 4) never enter the population and never count as agent success.

### 4.4 What this layer is not

- Not weight training, fine-tuning, or GRPO. No training infrastructure is used or needed.
- Not extra model calls: every update is a table write.
- Not a new agent role: one model, one checker, one controller.

## 5. Files and responsibilities

All paths below are under `projects/02-kernel-agent/` unless noted.

| File | Planned responsibility |
|---|---|
| `nkibench.py` | Consistent CLI traffic gate, input-preservation and measurement-validity checks; existing references/tolerances remain authoritative. |
| `traffic_eval.py` (new) | Common structured evaluator, resource estimates, per-shape results, independent grading of candidates. |
| `traffic_agent.py` (new) | Model-driven optimize/repair loop, best-candidate selection, exact token budget, append-only logs, termination. Reuse existing code extraction and diagnostic helpers where practical. |
| `test_time_learning.py` (new) | Strategy bandit (UCB1), attempt memory, parent population bookkeeping; pure functions, deterministic, unit-tested. |
| `traffic_cases.json` (new) | Declared optimization and held-out shapes, seeds, hostile-value families; held-out results excluded from adaptation. |
| `test_traffic_eval.py` (new) | Focused regression checks for false passes and numerical/input failures. |
| `validate_traffic_device.py` (new) | Compile/run selected candidates on assigned cores; compare outputs and collect profiler traffic when available. |
| `runs/traffic/<run_id>/` | Baseline, raw model replies, every candidate, JSONL records, verification reports, selected result. |
| `runs/traffic/<run_id>/learning/` | Per-run `bandit.json`, `memory.jsonl`, and `population/` snapshots; every winner records its parent hash. |
| `TRAFFIC-RESULTS.md` (new) | One-page reproduction note with results, provenance, failure counts, limits, and links to logs. |

Do not introduce a general tracing framework, new UI, weight-updating reinforcement learning, or multiple agent roles. Test-time learning limited to the three mechanisms in Section 4 (strategy bandit, attempt memory, parent population) is in scope: they are controller state, add no model calls, and never update model weights. Aggregate/per-input traffic is sufficient to guide this objective.

## 6. Task 1 — establish a trustworthy baseline (hour 0–1)

- [x] Record git revision, installed SDK, simulator API, model ID, tokenizer availability, context limit, and assigned device/core information.
- [x] Minute-zero gate: `python -c "import nki; print(nki.__version__)"` succeeds and the served model answers a minimal request. Resolve either failure before any other work.
- [ ] Read hardware capacity constants with installed-version semantics. New SDKs expose `sbuf_size_bytes` and `sbuf_fmax_bytes`; older `total_available_sbuf_size` can mean per-partition capacity. Do not interpret the latter as total capacity.
- [x] Run the existing selftest and baseline in the seat environment:

```bash
cd /workspace/projects/02-kernel-agent
python nkibench.py --selftest
python nkibench.py --level 4 --check reference_level4.py --seed 0
```

- [x] Compare each baseline byte count with the table above. Resolve discrepancies before model optimization.
- [ ] Create separate baseline and agent artifact directories. Mark seed source as `provided_reference`.
- [x] Prepare evaluation cases before choosing an optimized winner. Optimization: the four shipped shapes and seeds 0,1. Additional aligned evaluation: `(K,M,N)=(384,256,512),(128,384,1024),(512,256,1536)`, seeds 17,29,43. Ragged correctness: `(1,1,1),(129,127,513),(257,131,519)`.
- [ ] Include zero arrays, negative/mixed-sign inputs, repeated rows, alternating signs/cancellation, and moderate finite magnitudes that avoid reference overflow. Retain the existing tolerance definition; document maximum normalized error and dtype.

**Exit condition:** baseline correctness and transfer accounting are understood. Hardware/environment failure is reported explicitly, not sent to the model as a kernel repair problem.

## 7. Task 2 — make all acceptance paths agree (hour 0–2, checker owner)

- [x] Add focused regression cases: zero/unmeasured traffic cannot pass an optimization level; above-threshold traffic fails; numerically wrong candidates fail; mutated inputs fail; unsupported transfer coverage produces `unverified_traffic`.
- [x] Treat `counted['unmeasured'] > 0` as `unverified_traffic`, not a warning, before any traffic claim counts.
- [x] Fix `check_traffic_bar()` so missing/zero/incomplete measurements cannot silently skip an optimization-level gate. Retain exact original thresholds.
- [x] Make `verify()` snapshot inputs and enforce the same mutation, traffic, and known hardware-warning checks as agent evaluation before counting a passing case.
- [x] Restrict generated candidates to the supported explicit HBM transfer API for this experiment. Detect aliases/unsupported memory operations conservatively and refuse a verified traffic claim for unknown coverage. Count HBM reads and writes; exclude on-chip copies. Unknown dtype or memory-space attribution is an accounting failure.
- [x] In `traffic_eval.py`, return independent fields: `rules_ok`, `numerics_ok`, `inputs_ok`, `traffic_ok`, `hazards_ok`, `resource_status`, `per_case`, `failure_kind`, `feedback`. Preserve bytes as integers.
- [ ] Record static resource estimates as estimates. Compiler/device results alone can upgrade hardware status.
- [x] Check that an unchanged level-4 source, adapted only to the required level-7 entry name, fails the original level-7 traffic bar on the multi-tile cases.
- [x] Add focused regression cases for the learning layer: an invalid candidate earns reward 0; the population never admits an invalid member; UCB tie-breaks are deterministic across reruns.
- [x] Run the focused regression checks and existing selftest once after the changes; repeat only after relevant edits/failures.

**Exit condition:** the CLI and loop cannot disagree about whether a numerically correct but traffic-heavy candidate passed.

## 8. Task 3 — implement the real agent loop (hour 1–3, agent owner)

- [x] Give `traffic_agent.py` explicit options: `--seed-kernel`, `--rounds`, `--samples`, `--repeat`, `--context`, `--cases`, `--output`, `--max-tokens`, `--base`, `--model`, `--population-size`, `--memory-k`, and `--learning on|off`. Use the existing endpoint environment defaults.
- [x] Count the rendered request with the provided model's tokenizer/chat template before dispatch. Keep input at or below 8,192 tokens and reserve the configured completion budget inside server context. Use API `usage` for actual request/completion accounting. Do not present character counts as token counts.
- [x] Budget approximately 4,500 input tokens and 3,000 completion tokens within an 8,192-token server context, adjusting from exact counts. Trim old history, oldest memory lines, and redundant API guidance first; never silently truncate the kernel.
- [x] Ask the model for one complete candidate kernel, a short strategy label, and confidence of passing unseen cases. Provide the current kernel, shape/dtype contract, traffic breakdown, capacity facts, relevant API guidance, and the most recent useful failure.
- [x] The initial optimization instruction is: reduce repeated input loads, retain data in on-chip memory when it fits, and store completed output tiles once. Allow known optimization recipes; do not substitute a finished human-written optimized candidate for the model response.
- [x] Implement the following control flow:

```text
For each independent run (fresh bandit table, empty memory, population = {seed}):
    evaluate and record the provided seed
    repair_candidate = none
    repeat up to round limit:
        arm    = bandit selects a strategy (UCB1, deterministic tie-break)
        parent = population pick (best member; second member with small logged probability)
                 unless a bounded pending repair is next, which pins parent and arm
        request candidate kernel(s) from the model (prompt carries top memory lines)
        save raw replies and exact token usage before evaluating
        evaluate each candidate and save every result
        append memory line (strategy, parent, outcome, templated lesson)
        update bandit reward for the arm (0 if invalid; else gain in worst-case waste)
        if valid: insert into population when it beats the worst member; keep size P
        give a promising invalid candidate at most two repair rounds
        if all target shapes are at the floor: stop optimization
        if repeated source/failure: penalize the arm; consider the next strategy
    evaluate frozen winner on held-out cases without adaptation
    report success/failure and verification scope
```

- [x] Use float32 input/output and float32 accumulators. No dtype change may count as reducing transfer waste in this experiment.
- [x] Record `run_id`, `round`, `sample`, `strategy_arm`, `parent_hash`, `source_hash`, `population_snapshot`, the model's strategy label, confidence, raw reply path, code path, per-case bytes/floor/error, failure category, input/completion tokens, prompt-section token estimates (including memory lines), bandit-state reference, elapsed time, and acceptance decision.
- [x] Keep a failed latest candidate separate from the best valid kernel so repair can progress without losing the current winner. Hash duplicates to avoid cycling.
- [x] Stop at the byte floor rather than the first threshold pass; derive level 5/6/7 status from the same verified per-shape measurements.

**Exit condition:** a live model-produced candidate is evaluated and its result is used in the next request. Offline replay is only a wiring check and is excluded from results.

## 9. Task 4 — drive down bytes (hour 3–4.5)

- [x] Before round 1, write and run a clearly labelled probe kernel (human-authored, excluded from agent success counts) that allocates the largest resident layout and slices it as `nc_matmul` operands. If the simulator or compiler rejects it, move retain-one ahead of retain-both in the strategy menu before spending model rounds. **Result: PASSED — every shape exactly at the byte floor in simulation (see `probe_resident_layout_result.json`); retain_both stays first-class.**
- [x] Launch a short pilot using the supplied correct kernel. The following is the planned CLI after Task 3, not an existing command:

```bash
python traffic_agent.py --seed-kernel reference_level4.py --rounds 6 --samples 2 --repeat 1 --context 8192 --max-tokens 3000 --cases traffic_cases.json --population-size 3 --memory-k 6 --learning on --output runs/traffic/pilot
```

**Pilot result (Qwen3-8B, 6 rounds x 2 samples):** round 1 produced a "load both inputs once" kernel that measured **exactly 1.00x on the one shape where its [K, M] SBUF allocation is legal** and raised `partition dimension 256 exceeds maximum 128` on the rest. The raw exception caused the next attempt to abandon the approach (PSUM straight to HBM, a different failure). No valid improvement yet; seed fallback held. This is the repo's recurring lesson in the wild: the raw verdict is not an instruction, so `enrich_feedback()` now names the one change (partition wall and PSUM->HBM wall).

- [x] Prefer retaining both inputs when the resource estimate permits; send actual failure feedback if the compiler/simulator rejects the layout. Next try retaining one complete input plus streaming blocks.
- [x] If the model repeatedly damages arithmetic, narrow the edit to allocations, input loads, and loop placement; preserve the original matmul and final store pattern. Log changes to prompts and restart independent evaluation after tuning.

**Tuning log:** `enrich_feedback()` added to `traffic_agent.py` — simulation exceptions are translated into one named change (partition-dimension wall; PSUM->HBM wall). Prompt layout otherwise unchanged; the recipe appears only in repair feedback, never in generation prompts.
- [x] Maintain a fallback with a measured improvement even if the floor remains unreached. Do not replace a verified result with an unverified lower byte estimate.

**Fallback status after three pilots:** no valid improvement was ever produced, so the fallback was the fully verified seed (2.0x) throughout; no below-floor candidate was ever allowed to replace it.
- [x] Freeze controller, prompts, checker, bandit parameters, memory cap, and population size by hour 4.5. The remaining time belongs to verification and evidence.

**Frozen for the five repeats:** rounds=8, samples=2, population=3, memory-k=6, exploration=1.0, second-parent-prob=0.15, rng seeds 0..4 (one per repeat), context 8192, max-tokens 3000; winners re-evaluated at the second optimization seed. The pilots used rounds=6 while tuning; the repeats are mutually comparable at rounds=8.

## 10. Task 5 — verify finalists and repeat (hour 4.5–7)

- [ ] Run five independent searches from the same supplied seed using the frozen settings. Do not seed later runs with an earlier optimized result. Start every run with a fresh bandit table, empty memory, and population = {seed}; do not carry learning state across runs. Use sampling for Qwen; identical greedy repeats are not independent evidence.
- [ ] Evaluate frozen winners on the held-out cases and hostile values. Record failures without feeding the held-out answers into optimization. Any subsequent repairs need a fresh held-out split or must be labelled development results.
- [ ] For submission entry names, export the same winner by changing only the top-level function name to the required level-5/6/7 entry. Re-evaluate each export; do not infer a pass from the filename.
- [ ] Run an inexpensive diagnostic sweep of all registered NKI levels with the existing agent and a fixed two-round/one-sample budget. Preserve counts for parsing/API/numerical/traffic/unsupported-shape/environment failures; include timeout/unattempted labels if the deadline intervenes. Keep these separate from five-run matmul results.
- [ ] Optional, only if time remains: one control run with `--learning off` (uniform arm choice, population size 1), labelled preliminary. Do not claim a learning-layer effect from one pair of runs.
- [ ] Compile and execute the seed plus best candidate on assigned free cores only, following the seat configuration. Verify outputs on the actual device. Use the AWS debugging-skill recipe: `NEURON_CC_FLAGS="--target trn2 --lnc 1"`, `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, `NEURON_RT_VISIBLE_CORES` pinned to a free core; capture traces with `NEURON_RT_INSPECT_ENABLE=1` and `NEURON_RT_INSPECT_DEVICE_PROFILE=1`. If profiling exposes actual HBM traffic or spills, record those separately from source-level transfer counts. A simulator count cannot establish absence of compiler-inserted spills.
- [ ] If hardware is available, time baseline and winner after warm-up under the same settings. Faster execution is secondary to bytes and must be measured independently.
- [ ] If device validation fails, keep simulation results labelled `simulator_verified`; report compilation/runtime failure. Do not describe the candidate as hardware verified.

**Expected deliverable:** per-run and per-shape minimum valid ratios; target = 1.00×. No promised success rate or speedup before measurement.

## 11. Task 6 — submission and demo (hour 7–8)

- [ ] Package the agent, shared checker, evaluation cases, all attempts including failures, selected kernels, exact token logs, environment versions, and reproduction commands.
- [ ] Write `TRAFFIC-RESULTS.md` with number of runs, spread, all-shape success rates, attempts to first improvement, bandit arm counts vs outcomes, memory sizes, bytes before/after, unchanged thresholds/tolerances, and simulation/device status.
- [ ] Include a failure taxonomy with counts. Distinguish rule violations, incorrect accumulation/layout, shape failures, capacity/compile errors, incomplete traffic coverage, no improvement, and empty/truncated responses.
- [ ] Report confidence estimates alongside held-out outcomes; label their empirical calibration preliminary.
- [ ] Show one recorded real trajectory: supplied correct kernel → measured repeated reads → model rewrite or failed attempt → useful feedback → verified improvement. If no natural failed candidate occurs, demonstrate a separately labelled planted checker failure; never imply it came from the model.
- [ ] Disclose public starting kernels and human-authored components. Present 1.00× only if measured with complete accounting; otherwise present the achieved ratio.
- [ ] Present the learning layer as test-time learning and cite MemCon (arXiv 2607.13591) for the bandit and AlphaEvolve (arXiv 2506.13131) for the population. State plainly that no model weights were trained.

## 12. Deadline decisions

- No working instrumentation after hour 1: stay with the supported `dma_copy` path and analytical per-input accounting; defer general tracing.
- No improved model candidate by hour 4: reduce to one named reuse change and a short API example; keep the live model loop and original grader.
- Byte floor reached: stop byte optimization immediately.
- Device tooling blocked after 45 minutes of focused diagnosis: complete simulation evidence and explicitly label the missing device result.
- Ragged fallback fails: report it in the mandatory correctness assessment and narrow the supported-shape claim; do not represent the submission as satisfying unseen arbitrary shapes.
- Team-size exception unresolved: the technical deliverable can proceed, but event registration compliance remains unresolved until an organizer confirms it or the team meets the documented size.
- Learning layer stalls or corrupts selection: disable it with `--learning off`; the base loop and checker remain the submission.

## 13. Reference checks for implementation

- AWS NKI memory capacities: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/nki.language.tile_size.html
- AWS simulator limitations: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/nki_simulator.html
- Public optimization recipes: https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/nki/guides/tutorials/matrix_multiplication.html
- Neuron Agentic Development (AWS, Apache-2.0): official agents and skills for NKI writing, debugging, profiling — https://github.com/aws-neuron/neuron-agentic-development
  - The writing skill's matmul rules match this harness and confirm the resident-layout approach: stationary `[K<=128, M<=128]`, moving `[K<=128, N<=512]`, K loop always `nl.affine_range`, `accumulate=(k>0)` instead of memset, never write PSUM to HBM mid-accumulation, and slicing a larger SBUF tensor as an `nc_matmul` operand is a documented pattern.
  - The debugging skill gives the device recipe: `NEURON_CC_FLAGS="--target trn2 --lnc 1"`, `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, `NEURON_RT_VISIBLE_CORES` pinned to a free core, torch_xla execution, CPU-side reference, multiple numerical checks; traces via `NEURON_RT_INSPECT_ENABLE=1`, `NEURON_RT_INSPECT_DEVICE_PROFILE=1`, `NEURON_RT_INSPECT_OUTPUT_DIR`.
- MemCon: tabular bandit over memory operations, zero extra model calls — https://arxiv.org/abs/2607.13591
- AlphaEvolve: evolutionary population with an automated evaluator — https://arxiv.org/abs/2506.13131

Match API details to the installed SDK. The current docs may describe newer APIs than the hackathon seat provides.
