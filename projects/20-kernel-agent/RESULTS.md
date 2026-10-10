# Project 2 — teaching a small model to write chip code (team 20)

**In one line:** the same small model (Qwen3-8B on one Trainium2 chip) went from solving almost nothing to
solving **all 11 levels**: we changed **what the checker tells it after a failed try**, let it **reuse code
it had already got right**, and made the loop **7× faster**. Level 8 (full attention) needed an explicit
stage-by-stage plan in the prompt.

![How the loop works](assets/agent_loop.png)

## How it works

1. The model writes a small program ("kernel") for the chip.
2. A **static check** (`lint.py`) reads it first and lists every memory, shape and API mistake at once.
3. The **checker** (`nkibench.py`) runs it in the simulator and compares the output with the right answer.
4. The checker's message goes back to the model; it tries again (4 tries per round, up to 8 rounds).
5. A level counts as **solved** only when every test input is right — and each solution was re-checked separately.

## Results

![Before vs after](assets/chart_1_before_after.png)

| level | what the program does | before | after (runs solved) |
|---|---|---|---|
| 1 | average pooling | never | **2 of 2** |
| 2 | transpose inside each row | 2–4 of 5 | **5 of 6** |
| 3 | matrix multiply, one tile | never | **2 of 2** |
| 4 | matrix multiply, tiled | never | **5 of 6** |
| 5–7 | the same, moving less data | never reached | **2 of 2 each** |
| 8 | full attention | never | **2 of 2** with `--attention-plan`; never without it |
| 9–10 *(added)* | transpose, softmax — stepping stones to 8 | stuck | **2 of 2 each** |
| 11 *(added)* | attention scores q·kᵀ/√d | stuck | **3 of 3** under a traffic bar (earlier: 4 of 4 with building blocks, 2 of 4 without) |

![How fast](assets/chart_2_speed.png)

**Is it real?**
* **Unseen and hostile inputs** (`checks/holdout_check.py`): 228 cases — new sizes, a dimension of 1, ×50/×100
  scaling, large means, constant rows. **210 pass, none gives wrong numbers**; the 18 failures are sizes that
  aren't multiples of the tile size. Level 8 adds 24 more (`checks/check_attention.py`: zero-Q, constant-V,
  permuted K/V, large logits, odd shapes): 24/24.
* **On the real Trainium2 chip** (`checks/device_check.py`, `results/device/`): every level's current kernel is
  correct on every official shape — **37 of 37 cases** (errors ≤ 3e-5 of the output's size): 34 in the full
  run, plus the re-solved level 11 (3/3) in a second run. (`all_levels.json` lists the level-8 kernel under
  its old file name; it is the same file as `solved/level08_attention.py`.) This caught one thing the simulator missed: the first level-11
  kernel used `np.sqrt` inside the kernel, which the simulator accepts and the device compiler rejects.
  Lint now flags it; the agent's re-solve compiles and runs.
* **Tolerance:** max |error| ≤ 0.02 × the reference output's RMS. Correct kernels land at 1e-7 – 3e-5 (CPU and
  device); every wrong variant in the mutation suite lands at 4.4 – 66 — two orders of magnitude either side.
* **Does the checker catch real bugs?** `checks/mutation_check.py` puts 13 attention bugs from our logs into the
  level-8 kernel: **13/13 named in the right category**, with one documented blind spot (skipping the max
  subtraction passes, because no test input overflows exp).

![Unseen tests](assets/chart_5_unseen.png)

## What made the difference

1. **Name the fix, not the verdict.** "Wrong" became "this is the structure that fixes it", using the
   error's own numbers ("32,768 values into a space for 65,536").
2. **Find every mistake at once.** The simulator stops at the first error, so each round fixed one mistake and
   revealed the next. `lint.py` lists all of them, checks invented commands against the installed NKI, and
   quotes the matching AWS NKI doc rule. Checked on our logs: no false alarm on any correct program; it catches
   every logged memory error and 10 of 13 invented-command errors.
3. **Reuse what already works.** Levels 5–7 start from the agent's own level-4 program; levels 8 and 11 see
   its solved transpose, softmax and matmul programs. Level 11 went from stuck to 4 of 4.
4. **Say which step is wrong.** When attention gives wrong numbers, the checker compares the output with the
   usual wrong versions (scale inverted, Pᵀ·V, softmax on the wrong axis…) and names the one it matches.
5. **Fix the task text.** Level 2's description said the transpose crosses partitions; the reference keeps
   them. Corrected, level 2 went from about 2 in 3 to 5 of 6.
6. **Spell out the stages for attention.** Lighter guidance never solved level 8. Team 20's seat-97
   work added the attention contract and a 13-stage buffer plan (`--attention-plan`): **2 of 2 runs on round
   0, all 4 samples correct**. The kernel also passes 24 extra simulator cases (zero-Q, constant-V,
   permuted K/V, large logits, odd shapes) and the official shapes on the real chip
   ([results/seat97-repair/](results/seat97-repair/)).
7. **Make the loop fast.** The model server used half the chip (TP=2). On the whole chip (TP=4), level 11 went
   from **5.6 to 37.7 tokens/s per request** and from **109 s to 15 s per round** (medians of 96 and 256
   logged requests) [R9].

![Inference speed](assets/chart_6_inference.png)

## Kernels we optimized by hand (separate from the agent's)

`optimized/` holds four of the agent's kernels improved by us — **not agent output**, and `solved/` is untouched.
All pass the checker and run correctly on the chip (14/14 cases); speed is not yet measured.

| level | change | effect |
|---|---|---|
| 11 | scale while leaving PSUM, store once | HBM traffic 1.2–1.6× → **1.00×** the floor |
| 2 | one strided copy instead of F1×F2 single-column copies | 12–128 copy instructions → **1** |
| 10 | one fused exp + row-sum (`activation_reduce`) | no shifted matrix, two fewer copies |
| 6 | first K chunk overwrites PSUM instead of zeroing it first | one reset per output tile removed |

## Token budget

Every attempt logs its real prompt and completion tokens (`logs/attempts.tar.gz`). Median input per attempt:
level 11 **954**, level 8 with four building blocks **2,903** (repair prompts to ~3,300 of 8,192), level 8
with the stage plan **1,441**. The bloated level-8 prompt failed; the shorter, explicit one solved it.

## What we measured about the loop itself

Every chart below is computed from the attempt log in this repo (`analysis/make_charts.py`).

![Where effort was wasted](assets/chart_3_waste.png)

* **40%** of rounds had identical samples, and sampling is deterministic across servers: four seats ran
  **byte-identical** level-8 traces. Per-request seeds crash this vLLM-Neuron build and per-server `--seed`
  did not change the samples, so parallel seats did not add independent tries. **Open problem.**
* **17%** of retries returned the code unchanged — the loop now says "you changed nothing"; **30%** of rounds
  failed exactly like the round before; **18%** of failures were invented API calls (now caught by lint
  against the installed NKI).

![Failure taxonomy](assets/chart_4_failures.png)

**Failure taxonomy** (2,372 failed attempts, by the first problem reported): wrong shape or layout dominates
every matmul and attention level; invented APIs dominate level 1 and 10; level 8's 1,096 failures are mostly
the q·kᵀ layout (476, `logs/level8_failures_summary.json`).
* A cut-off reply that looped on one comment line "ran", returned nothing and **outscored honest attempts**.
  Fixed: returning nothing is not running, and cut-off replies rank last.

## Honest limits

* Few runs per level (2–6), one model. "2 of 2" means two tries, not a guarantee.
* The feedback describes the solution's structure in detail: this is the checker teaching the method.
* **Level 8 was solved only with `--attention-plan`**, which names every buffer and all 13 stages — close to
  dictating the kernel. Without it, no run solved it (seat 97, and every run on seats 95–99). Two runs is a
  smoke test, not a rate.
* We checked correctness on the chip, **not kernel speed**: an NKI 0.6 call re-compiles every time
  (~1.7 s), so wall-clock time measures the compiler. Speed needs `neuron-profile` on a compiled kernel.
* Level 6 zeroes PSUM before accumulating (AWS documents this as a hazard on some chips); it was correct
  on our trn2 for all 4 shapes, and `optimized/` removes the reset anyway.
* Level 11 got a traffic bar (1.1× the floor) after its first solve stored the scores twice; the 3 of 3
  above is under the bar.
* Levels 5–7 load whole inputs on chip and the matrix programs need tile-multiple sizes: fine for the
  tests, not general.
* Most changes are switches (`--lint`, `--level-hints`, `--blocks`, …). Always on: the scoring fixes above,
  the level-2 text fix, and stricter re-checks.

## Next steps: solve level 8 without the stage plan (from an independent audit of its failures)

* Level 8 fails on composition, not maths: it pastes the scores block's HBM stores into the middle of
  attention, sizes P as (seq, dim), and transposes V instead of P. Give **composable SBUF-to-SBUF stages**
  as building blocks, not standalone kernels.
* Name the exact P·V repair ("transpose the probabilities, keep v as is") instead of the generic Aᵀ·B advice.
* Detect A→B→A cycles across a run (code hashes), not only an unchanged reply.

## Rerun it

```bash
TP=4 MAX_MODEL_LEN=8192 BLOCKS=1024 VLLM_EXTRA_ARGS="--seed 1" ./serve.sh     # whole chip: 7x faster
cd /workspace/projects/20-kernel-agent
python nkibench.py --selftest
COMMON="--rounds 8 --samples 4 --context 8192 --model Qwen/Qwen3-8B --level-hints --lint"
python agent.py --level 2 $COMMON --repeat 6 --echo-check
python agent.py --level 7 $COMMON --repeat 2 --seed-from solved/level04_matmul_tiled.py
python agent.py --level 11 $COMMON --repeat 3 --blocks solved/level09_transpose_tensor_engine.py,solved/level03_matmul_single_tile.py
python agent.py --level 8 $COMMON --repeat 2 --attention-plan
python checks/mutation_check.py; python checks/holdout_check.py; python checks/check_attention.py solved/level08_attention.py
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_RT_NUM_CORES=1 python checks/device_check.py   # a free NeuronCore
```

## Files

```
projects/20-kernel-agent/
├── RESULTS.md            this note
├── agent.py              the agent loop (prompting, retries, sample selection, telemetry)
├── nkibench.py           the checker: references, test shapes, rules, simulator, tolerance, traffic bar
├── lint.py               static check before simulation: every memory/shape/API mistake at once
├── checks/               is it real?  holdout_check (228 unseen/hostile cases) · check_attention (24 level-8
│                         cases) · mutation_check (13 seeded bugs) · device_check (+ level-8 device runner)
├── solved/               every kernel the agent wrote that passed, with its re-check printout
├── optimized/            four of those kernels optimized by hand (team-written, labelled)
├── results/              device/ (real-chip results) · seat97-repair/ (level-8 evidence)
├── logs/                 attempt log (1,418 attempts) + summary of the 1,096 failed level-8 attempts
├── analysis/             trace_analysis (failure taxonomy), make_charts, ab (A/B runner)
├── assets/               the diagram and charts
└── reference_level1-4.py, kernelbench.py, try_level.py, CHALLENGE-kernel-agent.md   (from the original repo)
```

## References

**AWS Neuron documentation** (the rules the checker and lint enforce, and the source of their doc excerpts)

* [R1] `nki.isa.nc_matmul` — computes stationaryᵀ·moving; SBUF operands, PSUM result; accumulation rules:
  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul.html
* [R2] `nki.isa.nc_transpose` — Tensor engine (SBUF → PSUM, ≤128×128) vs Vector engine (≤32×32):
  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_transpose.html
* [R3] `nki.isa.activation` / `activation_reduce` — fused bias and row reduction (used in `optimized/`):
  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.activation.html
* [R4] `nki.isa.dma_copy`, `nki.isa.tensor_copy` — which memories each may touch, element-count rules:
  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.dma_copy.html
* [R5] NKI programming guide and tiling (128-partition limit, PSUM bank size):
  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/
* [R6] AWS agent skills for NKI (memory patterns, transpose and layout, language constraints):
  https://github.com/aws-neuron/neuron-agentic-development
* [R7] Neuron Runtime configuration (`NEURON_RT_NUM_CORES`, used to grab a free core for the device runs):
  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/guides/configuration-guide.html

**Evidence in this repo** (every headline number, and where to check it)

| claim | evidence |
|---|---|
| levels solved, rounds needed | `logs/attempts.tar.gz` (per-attempt `reward`, `round`, `selected`); `solved/*.check.txt` |
| level 8 solved with the stage plan | `results/seat97-repair/agent-planned.jsonl`, `.log`, `solved/level08_attention.provenance.json` |
| 210/228 unseen and hostile inputs | `checks/holdout_check.py` |
| 37/37 on the real chip; optimized 14/14 | `results/device/all_levels.json`, `results/device/level11_resolve_and_optimized.json` |
| 13/13 seeded bugs caught | `checks/mutation_check.py` (re-runs in ~1 min on a seat) |
| tolerance margins | `solved/*.check.txt`, `results/device/*.json` (`max_rel_err`), mutation suite output |
| [R8] lint has no false alarm on correct kernels | lint returns nothing on all 11 `solved/` and 4 `optimized/` kernels |
| [R9] 7× faster inference | `chart_6_inference.png`, computed from `prompt_tokens`/`completion_tokens`/`seconds` in the log |
| failure taxonomy | `chart_4_failures.png` + `logs/level8_failures_summary.json`, computed by `analysis/make_charts.py` |

## Rubric checklist

| asked for | where |
|---|---|
| the agent | `agent.py` (+ `lint.py`) |
| verification harness, tolerance and reasoning | `nkibench.py`; tolerance and its justification under "Is it real?" |
| eval set incl. hostile values | `checks/holdout_check.py` (228 cases), `checks/check_attention.py` (24), `checks/mutation_check.py` (13 bugs) |
| failure taxonomy with counts | chart "What the model got wrong" (`assets/chart_4_failures.png`), `logs/level8_failures_summary.json`, `analysis/trace_analysis.py` |
| token instrumentation | "Token budget" and the inference chart above; per attempt in the log |
| attempt log | `logs/attempts.tar.gz` (+ `logs/level8_failures_summary.json`), `results/seat97-repair/agent-planned.jsonl` |
| one-page reproduction note | "Rerun it" above |
| does the agent know when it failed? | it reports SOLVED only when every shape passes the harness, otherwise the best score and the failure; nothing is called verified that the harness did not pass |
