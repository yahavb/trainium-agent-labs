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
| 11 *(added)* | attention scores q·kᵀ/√d | stuck | **4 of 4** with building blocks, 2 of 4 without |

![How fast](assets/chart_2_speed.png)

**Is it real?** 228 inputs the agent never saw (`holdout_check.py`): **210 pass, none gives wrong numbers**;
the 18 failures are sizes that aren't multiples of the tile size. On the real chip (`device_check.py`), the
level 3, 4, 7, 9 and 10 programs gave **15 of 15 correct**.

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
7. **Make the loop fast.** The model server used half the chip. On the whole chip: **5.6 → 39 tokens/s per
   request**, a round went from **~120 s to 18–30 s**.

## What we measured about the loop itself (866 logged attempts)

![Where effort was wasted](assets/chart_3_waste.png)

* **70%** of rounds had 4 identical tries, and sampling is deterministic across servers: four seats ran
  **byte-identical** level-8 traces. Per-request seeds crash this vLLM-Neuron build and per-server `--seed`
  did not change the samples, so parallel seats did not add independent tries. **Open problem.**
* **39%** of retries returned the code unchanged — the loop now says "you changed nothing".
* A cut-off reply that looped on one comment line "ran", returned nothing and **outscored honest attempts**.
  Fixed: returning nothing is not running, and cut-off replies rank last.

## Honest limits

* Few runs per level (2–6), one model. "2 of 2" means two tries, not a guarantee.
* The feedback describes the solution's structure in detail: this is the checker teaching the method.
* **Level 8 was solved only with `--attention-plan`**, which names every buffer and all 13 stages — close to
  dictating the kernel. Without it, no run solved it (seat 97, and every run on seats 95–99). Two runs is a
  smoke test, not a rate.
* We checked correctness on the chip, **not kernel speed**; levels 6 and 11 were not run on the chip.
  Level 6 zeroes PSUM before accumulating, which AWS documents as a hazard on older chips — unverified here.
* Level 11's kernel is correct but writes its output twice: 1.2–1.6× the minimum traffic (level 11 has no
  traffic bar). A dtype bug in its reference had reported this as 1.00×; fixed.
* Levels 5–7 load whole inputs on chip and the matrix programs need tile-multiple sizes: fine for the
  tests, not general.
* Most changes are switches (`--lint`, `--level-hints`, `--blocks`, …). Always on: the scoring fixes above,
  the level-2 text fix, and stricter re-checks.

## Next steps (from an independent audit of the level-8 failures)

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
python agent.py --level 11 $COMMON --repeat 4 --blocks solved/level09_transpose_tensor_engine.py
python agent.py --level 8 $COMMON --repeat 2 --attention-plan
python mutation_check.py; python holdout_check.py; python device_check.py; python analysis/trace_analysis.py attempts.jsonl
```

## Files

| file | what it is |
|---|---|
| `agent.py` | the loop, with our changes |
| `nkibench.py` | the checker (bug fixes, stricter re-checks, levels 9–11) |
| `lint.py` | the static check: every memory, shape and API mistake at once, with AWS doc excerpts |
| `solved/` | every program the agent wrote that passed, with its re-check printout |
| `holdout_check.py`, `device_check.py` | the two "is it real?" checks |
| `ab.py`, `analysis/` | fair A/B runner, attempt-log analysis, chart script |
| `mutation_check.py` | seeds 13 observed attention bugs into the level-8 kernel; the checker must name each |
| `tests/`, `results/seat97-repair/` | level-8 independent checks (CPU and device) and the seat-97 evidence |
| `logs/attempts.tar.gz` | the attempt log: every attempt (2,106) with its code, score and the feedback it got |
