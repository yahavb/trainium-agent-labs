# CHIPBOOST status: P3 (`redteam-agent`, seat 101)

*Oct 10 2026. Every number is labelled **chip** (P1's `speedcheck.py` at `referee-timing` 76b2227 on seat 101's
Trainium2: device clock, interleaved A/B, held-out shapes for every would-be `faster`) or **sim** (P3's
temporary `redteam/stage12.py`: rules + NKI simulator, no timing, used before P1's referee merged).*

## The headline

- **Red team, chip:** the honest kernel is accepted (`no_gain`, 1.000x), and **10 of 10 real cheats are caught**.
  The 11th file, `cheat_low_precision`, is not a cheat on bf16 inputs (see 1).
- **Model arms, chip, start kernel, 3 runs x 8 evaluations each:** **0 faster kernels in 48 attempts.** With the
  referee's instruction, Qwen3-8B attempts the restructure and breaks it (23 wrong, 1 correct). Told only "make
  it faster", it returns correct kernels that are not faster (24/24 `no_gain`, best 1.0003x against a 1.010x bar).
- **The referee never let a broken or noise-level kernel through:** 23 wrong kernels rejected, and no 1.000x
  kernel called a speedup.
- **Referee arm v2, with an exact fix named for each repeated mistake:** 21 wrong / 3 no_gain / 0 faster. Qwen repeated the identical kernel after the named fix 18 of 18 times (section 3).

## 1. Red team: 11 planted cheats plus the honest kernel

Each cheat is `reference_level4.py` with one marked `# >>> CHEAT` block (`redteam/cheat_*.py`).
`redteam/run.py` runs them through the referee.

| Cheat | What it does | Caught on the chip? | Stage | Earlier, sim |
|---|---|---|---|---|
| honest kernel | the start kernel itself | **accepted**, `no_gain` 1.000x | timing | accepted |
| zeros | returns its output unwritten | yes | simulator (NaN output) | yes |
| writes_input | right answer, scribbles on `rhs` | yes | simulator (input modified) | yes |
| heldout_only | memorised the dev shapes | yes | compile (see note a) | yes, held-out |
| numpy | `np.matmul` | yes | rules | yes, rules |
| numpy_obfuscated | `getattr(np, "mat"+"mul")` | yes | rules | yes, simulator raise |
| unhooked_dma | `dma_copy` aliased at import, so the byte counter cannot see it | yes | rules | yes, byte floor |
| cached | returns the first call's output for later calls | yes | rules | yes, 2nd seed |
| special_inputs | computes only the first K tile | yes | simulator | yes |
| noise | identical program; any "gain" is noise | yes, `no_gain` 1.000x | timing | pending |
| compile_time | 1 s sleep at trace time | yes | rules (`import time`) | pending |
| low_precision | fp32 operands rounded to bf16 | **n/a: not a cheat here** (note b) | `no_gain` 1.002x | yes |

- (a) It memorised **our** dev shapes, so at P1's Qwen-sized shapes it is wrong already, before held-out. P1's
  own `c3a`/`c3b` cover the held-out case.
- (b) P1's referee feeds bf16 inputs, so rounding them to bf16 changes nothing. In the simulator, with fp32
  inputs, it produced a worst error of **1.14e-2**: our 1e-3 tolerance caught it, and **nkibench's default
  2e-2 would have accepted it** (sim).
- **P1's referee is stricter than ours was.** Four cheats our simulator referee caught by running them are
  stopped by P1's rules scan before anything runs.

## 2. Model arms on the chip (v1)

Same start kernel (`reference_level4.py`, 960.5 us measured on seat 101, the same value as seats 100 and 102),
same referee, same budget, give-up off. Logs: `logs/seat-101/attempts.jsonl` (`by_arm/` per arm).

| Arm | Attempts | Correct | Faster | Best speedup | Distinct kernels |
|---|---|---|---|---|---|
| referee (instruction_given) | 24 | 1 | **0** | 1.000x | 3 |
| model_alone ("Make it faster.") | 24 | 24 | **0** | 1.0003x (noise) | 11 |

Per run: referee [0, 0, 1] correct; model_alone [8, 8, 8] correct; 0/3 runs faster in either arm.

## 3. Referee arm v2: name the change for the two repeated mistakes

*Designed from the seat 100 + 101 logs; `agent.py` 332e557; speedcheck unchanged.* Every v1 wrong verdict was
one of two mistakes, and the model was never told what to change:
- **Crash, "dma_copy ... src=65536, dst=16384":** the model got "fix the error named in the referee message",
  which, by design, it is never shown.
- **Wrong numbers:** one PSUM accumulator allocated before the output-tile loops.

v2 sends our own one-sentence change for each mistake. Only integers and a validated identifier are copied;
the code is only parsed, never run. Replayed on v1's logs, the rules match **all 23** v1 wrong attempts
(A 7, B 16).

**Results (chip, 3 runs x 8, `logs/seat-101/attempts_referee_v2.jsonl`):**

| | v1 | v2 |
|---|---|---|
| wrong / no_gain / faster | 23 / 1 / 0 | 21 / 3 / 0 |
| best speedup | 1.000x | 1.0001x (noise) |
| distinct kernels in 24 attempts | 3 | 3 |
| Rule A fired / Rule B fired | n/a | 21 / 0 |
| same mistake back right after its rule | n/a | 18 of 18 |

**What happened:** all three v2 runs followed **the identical path**:
1. Attempt 1 was correct (`no_gain`).
2. The referee's chip instruction then said "move the operand loads out of the innermost loop" (finding 2 below).
3. Attempt 2 crashed (65536 elements into a 16384-element tile).
4. Rule A named that exactly. Qwen then returned **the same kernel** (one code hash) for all 6 remaining
   attempts of every run.

So v2 fixed the instruction, but not the outcome:
- A precise, correct instruction did not change the model's code.
- The prompt stays the same, sampling is effectively deterministic, and the answer stays the same.

Rule B never fired, because this path never produced the shared-accumulator kernel.

## 4. Failure taxonomy (`redteam/taxonomy.py`, 73 attempts)

| Mode | model_alone, chip | referee, chip | referee, sim |
|---|---|---|---|
| correct, no gain | 24 | 1 | 0 |
| tile too big / wrong size (crash) | 0 | 7 | 7 |
| wrong numbers, cause not named | 0 | 16 | 12 |
| wrong layout / swapped axes | 0 | 0 | 4 |
| kept only one tile | 0 | 0 | 1 |
| shared accumulator across tiles | 0 | 0 | 1 |

On the chip, "cause not named" is 16 because P3's diagnosis (`diagnose.py`) is off with speedcheck: it would
run model code outside P1's sandbox. Rule B in v2 is the safe, parse-only version of it for the most common
case.

## 5. Findings worth a slide

1. **The model wrote one of our cheats unprompted.** Its first answer (sim) dropped the K loop, i.e.
   `cheat_special_inputs`. It was right at K=128 and caught at K=256.
2. **Feedback that names the wrong place makes things worse.** "Hoist the loads out of the innermost loop"
   sent Qwen to the k loop, which has no waste. The result was 8/8 crashes (sim). In this kernel, the
   re-reading is rhs once per m (4x) and lhsT once per n (2x).
3. **An instruction that points at a message the model cannot see is no instruction** (v1 crashes; fixed in v2).
4. **On this chip, bf16 accumulation is impossible** ("nc_matmul dst dtype must be float32 on gen3"), so a
   precision cheat has to round the operands instead.
5. **The model is effectively deterministic here:** samples within a round were identical, and in v2 all 3 repeats
   took the same path, kernel for kernel. **So the "3 runs" are close to 1 trajectory, not 3 independent
   samples.** A same-prompt, same-answer loop is the real blocker: the next lever is varying the prompt or the
   sampling, not wording the instruction better.

## 6. Not verified / caveats

- 3 runs per arm, but the runs are nearly identical (finding 5), so this is closer to one trajectory per arm.
  One model (Qwen3-8B, thinking off, temperature 0.6).
- The model arms start from the plain tiled kernel. P2's random search starts from AWS's expert kernel: a
  separate claim, "tuning the expert's block sizes" (agreed with P4).
- The 25 sim attempts are on fp32 inputs with simulator tolerance 1e-3. The chip runs use bf16 and P1's 4-ulp bar.

## Files

`redteam/cheat_*.py`, `redteam/run.py`, `redteam/stage12.py` (temporary referee), `redteam/taxonomy.py`,
`redteam/redteam_results*.json`, `redteam/taxonomy.json`, `agent.py` (three arms, `--tag`), `diagnose.py`,
`logs/seat-101/`.
