# Menu-guided repair (the Jev / System-One pattern) — results, checks, limits

**One-line result.** The model, acting as a **classifier over a fixed menu of repair
guidances**, selected and applied the fixes that took a wrong-numbers kernel to the exact byte
floor (1.00x on all four shapes, accepted, re-verified at two evaluation seeds) — from the raw
near-miss start, in one run. A second start exposed a real failure mode (applier overshoot, then
a stuck classifier), and the one simplification we checked — removing the worked-example block
from the applier prompt — **failed its check** and was rejected. Details and honest limits below.

**Disclosure.** The menu, the `exact` change texts, and the deterministic expected-guidance
rules were authored by the team from the day's runs. Selection (classify → choose one id) and
application are model-driven. All results are simulator-verified, one run per start state, on
the four benchmark shapes. The pattern follows TypeSafe's Jev System One split (a classifier
owns the bounded choice; a generative model executes the chosen action), implemented with the
local Qwen3-8B — not the hosted Jev.

## What was built

- `guidance_menu.py` — 9 options (8 repair actions + 1 back-out control). Each option has
  `when` (the selection trigger, written as the classifier's criteria) and `exact` (the
  complete change the applier must make).
- `guidance_repair.py` — per round: (1) evaluate the kernel and run the behavioral diagnosis;
  (2) CLASSIFIER prompt = state + menu → reply `CHOICE: <id>`, `REASON`, `CONFIDENCE 1-5`;
  (3) APPLIER prompt = chosen `exact` text + kernel (+ demo block) → full updated kernel.
  Every choice is scored against a deterministic expected guidance derived from the same state.

## Run 1 — seat 223, raw near-miss start: **solved**

Start: the model's own earlier candidate `r8_s0` (1.857x on bytes, wrong numbers on three of
four shapes; the "stationary operand never advanced" signature).

| round | state | classifier choice | expected? | outcome |
|---|---|---|---|---|
| 0 | wrong numbers (1.857x bytes) | `stack_stationary_operand` | match | valid 1.857x in ONE application |
| 1 | valid 1.857x (gate-only fail) | `continue_from_best` | **miss** (read "not accepted" as "not valid") | back-out absorbed it — same kernel kept, one round spent |
| 2 | same state | `stack_moving_operand_symmetrically` | match | round 3: **1.00x accepted on all shapes** |
| 3 | — | — | — | SOLVED; winner written, re-verified seeds 0 and 1 (`accepted True`, worst 1.0) |

Classifier agreement 2/3, 0 parse fallbacks. Prompts: classifier ~1055–1372 tokens, applier
~1738–1755 tokens (with demo). Winner: `agent_kernel_menu_guided_1p00x.py` + two seed evals.

## Run 2 — seat 220, restacked start: **not solved**

Start: `r1_stacked` (two shapes at 1.00x; two failing with
`dma_copy requires src and dst to have the same number of elements`).

| round | classifier choice | expected? | outcome |
|---|---|---|---|
| 0 | `fix_cache_stride_and_width` | match | applier fixed the copy arithmetic BUT overshot: stationary slices came out 256/512 wide → `Matmul stationary free dimension ... exceeds gemm_stationary_fmax=128` |
| 1–4 | `stack_stationary_operand` ×4 | miss ×4 | state unchanged each round; no progress; budget exhausted |

Classifier agreement 1/5. Final: best progress 0.70 (the start state). Three gaps exposed:
(a) a **right choice can still fail in application** — the fix text was followed but the
geometry overshot; (b) the state line `[not accepted: ...]` is ambiguous between wrong numbers
and gate-only; (c) the menu has **no crisp trigger** for the `gemm_stationary_fmax` error, and
nothing in the driver forces a different choice after a no-op round — the classifier repeated
the same id, confidence 5/5 every time.

## The simplification check — removing the worked-example block: **rejected**

Hypothesis: the `demo.json` block (~300 tokens, ~17% of the applier prompt) is redundant with
the chosen guidance's `exact` text, and its meta label ("TEAM-AUTHORED... NOT produced by the
model...") is noise.

Check (once, per the rule): identical start (`r8_s0`), identical driver, single change —
`--no-demo` (applier prompt without the block).

- Baseline (with demo): solved — valid 1.857x in one application, floor at round 3.
- Check (no demo): the classifier chose the same guidance correctly at round 0, but the
  applier's kernel came out **wrong**: numerical mismatches persisted on all three broken
  shapes, and the K=512 shape moved only **0.50x the floor** — impossible for a correct
  computation, still zeroed by the checker. All six rounds then chose the same guidance
  against the unchanged state (agreement 1/6); no solve. The applier prompt shrank by
  ~340–400 tokens (1353–1416 vs the baseline's 1738–1755) — that was the size of the
  removed anchor.

**Verdict: keep the block.** A concrete, correct code fragment anchors the geometry (widths,
strides, offsets) in a way the rules-style `exact` text alone did not, for this model. Removing
"useless-looking" instructions is not decidable by reading them; it has to pass a run. This one
failed. (The check also reproduced Run 2's loop pathology independently.)

## Limitations

1. **The ceiling is the menu.** The model cannot invent a repair; new failure classes need a
   human to extend the menu (Run 2's fmax error showed exactly this).
2. **The knowledge sits in the `exact` texts** (human-authored); the model selects and applies.
3. **"Accuracy" is agreement with our own rules.** The expected-guidance mapping has gaps (the
   fmax case was mislabeled `continue_from_best`); numbers need manual adjudication, not blind
   trust.
4. **Selection and application blur in outcomes.** Run 2 round 0 was a correct choice, failed
   application.
5. **No loop-breaker; uncalibrated confidence.** The model re-picked a failed guidance four
   times, confidence 5/5. The 1–5 number is self-reported text, not a calibrated probability —
   the opposite of what real System One models provide.
6. **State wording is load-bearing.** Run 1's one miss came from reading a display line; small
   models read state literally.
7. **Most options are untested.** Only 3 of 8 repair entries were exercised across both runs.
8. **Repair, not discovery.** The solved run started from a near-miss whose byte-saving
   structure the model had found earlier (unguided); starting from the plain 2.00x seed is
   untested.
9. **Scale.** Prompt cost grows with the option count (9 options ≈ 800 tokens of triggers) on
   an 8k-context, 8B model. One solve and one failure is not a rate; simulator-only.

## Files

| file | what it is |
|---|---|
| `guidance223_rounds.jsonl`, `guidance223_summary.json`, `guidance223_replies/` | Run 1: every state, choice, reason, confidence, tokens; raw classifier/applier replies |
| `agent_kernel_menu_guided_1p00x.py` + `_seed0.json` / `_seed1.json` | Run 1 winner and two-seed verification |
| `guidance220_rounds.jsonl`, `guidance220_log.txt`, `guidance220_replies/`, `guidance220_best.py` | Run 2: the honest failure, complete |
| `guidance223_nodemo_rounds.jsonl`, `guidance223_nodemo_log.txt`, `guidance223_nodemo_best.py`, `guidance223_nodemo_r1_kernel.py`, `guidance223_nodemo_replies/` | the failed removal check, complete |
| `analyze_guidance_runs.py` | reproduces the tables above from the logs |
| `../guidance_menu.py`, `../guidance_repair.py` | the menu and the classify-then-apply driver |
