# API-level error messages (v1), levels 1-4

- **Who / seat:** Rishabh (+ Claude), seat 48 (its own model server, nothing else running)
- **Commit measured:** uncommitted on `team-main` (base `8f1ca41`): `agent.py` md5 95739ffb…,
  `nkibench.py` md5 68e795d5…
- **Hypothesis:** messages that map what the model was reaching for to the real NKI call move the
  levels stuck on API mistakes.
- **Change:** see `claims/rishabh-api-messages.md` (v1): invented names map to the intended call,
  `nl.sbuf` indexing gets an allocation fix, and NaN output says "never written".
- **Command:** `python agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5`, **stopped
  after 2 full runs plus levels 1-3 of run 3** (user out of time). Levels 1, 3 and 4 have zero spread,
  so one run is informative for them.
- **Log:** `logs/attempts-rishabh-api-messages.jsonl` (240 attempts)

## Scores

```
before (baseline): L1 0.30 x4   L2 [1.00, 0.30, 0.30, 1.00]   L3 0.30 x4   L4 0.62 x4
after  (v1):       L1 0.30 x3   L2 [0.30, 1.00, 0.30]          L3 0.30 x3   L4 0.62 x2
```

**No score moved.** Level 2's 1/3 against the baseline's 2/4 is within its noise.

## What did move: which error the model gets stuck on

- **Level 1:** invented `nisa.multiply` / `scalar_mul` fell from 80 of 160 attempts (50%, full baseline log) to 12 of 96 (12%). Qwen
  now calls the real `nisa.tensor_scalar` (24x "missing required positional argument") and hits the
  copy-size mismatch (36x). It moved one wall forward, to the next error.
- **Levels 2-4:** unchanged walls: copy-size mismatch, reshape / 1-D tile, partition > 128. Those
  messages didn't change in v1.

## Verdict

**Kept as a checker improvement, not a score improvement.** The targeted errors mostly disappear,
and the model moves to the next one. That fits the loop-tool session's finding: Qwen applies the
LATEST named fix and drops earlier ones, so message tuning alone cycles rather than converges. What
moved level 4 today was a different prompt structure (`results/2026-10-10-rishabh-loop-tool-staged.md`:
an explicit step list and one hole per prompt, 5/5), not messages.
