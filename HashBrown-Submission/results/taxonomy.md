# taxonomy.md — failure taxonomy from the honest baseline run, under the hardened checker

Mandatory deliverable (`CHALLENGE-kernel-agent.md`: "The failure taxonomy — what it got
wrong, grouped, with counts"). Source: every one of the 252 attempts logged in
`/workspace/baseline-copy/attempts.jsonl` (the honest Qwen3-8B baseline, `--all --rounds 8
--samples 4 --context 8192 --repeat 3`, started 2026-10-10 18:59 UTC, finished ~20:00 UTC,
scored by the ORIGINAL pre-hardening checker), re-scored here under the CURRENT hardened
checker (`agent.grade(code, level, augment=False)` — hardened A0, not an augmentation tier).
Full methodology and the false-verified analysis are in `HARDENING.md`'s Phase 4b Findings
log entry; this file is the condensed counts table.

## Headline number: 252 attempts re-scored, 3 correct, 249 wrong

Only level 2 ever produced a genuinely correct kernel (3 of 252 attempts, all round 0 of
different repeats). **Every other attempt — all 249 of them, across all 4 levels — failed
with a Python exception raised DURING simulation, before the checker's numeric comparison
(`describe_mismatch`) ever ran.** Not one non-passing attempt reached "wrong answer, right
shape" — they are all "the kernel doesn't even run as written." This reshapes what "harden
the checker" means for this particular baseline: tolerance, hostile values, and ragged shapes
are all downstream of a numeric comparison that this agent/model combination essentially
never reaches.

## Counts by level and category

| level | category | count | what it means |
|---|---|---|---|
| 1 | invented_api_call | 48 | called a `nki.isa`/`nki.language` attribute that doesn't exist (e.g. a guessed name) |
| 1 | dma_shape_mismatch | 24 | `dma_copy`'s source and destination tile don't have the same element count |
| 1 | invalid_keyword_arg | 12 | passed a keyword argument an NKI function doesn't accept |
| 1 | wrong_buffer_placement | 12 | allocated a tile in the wrong memory region (e.g. sbuf where psum is required) |
| **1 total** | | **96** | **0 correct** |
| 2 | (correct) | 3 | genuinely right, every shape, on 3 attempts (round 0 of 2 of the 3 repeats) |
| 2 | dma_shape_mismatch | 17 | same as above |
| 2 | out_of_bounds_index | 16 | indexed past the end of a tensor along some dimension |
| **2 total** | | **36** | **3 correct (8.3%)** |
| 3 | reshape_disallowed | 48 | called `.reshape()` on an NKI tensor, which this dialect does not support |
| 3 | partition_exceeds_max | 21 | tried to allocate/copy a tile with partition dimension > 128 |
| 3 | dma_shape_mismatch | 3 | same as above |
| **3 total** | | **72** | **0 correct** |
| 4 | partition_exceeds_max | 29 | same as above — level 4's larger shapes (K or M up to 512) make this the dominant failure |
| 4 | assign_shape_mismatch | 18 | assigned a value of one shape into a destination slice of a different shape |
| 4 | wrong_buffer_placement | 1 | same as above |
| **4 total** | | **48** | **0 correct** |

**252 total.** No attempt, at any level, fell into `rule_violation`, `non_finite_output`,
`zero_output`, `numerical_mismatch`, `input_tamper`, `hardware_hazard`, or `traffic_bar` —
every one of the checker's NUMERIC-layer categories is empty for this baseline, because no
attempt's kernel got far enough to be numerically compared at all. The static rule-checker
layer is also silent here (0 rule_violation) — Qwen3-8B's attempts are not committing the
"hand the whole op to numpy" cheat; they're failing purely on NKI's own API/shape mechanics.

## Three families, out of 249 failures

The 11 fine-grained categories above collapse into three families:

| family | categories rolled up | count | % of 249 failures |
|---|---|---|---|
| **shape / tiling** | `dma_shape_mismatch` (44), `out_of_bounds_index` (16), `reshape_disallowed` (48), `partition_exceeds_max` (50), `assign_shape_mismatch` (18) | **176** | **70.7%** |
| **invented API** | `invented_api_call` (48), `invalid_keyword_arg` (12) | **60** | **24.1%** |
| **buffer placement** | `wrong_buffer_placement` (13) | **13** | **5.2%** |

**70.7% of all failures are shape/tiling mistakes** — getting a tile's dimensions, a
destination slice, or a partition size wrong, not using a nonexistent name or the wrong
memory region. That's consistent with the ladder's own stated traps (rule 4 in
`CHALLENGE-kernel-agent.md`: "handle shapes that don't divide evenly... this is where most
generated kernels quietly break") — except here the model is breaking on shape mechanics
before it even gets to data that doesn't divide evenly; it can't reliably produce a
correctly-shaped tile at all. **Invented API calls (24.1%)** are the second-largest family —
guessing a plausible-sounding `nl`/`nisa` name or keyword argument that doesn't exist.
**Buffer placement (5.2%)** — sbuf vs. psum confusion — is a distant third.

## False-verified count: 0

**Of the 3 attempts the ORIGINAL (pre-hardening) checker scored 1.0, all 3 also score 1.0
under the hardened checker.** Zero false-verified. See `results/false_verified.csv` (empty
except a header note) and `HARDENING.md`'s Phase 4b Findings log entry for why this is a
real, not a null, result: the baseline agent so rarely produced a working kernel at all (3 of
252 attempts) that there was almost nothing for the Phase 3 hardening (tighter tolerance,
level 3's second shape, tamper-check parity) to have a chance to catch — the three solved
kernels happen to be genuinely, unambiguously correct level-2 transposes, not near-misses
that the old 2e-2 tolerance was quietly waving through.

**Interpretation: false-verified is 0 because 249/252 attempts crash before numeric checking
is ever reached — so for THIS model, at THIS skill level, the closed loopholes are
PREVENTIVE, not (yet) REACTIVE.** Phase 3's fixes (level 3's second shape, the measured
tolerance, tamper-check parity) and Phase 4's augmentation tiers all operate on a kernel that
already runs and returns a same-shaped, finite answer — exactly the stage 249 of 252 Qwen3-8B
attempts never reach. That does NOT mean the hardening was wasted work: `cheats/c1_do_nothing.py`
through `cheats/c11_assume_divisible.py` are hand-written, deliberately-passing kernels that
DO reach that stage, and the hardened checker catches every one of them (`results/cheats_before.csv`,
`results/cheats_after.csv`). The loopholes are real and closed; this baseline model simply
isn't skilled enough, yet, to produce the kind of plausible-looking-but-wrong kernel that
would need them. A stronger model or a better-tuned prompt that gets PAST the shape/API stage
is exactly the scenario this hardening was built for, and the cheats prove it would hold.

## Level 4 variance — contradicts the repo's "zero variance" claim

`HARDENING.md` section 9 ("Known context from STATE.md") states: "Baselines (Qwen3-8B, 5
runs): ... L4 0/5 always 0.62. Levels 1, 3, 4 have zero variance." **This run's actual level 4
rewards across its 3 repeats were `[0.62, 0.3, 0.62]`** (from `results/baseline.log`'s "over 3
runs" summary) — NOT always 0.62. One of the three repeats scored only 0.30 (the same score
level 1 and level 3 get every time — i.e., a repair loop that never got past the rules/parses
stage, consistent with `give-up-after` triggering early). Logged here as a direct
contradiction of the repo's prior claim, not a new regression: the claim was presumably true
for whatever 5 runs it was based on, but this run's variance is real and should update that
record rather than be treated as noise. See `HARDENING.md`'s Findings log for the full
discussion.
