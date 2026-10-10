# Task 4 pilot logs — per-attempt scores and learning state (seat-220)

Read this first: **no model weights are trained anywhere in this project.** The learning is
test-time only — a UCB bandit over strategy arms, an attempt memory, and a parent population
(the model authors every kernel). `reward` is the attempt's score: `0` for an invalid
candidate, else the clamped improvement in worst-case waste toward the byte floor (`1.00x`).
`valid=yes` means correct per the checker (rules, numerics, inputs untouched, no hazard); the
traffic gate is scored separately. `worst` is the worst-case waste across the shapes that
simulated; blank means every shape raised. `conf` is the model's own confidence (0-100).
`duplicate` means the model returned the same source hash as an earlier attempt within the run.

Three pilots, rounds=6, samples=2, Qwen3-8B, all on seat-220:

- `pilot` — raw simulator exceptions as repair feedback, repairs chained within the budget.
- `pilot_tuned` — `enrich_feedback()` names the two observed walls; repairs still chained.
- `pilot3` — full `enrich_feedback()` (delegates to the general agent's translator) and no
  repair chaining; the frozen design used for the five Task 5 repeats.

In all three: every bandit arm ends at mean reward 0.00 — no **valid** improvement was ever
produced, so the fallback stayed the verified seed at 2.00x. The only 1.00x kernels the model
wrote were invalid (they hit the partition wall on multi-tile shapes). The model self-reports
95-100 confidence on every attempt, including kernels that raise on three of four shapes:
calibration material for the write-up.

### pilot
| step | arm | decision | valid | conf | worst | reward | strategy |
|---|---|---|---|---|---|---|---|
| 0.0 | bounded_blocking | inserted | yes | 100 | 2.00x | 0.00 | Hoist SBUF tile allocations out of k-loop to reuse tiles a |
| 0.1 | bounded_blocking | duplicate | - | 100 |  |  | Hoist SBUF tile allocations out of k-loop to reuse tiles a |
| 1.0 | retain_both | evaluated | - | 100 | 1.00x | 0.00 | Load both inputs once into SBUF and reuse them for all til |
| 1.1 | retain_both | duplicate | - | 100 |  |  | Load both inputs once into SBUF and reuse them for all til |
| 2.0 | retain_both | evaluated | - | 100 |  | 0.00 | Avoid intermediate SBUF copy by directly copying from PSUM |
| 2.1 | retain_both | evaluated | - | 100 |  | 0.00 | Eliminate intermediate SBUF copy by directly copying from  |
| 3.0 | retain_both | duplicate | - | 100 |  |  | Avoid intermediate SBUF copy by directly copying from PSUM |
| 3.1 | retain_both | duplicate | - | 100 |  |  | Eliminate intermediate SBUF copy by directly copying from  |
| 4.0 | retain_lhs | evaluated | - | 95 | 1.86x | 0.00 | Hoist lhsT tile loading outside k-loop |
| 4.1 | retain_lhs | duplicate | - | 95 |  |  | Hoist lhsT tile loading outside k-loop |
| 5.0 | retain_lhs | evaluated | - | 95 | 1.86x | 0.00 | Reuse rhs tile buffer to avoid redundant allocations |
| 5.1 | retain_lhs | evaluated | - | 100 | 1.86x | 0.00 | Pre-allocate rhs tile to avoid repeated allocation/dealloc |
bandit (arm: plays, mean reward): bounded_blocking: n=1 mean=0.00, retain_both: n=3 mean=0.00, retain_lhs: n=2 mean=0.00, retain_rhs: n=0, tidy_only: n=0
memory entries: 14

### pilot_tuned
| step | arm | decision | valid | conf | worst | reward | strategy |
|---|---|---|---|---|---|---|---|
| 0.0 | bounded_blocking | inserted | yes | 100 | 2.00x | 0.00 | Hoist SBUF tile allocations out of k-loop to reuse tiles a |
| 0.1 | bounded_blocking | duplicate | - | 100 |  |  | Hoist SBUF tile allocations out of k-loop to reuse tiles a |
| 1.0 | retain_both | evaluated | - | 100 | 1.00x | 0.00 | Load both inputs once into SBUF and reuse them for all til |
| 1.1 | retain_both | duplicate | - | 100 |  |  | Load both inputs once into SBUF and reuse them for all til |
| 2.0 | retain_both | evaluated | - | 95 |  | 0.00 | Stack K rows along the free dimension to avoid exceeding p |
| 2.1 | retain_both | evaluated | - | 95 |  | 0.00 | Stack K rows along free dimension to avoid exceeding parti |
| 3.0 | retain_both | evaluated | - | 95 |  | 0.00 | Stack K tiles along the free dimension |
| 3.1 | retain_both | evaluated | - | 95 |  | 0.00 | Stack K dimension across multiple SBUF slices |
| 4.0 | retain_both | evaluated | - | 95 |  | 0.00 | Change the way tiles are sliced to avoid out-of-bounds acc |
| 4.1 | retain_both | duplicate | - | 100 |  |  | Correct tile slicing to avoid out-of-bounds access |
| 5.0 | retain_both | duplicate | - | 100 |  |  | Correct tile slicing to avoid out-of-bounds access |
| 5.1 | retain_both | duplicate | - | 100 |  |  | Correct tile slicing to avoid out-of-bounds access |
bandit (arm: plays, mean reward): bounded_blocking: n=1 mean=0.00, retain_both: n=5 mean=0.00, retain_lhs: n=0, retain_rhs: n=0, tidy_only: n=0
memory entries: 14

### pilot3
| step | arm | decision | valid | conf | worst | reward | strategy |
|---|---|---|---|---|---|---|---|
| 0.0 | bounded_blocking | inserted | yes | 100 | 2.00x | 0.00 | Hoist SBUF tile allocations out of k-loop to reuse tiles a |
| 0.1 | bounded_blocking | duplicate | - | 100 |  |  | Hoist SBUF tile allocations out of k-loop to reuse tiles a |
| 1.0 | retain_both | evaluated | - | 100 | 1.00x | 0.00 | Load both inputs once into SBUF and reuse them for all til |
| 1.1 | retain_both | duplicate | - | 100 |  |  | Load both inputs once into SBUF and reuse them for all til |
| 2.0 | retain_both | evaluated | - | 95 |  | 0.00 | Stack K rows along the free dimension to avoid exceeding p |
| 2.1 | retain_both | evaluated | - | 95 |  | 0.00 | Stack K dimension in SBUF to avoid exceeding partition dim |
| 3.0 | retain_both | evaluated | - | 95 |  | 0.00 | Stack K rows along free dimension to avoid exceeding parti |
| 3.1 | retain_both | duplicate | - | 95 |  |  | Stack K rows along free dimension to avoid exceeding parti |
| 4.0 | retain_lhs | evaluated | - | 95 | 1.00x | 0.00 | Load lhsT once and reuse it for all rhs tiles |
| 4.1 | retain_lhs | duplicate | - | 95 |  |  | Load lhsT once and reuse it for all rhs tiles |
| 5.0 | retain_lhs | evaluated | - | 95 | 1.00x | 0.00 | Pre-allocate and reuse rhs tile buffer |
| 5.1 | retain_lhs | duplicate | - | 95 |  |  | Pre-allocate and reuse rhs tile buffer |
bandit (arm: plays, mean reward): bounded_blocking: n=1 mean=0.00, retain_both: n=3 mean=0.00, retain_lhs: n=2 mean=0.00, retain_rhs: n=0, tidy_only: n=0
memory entries: 14

