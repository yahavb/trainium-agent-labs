# Formal level check — levels 5, 6, 7 (relaxed bar: any valid below 2.00x)

**Bottom line.** The menu-guided agent loop produced, and the official checker accepted, a
kernel at **exactly the byte floor (1.00x, all four shapes, seeds 0 and 1, CLI exit 0)** for
**each of levels 5, 6 and 7** — one native kernel per level, each with that level's entry-point
name. Under the relaxed bar ("anything below 2.00x counts") there are also two verified **1.857x**
intermediates. The from-seed route (starting from the shipped 2.00x kernel) **still fails** —
that failure story is documented below, because it is the most useful robustness evidence here.

## Verdict table

| level | kernel | native entry | how it was produced | worst waste | formal acceptance |
|---|---|---|---|---|---|
| 5 | `agent_kernel_menu_guided_1p00x.py` | `nki_matmul_hoist_load_` | menu-guided loop from the model's own near-miss (`r8_s0`); classifier 2/3 (one back-out absorbed) | **1.00x** | accepted, seeds 0 and 1; `nkibench.py --level 5 --check` exit 0 both |
| 6 | `agent_kernel_l6_nearmiss_1p00x.py` | `nki_matmul_block_free_dimension_` | near-miss path; the floor application landed in the 2-round continuation retry (classifier 1/1) | **1.00x** | accepted, seeds 0 and 1; CLI exit 0 both |
| 7 | `agent_kernel_l7_nearmiss_1p00x.py` | `nki_matmul_fully_optimized_` | menu-guided loop from the near-miss with the level-7 entry; classifier 2/2 | **1.00x** | accepted, seeds 0 and 1; CLI exit 0 both |

Valid intermediates below 2.00x: `agent_kernel_l6_best_1p86x.py` (level-6 run's best) and
`agent_kernel_verified_1p86x.py` (surgical family), both verified.

## Robust check (multi-core, official checker internals)

Run on the level-7 kernel (`robust_check.py --jobs 6`; the three level kernels share the same
algorithm). Sources: `robust_l7_winner.json`, `robust_l7_winner_log.txt`.

| case family | passed | note |
|---|---|---|
| optimization shapes, level-5 bar | **8/8** | seeds 0 and 1 |
| optimization shapes, level-6 bar | **8/8** | |
| optimization shapes, level-7 bar | **8/8** | the tightest gate |
| held-out aligned shapes (seeds 17/29/43) | **9/9** | beats the level-5 bar on shapes never used for optimization |
| hostile value families (zeros, negative-mixed, repeated rows, cancellation, large finite) | **20/20** | five families x four shapes |
| ragged correctness shapes (seeds 7/11/23) | **0/9** | **honest failure**: the kernel assumes tile-multiple shapes; K=1/129/257, M=127/131, N=513/519 break it (the challenge's "partial final tile" trap — still open) |

## The from-seed route — honest failure

Starting from the shipped 2.00x seed (renamed per level), all three formal runs failed:

1. Round 0 is valid at 2.00x; the classifier chose `stack_moving_operand_symmetrically`, whose
   precondition — an existing lhs cache — the seed does not meet. The applier applied it anyway:
   the kernel broke (`UnboundLocalError: k_tiles`; on level 6 a `gemm_stationary_fmax` overshoot
   instead), and the classifier then misread the broken states as "correct on every shape but too
   much traffic" and repeated the same choice (levels 5 and 7: five repetitions; level 6 chose
   the correct `keep_k_on_partition_axis` only in its final round).
   Logs: `formal_seed_l{5,6,7}_log.txt`, `formal_seed_l{5,6,7}_rounds.jsonl`.
2. With the menu's precondition fix (verified, `b4ee80e`), the applier added **both caches — but
   inside the output-tile loops**, reloading the full lhs and rhs for every tile: traffic got
   *worse* (2.67x on shape 2, 4.00x on shape 4 — the byte counts match tile-count x full-operand
   arithmetic exactly). The classifier repeated `stack_moving` four more times; the menu's
   `hoist_invariant_load` entry (whose "when" matches exactly this state — the loads do not
   depend on m/n) was never chosen. Logs: `recover_seed_l5/l7` (archived with this report).

So: **the near-miss start is load-bearing.** The loop reliably repairs and completes the model's
own partially-found byte-saving structure; it has not yet transformed the clean 2.00x kernel.

## Disclosure

- The menu, the `exact` texts and the deterministic expected-guidance rules are team-authored;
  selection and application are model-driven. Each run's per-round log records the classifier's
  choice, reason, self-reported confidence and both replies.
- "Classifier agreement" numbers score the model against our own rules; known quirks are
  adjudicated in the logs (e.g., from a clean seed, caching rhs first vs lhs first are both
  defensible orders — the rules arbitrarily expect lhs first).
- The level-6 floor landed on a retry: the same guidance application on the same state produced
  a broken kernel in the first attempt and the floor in the retry (the applier is
  near-deterministic, not deterministic). Disclosed rather than cherry-picked.
- Everything is `simulator_verified`; no device execution. The ragged-shape failures above are
  real checker results, not a gap in coverage.

## Files

| file | what it is |
|---|---|
| `agent_kernel_menu_guided_1p00x.py` (level 5), `agent_kernel_l6_nearmiss_1p00x.py`, `agent_kernel_l7_nearmiss_1p00x.py` | the three native level kernels (all at 1.00x) |
| `agent_kernel_l{5,6,7}_*_seed{0,1}.json` | structured evaluator results, both seeds |
| `agent_kernel_l{5,6,7}_*cli_seed{0,1}.txt` | official `nkibench.py` CLI runs (exit 0) |
| `agent_kernel_l6_best_1p86x.py` | the level-6 run's verified 1.857x intermediate |
| `recover_l6_rounds.jsonl`, `recover_l6_retry_rounds.jsonl`, `recover_l7_rounds.jsonl`, `recover_l7_summary.json` | the run logs behind the level 6 and 7 kernels |
| `robust_l7_winner.json` / `_log.txt` | the full robust matrix |
| `formal_seed_l{5,6,7}_*` | the from-seed failures (logs + round logs) |
| `../robust_check.py`, `../guidance_repair.py`, `../guidance_menu.py` | the harnesses that produced all of it |
