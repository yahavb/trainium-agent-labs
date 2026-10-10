# NOTE.md — checker hardening, one-page reproduction note

Hack the Chip 2026 (NYU × Annapurna Labs), Project 2 kernel-agent checker. Full detail and
measurements: `HARDENING.md` (living plan + Findings log), `CHECKER.md` (per-rule
reference), `results/eval_set.md`, `results/taxonomy.md`.

## What we ran

Pod `seat-66` (1/16 `trn2.48xlarge`, NKI 0.6.0), Qwen3-8B served locally (`--context 8192`).
Two full `agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 3` runs: **baseline**
(`/workspace/baseline-copy`, original pre-hardening checker, 2026-10-10 18:59–20:00 UTC, 252
attempts) and **augmented regression** (`/workspace/hardened-copy`, hardened checker +
`--augment` — a real flag added to `agent.py`'s `grade()`, not a workaround,
20:08–21:07 UTC, 256 attempts).

## The cheats, and what catches each

Ten hand-written kernels pass the ORIGINAL checker; the HARDENED checker catches all ten.
C1-C8 (do_nothing, partial_rows, hardcoded_shape, constant_output, input_tamper,
almost_right, edge_skip, path_mismatch) are caught by **three structural fixes alone, at A0,
no augmentation needed**: level 3 given a second declared shape, tolerance measured and
tightened (1e-4, from a 3.1e-6 noise floor), and `nkibench.verify()`/`--check` given the same
input-tamper check `agent.grade()` always had. C9 (memorise_all_shapes) and C11
(assume_divisible) are smarter — each passes A0 and is caught at exactly the augmentation
tier it was designed against (A2a, A2b) and stays caught after. C10 and C12 were attempted
and dropped: NKI blocks reading a concrete scalar out of a tile (confirmed via two failing
probes), and no float32 precision cliff exists in our 1e-6..1e4 test range for these four ops
(avgpool/transpose/matmul have no cancellation-prone formula).

## Cost and recommended default

Full augmentation runs ~8x A0's checks; wall-clock is dominated by level 4 (0.70s→6.49s,
~9x), levels 1-3 stay under 0.2s. **A2a/A2b are the only tiers with a demonstrated catch and
are nearly free** (±0.01s/kernel); A1/A3/A4 cost real time for zero new catches on this cheat
set. **Recommended default: A0+A1+A2a+A2b every round; A3+A4 for periodic audits.**

## Runs, spread, and regressions

Baseline (3 repeats): L1 0/3 (0.30 flat), L2 2/3 (1.00, 1.00, 0.30), L3 0/3 (0.30 flat), **L4
0/3 `[0.62, 0.30, 0.62]`** — contradicts `HARDENING.md` section 9's prior "L4 has zero
variance" claim; corrected there, not deleted.

Augmented regression vs. baseline: **L1/L3 identical.** **L2 dropped 2/3→1/3 — confirmed
NOT a real regression**: of 48 level-2 attempts, exactly one passes the original 4 shapes on
its own, and it also scored 1.0 live under full augmentation; zero kernels passed originally
and failed on an augmented shape/value. Ordinary sampling variance at n=3 on a level already
known to be luck-limited. **L4 showed `[0.83,0.83,0.83]` as first measured — this was a
reward-inflation BUG, found by measurement and fixed.** `agent.grade()` was granting credit
per sub-check (base + 6 A3 variants + 1 A4) rather than per shape, so a passing shape was
worth 8x a failing one. Fixed: every shape is now exactly one unit of credit, passed only if
its base check and every variant and A4 all pass; `correct` prorates over shape count, never
sub-checks (WEIGHTS untouched). Re-scored the exact kernel that exposed the bug:
`augment=True` now gives **0.60** (1 of 5 shapes, correctly including the failing A2a
shape), `augment=False` gives **0.625**, exactly matching baseline's own best level-4
attempt — same failure, same shape, proving the kernel never changed, only the accounting
did. Re-verified post-fix: every honest reference still scores 1.0 under `augment=True`,
and all 10 cheats are still caught. Every honest reference kernel also scores 1.0 at every
cumulative tier independent of the agent (`results/cost.csv`) — including A2b's ragged
shapes, run against every honest kernel first. **No case, in either direction, of the
checker giving a wrong verdict on a kernel that should have scored differently.**

## False-verified and taxonomy

Re-scoring all 252 baseline attempts under the hardened checker: **false-verified = 0** (the
only 3 that ever scored 1.0, all level 2, still do). Why that's not fully reassuring: **249
of 252 attempts raised a Python exception during simulation and never reached the
numeric-comparison stage at all** — shape/tiling 176 (70.7%), invented API 60 (24.1%),
buffer placement 13 (5.2%). **Interpretation: Phases 2-4's closed loopholes are PREVENTIVE,
not yet REACTIVE, at this model's skill level — proven by the hand-written cheats catching
10/10 regardless of whether this baseline ever produced a kernel that needed catching.**

## Limitations

No on-device timing (`nki.simulate` only). `bfloat16` tolerance (3e-2) is an unmeasured
placeholder — no current shape uses it. Level 4 has no A2b coverage (its declared class
forbids non-tile-multiple shapes — a scope boundary, not a gap). Level 8 (attention) has no
augmentation or cheat coverage at all. Element-at-a-time kernels (level 2's own shipped
reference is one) need op/transfer counting to penalize, not input augmentation.
