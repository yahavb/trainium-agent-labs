# eval_set.md — every shape, dtype, value pattern and seed the hardened checker uses

Mandatory deliverable (`CHALLENGE-kernel-agent.md`: "Your eval set — the shapes and values you
tested, including the hostile ones."). This is the literal, exhaustive list behind the tables
in `HARDENING.md` section 8 and `CHECKER.md` section 4. Source of truth: `nkibench.py`'s
`LEVELS` dict (declared shapes) and `augment.py` (A1-A4). All values float32 unless noted —
see "dtype" below.

## dtype

Every level's input generator (`_args_pool`, `_args_transpose`, `_args_matmul`,
`_args_attention` in `nkibench.py`) casts its NumPy-generated values to **float32** with
`.astype(np.float32)`. **No current shape, at any tier, uses bfloat16.** The checker's
bfloat16 tolerance (`TOL_BY_DTYPE["bfloat16"] = 3e-2`) is consequently never exercised —
see `CHECKER.md` section 2a.

## Seeding

`nkibench.make_inputs(spec, level_n, seed=0)` builds `np.random.default_rng(seed + level_n)`
— the RNG is re-seeded identically for every SHAPE of a given level (confirmed empirically:
every level-1 shape's first element is the same value, `0.3455841839313507`, at `seed=0`,
because the stream restarts from the same seed each time and the shapes differ only in how
much of it gets consumed/reshaped).

- **A0 (unaugmented):** `seed=0` always.
- **A1 and beyond:** the CLI (`nkibench.py --augment`) picks a fresh, time-based seed via
  `augment.fresh_seed()` and PRINTS it (rule 8: logged, so reproducible after the fact).
- **This report's cumulative-tier run** (`run_cheats.py --tiered`, behind
  `results/cheats_after.csv` and `results/cost.csv`) used a FIXED seed, `A1_SEED = 424242`,
  for every tier from +A1 onward, instead of a time-based one, specifically so this document
  and that CSV are reproducible by anyone re-running `python run_cheats.py --tiered`.
- **A3's hostile-value RNG** (`augment.hostile_args(args, seed)`) is seeded with the SAME
  `seed` the shape itself was built with (0 at A0, 424242 in this report).
- **A4's second instantiation** uses `seed + 10_007` — an arbitrary large odd offset, chosen
  only so the two runs draw from clearly different parts of the stream.

---

## Level 1 — average pooling 2D (`tensor_avgpool_kernel`)

Input: one tensor, shape `(C, H, W)`, plus a scalar `pool_size`. Reference: `ref_avgpool2d`
(truncates to `H//p*p, W//p*p`, then averages each `p×p` window).

| tier | shape (C, H, W) | pool_size | note |
|---|---|---|---|
| A0 | (32, 32, 32) | 2 | declared |
| A0 | (128, 16, 16) | 4 | declared — C=128 is PMAX exactly |
| A0 | (8, 24, 24) | 3 | declared |
| A0 | (64, 8, 8) | 2 | declared |
| A2a | (16, 20, 20) | 5 | evenly dividing, new combination |
| A2b | (1, 8, 8) | 2 | **C=1**, a dimension of exactly 1, on the partition axis |

A2b note: no prime-dimension / non-divisible-window case is included for level 1, because
`ref_avgpool2d` itself truncates a non-divisible `H`/`W` to the nearest multiple of
`pool_size` (its own defined behavior, not an edge case to catch) — see `HARDENING.md`'s
Phase 3 findings. The dimension-of-1 case (C=1) is the one ragged shape that genuinely tests
something (the partition axis, un-truncated).

**A3 (applied to every shape above, 6 variants each, same `(C,H,W)`, pool_size unchanged):**
`large_magnitude` (±1e4 + standard-normal noise), `large_mean` (1e4 + 0.01×noise),
`zeros`, `negatives` (`-|standard_normal|`), `tiny` (uniform 1e-6 to 1e-3, random sign),
`identical_row` (entire first channel set to one repeated value).

**A4:** run twice per shape above, with two different random `(C,H,W)` tensors of the same
shape (seed vs. seed+10007); fails if the two pooled outputs are identical.

---

## Level 2 — 2D transpose (`tensor_transpose2D_kernel_`)

Input: one tensor, shape `(sz_p, F)`, plus `shape2D = (F1, F2)` with `F1*F2 == F`. Reference:
`ref_transpose2d` (transposes the two FREE axes inside each partition row; partition axis
untouched).

| tier | shape (sz_p, F) | shape2D | note |
|---|---|---|---|
| A0 | (32, 12) | (3, 4) | declared |
| A0 | (128, 64) | (8, 8) | declared — sz_p=128 is PMAX exactly; the one SQUARE shape (F1=F2) |
| A0 | (64, 128) | (4, 32) | declared |
| A0 | (8, 35) | (5, 7) | declared |
| A2a | (16, 24) | (6, 4) | evenly dividing, new combination |
| A2b | (37, 12) | (3, 4) | **sz_p=37 is prime** |
| A2b | (1, 12) | (3, 4) | **sz_p=1**, a dimension of exactly 1 |
| A2b | (32, 37) | (37, 1) | **F1=37 is prime; F2=1** |

**A3 (applied to every shape above, 6 variants each, same `(sz_p, F)` and `shape2D`):** same
6 kinds as level 1.

**A4:** run twice per shape above, same mechanism as level 1.

---

## Level 3 — matmul, single tile (`nki_matmul_basic_`)

Inputs: `lhsT` shape `(K, M)`, `rhs` shape `(K, N)`. Reference: `ref_matmul`
(`lhsT.T @ rhs`, computed in float32). Declared class: single-tile, i.e. `K<=128` (PMAX),
`M<=128` (GEMM_STATIONARY_FMAX), `N<=512` (GEMM_MOVING_FMAX) — no tiling loop needed.

| tier | K | M | N | note |
|---|---|---|---|---|
| A0 | 128 | 64 | 512 | declared (original) — K at PMAX, N at GEMM_MOVING_FMAX |
| A0 | 64 | 32 | 256 | declared, **added in Phase 3** specifically so a kernel that hardcodes the FIRST shape's exact values (`cheats/c3_hardcoded_shape.py`) cannot also pass this one — see `HARDENING.md` |
| A2a | 96 | 48 | 384 | evenly-shaped, new combination, still single-tile |
| A2b | 37 | 1 | 17 | **K=37 and N=17 are prime; M=1** |

**A3 (applied to every shape above, 6 variants each, same K/M/N):** same 6 kinds, applied
independently to `lhsT` and `rhs`.

**A4:** run twice per shape above, same mechanism.

---

## Level 4 — matmul, tiled (`nki_matmul_tiled_`)

Inputs: `lhsT` shape `(K, M)`, `rhs` shape `(K, N)`. Reference: `ref_matmul`. Declared class:
`K` and `M` are multiples of 128, `N` a multiple of 512 — `reference_level4.py` itself asserts
this (it's an honest limitation of the level's own definition, not an omission here).

| tier | K | M | N | note |
|---|---|---|---|---|
| A0 | 128 | 128 | 512 | declared — exactly one tile in every dimension |
| A0 | 256 | 256 | 1024 | declared |
| A0 | 512 | 128 | 512 | declared |
| A0 | 256 | 512 | 1024 | declared — the shape used to expose redundant HBM traffic (2.0x the floor on the shipped tiled kernel) |
| A2a | 384 | 384 | 512 | evenly-shaped (3 tiles × 3 tiles × 1 tile), new combination |
| A2b | — | — | — | **deliberately empty.** No shape outside exact tile multiples is legal input for this level's own declared class; see `CHECKER.md` section 4 |

**A3 (applied to every shape above, 6 variants each):** same 6 kinds.

**A4:** run twice per shape above, same mechanism.

---

## Level 8 — single-head attention (`nki_attention_`) — declared but NOT covered here

Declared shapes exist in `nkibench.LEVELS[8]` (`seq=128,dim=64`; `seq=64,dim=128`;
`seq=96,dim=32`), but `augment.extra_shapes()` has no case for level 8 — it falls through to
the generic `return []`, so **no A2a/A2b shapes, and by extension no cheat-set measurement,
exist for this level.** It was never extended because no cheat was built against it (see
`CHECKER.md` section 5). This is a known gap for anyone picking up this harness to extend the
ladder past level 4.

---

## Tolerance applied against every comparison above

`TOL_BY_DTYPE = {"float32": 1e-4, "bfloat16": 3e-2}` — see `CHECKER.md` section 2a for the
full measurement (`1e-4` is ~32x above the worst observed honest-kernel error, 3.103e-6, and
~10x below the known cheat `cheats/c6_almost_right.py`'s error at this data scale). Every
value in this eval set is float32, so `1e-4` is what's actually enforced everywhere; the
bfloat16 figure is provisional and unused today.
