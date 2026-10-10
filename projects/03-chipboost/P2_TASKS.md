# P2 (`kernels-search`): tasks for the second coding agent

> **CHANGED 13:15. If you started earlier, run `git fetch origin && git rebase origin/kernels-search`,
> then apply these.** (They come from reading P1's referee on branch `referee-timing`.)
> 1. The matmul entry point is **`nki_matmul_tiled_`**: P1's referee and P3's 11 cheat kernels already
>    use it. RMSNorm: `qwen3_rmsnorm`, copy: `copy_floor`, SwiGLU: `qwen3_swiglu`. Use `shapes.entry(op)`.
> 2. Timing is at **256 tokens** (P1 measured the timer there). Primary shape: **K=4096, M=256, N=6144**,
>    so tile counts are M 2, N 12, K 32: **72 triples**, and the default caps (16, 2, 8) act as (2, 2, 8).
> 3. The referee's real API is `speedcheck.check_isolated(path, op="matmul", baseline=None,
>    heldout=True)`: one fresh process per candidate, and it returns a **complete** schema record. There
>    is no `shapes=` argument; `--no-heldout` maps to `heldout=False`. The record's `seat` comes from the
>    `CHIPBOOST_SEAT` env var (default 100), so overwrite it with `--seat`.
> 4. `shapes.reference` now returns the **float32** truth (the referee compares against that). Use
>    `shapes.work(op, case)[1]` as the byte floor, not `nkibench.minimum_hbm_bytes(args, want)`, which
>    would count the float32 reference as the output size. Held-out cases carry `hostile=True`.
> 5. `check_kernels.py`: also check `kernels/matmul_expert_aws.py` (op `matmul`).
> 6. vLLM holds NeuronCores **0-1** on the seat pods (P1 measured), not 2-3: anything on the device uses 2.
> 8. **From REVIEW.md (on master, merged into this branch):** loops call the referee with
>    `heldout=False` (about half the compiles, and held-out shapes never leak into messages). `search.py`
>    does **not** run its own end-of-run held-out check: P2's `heldout_grid.py` does it for every arm's best
>    kernel at once, from the logs, and writes the dashboard's panel 5. `--budget` counts referee calls, the
>    same budget as `agent.py`. The SBUF filter leaves 62 of the 72 triples; REVIEW counted 62 independently.
> 7. **Logs (P4's dashboard):** each seat writes `logs/seat-<N>/attempts.jsonl`, so merges never
>    conflict. `search.py`'s `--out` defaults to `logs/seat-<seat>/attempts.jsonl` (create the folder).
>    `check_kernels.py --json` is P2 data for the dashboard: not attempts.

You are helping P2 of the CHIPBOOST hackathon team. Read `README.md`, `TEAM.md` and `schema.py` in
this folder first, 5 minutes, then build the three files below. **Nobody else edits these files, and
you edit nothing else.**

| You own (create) | Do NOT touch (others are editing them) |
|---|---|
| `search.py`, `check_kernels.py`, `pod_check.sh`, `tests/test_search.py` | `shapes.py`, `kernels/*`, `tools/*`, `schema.py`, `TEAM.md`, `README.md`, anything in `../02-kernel-agent/` |

## Setup (do this first)

```bash
# Separate working copy on your own branch, so you never collide with the other agent's working tree.
# Same machine as the main checkout:
git -C /path/to/trainium-agent-labs fetch origin
git -C /path/to/trainium-agent-labs worktree add ../tal-p2-tools -b p2-tools origin/kernels-search
cd ../tal-p2-tools/projects/03-chipboost
# (Different machine instead:  git clone -b kernels-search https://github.com/likhith2366/trainium-agent-labs.git
#  && cd trainium-agent-labs && git checkout -b p2-tools && cd projects/03-chipboost)

python3 -m venv ../../.venv && ../../.venv/bin/pip install numpy ml_dtypes   # .venv/ is gitignored
../../.venv/bin/python shapes.py          # must print the shape table
```

- **NKI is NOT installable on the laptop.** Anything that simulates a kernel only runs in the seat pod.
  Make every script fail with one clear sentence when `import nki` fails, and give it a mode that runs
  without NKI (`--stub` / `--dry-run`) so you can test it here.
- Code must run on **Python 3.9 (laptop) and 3.13 (pod)**: no `match`, no `X | Y` type hints.
- Style: match the repo. Module docstring with usage lines and the *why*, `argparse`, plain functions,
  comments that explain why rather than what, 4-space indent.
- Commit only your own files by path (`git add search.py check_kernels.py ...`), never `git add -A`.
  Push with `git push -u origin p2-tools`. Do **not** merge into `kernels-search` or `master`; P2 merges.

## Contracts you code against (they exist on the branch now, except where marked)

**`shapes.py`** (P2): `OPS[op]` for op in `matmul`, `rmsnorm`, `copy`, `swiglu`, each with `level`,
`entry`, `names`, plus the referee's keys (`make_inputs`, `ref`, `flops`, `sim_shapes`, `time_shapes`,
`heldout_shapes`, `tol`). Helpers for tools: `cases(op, which)` (dicts; `which` is `dev`, `timing` or
`heldout`), `entry(op)`, `make_inputs(op, case, seed=0)` (tuple of bf16 NumPy arrays in argument
order), `reference(op, args)` (float32), `tolerance(op)`, `label(op, case)`, `work(op, case)` (flops,
minimum bf16 HBM bytes). `cases("matmul", "timing")[0]` is the **primary shape**: K=4096, M=256, N=6144.

**`../02-kernel-agent/nkibench.py`** (import after `sys.path.insert` of that folder, as `shapes.py` does):
`load_kernel(path, entry)`, `simulate_and_count(kernel, args)` -> `(out, counted)` with `counted["bytes"]`,
`counted["transfers"]`, `counted["warnings"]`; `check_inputs_untouched(before, args)`;
`describe_mismatch(got, want, tol)` -> `None` if correct, else the message; `minimum_hbm_bytes(args, want)`;
`NkiMissing` (raised when nki is absent).

**`schema.py`** (shared): `ATTEMPT_FIELDS`, `validate(rec)` -> list of problems. Every log line you
write must have every field (use `None` where not applicable) and pass `validate`.

**`speedcheck.py`** (P1, on branch `referee-timing`, not merged into this branch yet):
`speedcheck.check_isolated(path, op="matmul", baseline=None, heldout=True)` runs the referee in a fresh
process and returns a complete schema record (`verdict`, `referee_message`, `instruction_given`,
`sim_ok`, `chip_ok`, `time_us_median`, `time_us_iqr`, `baseline_us_same_session`, `speedup`, `source`,
and the rest). Its `seat` comes from `CHIPBOOST_SEAT`; overwrite it.

**`kernels/matmul_expert.py`** (P2, on the branch): defines `nki_matmul_tiled_(lhsT, rhs)` and has
exactly these three module-level lines, which `search.py` rewrites:

```python
TILES_IN_BLOCK_M = 16   # search.py rewrites these three lines
TILES_IN_BLOCK_N = 2
TILES_IN_BLOCK_K = 8
```

They are **caps**: at each shape the kernel uses the largest divisor of the tile count that is <= the cap
(tile counts: M/128, N/512, K/128), so every candidate is valid at every shape.

## Task 1: `search.py`, arm (c): random search, no AI

```
python search.py --budget 24 --seed 0 --seat 102 [--out logs/seat-102/attempts.jsonl] [--heldout-every] [--stub] [--dry-run]
```

1. **Space.** At the primary shape, tile counts are M 2, N 12, K 32. A candidate is a triple
   `(tm, tn, tk)` with `tm | 2`, `tn | 12`, `tk | 32`; that is 72 triples. Compute it from
   `shapes.cases("matmul", "timing")[0]`, don't hard-code it.
2. **SBUF filter.** Drop triples that will not fit on chip. Put the estimate in ONE function. The formula
   below is confirmed against the tutorial source for our fp32-accumulating expert kernel: bytes per
   partition, bf16 operands, fp32 accumulators for the whole of M:

   ```python
   def sbuf_bytes_per_partition(tm, tn, tk, M, itemsize=2, acc_bytes=4):
       block_m, block_n = tm * 128, tn * 512
       return (tk * block_m * itemsize          # lhsT block
               + tk * block_n * itemsize        # rhs block
               + (M // 128) * block_n * acc_bytes)  # result accumulator
   SBUF_LIMIT = int(0.75 * 192 * 1024)   # 192 KiB per partition, 25% headroom for the compiler
   ```

   Print how many triples the filter dropped. No silent caps.
3. **Order.** Attempt 0 is the expert kernel exactly as shipped (caps 16, 2, 8). Then `budget - 1`
   distinct random triples from the filtered space, shuffled with `random.Random(seed)`.
4. **Each attempt.** Write the candidate to `search_runs/<run_id>/cand_m{tm}_n{tn}_k{tk}.py` by
   regex-replacing the three cap lines in `kernels/matmul_expert.py` (fail loudly if the three lines are
   not found exactly once each). Call the referee:
   `speedcheck.check_isolated(path, op="matmul", baseline="kernels/matmul_start.py",
   heldout=args.heldout_every)` (default False: see item 8 at the top).
5. **Log.** One JSON line per attempt to `--out`: the referee's record, with these overwritten:
   `seat`, `kernel="matmul"`, `arm="random_search"`, `run_id` (`--run-id`, default
   `matmul-random-<seed>-<unix time>`), `attempt_no`, `round=attempt_no`, `prompt_tokens=None`,
   `prompt=None`, `response=None`, `code` (the candidate source), `code_hash` (sha1 of the source),
   `timestamp`. Call `schema.validate(rec)` and raise if it returns problems.
6. **Print.** One line per attempt: `#3  caps m4 n2 k8  faster  412.3us  1.18x  best 1.18x`. At the
   end: best triple, best speedup, evaluated, filtered.
7. **`--stub`.** Also used automatically if `import speedcheck` fails, with a warning. A fake referee:
   deterministic from the triple and seed, `source="sim"`, `referee_message` and `instruction_given`
   start with `(stub)`, about 20% wrong, the rest timed with a smooth function of the triple plus noise.
   It must be impossible to mistake stub output for a real result. In stub mode, if
   `kernels/matmul_expert.py` does not exist yet, use an inline template containing the three cap lines.
8. **`--dry-run`.** Print the candidate list with SBUF estimates and exit. No referee calls.

**Test (`tests/test_search.py`, plain `assert`s, run with `python tests/test_search.py`):** the space
has 72 triples before filtering; the default caps map to (2, 2, 8) at the primary shape; the
same seed gives the same order; `--stub` with budget 5 writes 5 lines that pass `schema.validate`;
the cap-line rewrite fails loudly on a template missing a line.

## Task 2: `check_kernels.py`: every kernel x every shape, in the simulator

```
python check_kernels.py [--which dev,heldout] [--only matmul_start,rmsnorm_start] [--timing] [--json out.json] [--dry-run]
```

- Kernels, skipping files that do not exist yet with `not written yet`:
  `matmul`: `kernels/matmul_start.py`, `kernels/matmul_expert.py`, `kernels/matmul_expert_aws.py`;
  `rmsnorm`: `kernels/rmsnorm_start.py`; `copy`: `kernels/copy_floor.py`;
  `swiglu`: `kernels/swiglu_start.py`.
- For each kernel x each case of each requested list: `shapes.make_inputs`, snapshot the inputs,
  `shapes.reference`, `nkibench.load_kernel(path, shapes.entry(op))`, `nkibench.simulate_and_count`,
  then `nkibench.check_inputs_untouched(...) or nkibench.describe_mismatch(got, want, shapes.tolerance(op))`.
  A simulator warning containing `incorrect results on hardware` is also a FAIL.
- Print a table: kernel, list, shape label, PASS/FAIL, sim seconds, bytes moved / floor
  (`counted["bytes"] / shapes.work(op, case)[1]`), transfers, then the first line of any
  failure message. End with `N passed, M failed` and exit non-zero if anything failed.
- `--timing` adds the timing shapes, which are big (print a warning that they take minutes).
  `--json` writes one object per row; the dashboard owner (P4) can use it for the held-out map.
- Without NKI, print `nki is not installed here: run this in the seat pod` and exit 2. `--dry-run` only
  builds inputs and references for every case, so it can be tested on the laptop.

## Task 3: `pod_check.sh`: P2's one command in the pod

```bash
#!/usr/bin/env bash
# P2's one command in the seat pod: harness selftest, Qwen3 config, then every kernel in the simulator.
set -uo pipefail
cd "$(dirname "$0")"
python ../02-kernel-agent/nkibench.py --selftest | tail -3
python shapes.py --verify-config
python check_kernels.py --which dev,heldout "$@"
```

`chmod +x pod_check.sh`.

## Done when

- `python tests/test_search.py` passes on the laptop.
- `python search.py --stub --budget 5 --out /tmp/x.jsonl && python schema.py --check /tmp/x.jsonl`
  prints `valid`.
- `python check_kernels.py --dry-run` lists every kernel x case.
- Branch `p2-tools` is pushed. Report back with the commit hash, plus anything in these contracts that
  looked wrong.
