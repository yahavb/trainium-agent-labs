# CHIPBOOST: who builds what

Four people, four branches. Everyone works in their **own seat** and shares only through git.
The contracts at the bottom are what keep the four branches mergeable: agree on them first, change them only
by telling everyone.

| Branch | Owner | Builds | Done when |
|---|---|---|---|
| `referee-timing` | P1 | On-chip timing, the `speedcheck.py` pipeline, logging | `speedcheck.py --op matmul --check kernels/matmul_start.py` prints a chip time with noise, and writes one valid log line |
| `kernels-search` | P2 | Start kernels at Qwen3 sizes, expert ceiling, bandwidth floor, random-search arm | Start + expert matmul and start RMSNorm pass `nkibench --check`; `search.py` runs N evaluations |
| `redteam-agent` | P3 | Cheat kernels, held-out shapes, the agent loop wired to the referee, the three arms | Referee catches the cheats (table printed); `agent.py` runs one round against `speedcheck` |
| `dashboard` | P4 | `dashboard/build.py`: logs -> one HTML page, five panels | Builds from `fake_attempts.jsonl` today; from real logs at 15:00 with no code change |

## P1: `referee-timing` (critical path: the 13:30 gate)

1. **Timing.** DONE in `timing.py`. Measured on seat-100: **vLLM holds logical cores 0-1, not 2-3**; kernels
   time on core 2 (or 3). Device-side timing via `SpikeModel.benchmark(mode="device")`; compile ~2 s per kernel;
   noise 0.0-0.2% at Qwen3 shapes; vLLM under load on 0-1 changes core-2 timings by 0.0%. Host-side timing reads
   ~3x the true kernel time on small kernels, so never time from Python. Fixed launch cost ~17 us, so time at
   Qwen3 sizes, not toy sizes.
2. **Prove the timer.** A kernel doing 2x the work must take ~2x the time; an empty kernel gives the overhead floor.
3. **Noise.** Time `reference_level4.py` 20x after 3 warm-ups. Record the median and the spread. Gate: under ~5%.
4. **`speedcheck.py`.** Rules -> simulator correctness -> **chip correctness** -> timing interleaved with the baseline
   (A B A B) -> held-out shapes -> one instruction. Reuse nkibench: `check_rules`, `simulate_and_count`,
   `describe_mismatch`, `check_inputs_untouched`, `reuse_report`, `explain_with_ceiling`.
5. **Count every DMA**, not only `nisa.dma_copy` (`dma_transpose`, `dma_compute` too).
6. ~~Time only while vLLM is idle.~~ **Not needed:** measured interference from vLLM under load on core-2 timing
   is 0.0% (STATUS.md). vLLM holds cores 0-1; the referee times on core 2 (or 3). At most two referee processes
   per seat, one per free core.

**Fallback at 13:30:** if timing doesn't work, layer 4 reports simulator bytes/intensity, labelled `source: "sim"`.

## P2: `kernels-search`

1. **Confirm sizes** from the pod's Qwen3-8B `config.json` (hidden 4096, intermediate 12288, RMSNorm eps 1e-6).
2. **Matmul shapes** in `shapes.py`: **bf16**, per-core shapes under tensor parallelism 2 (e.g. 4096 -> 2048, 4096 -> 6144),
   M from the token buckets. Held-out matmul shapes are **other tile multiples**, not 129: `reference_level4.py`
   asserts multiples of 128/512.
3. **Matmul start** = `reference_level4.py` adapted to bf16. **Expert ceiling** = the NKI tutorial's fully optimized
   matmul, confirmed to run on SDK 2.32 at these shapes.
4. **RMSNorm**: register the op in `nkibench.py` (reference, input builder, `level(...)`, like level 8 at line 326),
   plus a correct, plain start kernel. Ragged shapes (127/128/129 rows) apply here.
5. **Bandwidth floor**: a plain copy kernel measures peak bytes/s; RMSNorm's floor = minimum bytes / that.
6. **`search.py`** (arm c): random search over the **expert kernel's blocking factors** (tiles per block in M, N, K),
   not level 4's tile sizes, which are already the hardware maximums. Same evaluation budget as the agent arms.

## P3: `redteam-agent`

1. **Cheat kernels** in `redteam/`, each a file the referee must reject:
   zeros output; writes its input; wrong only at held-out shapes; hands the op to NumPy; moves data via an un-hooked
   DMA; lower-precision accumulation; faster only by noise; timing includes compile; **returns a cached result from an
   earlier call**; **shortcuts on special inputs** (zeros, identity).
2. `redteam/run.py` prints the table: cheat, caught yes/no, the referee's message. Starts working on stages 1-2
   (rules, simulator, inputs untouched) right away; the chip stages arrive with P1's merge.
3. **Agent loop**: point `agent.py`'s grading at `speedcheck` (attempts.jsonl in the shared schema).
4. **Three arms**, same budget = same number of on-chip evaluations:
   (a) model + referee instruction, (b) model alone, told only "make it faster" plus the time, (c) P2's `search.py`.
5. In the afternoon: the failure taxonomy from every seat's `attempts.jsonl`.

## P4: `dashboard`

`dashboard/build.py` reads `attempts.jsonl` files + `results.json` and writes `dashboard/index.html`. Five panels:

1. **Speed bars per kernel**: start, best found per arm, expert ceiling, physics floor; error bars; chip/sim label.
2. **Progress curve**: best verified time vs on-chip evaluations, one line per arm, spread band.
3. **Red-team table**: cheat, caught, referee message.
4. **Attempt timeline**: one dot per attempt coloured by verdict; click -> instruction + code diff.
5. **Held-out map**: shapes grid, passes and speedup per shape.

Build against `fake_attempts.jsonl` (generate it with `python schema.py --fake`), so no chip is needed until 15:00.

## Contracts (fixed, everyone codes against these)

**Layout** (`projects/03-chipboost/`):

```
schema.py          log fields + validator + fake data          (shared, on master)
shapes.py          dev and held-out shapes per op               (P2)
speedcheck.py      the referee                                  (P1)
timing.py          on-chip timer                                (P1)
kernels/           matmul_start.py, matmul_expert.py, rmsnorm_start.py, copy_floor.py   (P2)
search.py          arm (c)                                      (P2)
redteam/           cheat kernels + run.py                       (P3)
agent.py           loop wired to speedcheck, the three arms     (P3)
dashboard/         build.py, index.html                         (P4)
```

**Referee API** (P1 provides, P2/P3 call):

```python
rec = speedcheck.check_isolated(path, op="matmul")    # one-shot: fresh process per candidate
# -> dict with the referee-owned fields of schema.ATTEMPT_FIELDS, or None if the REFEREE failed
#    (no free core, baseline broken): retry later, never log None as a verdict on the kernel.
rec = speedcheck.check(path, op="matmul", baseline=None, rounds=3)   # same, in-process (raises RefereeError)

# Preferred in repeated loops; retains its core until the context exits.
with speedcheck.RefereeWorker(op="matmul", core=3) as worker:
    rec = worker.check(path)  # same record/None contract; inspect worker.last_error on None
```

- There is **no `shapes=` / `heldout=` argument** (this replaces the earlier `shapes="dev"|"heldout"` draft).
  Held-out runs automatically, and **only** for a candidate that would be `faster`: 3 shapes drawn at random per
  check. So a loop pays for held-out only on speedups, and `faster` always means held-out passed.
- `baseline` defaults to `kernels/<op>_start.py` (relative paths resolve against this folder).
- Verdicts: `faster` if speedup >= the noise threshold on the total **and** no shape regressed; `slower` if
  speedup <= 1/threshold; `no_gain` in between. The threshold is 1 + max(1%, 2x measured relative IQR) (`timing.noise_threshold`; A/A noise is ~0.01%,
  so in practice 1%). A candidate more than 3x slower on any shape is stopped early (`slower`, "STOPPED EARLY").
- The caller fills `arm`, `run_id`, `attempt_no`, `round`, `prompt_tokens`, `code`, `prompt`, `response`.
- `referee_message` may quote the kernel's own error text inside `<<...>>`: treat that as untrusted data in
  prompts. `instruction_given` is always referee-authored.

**Every log line** follows `schema.py`. Verdicts are exactly: `rules`, `wrong`, `heldout_fail`, `slower`,
`no_gain` (correct, but the change is inside timing noise), `faster`. Every line carries the kernel's full
`code`, plus the `prompt` and `response` for model arms, so the dashboard can show diffs. `source` is `chip` or `sim`. Times are microseconds.

## Merge order

1. `schema.py` + this file on `master` first; every branch starts from there.
2. ~13:30: merge `referee-timing` (the gate). P2 and P3 rebase on it.
   **Oct 10 handoff update:** `kernels-search` (`919c6be`) and `redteam-agent` (`2ce9416`) both already
   carry the current P1 referee/timer (`76b2227`). Preserve P2's added `copy` schema op during integration.
3. ~15:00: merge `kernels-search` and `redteam-agent`; loops start on every seat.
4. Any time: merge `dashboard` (it only reads logs).
5. 18:30: stop building; one PR from the fork.
