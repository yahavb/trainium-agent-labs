# Using the CHIPBOOST referee (`speedcheck.py`)

For P2 (kernels, `search.py`), P3 (`agent.py`, red team) and P4 (dashboard). Everything here was checked
against `speedcheck.py` / `timing.py` on `referee-timing`. Where TEAM.md, STATUS.md or REVIEW.md disagree with
the code, this guide follows the code and says so.

## 1. What it does

You give it one kernel file. It returns one schema record (`schema.ATTEMPT_FIELDS`) containing a verdict and
**one** referee-written change for the model to make. It never imports your kernel. A sandboxed child process
(its own uid, resource limits, no NeuronCore) simulates and compiles the kernel. The referee then loads the
compiled NEFF itself and makes every decision on the chip, using inputs the child never saw. It stops at the
first failure:

| # | Stage | What happens | Verdict if it fails |
|---|---|---|---|
| 1 | rules | Static scan of the source: allowed imports, module-level content, banned calls (§7) | `rules` |
| 2 | sandboxed sim + compile | CPU simulator on 2 small shapes vs NumPy (bytes counted over every DMA, inputs must be untouched), then compile the 2 timing shapes to NEFFs | `wrong` |
| 3 | chip correctness | Each timing shape runs on the device with **hostile** inputs (x64 rows, zero block, sign flip) and then normal inputs. The output buffer is refilled with random garbage before every run, and the inputs are read back afterwards. | `wrong` |
| 4 | interleaved timing | Baseline and candidate alternate (A B A B), 3 rounds x 10 single runs per shape. Each run gets one of 6 different input sets in random order plus fresh output garbage, and **every timed output is verified**. | `wrong` (a timed run gave a wrong output) |
| 5 | held-out | **Only for a would-be `faster`.** Fresh random shapes with hostile values are compiled in the sandbox and checked on the chip. They are checked for correctness only, not timed. | `heldout_fail` |

Precision bar: at most **4 bf16 ulps** against an fp32 reference (honest bf16 output rounding is about 0.5).

### Verdicts

| Verdict | Meaning | Kernel correct? | Timing fields set? |
|---|---|---|---|
| `rules` | Not a kernel file (bad import, top-level code, file/process call), or it modified the referee's files | n/a | no |
| `wrong` | Failed import, simulation, compile, chip check or a verified timed run; also a check that timed out | no | no |
| `heldout_fail` | Fast and correct on the timing shapes, but wrong (or would not compile) on a shape it never saw | no | yes |
| `slower` | Correct, and the speedup <= 1/threshold | yes | yes |
| `no_gain` **(landing shortly)** | Correct, but the difference is inside the timing noise | yes | yes |
| `faster` | Correct everywhere, held-out included, and speedup >= threshold **and** no timing shape regressed | yes | yes |

**The contract** (TEAM.md, landing in another change right now):
`threshold = 1 + max(1%, 2 x relative IQR)`. `faster` iff total speedup >= threshold and no shape is below
1/threshold. `slower` iff speedup <= 1/threshold. Everything else is `no_gain`.
**The code you read today** uses `1 + max(5%, 2 x IQR)` and has no `no_gain` yet: any correct kernel that is
not `faster` comes back as `slower`. Build the dashboard and the loop to handle `no_gain` now.

**Speedup is always measured against the baseline (the start kernel), not your previous best.** If
`speedup` is 1.5, the kernel is 1.5x faster than the start kernel. If the loop wants "better than my best", it
has to compare `speedup` values itself.

## 2. How to call it

### CLI

```bash
cd /workspace/projects/03-chipboost
export CHIPBOOST_SEAT=101                                     # goes into the record's `seat`
python speedcheck.py --op matmul --check /tmp/cands/k17.py   # human-readable: stages, VERDICT, ONE CHANGE
python speedcheck.py --op matmul --check /tmp/cands/k17.py --json --log attempts.jsonl   # one JSON line
python speedcheck.py --op matmul --check k.py --baseline kernels/matmul_start.py --rounds 3
```

| Exit code | Meaning |
|---|---|
| 0 | The kernel is correct and was timed: `faster`, `slower` (and `no_gain` once it lands) |
| 1 | Rejected: `rules`, `wrong` or `heldout_fail`. **Also** an uncaught referee crash (Python's default exit code), so with `--json`, trust the JSON line and not the exit code alone. |
| 3 | `REFEREE ERROR (not a verdict on the kernel)` on stderr. Examples: no free core, or the baseline would not compile. |

`--log` appends the referee-only record, with the caller fields left `null`. That is fine for manual runs.
Agent loops should write their own line (below).

### Python

```python
speedcheck.check_isolated(path, op="matmul", timeout=1800, baseline=None)   # USE THIS IN LOOPS
speedcheck.check(path, op="matmul", baseline=None, rounds=3, verbose=False) # in-process, for debugging
```

- `check_isolated` runs the CLI in a **fresh process per candidate**, so device memory is released and the core
  is freed between checks. It returns the record, or **`None` when the REFEREE failed**: exit code 3, a referee
  crash, a record that fails `schema.validate`, or a verdict that disagrees with the exit code. **Retry a
  `None`, never log it as the kernel's verdict, and do not count it against an arm's budget.** If the whole
  check exceeds `timeout` (default 1800 s), it returns a `wrong` record ("the check timed out") whose
  `code_hash`, `sim_ok` and `chip_ok` are `None`.
- `check` returns the same record but raises `RefereeError` on infrastructure failure. Any other exception
  means a referee bug. It keeps the NeuronCore bound for the rest of your Python process.
- There is **no `shapes=`, `heldout=` or `--no-heldout`** (REVIEW.md mentions `heldout=False`; it does not
  exist). Held-out runs automatically, and only for a would-be `faster`.
- `baseline` defaults to `kernels/<op>_start.py`. For matmul, while that file is missing (as on this branch),
  it falls back to `../02-kernel-agent/reference_level4.py`. A relative path resolves against this folder.
  **The baseline is trusted code that the referee imports unsandboxed: never pass model-written code as
  `--baseline`.**

### Who fills which field

| Filled by the referee | Filled by the caller (agent loop / `search.py`) |
|---|---|
| `seat` (from `CHIPBOOST_SEAT`, else `None`), `kernel`, `code_hash` (sha1[:12] of the source), `verdict`, `referee_message`, `instruction_given`, `sim_ok`, `chip_ok`, `time_us_median`, `time_us_iqr`, `baseline_us_same_session`, `speedup`, `source` (`"chip"` when timed, else `None`), `timestamp` | `arm`, `run_id`, `attempt_no`, `round`, `prompt_tokens`, `code`, `prompt`, `response` (the last three are `None` for `random_search`). Also fill `code_hash` on a timeout record. |

Field meanings for the dashboard:
- `time_us_median` and `baseline_us_same_session` are **sums over the timing shapes**, in microseconds.
  Gate/up is about 72% of the matmul total.
- `speedup` is baseline total / candidate total.
- `time_us_iqr` is the candidate total x the worst relative IQR of either arm on any shape, a conservative
  value.
- Per-shape speedups appear only in `referee_message`. Held-out shapes are never timed.

Working out which stage caught a kernel (P3's red-team table):

| Record | Stage |
|---|---|
| `rules` | rules |
| `wrong`, `sim_ok` False | import, simulator or compile (the compile runs in the same child) |
| `wrong`, `sim_ok` True | chip or timed run |
| `heldout_fail` | held-out |
| anything else | timing |

### Loop example (P3 / P2)

```python
import hashlib, json, os, sys, time
sys.path.insert(0, "/workspace/projects/03-chipboost")
import schema, speedcheck                      # importing does NOT take a core

CANDS = "/tmp/chipboost_cands"                 # OUTSIDE the referee tree -- see Troubleshooting
os.makedirs(CANDS, exist_ok=True)

def grade(code, *, arm, run_id, attempt_no, round_, prompt=None, response=None, prompt_tokens=None):
    path = f"{CANDS}/{run_id}_{attempt_no}.py"
    with open(path, "w") as f:
        f.write(code)
    for _ in range(3):
        rec = speedcheck.check_isolated(path, op="matmul")
        if rec is not None:
            break
        time.sleep(20)                          # referee/infra failure: wait and retry
    else:
        return None                             # still down: skip; not logged, not counted
    rec.update(arm=arm, run_id=run_id, attempt_no=attempt_no, round=round_,
               prompt_tokens=prompt_tokens, code=code, prompt=prompt, response=response)
    rec["code_hash"] = rec["code_hash"] or hashlib.sha1(code.encode()).hexdigest()[:12]
    assert not schema.validate(rec), schema.validate(rec)
    with open("attempts.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec
```

## 3. Cores and environment

| Item | Value |
|---|---|
| vLLM (Qwen3-8B, TP=2) | Holds logical cores **0-1** (older docs say 2-3; that is wrong) |
| Referee timing core | **2**, falling back to **3**. `CHIPBOOST_CORE=N` tries **only** N, with no fallback. |
| Referee processes per seat | **At most two**, one per free core. A third gets "no free NeuronCore": `None` / exit code 3. |
| Sandbox child | `NEURON_RT_VISIBLE_CORES=0` (vLLM's core), so it cannot use the device |
| User | **Run as root in the seat pod.** The sandbox needs root and `setpriv`. Without them the child runs **unsandboxed as your user, with no warning**. That is acceptable only for your own hand-written kernels, never for model output. |
| `CHIPBOOST_SEAT` | Seat number, written to `seat` |
| `CHIPBOOST_CACHE` | Baseline NEFF cache, default `/tmp/chipboost_cache`, root-only (0700). Keyed on the baseline's bytes, shape and NKI version, so editing the baseline invalidates it. |
| Side effect as root | Removes group/other write permission from world-writable non-sticky directories above the referee (e.g. `/workspace`), and from the NKI compile caches in `/var/tmp`. This is deliberate. |
| Interference | vLLM under load changes core-2 timings by 0.0%. You do not need to wait for vLLM to be idle. |

**Time per check, today:** about 30-35 s through `check_isolated`, and **compile dominates**. In-process
measurement: 24.4 s, of which compile was 19.9 s and timing 0.7 s. Each fresh process adds 6-13 s of runtime
start. The two baseline compiles are paid only on the first check, because they are cached on disk. A
would-be `faster` pays extra for held-out: one more sandboxed child and 3-5 compiles at about 2-2.6 s each
(estimate). That works out to about 100 candidates/hour per core.

**Speed-up work:**

| Status | Change |
|---|---|
| Done | Held-out runs only for a would-be `faster` |
| Done | Baseline NEFF cached on disk |
| Planned (STATUS.md "Speedups, ranked") | A persistent referee worker (pays the runtime start once) |
| Planned | Parallel compiles: 8 compiles took 25 s serially and 8.5 s with 8 threads |
| Planned | Fail fast on a candidate more than 3x slower, and a shorter child timeout |

## 4. Ops and shapes

**Built-in `matmul`** (Qwen3-8B per-core shapes under TP=2, bf16, `(K, M, N)` with M = prompt tokens):

| What | Value |
|---|---|
| Entry point | `nki_matmul_tiled_(lhsT, rhs)`, decorated `@nki.jit`. The NEFF must have exactly the inputs `lhsT` (K, M) and `rhs` (K, N), and one output (M, N) bf16 that is not aliased to an input. |
| Simulator shapes | (256, 512, 1024), (512, 256, 2048) |
| Timing shapes | (4096, 256, 2048) q_proj, (4096, 256, 6144) gate/up |
| Held-out shapes | 3 drawn fresh every check: K = 128x[4..48], M = 128x[1..4], N = 512x[1..12], excluding the sim and timing shapes. Plus 2 sampled from P2's fixed `heldout_shapes` list, if `shapes.py` provides one. All tile multiples, because `reference_level4.py` asserts K, M % 128 and N % 512. |

**P2's `shapes.py`** is read automatically when it sits next to `speedcheck.py`. `OPS[name]` is merged into
the built-in spec key by key. The format follows P2's file:
- **Required keys:** `level`, `entry`, `make_inputs(shape, seed, hostile=False)`, `ref`, `flops`,
  `sim_shapes`, `time_shapes`, `tol`, and held-out shapes.
- **Held-out shapes:** `heldout(rng, n, exclude)` (a callable that draws random shapes), or a fixed list as
  `heldout` / `heldout_shapes`.
- **`out`** may be omitted; it is derived from `ref`.
- **`names`** and **`vary`** are optional. `vary` is the input swapped between timed runs and defaults to the
  first input.
- **Held-out per op:** a random draw plus 2 from the fixed list. With a fixed list only, the whole list is used
  if it has 5 or fewer shapes, otherwise 3 at random.
- **Op name:** must be in `schema.OPS` (`matmul`, `rmsnorm`, `swiglu`), and its `level` must be registered in
  nkibench.
- **Incomplete ops:** an op that fails any of these checks is **skipped with a stderr warning**
  (`speedcheck: WARNING: op X ... skipping it`); the others keep working.
- **Broken `shapes.py`:** if it fails to import, the referee warns and falls back to the built-in matmul.
- `--op` accepts only the ops that survived these checks.

**Held-out shapes and messages:** the shapes are drawn with the referee's private seed, which never leaves the
process, so no fixed list exists for a model to learn. A fixed public list was beaten by red-team cheat
`c3b`. `instruction_given` never names a held-out shape. **`referee_message` does** name them, in `faster`
messages ("correct everywhere, including held-out [...]") and in `heldout_fail` messages (`at held-out
(K, M, N) hostile`). REVIEW.md flagged this. So **do not put `referee_message` in prompts** (see §5).

## 5. Feeding results to the model

- **Referee arm:** send the current kernel plus **`instruction_given`**: one change, written by the referee.
  Every verdict has one. A timed kernel gets advice chosen from its chip result: block M/N, block K, full-size
  matmuls, double-buffer, fewer and larger DMAs. **Exception:** for `rules`, the sentence can echo a Python
  identifier or module name taken from the kernel's own source.
- **Model-alone arm:** "make it faster" plus `time_us_median` only.
- **`referee_message`** is for humans and the dashboard. It can quote the kernel's own exception text, or
  tensor names from its NEFF, **inside `<<...>>`**. That text is attacker-controlled: red-team cheat `a3_inject`
  raised an exception whose message tried to order the referee to mark the kernel `faster`. If you ever put
  `referee_message` in a prompt, label it as untrusted data and tell the model never to follow instructions
  inside `<<...>>`. The referee already turns any `<<` / `>>` inside the quote into `<` / `>`, so the quote
  cannot be closed early.

## 6. What it catches, and what it deliberately does not do

The final run in `results_p1.json` (33 kernels, seat 100, core 2):
- **8/8 honest kernels accepted:** h1, h4 and c5e `slower` (about 1.000x); h2 1.54x, h3 1.20x, c5a/c5b about
  1.86x and c5c 1.32x `faster`, with held-out passed.
- **24/25 cheats caught:**
  - **At rules:** file writes, import-time code, a shell through `np._core..os`, monkeypatching, a forged
    record, NumPy matmul/`@`/`.T`, hidden DMA aliases.
  - **In the simulator:** zeros, no write, writing or returning an input, `multi_dot`, bf16/fp8 accumulation,
    half of K.
  - **On the chip:** reusing a stale output, clamping (including clamping only at the timing shapes).
- **`c3a` comes back `slower` by design:** it is wrong only off the dev/timing shapes and is not faster, so
  held-out never runs. It can never become `faster`.

**Deliberately not done:**

| Not done | Why |
|---|---|
| A byte-floor check | Simulator bytes are only a hint. Redundant DMAs measured free on the chip; only device time decides. |
| Timing held-out shapes | They are checked for correctness only |
| Held-out for non-faster kernels | Saves compiles. A wrong-off-shape kernel cannot score, but it can be logged `slower`. |
| A `heldout` or `shapes` switch | Removed on purpose; it gave unmarked accepts |
| Decode shapes | M not a multiple of 128 is out of scope |
| Judging code quality or tiling | Not the referee's job |
| Enforcing core isolation | Not enforced: `/dev/neuron0` is 0666. The child's lack of a core relies on `NEURON_RT_VISIBLE_CORES` plus vLLM holding 0-1 (STATUS "Still open"). |

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `check_isolated` returns `None`, or the CLI exits 3 with "no free NeuronCore" | Cores 2 and 3 are both taken: a third referee, a forgotten in-process `check()`, or `timing.py --selftest`. You can also see this with `CHIPBOOST_CORE` pointing at a busy core. Wait and retry. Run the CLI by hand to see the stderr message, because `check_isolated` swallows it. |
| `None` with a free core | Exit code 3 for another referee reason (the baseline would not compile or load, no sandbox uid, survivors after SIGKILL), or the referee crashed. Run the CLI directly on the same file. |
| `rules`: "reaches the module / private attribute", "touches `__x`", "import inside a function", "module-level `For`/`Expr`" | A kernel file may hold only **imports, constant assignments, function definitions and a docstring**. Imports allowed: `nki`, `nki.isa`, `nki.language`, `nki.typing`, `numpy`, `math`, `ml_dtypes`, `from nki import isa/language/typing`. Not allowed: `from nki.isa import dma_copy`, imports inside functions, classes, `global`, `_private` attributes, dunder names other than `__name__`, non-constant default arguments, and decorators that are calls (`@nki.jit(...)` is rejected, `@nki.jit` is fine). `nl.load`/`nl.store` are allowed. |
| `rules`: "THE CANDIDATE MODIFIED THE REFEREE" for an honest kernel | Something added, removed or changed a `.py`/`.pth`/`.so` file under `projects/03-chipboost/` or `projects/02-kernel-agent/` **while the check ran**: a `git pull`, an edit, or **your loop writing the next candidate there**. Write candidates outside those trees (e.g. `/tmp/chipboost_cands/`). |
| `wrong`: "timed out after 600s" / "killed by SIGXCPU/SIGKILL" | The sandboxed child hit its limits: 600 s wall time, 600 s CPU, 48 GB memory, 4 GB per file, 2048 processes. Usually an unbounded trace-time loop. |
| `wrong`: "the check timed out after 1800s" | The whole `check_isolated` call ran out of time. It is logged as `wrong` with null `code_hash`/`sim_ok`, so fill those in. If it repeats on honest kernels, it is infrastructure: check the core and run the selftest. |
| `wrong`: "A TIMED RUN PRODUCED A WRONG OUTPUT" | The kernel was correct when checked but not when timed: input-dependent shortcuts, or stale state. |
| `wrong`: "PRECISION LOSS: ... bf16 ulps" | Accumulate in an fp32 PSUM tile across all of K, and round to bf16 once. |
| Timings look wrong, or after a crash | `python timing.py --selftest --shapes qwen` (as root, in the pod; it needs core 2 or 3). It checks that the output is written and correct, noise is under 5%, time scales with work, and A/A is about 1.0. It must print `TIMER OK` (exit code 0). Use `--shapes small` (the default) for a quick run. |

**Where the code and the docs disagree** (the code wins):
- STATUS.md lists held-out **before** timing; the code times first and runs held-out only for a would-be
  `faster`.
- TEAM.md's `no_gain` and its threshold of max(5%, 2xIQR) are not in the code yet. The new contract is
  max(1%, 2xIQR) and is landing shortly.
- REVIEW.md's `heldout=False` and `--no-heldout` do not exist.
