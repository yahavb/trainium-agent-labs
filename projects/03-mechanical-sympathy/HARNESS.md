# The harness: how an attempt is run, checked and recorded

This guide explains the test harness in this project: what each piece does, what
happens step by step during one attempt, and how to run it yourself. It also covers
the second, separate harness that grades the kernels written by the Qwen agent.

If you only have two minutes, read [The short version](#1-the-short-version) and
[Quick start](#10-quick-start).

## Contents

1. [The short version](#1-the-short-version)
2. [The pieces and where they live](#2-the-pieces-and-where-they-live)
3. [One attempt, step by step](#3-one-attempt-step-by-step)
4. [The CPU reference: fixture and manifest](#4-the-cpu-reference-fixture-and-manifest)
5. [The adapter: how a candidate plugs in](#5-the-adapter-how-a-candidate-plugs-in)
6. [The runner: what is timed and what is not](#6-the-runner-what-is-timed-and-what-is-not)
7. [The checker: how correctness is decided](#7-the-checker-how-correctness-is-decided)
8. [The attempt record: results/attempts.csv](#8-the-attempt-record-resultsattemptscsv)
9. [Where the harness stands today](#9-where-the-harness-stands-today)
10. [Quick start](#10-quick-start)
11. [Pitfalls](#11-pitfalls)
12. [Other timing tools in this project (not the harness)](#12-other-timing-tools-in-this-project-not-the-harness)
13. [The kernel-agent harness (a separate loop)](#13-the-kernel-agent-harness-a-separate-loop)
14. [Testing the harness itself](#14-testing-the-harness-itself)
15. [Glossary](#15-glossary)

---

## 1. The short version

The harness answers one question: **is this Trainium version of Samudra correct, and
if so, how fast is it?** It always answers in that order. A wrong answer never gets a
speed number.

```text
                 fixtures/cpu_reference.npz   (inputs + the CPU's answer)
                              |
                              v
  your adapter  -->  runners/trainium_runner.py  -->  candidate.npz  +  metrics.json
  (Trainium code)     loads it, compiles, warms up,     (Trainium's       (timings)
                      times run() N times                answer)
                                                            |
                                                            v
                                                       checker.py
                                              compares with the CPU answer
                                                            |
                                      +---------------------+---------------------+
                                      | fail                                      | pass
                                      v                                           v
                         row in attempts.csv,                     row in attempts.csv
                         timing columns left EMPTY                WITH median_ms, p95_ms,
                                                                  steps_per_second
```

`agent.py` is the one command that does the whole chain: it runs the runner, then
the checker, then writes one row to `results/attempts.csv`.

Three ideas hold the design together:

- **The CPU defines the right answer.** The official Samudra code, run on the CPU,
  produces the reference prediction. Trainium must match it within a tolerance.
- **The first correct Trainium result is "Trainium v0".** Every later speed-up is
  measured against v0, never against the CPU.
- **Compilation is never timed.** Trainium compiles a model the first time it runs,
  which can take minutes. The harness compiles first, warms up, and only then starts
  the clock.

---

## 2. The pieces and where they live

All paths are relative to `projects/03-mechanical-sympathy/`.

| Piece | File | What it does |
| --- | --- | --- |
| Fixture generator | `runners/data_smoke.py` | Runs one real Samudra forward pass on the CPU and saves the inputs and the answer to `fixtures/cpu_reference.npz` |
| Fixture | `fixtures/cpu_reference.npz` | The model inputs plus the CPU answer. About 43.7 MB, so it is **not in Git**; regenerate it on each pod |
| Manifest | `fixtures/manifest.json` | Describes the fixture: its SHA-256 hash, which arrays to compare, and the allowed error for each precision. In Git |
| Source pin | `samudra-source.json` | The exact Samudra commit used for the reference. `bootstrap_samudra.sh` checks it out into `.scratch/Samudra` |
| Adapter | e.g. `trainium/candidate_adapter.py` | **Your code.** Loads Samudra onto Trainium and runs it on the fixture inputs |
| Runner | `runners/trainium_runner.py` | Loads the adapter, compiles, warms up, times it and saves the prediction and timings |
| Checker | `checker.py` | Compares the prediction with the CPU answer. Pass or fail, with a score and the reason |
| Controller | `agent.py` | Runs runner → checker → writes one row. The command you normally use |
| Attempt log | `results/attempts.csv` | One row per attempt, pass or fail. In Git |
| Profiling (optional) | `runners/profiling.py`, `PROFILING.md` | Shows where the time goes. Never used for the reported numbers |
| Tests | `tests/test_agent.py`, `tests/test_checker.py`, `tests/test_trainium_runner.py` | Prove the harness behaves as described here |

---

## 3. One attempt, step by step

This is exactly what happens when you run:

```bash
python agent.py --experiment trainium-v0 --adapter trainium/candidate_adapter.py \
  --fixture fixtures/cpu_reference.npz --precision bf16-autocast
```

**Step 1: agent.py starts a row.** It records the time (UTC), your `--experiment`
name, the Git commit of this repository (`runner_commit`), the pinned Samudra commit
from `samudra-source.json`, `--hardware` (default `aws-trainium`) and `--precision`.
The row's status starts as `runner_failed`, so a crash in step 2 is still recorded.

**Step 2: agent.py starts the runner** as a separate Python process and gives it a
temporary folder for its output files.

**Step 3: the runner reads the fixture.** It opens `cpu_reference.npz` and splits
the arrays into three groups, using the names listed in the manifest:

- the **answer**, `prediction`, which is held back and never shown to the adapter;
- the **coordinates**, `time` and `variables`, which are labels copied to the output
  unchanged;
- **everything else**, which becomes the adapter's inputs (`prognostic`, `boundary`,
  `label_mask`, and the four grid arrays).

**Step 4: the runner calls `prepare(inputs, context)` in your adapter and times it.**
This is where the model is built, the weights are loaded, everything is moved to
Trainium and the graph is compiled. The time it takes is recorded as
`compile_seconds`, separately from everything else.

**Step 5: warm-up.** The runner calls `run()` then `synchronize()` `--warmup` times
(default 3). These calls are not timed. They let caches and memory settle.

**Step 6: timed runs.** The runner calls `run()` then `synchronize()` `--repeats`
times (default 20) and times each pair with a high-resolution clock. Each one gives
one latency sample in milliseconds.

**Step 7: the runner saves its results.** It takes the prediction returned by the
**last** timed call and copies it to the CPU. It stops with an error if any value is
NaN or infinite. It then writes two files:

- `candidate.npz`, with `prediction` plus the coordinates copied from the fixture;
- `metrics.json`, with every latency sample, `median_ms`, `p95_ms`,
  `steps_per_second`, `compile_seconds`, the input shapes and any extra metrics your
  adapter reports.

**Step 8: if the runner failed,** agent.py writes a row with status `runner_failed`
and the first 500 characters of the error in `notes`, then stops. You get the
runner's exit code.

**Step 9: agent.py runs the checker** on `candidate.npz` (section 7).

**Step 10: agent.py writes the row.** It always fills in `status`,
`correctness_score`, `input_shape` and the checker's reason (in `notes`). It fills in
`compile_seconds`, `timed_steps`, `median_ms`, `p95_ms` and `steps_per_second` **only
if the checker returned `passed`**. Otherwise those columns stay empty.

**Step 11: clean-up.** The temporary folder is deleted. **`candidate.npz` and
`metrics.json` are not kept**, only the CSV row. To keep them, run the runner and
the checker yourself (section 10, step 5).

agent.py exits with the checker's code: `0` for a pass, `1` for a correctness
failure, `2` if the checker or reference is not ready.

---

## 4. The CPU reference: fixture and manifest

### What is in the fixture

`runners/data_smoke.py --reference-output fixtures/cpu_reference.npz` runs the
**official** Samudra code (the commit pinned in `samudra-source.json`) on the CPU,
with the released Samudra2 one-degree EMA checkpoint and real OM4 ocean data. It
makes one forward pass, starting from 2014-10-10, and saves:

| Array | Meaning | Who uses it |
| --- | --- | --- |
| `prognostic` | The ocean state the model starts from (2 input steps) | Adapter input |
| `boundary` | Forcing at the surface and boundaries for that step | Adapter input |
| `label_mask` | Where the ocean is (land is masked) | Adapter input |
| `input_lat`, `input_lon`, `output_lat`, `output_lon` | Grid coordinates | Adapter input (our adapter does not need them) |
| `prediction` | **The CPU's answer**: the next 2 output steps | Checker only |
| `time` | Dates of the predicted steps | Copied into the candidate, must match |
| `variables` | Channel names such as `t+1:thetao_0` | Copied into the candidate, must match |

It covers one forward call: 2 input steps in, 2 output steps out (the manifest's
`data` block records this).

### What is in the manifest

`fixtures/manifest.json` is small and lives in Git. The checker reads it on every
run.

| Field | Meaning |
| --- | --- |
| `schema_version` | Must be `1` |
| `status` | `draft` or `frozen`. **Graded runs need `frozen`** (section 9) |
| `reference_artifact` | Fixture path, relative to the project folder |
| `reference_sha256` | Hash of the fixture. The checker refuses a fixture whose hash differs, so nobody can grade against a changed reference by accident |
| `prediction_key`, `coordinate_keys` | Which arrays to compare (`prediction`) and which must match exactly (`time`, `variables`) |
| `precision_tolerances` | `atol` and `rtol` for each precision name, currently `fp32` and `bf16-autocast` |
| `checkpoint`, `data`, `samudra_commit` | Where the reference came from, with hashes |
| `repeatability` | Proof that the CPU reference is stable: two runs gave byte-identical files |

### How the reference is made trustworthy

1. Run `data_smoke.py` twice. The two fixtures must be identical. Today they are:
   both runs have the same SHA-256 (`c6c43040…`).
2. Put the fixture's hash in `reference_sha256`.
3. Choose `atol` and `rtol` for each precision from CPU repeatability and the
   known precision of Trainium. **Never** choose them by looking at a Trainium result.
   A tolerance chosen that way would just be tuned until the candidate passes.
4. Set `status` to `frozen`.

---

## 5. The adapter: how a candidate plugs in

The adapter is the only file you write to try a new idea. The harness stays the
same, so every idea is measured the same way.

### The contract

An adapter is a Python file with one function:

```python
def prepare(inputs: dict[str, numpy.ndarray], context: dict) -> object:
    ...   # build the model, load weights, move to the device, COMPILE
    return prepared
```

`context` contains `workload`, `forecast_steps`, `precision`, `hardware` and the whole
`manifest`.

The returned object must have:

| Method | Required | What it must do |
| --- | --- | --- |
| `run()` | yes | Run the model once and return the prediction: an array or tensor, or a dict with a `prediction` key |
| `synchronize()` | strongly recommended | Wait until **all** device work has finished. Without it you time how fast work is *queued*, not how fast it *runs* |
| `close()` | optional | Release resources. Called even if a run fails |
| `metrics()` | optional | Return a dict of extra facts to put in `metrics.json` under `adapter_metrics` |

Two rules matter for honest numbers:

- **Compile inside `prepare`.** Call `run()` and `synchronize()` once at the end of
  `prepare` so the compile happens there, not in the first timed run.
- **`synchronize` must really wait.** On torch-xla, work is lazy: `run()` only
  records a graph. `torch_xla.sync()` launches it and `xm.wait_device_ops()` waits for
  it to finish.

### A minimal adapter

This one just returns the input, so it is useless as a model. It is the smallest
file that satisfies the contract, and the tests use the same pattern:

```python
class Prepared:
    def __init__(self, inputs):
        self.inputs = inputs

    def run(self):
        return self.inputs["prognostic"]

    def synchronize(self):
        pass

def prepare(inputs, context):
    return Prepared(inputs)
```

### The real adapter: `trainium/candidate_adapter.py`

What it does, in order:

1. Reads `--precision`. `fp32` uses no extra compiler flags. `bf16-autocast` sets
   `NEURON_CC_FLAGS="--auto-cast=matmult --auto-cast-type=bf16"`: the weights stay
   fp32, and the compiler runs matrix multiplies and convolutions in bf16. Anything
   else is rejected.
2. Sets `PJRT_DEVICE=NEURON` and, if not already set, `NEURON_RT_VISIBLE_CORES=2`
   (vLLM usually holds cores 0 and 1). This must happen before `torch_xla` is imported.
3. Adds the Neuron PJRT plugin folder (`XLA_PLUGIN_DIR`, default
   `/workspace/xla_plugin`) to the import path.
4. Builds `SamudraNet` from `trainium/samudra_core.py`, a standalone inference copy of
   Samudra's UNet with the same parameter names. It loads the checkpoint strictly and
   fails if any weight is missing or unexpected. Checkpoint: `SAMUDRA_CKPT` or the
   manifest's `checkpoint.path_on_seat`.
5. Moves the model and the three inputs (`prognostic`, `boundary`, `label_mask`) to
   the NeuronCore.
6. Calls `run()` and `synchronize()` once, so compilation happens inside `prepare`.
   About 5 minutes the first time, then served from the Neuron compile cache.

`run()` calls the model, then `torch_xla.sync()`. `synchronize()` calls
`xm.wait_device_ops()`.

Why a copy of the model? The official code uses `F.pad(mode="circular")`. torch-xla
turns that into a large boolean constant that the Neuron compiler (neuronx-cc 2.27)
cannot read (`NCC_EMOD021`). `samudra_core.wrap_lon` does the same wrap-around with
slices and `torch.cat`. On the CPU the result is bit-identical;
`trainium/verify_vs_original.py` checks this against a Samudra checkout.

---

## 6. The runner: what is timed and what is not

| Part | Timed? | Where it is reported |
| --- | --- | --- |
| Loading the adapter, reading the fixture | no | not reported |
| `prepare()`: build, load weights, move to device, compile | separately | `compile_seconds` |
| Warm-up calls | separately | `warmup_seconds` (in `metrics.json` only) |
| Each timed `run()` + `synchronize()` | **yes** | `latency_ms` (every sample), `median_ms`, `p95_ms` |
| Copying the prediction back to the CPU, saving files | no | not reported |

How the summary numbers are computed:

- `median_ms`: the median of the samples.
- `p95_ms`: the 95th percentile, taken as the sample at rank ceil(0.95 × N). With
  20 samples, that is the 19th smallest.
- `steps_per_second` = (repeats × forecast_steps) ÷ (sum of all timed samples, in
  seconds). This is the project's primary metric when the workload is a rollout.

Runner options (defaults in brackets): `--workload` [`single-step`] or `rollout`,
`--forecast-steps` [1], `--warmup` [3], `--repeats` [20], `--precision` (required),
`--hardware` [`$TRAINIUM_HARDWARE` or `aws-trainium`], and the opt-in
`--torch-profile DIR`. Profiling runs **after** the timed loop, so it never affects
the reported latency.

The runner refuses `--workload single-step` with `--forecast-steps` other than 1.

---

## 7. The checker: how correctness is decided

`checker.py` makes these checks in this order and stops at the first failure.

| # | Check | If it fails |
| --- | --- | --- |
| 1 | The manifest is valid: schema 1, `status: frozen`, numeric non-negative `atol`/`rtol` for the requested `--precision` | `configuration_error`, exit **2** |
| 2 | The fixture exists and its SHA-256 matches `reference_sha256` | `configuration_error`, exit 2 |
| 3 | Both files contain `prediction`, `time` and `variables` | `failed`: "missing required arrays", exit **1** |
| 4 | The prediction shapes are equal | `failed`: "prediction shape mismatch" |
| 5 | `time` and `variables` are exactly equal | `failed`: "coordinate mismatch" |
| 6 | Every candidate value is finite | `failed`: "candidate contains NaN or infinite values" |
| 7 | **Every** value is within tolerance (below) | `failed`: "one or more values exceed tolerance" |

### The tolerance rule

Both arrays are converted to float64. For every single value:

```text
|candidate - reference|  <=  atol + rtol * |reference|
```

This is the same rule as `numpy.allclose`. **One value outside the bound fails the
whole attempt.**

### What the checker reports

| Field | Meaning |
| --- | --- |
| `status` | `passed`, `failed`, `diagnostic` or `configuration_error` |
| `reason` | One sentence explaining the verdict |
| `correctness_score` | Fraction of values within tolerance, 0 to 1. Only 1.0 is a pass. It shows how close a failing attempt came |
| `rmse`, `normalized_rmse` | Root-mean-square error, and the same divided by the RMS of the reference |
| `max_abs_error` | The largest single difference |
| `values_compared`, `values_within_tolerance` | Counts behind the score |
| `reference_sha256`, `candidate_sha256` | Exactly which files were compared |
| `performance` | The runner's `metrics.json`, **only if the check passed** and `--performance-json` was given. Otherwise `null` |

### Diagnostic mode

```bash
python checker.py --manifest fixtures/manifest.json --candidate <file> \
  --precision bf16-autocast --diagnostic-only
```

This works on a `draft` manifest. It reports `rmse`, `normalized_rmse` and
`max_abs_error` but gives **no pass/fail and no performance**: `status` is
`diagnostic` and `correctness_score` is `null`. It exits 0. Use it to look at real
errors while the team decides the tolerances. It refuses `--performance-json`.

---

## 8. The attempt record: `results/attempts.csv`

One row per attempt, appended by `agent.py`. The header is written automatically if
the file is new or empty.

| Column | Filled when | Meaning |
| --- | --- | --- |
| `timestamp` | always | UTC start time |
| `experiment` | always | Your `--experiment` name |
| `runner_commit` | always | Git commit of this repository, or `unknown` |
| `samudra_commit` | always | From `samudra-source.json` |
| `hardware`, `precision` | always | From the command line |
| `input_shape` | runner succeeded | JSON with the shape of each input |
| `warmup_runs` | always | `--warmup` |
| `correctness_score` | runner succeeded | From the checker |
| `status` | always | `passed`, `failed`, `configuration_error` or `runner_failed` |
| `notes` | always | Your `--notes` plus the checker's reason or the runner's error |
| `compile_seconds`, `timed_steps`, `median_ms`, `p95_ms`, `steps_per_second` | **only when `status` is `passed`** | Timing from the runner |

Keep failed rows. They are part of the record of what was tried.

The kernel agent (section 13) has its own summary file,
`results/agent_attempts.csv`. Do not mix the two.

---

## 9. Where the harness stands today

As of this commit:

- The CPU fixture exists and is repeatable (two runs, same hash).
- `fixtures/manifest.json` is **`draft`**, and both precisions have
  `atol: null, rtol: null`.
- So **`agent.py` cannot record a pass yet.** Every graded attempt ends as
  `configuration_error` with exit code 2 and the note *"CPU reference is not frozen"*.
  This is intended: the harness refuses to grade until the team has agreed what
  "correct" means.
- `results/attempts.csv` has only its header row.
- No result has passed through this harness yet. The Trainium timings in the
  project README come from the forward-only benchmark (section 12), which does not
  check correctness.

To unlock the harness:

1. Run the adapter at both precisions and look at the errors in diagnostic mode
   (section 10, step 4).
2. Agree `atol` and `rtol` for `fp32` and `bf16-autocast` as a team, based on CPU
   repeatability and the known precision of each mode. A proposal for discussion:
   fp32 `rtol = atol = 1e-4`; bf16-autocast `rtol = 2e-2`, `atol = 0.1`. These are
   not agreed values.
3. Write them into the manifest, set `status` to `frozen`, open a pull request.
4. Run `agent.py` with `--precision bf16-autocast`. The first `passed` row is
   **Trainium v0**.

---

## 10. Quick start

Run everything from `projects/03-mechanical-sympathy/` on your own seat. Use the
Neuron Python (`/opt/conda/bin/python`) for Trainium steps.

**Step 0: one-time seat setup.** See `trainium/README.md` for details.

```bash
./bootstrap_samudra.sh                                   # pinned Samudra into .scratch/Samudra
pip install --no-deps --target /workspace/xla_plugin libneuronxla \
  --extra-index-url https://pip.repos.neuron.amazonaws.com    # Neuron plugin for torch-xla
export OMP_NUM_THREADS=11                                # your pod's CPU quota (cat /sys/fs/cgroup/cpu.max)
```

Find a free NeuronCore. vLLM normally holds 0 and 1:

```bash
for c in 0 1 2 3; do
  PYTHONPATH=/workspace/xla_plugin PJRT_DEVICE=NEURON NEURON_RT_VISIBLE_CORES=$c python -c \
    "import torch, torch_xla, torch_xla.runtime as xr; d=torch_xla.device(); print('core $c OK', xr.device_type(), (torch.ones(2,device=d)*2).cpu())" \
    2>&1 | grep -E "OK|not available" | head -1
done
```

**Step 1: make the CPU fixture** (in the Samudra environment):

```bash
cd .scratch/Samudra && source .venv/bin/activate
python ../../runners/data_smoke.py --samudra-root "$PWD" \
  --checkpoint ../../checkpoints/Samudra2/onedeg/ema_ckpt.pt \
  --data-root ../../data/om4-smoke \
  --reference-output ../../fixtures/cpu_reference.npz
cd ../..
sha256sum fixtures/cpu_reference.npz      # must equal reference_sha256 in the manifest
```

If the hash differs, the checker refuses the fixture. Find out why (a different
Samudra commit, checkpoint or data) before changing the manifest.

**Step 2: check the harness itself** (any Python with NumPy, no Trainium needed):

```bash
python -m unittest tests.test_checker tests.test_trainium_runner tests.test_agent
```

**Step 3: run the adapter once and keep its files:**

```bash
NEURON_RT_VISIBLE_CORES=2 python runners/trainium_runner.py \
  --adapter trainium/candidate_adapter.py --fixture fixtures/cpu_reference.npz \
  --candidate-output runs/trainium/bf16/candidate.npz \
  --metrics-json runs/trainium/bf16/metrics.json \
  --precision bf16-autocast
```

**Step 4: look at the error** (works while the manifest is a draft):

```bash
python checker.py --manifest fixtures/manifest.json \
  --candidate runs/trainium/bf16/candidate.npz --precision bf16-autocast --diagnostic-only
```

Repeat steps 3 and 4 with `--precision fp32` and a different output folder.

**Step 5: once the manifest is frozen, grade it.** The one-command way, which writes
the CSV row and deletes the intermediate files:

```bash
python agent.py --experiment trainium-v0 --adapter trainium/candidate_adapter.py \
  --fixture fixtures/cpu_reference.npz --precision bf16-autocast --notes "first graded run"
```

Or by hand, which keeps `candidate.npz`, `metrics.json` and `check.json` but does
**not** write a CSV row:

```bash
python checker.py --manifest fixtures/manifest.json \
  --candidate runs/trainium/bf16/candidate.npz --precision bf16-autocast \
  --performance-json runs/trainium/bf16/metrics.json --json-out runs/trainium/bf16/check.json
```

**Step 6: commit the CSV row** on a branch and open a pull request
(`GIT_WORKFLOW.md`). `runs/` and the fixture stay out of Git.

---

## 11. Pitfalls

Each of these was checked against the code. Most were also tried on a toy fixture.

- **Precision names must match the manifest exactly.** The checker looks up
  `--precision` as a key in `precision_tolerances`. `fp32` and `bf16-autocast` work.
  The adapter also accepts `float32` and `bf16`, but the checker then stops with
  *"Manifest has no tolerance entry for precision 'float32'"*. (`PROFILING.md` uses
  `--precision float32` in one example; that is fine for profiling, but not for a
  graded run.)
- **`--workload rollout` does not make the adapter run a rollout.** The runner
  passes `forecast_steps` to the adapter in `context` and multiplies by it when
  computing `steps_per_second`. `candidate_adapter.py` ignores it and always runs one
  step. With `--workload rollout --forecast-steps 10`, it still passes the check
  (same shape as the one-step fixture) and reports 10× too many steps per second. The
  fixture also contains only one step. Use `single-step` until there is a rollout
  adapter **and** a rollout reference.
- **`agent.py` deletes `candidate.npz` and `metrics.json`.** Only the CSV row
  survives, and it has no individual latency samples. Run the runner by hand
  (step 3) when you need the files.
- **Wrong core, or CPU fallback.** Without `PJRT_DEVICE=NEURON`, torch-xla can
  quietly run on the CPU. A core held by vLLM gives "not available". Check with the
  loop in step 0.
- **The first run takes minutes.** Compilation takes about 5 minutes for the
  one-degree model and is cached afterwards (`/var/tmp/neuron-compile-cache`). It is
  in `compile_seconds`, never in the latency.
- **Whole-model bf16 crashes the compiler** on the one-degree grid (`NCC_IBIR229`).
  Use `bf16-autocast`, which keeps the weights fp32.
- **CPU threads.** The pod shows 192 cores but is limited to about 11. Too many
  threads makes CPU runs, including the fixture generator, many times slower.
- **Never tune tolerances from Trainium output.** That turns the checker into a
  rubber stamp.
- **Do not report CPU speed as the Trainium baseline.** Trainium v0 is the baseline.

---

## 12. Other timing tools in this project (not the harness)

The project also has faster, simpler timing tools. They are useful for exploring,
but **they do not check the answer against the CPU fixture**, so their numbers are
speed-only.

| Tool | What it measures | Correctness check |
| --- | --- | --- |
| `runners/benchmark_forward.py` + `runners/forward_timing.py` | Forward calls over the eight-year evaluation range (299 calls) on CPU or Trainium, using the official Samudra model | None against the fixture. It refuses timings that include compilation or a CPU fallback |
| `trainium/bench.py` | One forward pass of `samudra_core` with random or real weights | Its own check against a CPU run of the same model (relative RMS ≤ 3%, land exactly zero), not the frozen fixture |
| `trainium/sweep.py`, `trainium/probe_ops.py` | Grid sizes; single operations | Their own quick checks |

The numbers in the project README (FP32 → BF16 autocast → BatchNorm folding →
latitude tiling, 73.29 s → 42.58 s for 299 calls) come from `benchmark_forward.py`.
The README says so: *"These speed-only results have not passed the project's forecast
correctness acceptance workflow."* To make any of those ideas official, put it in an
adapter and send it through `agent.py`.

---

## 13. The kernel-agent harness (a separate loop)

The project has a second harness, `trainium/agent_loop.py`. It grades **small NKI
kernels written by a language model**, not Samudra itself. It never runs Samudra and
never changes Samudra's speed.

### What it does

```text
          +---------------------- prompt -----------------------+
          |                                                     |
          v                                                     |
  Qwen3-8B (vLLM on this seat)  --writes-->  NKI kernel source   |
                                                  |              |
                                                  v              |
                               staged checker, NKI CPU simulator |
                                                  |              |
                              reward + one-sentence reason ------+
                              (the next prompt is a repair request)
```

The task is one fused operation that appears 18 times in Samudra's UNet:
**InstanceNorm followed by a GELU capped at 10.** For an input `x` of shape `[C, N]`,
for each row: subtract the mean, divide by `sqrt(variance + 1e-5)`, apply GELU, then
take `minimum(·, 10)`.

### The staged checker and its reward

Each candidate goes through four stages, cheapest first. The reward is the sum of
the weights of the stages passed:

| Stage | Weight | Passes if |
| --- | --- | --- |
| parses | 0.1 | The code is valid Python |
| rules | 0.2 | Only `nki`, `nki.language`, `nki.isa` are imported; there is a top-level `instnorm_gelu_kernel(x)` with `@nki.jit`; no numpy or torch inside |
| runs | 0.2 | The NKI **CPU simulator** runs it on every test shape without raising |
| correct | 0.5 | Relative RMS error ≤ 2% against a PyTorch reference on every shape |

So a reward of **0.3** means "valid code that crashes in the simulator", and **1.0**
means correct. The test shapes are `128×512`, `200×1000` (not a multiple of 128 rows)
and `280×2048` (Samudra's channel count). Every 37th row has a large outlier, so a
kernel that forgets the cap fails. The inputs are generated from a fixed seed.

Each candidate is written to a fresh file name with bytecode caching off, so a
stale compiled copy of an earlier candidate can never be graded by mistake.

`python trainium/agent_loop.py --selftest` proves the checker before any model is
used. It has 15 planted cases. Twelve are bugs (syntax error, torch import, missing
decorator, wrong name, empty answer, forgotten cap, missing GELU, rows past 128 never
written, NaN, wrong shape, a plain copy of the input, and a copy kernel run through the
simulator), and each must get the expected reward and message. Three are small
differences that must still pass (the exact reference, tanh-approximate GELU, variance
divided by N−1), which shows the 2% tolerance is not too strict.

**Speed is not measured.** The docstring mentions a stage 2, `device_check.py`, to
time correct kernels on the chip. That file is **not in this repository**.

### The loop

Each **run** has up to 8 **rounds** with 4 **samples** each. After a round, the best
sample of that round, together with its checker message, becomes the next prompt (a
repair request). A run stops as soon as a sample reaches 1.0; that kernel is saved to
`kernels/instnorm_gelu_<run>.py`. `--repeat 3` does three independent runs, which
gives a solve rate.

Every attempt is one JSON line in the `--log` file, with: `run`, `round`, `sample`,
`reward`, `parts` (which stages passed), `feedback`, `source` (the full kernel),
`gen_s` (model time), `check_s` (checker time), `chars`, `prompt_chars`,
`reply_chars`, `feedback_version`, `prompt_version` and `time`.

### The three versions

Each version changes only what Qwen is told.

| Version | Flags | Change |
| --- | --- | --- |
| v1 | `--feedback v1 --prompt v1` | The organizers' error enrichment, the task and an example kernel |
| v2 | `--feedback v2 --prompt v1` | Feedback also quotes the failing line and gives its exact rewrite (`keepdims=True`, `(rows, 1)` shapes, the real function for an invented name) |
| v3 | `--feedback v2 --prompt v2` | Adds a verified NKI API card to every prompt. `trainium/card_check_kernel.py` is a kernel built only from the card; it scores 1.0, which proves the card is right. The model never sees it |

Results so far (8 rounds × 4 samples × 3 runs each):

| Version | Seat, Qwen tensor parallel | Attempts | Solved runs | Notes |
| --- | --- | --- | --- | --- |
| v1 | seat-212, TP=2 | 96 | 0 / 3 | From round 3 on Qwen resent the same code. Best 0.30 |
| v2 | seat-212, TP=2 | 96 | 0 / 3 | Applied the exact `keepdims` fix in round 2 in all 3 runs, then crashed on Python `/` between tiles (34 of 96 attempts) and other invented calls. Best 0.30 |
| v3 | seat-213, TP=4 | 72 | **1 / 3** | Run 1 solved in round 2 by renaming one argument (`data=` → `data1=`). The two other runs got stuck: one on invented names (`nl.constant`, `nl.tensor_scalar`) and then a misused `gather_flattened`, the other on `ndarray(data=...)` for six rounds |

The v3 log, attempts and solved kernel are in `results/agent-v3-seat213-2026-10-10/`,
with a summary row in `results/agent_attempts.csv`. The v1 and v2 logs are on seat
212, not in Git. Three runs per version is a small sample: treat 1/3 against 0/3 as
a promising sign, not a proven effect.

### How to run it

The loop needs the seat's vLLM server (`./serve.sh`, Qwen3-8B) and the organizers'
`/workspace/projects/02-kernel-agent` (change it with `KERNEL_AGENT_DIR`). It does not
need a free NeuronCore.

```bash
cd trainium
python agent_loop.py --selftest                     # must end with SELFTEST OK
python agent_loop.py --rounds 8 --samples 4 --repeat 3 \
  --feedback v2 --prompt v2 --log attempts_v3.jsonl
python analyze_attempts.py attempts_v1.jsonl attempts_v2.jsonl attempts_v3.jsonl
```

### How the two harnesses relate

| | Samudra harness (`agent.py`) | Kernel-agent harness (`agent_loop.py`) |
| --- | --- | --- |
| What is graded | A full Samudra forward pass on Trainium | One small NKI kernel |
| Who writes the candidate | A person (an adapter file) | Qwen3-8B |
| Where it runs | A real NeuronCore | The NKI simulator on the CPU |
| Reference | The CPU fixture from the official Samudra code | A PyTorch function on fixed random inputs |
| Correct means | Every value within atol + rtol × abs(reference) | Relative RMS error ≤ 2% on 3 shapes |
| Speed | Measured after a pass | Not measured |
| Record | `results/attempts.csv` | JSONL log + `results/agent_attempts.csv` |

A correct agent kernel does not speed up Samudra by itself. To do that, someone
would have to put it into an adapter in place of the matching PyTorch operations and
send that adapter through `agent.py`. Nobody has done this yet.

---

## 14. Testing the harness itself

```bash
python -m unittest tests.test_checker tests.test_trainium_runner tests.test_agent
```

The 11 tests check that the checker:

- accepts values within tolerance;
- rejects a shape mismatch, NaN or infinite values, too much error, and a coordinate
  mismatch;
- refuses a draft reference;
- picks the tolerance for the requested precision, and fails if that precision has
  none;
- in diagnostic mode, reports the errors with no pass and no performance.

They also check that an adapter's output written by the runner passes the checker,
and that `agent.py` writes timing for a correct adapter and leaves it empty for a
wrong one. They need only NumPy and run in a few seconds without Trainium.

---

## 15. Glossary

| Term | Meaning |
| --- | --- |
| Adapter | Your Python file that implements `prepare()` / `run()` / `synchronize()` |
| Fixture | `cpu_reference.npz`: the inputs and the CPU's answer |
| Manifest | `manifest.json`: the fixture's hash and the tolerances |
| Frozen | The manifest's tolerances are agreed; graded runs are allowed |
| Trainium v0 | The first attempt that passes the checker. The speed baseline |
| bf16-autocast | Weights stay fp32; the compiler runs matrix multiplies and convolutions in bf16 |
| NeuronCore | One compute unit of the Trainium2 chip. With LNC=2, a seat sees 4 logical cores with 24 GB each |
| Compile cache | Where neuronx-cc stores compiled graphs (`/var/tmp/neuron-compile-cache`), so a second run starts fast |
| NKI | Neuron Kernel Interface: Python for writing low-level Trainium kernels |
| NKI simulator | Runs an NKI kernel on the CPU to check it, without a chip |
