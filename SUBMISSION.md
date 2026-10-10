# Project 2: an NKI kernel agent that knows when it failed

Team submission, NYU × Annapurna Labs Trainium hackathon, 2026-10-10. Team: [[TBD: team name, members]].

**Start here.** Section 1 is the one-page note: what we ran, on what, what came out, how many runs, the
spread, and how to reproduce it. The sections after it are the evidence. Unless a line says *on chip*,
every number comes from the NKI 0.6.0 CPU simulator (`nki.simulate`), graded by our checker.

---

## 1. One-page note

**What we ran.** An agent loop that asks a model for an NKI kernel, grades it, turns the checker's verdict
into one repair instruction, and tries again: at most 8 rounds of 4 samples per level, inside an
8,192-token context. The final agent is `feedback_v7.py`, which layers v2–v7 over the organizers'
`agent.py` (configuration and every change: [V7.md](projects/02-kernel-agent/V7.md)). [[TBD: plus E-F if adopted]]

**On what.**

| | |
|---|---|
| model | Qwen3-8B, thinking off, served by vLLM on the seat pod's Trainium2 chip: one chip at LNC=2, tensor parallel 2, max-model-len 8192, max-num-seqs 4 |
| checker | `nkibench.py` on NKI 0.6.0, simulating trn2 (on a seat pod NKI picks trn2 from the hardware, which we confirmed with a probe kernel; the v7 runs also set it explicitly; the baseline and experiment A re-graded under trn2 and trn3 give identical scores, [analysis/sim_target_check.md](analysis/sim_target_check.md)), with an on-chip allocation audit and a held-out set: [CHECKER.md](projects/02-kernel-agent/CHECKER.md), [EVAL.md](projects/02-kernel-agent/EVAL.md) |
| where | [[TBD: 5]] seat pods in parallel, one agent process per model server |
| speed | about 50 s per round of 4 samples, bound by generation: 13.9 tok/s for one stream, 22.1 tok/s in total for four |

**What came out.** [[TBD: replace with `scripts/summarize.py` output for the final run]]

| level | operation | baseline: solved, 5 scores | final: solved, 5 scores | attempts to first 1.0 | held-out verdicts | tokens per attempt, in / out |
|---|---|---|---|---|---|---|
| 1 | average pool 2D | 0/5 · .30 .30 .30 .30 .30 | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| 2 | 2D transpose | 3/5 · 1 .30 1 .30 1 | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| 3 | matmul, one tile | 0/5 · .30 .30 .30 .30 .30 | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| 4 | matmul, tiled | 0/5 · .62 .62 .50 .62 .62 | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |

**How many runs, and the spread.** Every cell is 5 runs of one configuration. We report the rate, never the
best run, and next to it the number of **distinct trajectories**: the seat's model server decodes greedily in
practice (4 concurrent requests at temperature 0.7 come back byte-identical, and so does `n=4`), so 5 runs
are often the same run 5 times. [[TBD: E-div result: whether per-sample prompt variation made the runs distinct]] [[TBD: one sentence on the spread of the final run.]] The baseline was run twice,
on two seats with the organizers' agent: L1 0/5, L2 3/5 and 2/5, L3 0/5, L4 0/5 both times, with identical
scores on L1, L3 and L4. Both logs were re-graded from scratch with the current checker under trn2, and
every attempt matched ([analysis/calibration_baseline_seat116.md](analysis/calibration_baseline_seat116.md),
[analysis/calibration_replica_seat119.md](analysis/calibration_replica_seat119.md)).

**Reproduce.** Everything except the model runs on a laptop; the agent itself runs on a seat pod.

On a seat pod (`kubectl exec -it seat-<N> -- bash`; wait for the `root@seat-<N>:/workspace#` prompt). `/workspace`
is the organizers' repository; put ours next to it:

```bash
git config --global --add safe.directory '*'
git clone https://github.com/liuyq123/trainium-agent-labs.git /workspace/team   # [[TBD: the URL judges will use after the PR]]
cd /workspace && MAX_MODEL_LEN=8192 ./serve.sh      # the model server: about 4 minutes, keeps this shell
```

In a second shell on the same pod:

```bash
cd /workspace/team/projects/02-kernel-agent
python nkibench.py --selftest                                                        # SELFTEST PASSED
for l in 1 2 3 4; do python nkibench.py --level $l --eval reference_level$l.py | head -1; done   # 20/20 16/16 4/4 16/16
# the exports in V7.md "Run it", with per-level file names, then for each level N:
export NKI_VERDICTS=nki_verdicts_LN.jsonl USAGE_LOG=usage_LN.jsonl
nohup python3 feedback_v7.py --level N --rounds 8 --samples 4 --context 8192 --repeat 5 \
    --log attempts_LN.jsonl --verdicts verdicts_LN.jsonl > run_LN.log 2>&1 < /dev/null &
```

Without a seat, the checker and the agent's loop still run (no model: `--offline` replays the reference
kernels). Set up NKI 0.6.0 per [SETUP_PYTHON.md](SETUP_PYTHON.md) (on a Mac, its Docker step), then, with
`~/venvs/nki/bin` on your PATH and `export NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, the same `--selftest` and
`--eval` lines, and V7.md's exports with `python3 feedback_v7.py --offline --all`. Offline runs make no model
calls, so they write no `USAGE_LOG`.

The tables, from the logs (on a laptop; `python3` with matplotlib for the token chart):

```bash
python3 -m venv .venv && .venv/bin/pip install matplotlib
.venv/bin/python scripts/report.py analysis/final analysis/logs/final      # checks, summary, taxonomy, token chart
.venv/bin/python scripts/report.py /tmp/baseline analysis/logs/baseline    # the baseline: L1 0/5, L2 3/5, L3 0/5, L4 0/5
```

---

## 2. The checker: what it accepts, what it rejects, and why

Full account: [CHECKER.md](projects/02-kernel-agent/CHECKER.md). The decisions that matter:

- **Order.** Parse, static rules, load from a fresh file, simulate with the allocation audit on, compare
  numerics on every loop shape, score. When a level ends, the held-out set runs once.
- **Tolerance: max |error| ≤ 2e-2 × RMS(reference output).** Relative to the output's RMS, because
  per-element relative error blows up where matmul and avgpool outputs cross zero. Measured, not chosen:
  rounding the inputs to bf16 moves the reference by at most 0.0134 RMS; the smallest error any real bug
  produced was 0.393 RMS.
- **Hardware legality before numerics.** The simulator allocates tiles without checking hardware limits,
  so a kernel can pass numerically and still be illegal. The audit records every on-chip allocation and
  rejects a partition dimension over 128, PSUM over 16 KiB per partition, and SBUF over 192 KiB per
  partition. 192 KiB is deliberately conservative (trn2 has 224 KiB): it can only reject a legal
  kernel, never pass an illegal one.
- **A held-out set the model never sees.** New shapes per level times four kinds of values: a fresh
  normal draw, a ramp (every element distinct), ×1e4 (float16 overflows), and float16 inputs, plus an
  output dtype check. The reference kernels pass all of it (20/20, 16/16, 4/4, 16/16). Four deliberately
  wrong kernels that pass every loop shape are all caught. [EVAL.md](projects/02-kernel-agent/EVAL.md)
- **Bugs we found in our own checker, and what we re-checked.**
  - *Path cache.* NKI caches a kernel by its file path. Grading several candidates from one path in one
    process graded a later candidate as the first one, numerics included, which can make both false solves
    and false failures. Fixed by a fresh file per candidate (c39c0ce; diagnosis in ded1ef2). Every log from
    before the fix was re-graded from scratch: baseline 424/424, replication 420/420, experiment A 160/160,
    v3 100/100 attempts identical to what was logged.
  - *Audit left installed.* The audit's wrappers stayed in place after a simulation and broke the trn2
    compiler in the same process. They are now installed only while a simulation runs (183b384).
  - *Simulation target.* Without a Neuron device, NKI 0.6.0 simulates trn3, which accepts a 1024-wide moving
    tile that trn2 rejects. On a seat NKI picks trn2 from the hardware; the v7 runs also set it, and the
    earlier logs give the same scores under both targets (530ab9a).
- **What it does not check.** Speed on the device (every number is simulated; levels 5–7's traffic bars
  are a model, not a measurement). Level 3 has a single loop shape, because the organizers' reference
  asserts it. The static rules are a text scan. The output dtype is checked only in the held-out set. SBUF
  is held to 192 KiB per partition, below trn2's 224 KiB.

## 3. The agent: what changed, and the rule behind every message

The loop is the organizers' `agent.py`: up to 8 rounds of 4 samples, the round's best kernel is repaired
next, the same failure twice adds a one-line-per-attempt ledger of what already failed. On top of it, the
final agent stacks these layers (full list and sources: [V7.md](projects/02-kernel-agent/V7.md)):

| layer | what it changes | the failure it answers (baseline counts) |
|---|---|---|
| first prompt (`PROMPT1=v2`) | states the exact `def` line and a short block of real NKI calls, each checked to lower for trn2 | invented APIs: 80 of 160 level-1 attempts |
| worked example (`CARD=category`) | one example chosen by the operation's category: a row mean for matmuls, a channel mean for reductions | 1-D tiles, reshape instead of slicing (70 of 112 level-3 attempts) |
| repair messages (v2–v5) | quote the failing line; for a known error class give the fix as lines to paste, in the model's own variable names; a too-large matmul operand gets the whole three-loop tiling at once | partition over 128 (55 of 84 level-4 attempts) and copy-size mismatches |
| repair prompt (`restructure`) | "use the code given; restructure around new loops or tiles if needed" replaces "keep everything else identical" | the old line forbade the loop restructuring tiling needs |
| invented-name map (our experiment A) | known invented calls (`nisa.multiply`, `transpose_moving`, …) mapped to the real 0.6.0 call | invented APIs |
| grading fixes, compiler gate | a fixed `grade()` with a 120 s timeout; a full-score kernel using a form the trn2 compiler rejects is held at 0.95 and given the rewrite | compiler-only failures |
| verdict | after each level: a confidence, then extra hostile cases and lowering for trn2 | see §5 |
| sampling (`SAMPLING=qwen`) | Qwen3's thinking-off sampling settings | none in practice: the server decodes greedily |

[[TBD: ablation table: which of these layers made levels 3 and 4 solvable]]

The rule for feedback: one failure, one instruction, naming the failing line. A verdict ("line 16 calls a
banned function") is rewritten as an instruction ("change this call; keep everything else"). **For known
error classes the instruction is code**: the lines to paste, written in the model's own variable names and
shape expressions, never the test shapes' numbers and never taken from a reference kernel. The model
copies them (§8 shows it doing so). We chose this deliberately and report it, because it means part of
the solution comes from the checker, not the model.

**What the model sees, and what it never sees.** The first prompt carries one worked example chosen by the
operation's category: a row mean for matmul levels, a channel mean for reductions, nothing otherwise. For
level 1 (average pooling) the channel mean is a strong hint, and we say so: it has the same pipeline (load
to SBUF, `nl.sum` with `keepdims`, scale with `tensor_scalar`, store) but not the windowed access pattern
that turns a mean into a pooling. It is not the answer; judge for yourself. What never reaches a prompt: the
NKI tutorial kernels, the organizers' reference kernels for levels 1–4, and our answers for levels 9–14. A
line-by-line scan of every request v7 sends (35 distinct prompts, first and repair rounds, levels 1–4 and
9–14) and of 2,851 string constants in the agent found 0 lines from any of them; the same scanner finds 95
on a positive control. ([analysis/prompt_leak_check.md](analysis/prompt_leak_check.md))
Thinking stays off: with it on, a round took 446 s instead of about 8 s on the same model, and every
sample was cut off before any code.

## 4. Experiments, one change at a time

Each change was measured with `--repeat 5` against the current reference and kept or rolled back by rules
written down before the results came in ([PLAN.md](PLAN.md) §4).

| id | change | level | result (solved, scores) | decision |
|---|---|---|---|---|
| E-A | invented NKI names mapped to the real 0.6.0 calls | L1 | 0/5, all 0.30; invented names 80 → 20, the failures moved one layer deeper | kept as groundwork |
| E-v3 | feedback_v3 | L4 | 1/2 before the seat went to v7: solved on round 6, then a 0.75 | superseded by v7 |
| E-F | wrong argument list: failing line + real signature + one instruction | L1 | 0/5, all 0.30, one trajectory; wrong-signature errors 8 → 3 per run, the run then stalls on copy sizes | not carried into v7: v7's level-1 failures are different, and with greedy decoding any message change can move v7's solved trajectories |
| E-v7 | feedback_v7 as a whole | L1, L3, L4, L2 | L3 5/5 (round 1), L4 4/4 (round 3, one trajectory), L1 [[TBD]], L2 [[TBD]]; all solves VERIFIED [[TBD: final after re-audit]] | adopted |
| E-div | v7, plus a one-line `(attempt k of n, run r)` tag on samples 2–4 so a greedy server returns different samples | L1–L4 | [[TBD]] | [[TBD]] |

Changes that looked reasonable and were worse, kept here because they cost us time: [[TBD: the 128-row
copy example (L2 2/5 → 0/5, L4 0.62 → 0.30, rolled back); thinking on; compiler -O3 not ready in 33 min.]]

## 5. Does the agent know when it failed?

[[TBD: confidence buckets against the held-out outcome, Brier score, and the number of "confident (≥ 0.5)
but wrong" claims, for the final run; agent.py's confidence and v7's verdict side by side.]]

Baseline, scored after the fact with the same confidence function: the 3 solved runs said 0.90 and all
passed the held-out set; the 17 unsolved said 0. Brier 0.001, or 0.010 over the 3 real predictions; no
confident-but-wrong claim. The replication adds 2 more solves, both 0.90 and both passing the held-out set.
Five predictions say little, which is why the final run matters.
([analysis/calibration_baseline_seat116.md](analysis/calibration_baseline_seat116.md))

## 6. Where the tokens went

[[TBD: analysis/token_budget_final.png, tokens per attempt split into instructions, reference, API card,
previous code, feedback and ledger, with the 8,192 line; exact counts from the server's usage log.]]

## 7. Failure taxonomy

[[TBD: final table from scripts/taxonomy.py.]] Baseline, 424 attempts in 25 named modes: index and size
arithmetic 52%, unfamiliar API 24%, tiling rules 18%, memory placement 5%. Four modes were never fixed by the
original feedback once they appeared (stuck rate 100%): out-of-bounds index, reshape instead of slicing,
invented function, partition dimension over 128.
([analysis/taxonomy_baseline_seat116.md](analysis/taxonomy_baseline_seat116.md))

## 8. A failure, and the recovery

[[TBD: one complete transcript: the failing kernel, the checker's words, the instruction sent back, the
round that fixed it, tokens per round.]]

## 9. Limits

- **The simulation target matters in general.** With no target set and no `neuron-ls` on the machine,
  NKI 0.6.0 simulates trn3. A level-4 kernel with a 1024-wide moving tile passes 2/4 shapes under trn3 and
  0/4 under trn2 ("moving free dimension 1024 exceeds max 512 for nc_version.gen3"). We set trn2
  explicitly and re-graded earlier runs under it. ([analysis/sim_target_check.md](analysis/sim_target_check.md))
- **Greedy decoding.** The seat's model server ignores the sampling temperature in practice, so repeated runs of
  one configuration are mostly copies, and "5/5" can mean one trajectory five times. We report distinct
  trajectories next to every rate. A request with a `seed` parameter returned HTTP 500 and took the server down once.
- [[TBD: the rest]]

## 10. Files

| what | where |
|---|---|
| attempt logs, every attempt with its score | earlier runs: [analysis/logs/](analysis/logs/README.md) (baseline, replication, experiments; what each is, its code version, md5, and whether it is comparable); final run: [[TBD: analysis/logs/final/]] |
| results table | [[TBD: analysis/summary_final.md]] |
| checker, eval set, tolerance | [CHECKER.md](projects/02-kernel-agent/CHECKER.md), [EVAL.md](projects/02-kernel-agent/EVAL.md), `nkibench.py` |
| agent | `agent.py`, `feedback_v2.py` … `feedback_v7.py`, [V7.md](projects/02-kernel-agent/V7.md) |
| hand-in kernels, one per level | [[TBD: nki_kernels/]] |
| how we ran the day | [PLAN.md](PLAN.md), [NOTES.md](NOTES.md) |
