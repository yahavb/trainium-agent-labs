# Project 2: an NKI kernel agent that knows when it failed

Team submission, NYU × Annapurna Labs Trainium hackathon, 2026-10-10. Team 24, "Saturday team" (seats
115–119): yl8406, tg3077, sz3941, sm14493, yx3019.

**Start here.** Section 1 is the one-page note: what we ran, on what, what came out, how many runs, the
spread, and how to reproduce it. The sections after it are the evidence. Unless a line says *on chip*,
every number comes from the NKI 0.6.0 CPU simulator (`nki.simulate`), graded by our checker.

**In one paragraph.** We built a checker-driven agent that writes NKI kernels with a model that sees 8,192
tokens (Qwen3-8B, served on the seat's Trainium2). The organizers' agent solved only level 2 (3 runs of 5).
The final version solves level 1 in 5 of 5 runs with five different kernels, levels 3 and 4 in 5 of 5, and
level 2 in 5 of 9, and in 7 of 9 once the checker says what a wrong level-2 kernel actually computed (v8.3)
[[TBD: final counts]]; every solve passed a
fresh-process re-audit and a held-out set of new shapes and hostile values [[TBD: final-run numbers]]. The
model never changed; what it was shown did. One worked example in the first prompt solves level 3; the code
in the repair messages solves level 4 (hollowed out, level 4 stopped solving). Two of level 1's solves show how: once a
one-line instruction, "Add keepdims=True to this call", sent in place of a raw simulator error; once a
fresh first attempt that was right in the simulator but not legal on trn2, for which a compiler gate handed
back the rewrite. Along the way we found that the seat's model
server is deterministic, so "5 runs" were often one run five times. We report distinct runs next to every
rate and changed the agent so its samples actually differ.

**Where each required item is.**

| asked for | where |
|---|---|
| the checker, and why it accepts and rejects what it does (README Part 4) | §2, [CHECKER.md](projects/02-kernel-agent/CHECKER.md) |
| the attempt log, every attempt with its score (Part 4) | [analysis/logs/](analysis/logs/README.md) |
| a one-page note: what ran, on what, how many runs, the spread (Part 4) | §1 |
| the agent (kernel-agent challenge) | §3, [V7.md](projects/02-kernel-agent/V7.md) |
| the verification harness, with its tolerance and the reasoning | §2, [EVAL.md](projects/02-kernel-agent/EVAL.md) |
| the eval set, hostile values included | [EVAL.md](projects/02-kernel-agent/EVAL.md) |
| the failure taxonomy | §7 |
| token instrumentation | §6 |
| a one-page reproduction note | §1, "Reproduce" |
| does the agent know when it failed | §5 |
| a failure and the recovery | §8 |

---

## 1. One-page note

**What we ran.** An agent loop that asks a model for an NKI kernel, grades it, turns the checker's verdict
into one repair instruction, and tries again: at most 8 rounds of 4 samples per level, inside an
8,192-token context. The final agent is `feedback_v8.py` at tag `final`: liuyq's v2–v7 layers over the
organizers' `agent.py`, and our v8 layer on top ([V7.md](projects/02-kernel-agent/V7.md),
[V8.md](projects/02-kernel-agent/V8.md)). Its numbers come from three commits that send byte-identical
requests wherever they overlap: levels 1, 3 and 4 were run on v8.2 (96a9fc9), level 2 on v8.3 (5c3aba2, which
adds error distillation for level 2), levels 5–7 on v8.4 (15fb0d5, which adds a warm start for levels 5–7)
([PLAN.md](PLAN.md) §3, [analysis/v83_l134_identity.md](analysis/v83_l134_identity.md)).

**On what.**

| | |
|---|---|
| model | Qwen3-8B, thinking off, served by vLLM on the seat pod's Trainium2 chip: one chip at LNC=2, tensor parallel 2, max-model-len 8192, max-num-seqs 4 |
| checker | `nkibench.py` on NKI 0.6.0, simulating trn2 (on a seat pod NKI picks trn2 from the hardware, which we confirmed with a probe kernel; the v7 runs also set it explicitly; the baseline and experiment A re-graded under trn2 and trn3 give identical scores, [analysis/sim_target_check.md](analysis/sim_target_check.md)), with an on-chip allocation audit and a held-out set: [CHECKER.md](projects/02-kernel-agent/CHECKER.md), [EVAL.md](projects/02-kernel-agent/EVAL.md) |
| where | seat pods 116–119 in parallel for the final version (seat 115 for liuyq's chip runs), one agent process per model server |
| speed | 50–80 s per round of 4 samples, up to 4 minutes when answers run to the 2,500-token limit; bound by generation: 13.9 tok/s for one stream, 22.1 tok/s in total for four. A level takes 1–12 minutes per run |

**What came out.** The final candidate is v8.2 (commit 96a9fc9; switches in §4). At least five runs per level, every one of them reported, rounds
counted from 0 (round 0 is the first prompt). [[TBD: refresh from `scripts/report.py` on analysis/logs/final/; v8.3 if it
replaces v8.2 on level 2]]

| level | operation | baseline (organizers' agent) | v7 | **v8.2: solved, first 1.0 at round** | distinct | held-out |
|---|---|---|---|---|---|---|
| 1 | average pool 2D | 0/5 · .30 .30 .30 .30 .30 | 0/1 · best 0.50 | **5/5** · rounds 2, 2, 0, 4, 2 | 5 different solving kernels | all VERIFIED (20/20 each); all 5 fully built for trn2 and matching in birsim on two shapes |
| 2 | 2D transpose | 3/5 · 1 .30 1 .30 1 (replication 2/5) | 0/3 | **5/9** · .50 1 .50 .50 1 .50 1 1 1 [[TBD: rounds]] | [[TBD: trajectories over 9 runs]] | 5 VERIFIED, 4 NOT SOLVED; both verdicts agree; every solve builds for trn2 and matches in birsim |
| 3 | matmul, one tile | 0/5 · .30 .30 .30 .30 .30 | 5/5 · round 0 | **5/6** · rounds 0, 2, 0, 0, 0; the sixth, inside the one-command `--all` run, stopped at 0.30 after four identical failures | 3 solving kernels | [[TBD: held-out]]; builds for trn2, matches in birsim |
| 4 | matmul, tiled | 0/5 · .62 .62 .50 .62 .62 | 5/5 · round 2 | **5/5** · round 2 every run | **1 trajectory**: five copies of one path, the same kernel v7 found | [[TBD: held-out]]; builds for trn2, matches in birsim |

We also ran the final version once end to end, one command for all four levels (`--all --repeat 1`): levels 1,
2 and 4 solved, level 3 did not, and that run is counted above. Every solving kernel passed a fresh-process re-audit on trn2, and every one of v8.2's and v8.3's solves on
levels 1–4 also builds in full for trn2 and matches in birsim, the compiler's instruction-level simulator
(§10). Against our held-out set, no verdict was confident (≥ 0.5) and wrong; against a full trn2 build, v7's
verdict was once, on liuyq's seat-115 run (§5).
Level 4's 5/5 is one path, not five: each run's first sample is the same request, so every run repairs the
same kernel.

**What we learned.**

1. **The checker and the prompt are the agent.** The model did not change; what it was shown did. Level 3
   went from 0/5 to 5/5 on a worked example in the first prompt, and the ablation says that example is the
   one layer level 3 cannot do without (§3).
2. **A simulator pass is not a chip pass.** Our allocation audit rejects kernels the simulator runs but
   no chip can hold (§8 shows one in a live run). liuyq's compiler gate finds level-1 kernels the simulator
   accepts and the trn2 compiler rejects.
3. **The model server is deterministic**: the same request returns the same text, even at temperature
   0.7, so 5 runs were often one run five times. We report distinct trajectories next to every rate, and v8 makes the samples differ (§4).
4. **The token budget was not what bound us; reading `finish_reason` was.** Prompts stay near 1,100
   tokens, repairs near 750, far below 8,192 (§6). But one layer dropped the `finish_reason` check, and a
   level-1 run lost 22 minutes to cut-off answers graded as syntax errors (§9).

**How many runs, and the spread.** Every cell is at least 5 runs of one configuration. We report the rate, never the
best run, and next to it the number of **distinct trajectories**: the seat's model server is deterministic
(4 concurrent identical requests at temperature 0.7 come back byte-identical, and so does `n=4`; only a
different request, or different sampling settings, gives different text), so 5 runs are often the same run
5 times. Tagging samples 2–4 (E-div) and restarting them from the first prompt (E-mix) made v8.2's runs
differ where it matters: five level-1 runs, five different kernels. Level 4's runs still repeat one path.
[[TBD: one sentence on the spread of the final run.]] The baseline was run twice,
on two seats with the organizers' agent: L1 0/5, L2 3/5 and 2/5, L3 0/5, L4 0/5 both times, with the same
scores on L1, L3 and L4 (in a different order on L4). Both logs were re-graded from scratch with the current checker under trn2, and
every attempt matched ([analysis/calibration_baseline_seat116.md](analysis/calibration_baseline_seat116.md),
[analysis/calibration_replica_seat119.md](analysis/calibration_replica_seat119.md)).

**Reproduce.** Everything except the model runs on a laptop; the agent itself runs on a seat pod.

On a seat pod (`kubectl exec -it seat-<N> -- bash`; wait for the `root@seat-<N>:/workspace#` prompt). `/workspace`
is the organizers' repository; put ours next to it:

```bash
git config --global --add safe.directory '*'
git clone https://github.com/liuyq123/trainium-agent-labs.git /workspace/team
git -C /workspace/team checkout final   # the final version
cd /workspace && MAX_MODEL_LEN=8192 ./serve.sh      # the model server: about 4 minutes, keeps this shell
```

In a second shell on the same pod:

```bash
cd /workspace/team/projects/02-kernel-agent
python nkibench.py --selftest                                                        # SELFTEST PASSED
for l in 1 2 3 4; do python nkibench.py --level $l --eval reference_level$l.py | head -1; done   # 20/20 16/16 4/4 16/16
# the three exports in V8.md "Run it" (v8.2's switches), then for each level N:
nohup python3 feedback_v8.py --level N --rounds 8 --samples 4 --context 8192 --repeat 5 \
    --log v82_LN.jsonl --verdicts verdicts_v82_LN.jsonl > run_LN.log 2>&1 < /dev/null &
```

Without a seat, the checker and the agent's loop still run (no model: `--offline` replays the reference
kernels). Set up NKI 0.6.0 per [SETUP_PYTHON.md](SETUP_PYTHON.md) (on a Mac, its Docker step), then, with
`~/venvs/nki/bin` on your PATH and `export NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, the same `--selftest` and
`--eval` lines, and V8.md's exports with `python3 feedback_v8.py --offline --all`. Offline runs make no model
calls, so they write no `USAGE_LOG`.

The tables, from the logs (on a laptop; `python3` with matplotlib for the token chart):

```bash
python3 -m venv .venv && .venv/bin/pip install matplotlib
.venv/bin/python scripts/final_auto.py analysis/logs/final analysis/final   # groups runs by version, then checks, summary, taxonomy, token chart
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
- **Levels 5–7** compute the same matmul as level 4 and add a traffic bar: a correct kernel that moves more
  HBM bytes than the level allows (1.60× the byte floor for level 5) scores 0.5 + 0.5 × the fraction of loop shapes that pass (3 of 4 → 0.875).
  Their held-out set checks values on new shapes, not traffic; the bar is enforced on the loop shapes, and the
  final solves' traffic was re-measured afterwards on a larger multi-block shape. [[TBD: that result, if L5–7 run]]
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
| sampling (`SAMPLING=qwen`) | Qwen3's thinking-off sampling settings | none on levels 3 and 4 (ablation below); on level 2 it cost the round-0 solves (2/20 → 0/20), so v8.2 alternates it with the original settings |

**Which layer did it: leave-one-out ablation.** Each row switches one v7 layer back to the organizers'
original and keeps the rest. Because the server is deterministic, one run is the trajectory, so each cell is one run.
Rounds count from 0.

| switched back | level 3 | level 4 |
|---|---|---|
| nothing (v7) | 1.0, round 0 | 1.0, round 2 |
| first prompt (`PROMPT1=theirs`) | 1.0, round 0 | 1.0, round 1 |
| **worked example (`CARD=theirs`)** | **0.30, not solved** | 1.0, round 4 |
| all-dims tiling message (`MESSAGES=v4`) | 1.0, round 0 | 1.0, round 4 |
| repair prompt (`REPAIR_PROMPT=theirs`) | 1.0, round 0 | 1.0, round 2 |
| sampling settings (`SAMPLING=theirs`) | 1.0, round 0 | 1.0, round 2; rounds 0–1 identical to v7 |

Level 3 needs exactly one layer, the worked example in the first prompt; without it, level 3 stays
unsolved at 0.30. Level 4 needs no single layer: the example and the all-dims tiling
message each save two rounds, and v7's longer first prompt costs one. On level 4, switching the sampling
settings back did not move the trajectory (rounds 0–1 identical).

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
For levels 5–7, v8.4's first prompt (WARM) carries one more thing: the agent's own level-4 solve,
unchanged but for the entry name, with the checker's traffic verdict on it. The same scan finds no line from
any forbidden source in those prompts, except, on level 6, the entry signature the level itself prescribes
(`def nki_matmul_block_free_dimension_(lhsT, rhs):`, a name the organizers took from the tutorial).
([analysis/v83_l134_identity.md](analysis/v83_l134_identity.md))
Thinking stays off. The organizers measured it on this model: with thinking on, a round took 446 s and every
sample was cut off before any code ([projects/02-kernel-agent/README.md](projects/02-kernel-agent/README.md)).

## 4. Experiments, one change at a time

Each change was measured with `--repeat 5` against the current reference and kept or rolled back by rules
written down before the results came in ([PLAN.md](PLAN.md) §3).

| id | change | level | result (solved, scores) | decision |
|---|---|---|---|---|
| E-A | invented NKI names mapped to the real 0.6.0 calls | L1 | 0/5, all 0.30; invented names 80 → 20, the failures moved one layer deeper | kept as groundwork |
| E-v3 | feedback_v3 | L4 | 1/2 before the seat went to v7: solved in round 5, then a 0.75 | superseded by v7 |
| E-F | wrong argument list: failing line + real signature + one instruction | L1 | 0/5, all 0.30, one trajectory; wrong-signature errors 8 → 3 per run, the run then stalls on copy sizes | not carried into v7: v7's level-1 failures are different, and with a deterministic server any message change can move v7's solved trajectories |
| E-v7 | feedback_v7 as a whole | L1, L3, L4, L2 | L3 5/5 (round 0), L4 5/5 (round 2, one trajectory), L1 0/1 (best 0.50), L2 0/3; every solve re-audited and VERIFIED | adopted, then built on |
| E-div | v7, plus a one-line `(attempt k of n, run r)` tag on samples 2–4 so a deterministic server returns different samples | L3, L4 | L3 5/5 (3 distinct runs, as under v7); L4 5/5 (5 distinct runs; v7: 1), round 2 each | kept (in v8) |
| v8 | E-div + skeleton feedback + cut-off answers not graded + level-1 call fixes; liuyq's level-1 compiler gate in the base | L1–L4 | **L1: our first level-1 solve**, in its first repair round; VERIFIED by both verdicts, held-out 20/20, lowers for trn2, re-audit PASS (1 of 3 runs). L4: 0.62 twice (v7: 5/5). L3: solved, two rounds later than v7. L2, round 0 only: 0/20 | skeleton dropped (see below); the rest kept |
| L2 sampling | v8 with the first prompt and the sampling settings back to the organizers' | L2, round 0, 20 samples | 2/20, the same as the organizers' agent re-run today (2/20); with v7's sampling settings 0/20 | the level-2 regression was v7's sampling settings |
| v8.1 | v8 without the skeleton, plus samples 1 and 3 on the original sampling and 2 and 4 on v7's, plus one sentence restating the level-2 task when the model transposes the whole input | L2 | the first level-2 solve in a repair round today (round 1), through the existing numeric-mismatch message; stopped after one run for v8.2 | folded into v8.2 |
| v8.3 (5c3aba2) | v8.2 plus **error distillation on level 2**: when a level-2 kernel runs but its numbers are wrong, the checker runs it once more on `arange` input, reads off where each output element came from, and says what the kernel actually did ("your output at row position i*B+j holds x[i]: the source index uses only i") and what the task needs, without code. Only level 2's mismatch path changes: on levels 1, 3 and 4, v8.3 sends byte-identical requests to v8.2 (27 requests, 18 of them repair rounds, compared by two of us separately: [analysis/v83_l134_identity.md](analysis/v83_l134_identity.md)), so v8.2's runs on those levels are v8.3's runs | L2 | **7/9** so far, against v8.2's 5/9 on the same seats. In 3 of the 5 solves the best kernel's feedback carried the distilled explanation in the round before (e.g. 0.50 → 1.00); the one miss stalled on `WRONG SHAPE: returned ()`, which is not a numeric error, so the distillation never spoke | final if it holds at 17:45 [[TBD]] |
| v8.4 (15fb0d5) | v8.3 plus **WARM** on levels 5–7 only: round 0 starts from the agent's own verified level-4 kernel and the checker's traffic verdict on it ("CORRECT, BUT TOO MUCH HBM TRAFFIC FOR THIS LEVEL: moving 2.00x the byte floor, and level 5 requires 1.60x or better"), and the agent repairs from there. On levels 1–4 v8.4 sends byte-identical requests to v8.3 (63 requests, 51 of them repair rounds) | L5–L7 | [[TBD]] | [[TBD]] |
| **v8.2** (final candidate, 96a9fc9) | v8.1 plus **E-mix**: in every repair round, sample 1 repairs the best kernel as before and samples 2–4 start over from the first prompt, each tagged differently. Level 2 had only ever been solved in round 0, so its repair rounds now also buy fresh first attempts | L1–L4 | **L1 5/5** (rounds 2, 2, 0, 4, 2; five different kernels, all built in full for trn2 and matching in birsim), **L2 5/9**, **L3 5/6** (the sixth inside the `--all` run), **L4 5/5** (one trajectory); every solve re-audited, VERIFIED on the held-out set. Two level-1 solves are told in §8 | final candidate |

**The skeleton gamble, and why it lost.** v8 hollowed out the code in the repair messages (`t[<…>]`) so that the
model would have to work out slices and shapes itself. On level 4 it could not. With and without the
skeleton, rounds 0 and 1 are identical (0.30 broadcast error, then 0.62 partition over 128). They part at
the message sent after round 1, v5's "Tile all three dimensions at once": given the code, every run solved
in round 2; given `for m in nl.affine_range(<…>):` and `nl.ndarray(<…>, …)`, the model filled the holes with
out-of-range slices and stayed at 0.30, out of bounds, for six rounds running (best 0.62, both runs). How it
filled them: `affine_range` written like Python's `range(start, stop, step)` (`nl.affine_range(0, lhsT.shape[1], TM)`),
loop indices used as element offsets (`lhsT[k:k+TK, m:m+TM]`), and `rhs` indexed with its axes swapped
(`rhs[n:n+TN, k:k+TK]` for a (K, N) tensor), which raised `dimension 0: index range [0, 511]` from round 2 to
round 7. ([analysis/taxonomy_versions.md](analysis/taxonomy_versions.md), "How SKELETON broke level 4") So the
code in v7's messages is not decoration; on level 4 it is the part of the solution the model does not find
on its own. We report that rather than claim the model wrote it.

What did not work, kept here because each one cost us time:

- **Both message fixes for level 1 moved the failure, not the score.** Experiment A cut invented calls from
  80 to 20 and E-F cut wrong-signature errors from 8 to 3 per run; both stayed at 0.30, stuck on the next wall.
- **Compiling the model server with -O3** to make rounds faster: still compiling after 33 minutes. Dropped;
  it would also have invalidated the baseline.
- **A `seed` parameter**, to get independent samples from a deterministic server: HTTP 500, and it took the server
  down once.
- **Splitting an experiment's runs across seats** to get results sooner: with a deterministic server the runs
  are mostly copies, so it buys speed without information. E-div (§4 table) is the fix we tried instead.

## 5. Does the agent know when it failed?

Two verdicts, one yardstick. When a level ends, the agent states a confidence from what the loop saw and
nothing else (0 if unsolved; 0.9 if solved, ×0.7 if the loop tested one shape, ×0.5 for a test-shape size
written into the code, ×0.6 for a hard-coded output dtype; [EVAL.md](projects/02-kernel-agent/EVAL.md)).
v7 adds its own verdict after extra hostile cases and lowering for trn2. Both are then scored against our
held-out set, which neither has seen.

| runs | level | held-out | ours: confidence, Brier | v7's: confidence, Brier | confident (≥ 0.5) but wrong |
|---|---|---|---|---|---|
| v7, round 2 | 3 | 5/5 VERIFIED | 0.63, 0.137 | 0.74, 0.067 | 0 and 0 |
| v7, round 2 | 4 | 5/5 VERIFIED | 0.90, 0.010 | 0.88, 0.014 | 0 and 0 |
| [[TBD: final run, every level]] | | | | | |

Against our held-out set, neither verdict ever claimed a kernel that then failed; ours is too cautious on
level 3 (§9). Against a full trn2 build, v7's verdict did once (below). The first
level-1 solve (v8) was claimed VERIFIED by both verdicts, and it holds: held-out 20/20, 35 extra hostile
cases, lowered for trn2, re-audit PASS.
([analysis/round2_v7/](analysis/round2_v7/README.md))

**Beyond the simulator: the full build and the chip.** The simulator accepts kernels that trn2 cannot run.
We found four such forms: `nl.divide`, a (rows, 1) column passed to `tensor_tensor`, a compute
instruction on one partition starting at partition c (level 1's per-channel reduce, a level-2 per-row
copy), and tiles beyond on-chip limits (the allocation audit, §2). So solves were also built in full for
trn2 (neuronx-cc to a NEFF, then birsim, the compiler's instruction-level simulator) and run on a
NeuronCore of seat-115. Over three chip runs, 63 kernel-shapes with the organizers' references: every
kernel that builds ran correctly, every kernel that does not build failed on the chip, and the simulator
alone passed all of them. This caught our own false claims. v7's verdict lowers but does not build, and it
called a seat-115 level-2 solve VERIFIED that the full build rejects; all 7 level-1 simulator solves of the
4090 rehearsal fail the full build too. Hand-in candidates, each written by the model in an agent run and
not edited:

| level | generated on | simulator | full build + birsim | NeuronCore | held-out |
|---|---|---|---|---|---|
| 1 | 4090 stand-in, v7 + level-1 rule | 4/4 | yes | 4/4 | 20/20 |
| 1 | Trainium (seat-115), v7 + rule (rule not fired) | 4/4 | yes | not run | 20/20 |
| 2 | 4090 stand-in, v7 | 4/4 | yes | 4/4 | 16/16 |
| 3 | Trainium (seat-115), v7 | 1/1 | yes | 1/1 | 4/4 |
| 4 | Trainium (seat-115), v7 | 4/4 | yes | 4/4 | 16/16 |

Levels 9 and 11 (liuyq's operations) pass the first three as well. The level-1 rule turned held kernels
into ones that build: 4/10 runs vs 0/30 without it (4090 generation, every solve built on seat-115).
([analysis/seat115_chip_and_l1rule.md](analysis/seat115_chip_and_l1rule.md))

Baseline, scored after the fact with the same confidence function: the 3 solved runs said 0.90 and all
passed the held-out set; the 17 unsolved said 0. Brier 0.001, or 0.010 over the 3 real predictions; no
confident-but-wrong claim. The replication adds 2 more solves, both 0.90 and both passing the held-out set.
Five predictions say little, which is why the final run matters.
([analysis/calibration_baseline_seat116.md](analysis/calibration_baseline_seat116.md))

## 6. Where the tokens went

![tokens per attempt, by segment](analysis/round2_v7/token_budget.png)

Round 2, levels 3 and 4, counted by the server ([analysis/round2_v7/](analysis/round2_v7/README.md)). A
first prompt averages 1,148 tokens: 77% the API card and the worked example, 14% instructions, 10% the NumPy
reference. A repair prompt averages 744: 47% the checker's feedback, 44% the previous kernel, 9%
instructions. Neither comes near 8,192. The budget goes to the two things the model cannot work out by
itself, real API calls up front and the checker's diagnosis afterwards, and the repair prompt drops the
docs. [[TBD: final-run chart and numbers]]

## 7. Failure taxonomy

Under v7 (round 2), level 3's walls are gone: the seat-116 baseline's reshape (48), copy size (26), 1-D tile (22) and
out-of-bounds (16) failures do not occur, the 7 failures left are all a tile in the wrong memory, and every
run solves in round 0. Level 4 still meets the baseline's wall first (partition over 128, 20 attempts in round
0; the baseline never got past it), then a broadcast shape mismatch (20 in round 1), and solves in round 2.
[[TBD: final table from scripts/taxonomy.py]]

**Across versions** (full table, by version and level: [analysis/taxonomy_versions.md](analysis/taxonomy_versions.md);
the later columns are partly snapshots of runs in progress, so compare which modes appear and vanish, not
raw counts):

1. **Three walls fell.** Level 3's baseline walls, reshaping (96), 1-D tiles (55) and copy-size mismatches (45)
   in 10 runs with no solve, are gone under v7, which solves every run in round 0: the worked example, by the
   ablation. Level 4's partition-over-128 wall (110 in the baseline, never passed) still shows up in round 0,
   and the all-dims tiling code gets every run past it by round 2. Level 1's invented calls fall from 160 to
   16–18.
2. **Failures moved rather than vanished.** On level 2, v7 traded the baseline's index errors (out of bounds
   63, copy size 71) for two modes the baseline never showed: invented names (30: `dtype`, `transpose`,
   `reshape`) and **transposing the whole input** (30 of v7's 108 failed level-2 attempts): out-of-range
   indices on the second axis, or "Partition dim size must be preserved, got 32 -> 3", because the model
   transposed all of `x` instead of the small matrix inside each row. The checker named the symptom and never
   the misreading; v8.1 adds one sentence restating the task when these errors appear (no code). On level 1,
   cut-off answers (12) appeared under v7 and took 22 of the 29 minutes of the run they hit.
3. **One wall came back** when the skeleton blanked the tiling code: 24 out-of-bounds attempts on level 4
   (§4).
4. **What is left.** Level 4's round-0 broadcast mismatch is in every version (18, 20, 18, 7, 7): cleared by
   repair, never prevented. Level 1 stays the hardest: no solve before v8. [[TBD: final level-1 count]]

Until v8.1, level 2's repair rounds had never once succeeded, in the baseline or under v7: every level-2
solve came from a first attempt. Under v8.3, four of five level-2 solves came after round 0, three of them
right after the distilled explanation (§4).

**Why the final version still misses.** Level 2, 2 of 9 runs: in one, no kernel ever ran, the repairs cycled
through runtime errors and the fresh samples kept redrawing the round-0 mistake; in the other, the misleading
`returned ()` message above, over a kernel that was wrong anyway. Level 3, the one run inside `--all`: a
different first answer to the same prompt, then four rounds on the same copy-size error, and the run stopped
early under the repeated-failure rule; we cannot rule out that a later fresh sample would have solved it.
([analysis/l2_l3_misses.md](analysis/l2_l3_misses.md))

Baseline, 424 attempts in 25 named modes: index and size
arithmetic 52%, unfamiliar API 24%, tiling rules 18%, memory placement 5%. Four modes were never fixed by the
original feedback once they appeared (stuck rate 100%): out-of-bounds index, reshape instead of slicing,
invented function, partition dimension over 128.
([analysis/taxonomy_baseline_seat116.md](analysis/taxonomy_baseline_seat116.md))

## 8. A failure, and the recovery

Level 4 (tiled matmul), feedback v3 (v7's predecessor), run 1: the first level-4 solve of the day. Every
message below is in the log; the full transcript, with the code that changed each round and the prompts
rebuilt to the logged length, is [analysis/recovery_v3_L4_run1.md](analysis/recovery_v3_L4_run1.md)
(log: [analysis/logs/ev3_L4/](analysis/logs/README.md), lines 29–52).

| round | best of 4 | what the checker said | what went back to the model |
|---|---|---|---|
| 0 | 0.62 | `dma_copy dst partition dimension 256 exceeds maximum 128`: one SBUF tile for a whole operand | loop over the partition dimension in chunks of at most 128, allocate each tile inside the loop |
| 1 | 0.62 | `Matmul stationary free dimension 256 exceeds gemm_stationary_fmax=128` | the loop over output rows, as code in the model's own names |
| 2 | 0.62 | `Matmul moving free dimension 1024 exceeds max 512 for nc_version.gen3` | the loop over output columns, as code |
| 3 | 0.62 | **ILLEGAL ON HARDWARE**: SBUF tiles of (256, 256) and (256, 1024) | allocate each tile inside the loop, with the chunk's own shape |
| 4 | 0.62 | the same, a second time | the same message, plus a ledger: "These approaches have already failed, so do something different" and one line per failed attempt |
| 5 | **1.00** | correct on every shape | |

Three things happen here. The model pasted the code-form messages almost verbatim, so each round removed
exactly one wall (rounds 1 to 3). At round 3 the simulator ran the kernel on every shape; only our
allocation audit knew that no chip holds a (256, 1024) SBUF tile, and it said so. The same failure twice
brought in the ledger, and the next answer rewrote the allocations as (128, 128) and (128, 512) tiles:
solved, after 21 attempts. Then, before seeing anything new, the agent stated a confidence of 0.90; the
held-out set (4 new shapes × 4 kinds of values) passed 16/16. Verdict: VERIFIED, and the claim was right.
The solving kernel is correct, not fast: 36.6 Flops/Byte, memory-bound.

### A second one, shorter: our first level-1 solve

v8 on level 1. Full transcript, rebuilt prompts and re-checks:
[analysis/recovery_v8_L1.md](analysis/recovery_v8_L1.md).

| round | best of 4 | what the checker said | what went back to the model |
|---|---|---|---|
| 0 | 0.30 | `SBUF and PSUM tensors must have at least 2 dimensions`: `nl.sum` without `keepdims` makes a 1-D tile. Two other samples ran into the 2,500-token limit and were not graded | "Add keepdims=True to this call"; for the two cut-off samples, "Your previous answer was cut off at the token limit…" |
| 1 | **1.00** | correct on every shape | |

This is the translation the challenge asks for, in its smallest form: the simulator's sentence says what is
wrong, the agent's sentence says what to change, and all four samples changed exactly that line. The
solving kernel reduces all channels at once, so the compiler gate had nothing to say. Before the held-out
set ran, the agent stated 0.90; held-out 20/20, re-audit PASS, VERIFIED by both verdicts. It also lowers for
trn2 (a lowering, not a full build). Round 0 took 247 s, almost all of it the two cut-off answers; round 1
took 81 s.

### A third: level 1 through the compiler gate

v8.2 on level 1 ([analysis/recovery_v82_L1.md](analysis/recovery_v82_L1.md)). Every layer that v8.2 added has a
part in it.

| round | best of 4 | from which sample | what happened |
|---|---|---|---|
| 0 | 0.50 | sample 1, first prompt, original sampling | a fourth sample hit the token limit and was not graded |
| 1 | 0.95 | **sample 2, a fresh first attempt** (E-mix); the repair sample only reached 0.30 | correct on every shape in the simulator, but one channel at a time (`t[c, ...]`), a form the trn2 compiler rejects. The gate held it at 0.95 and sent back the loop rewritten to do all channels at once |
| 2 | **1.00** | sample 1, the repair | the held kernel with the gate's rewrite applied |

Without E-mix the near-solution would not have been drawn: the repair sample was stuck at 0.30. Without the
gate it would have scored 1.0 as it stood, in a form that failed every full trn2 build liuyq ran on it
(89417dd); we did not full-build this particular kernel. Before the
held-out set ran, the agent stated 0.90; held-out 20/20, re-audit PASS, VERIFIED by both verdicts, and the
kernel builds for trn2 and matches in birsim (worst error 1.9e-7 of the RMS). 12 requests, 13,595 prompt and 10,935 answer tokens,
counted by the server.

## 9. Limits

- **The simulation target matters in general.** With no target set and no `neuron-ls` on the machine,
  NKI 0.6.0 simulates trn3. A level-4 kernel with a 1024-wide moving tile passes 2/4 shapes under trn3 and
  0/4 under trn2 ("moving free dimension 1024 exceeds max 512 for nc_version.gen3"). We set trn2
  explicitly and re-graded earlier runs under it. ([analysis/sim_target_check.md](analysis/sim_target_check.md))
- **A deterministic server, within one server state.** In the standard seat configuration (`--max-num-seqs 4`,
  which every number above uses), the model server returns the same text for the same request at temperature
  0.7, so repeated runs of one configuration are mostly copies, and "5/5" can mean one trajectory five times. We report distinct
  trajectories next to every rate. It is not the same text in every server state: in the one-command `--all`
  run, level 3's first prompt, byte for byte the same length as in the per-level runs, drew four different
  kernels, none of which solved (probably a different batch or prefix-cache state;
  [analysis/l2_l3_misses.md](analysis/l2_l3_misses.md)). On seat-115 (`--max-num-seqs 8`) sampling was applied (six identical
  requests at temperature 1.0 gave six different answers), but samples still repeated far more than on a
  GPU: 50% of sample pairs identical in first rounds and 87% in repair rounds, against 1% and 54% on an
  RTX 4090. A request with a `seed` parameter returned HTTP 500 and took the server down once.
- **One misleading checker message.** A level-2 kernel with no `return` gets "WRONG SHAPE: returned (),
  reference is (32, 12). Check the output-size arithmetic, not the values." The sizes were fine; the return was
  missing. One v8.3 run spent four rounds on it and did not recover; we found it too late to change the
  checker and re-measure, so it stays as a known weakness.
- **Simulator numbers.** Every score comes from `nki.simulate` on a CPU. Beyond it: every v8.2 and v8.3 solve
  on levels 1–4 was built in full for trn2 and matched in birsim; one level-1 solve, and liuyq's hand-in
  candidates for levels 1–4, 9 and 11, also ran on a NeuronCore (§5, §10). Speed on the chip was not measured.
  [[TBD: the final run's kernels, if built]] Chip timing was not measured: a standalone call costs ~1.5 s
  of launch.
- **Five runs per cell, and fewer distinct trajectories.** A 3/5 against a 4/5 is noise.
- **Level 3 has one loop shape**, because the organizers' reference asserts it; its held-out cases change
  only the values.
- **The confidence rule is stated, not fitted.** Its weights were fixed at 13:40, before any held-out
  result, and never tuned. It is underconfident on level 3: it docks every level-3 kernel ×0.7 for having
  been tested on a single loop shape, which the level's own contract fixes, so it said 0.63 for five kernels
  that were all right.
- **Part of each solution comes from the checker.** Code-form messages are pasted by the model (§3, §8),
  and the level-1 example is a strong hint (§3). The compiler gate goes furthest: once a kernel passes every
  shape, a form trn2 rejects is rewritten as code from the model's own lines (for level 1, the
  per-channel loop as one instruction over all channels). Rules come from logged failures and were checked
  with the full trn2 build; no reference or answer kernel is used.
- **v7 is a bundle of layers**, and only part of it is explained: the ablation (§3) shows level 3 rests on the
  worked example alone and level 4 on no single layer; what each layer does on levels 1 and 2 we did not ablate.
- **Levels 9–14 are liuyq's own held-out operations**, not the official ladder, and are reported apart from it.
- **The SBUF limit is conservative**: 192 KiB per partition, below trn2's 224 KiB.

## 10. Who did what, and where each number comes from

| part | by | where |
|---|---|---|
| the agent loop, the original checker, the reference kernels and the API card for levels 1–8 | the organizers | `agent.py`, `nkibench.py`, `reference_level*.py` |
| feedback layers v2–v7, the compiler gate with its level-1 rule, the verdict, levels 9–14 | liuyq | `feedback_v2.py` … `feedback_v7.py`, `gate_nki.py`, `verdict_nki.py`, `ops07.py`, `ops08.py`, [V7.md](projects/02-kernel-agent/V7.md) |
| checker additions: allocation audit, a fresh file per candidate, the held-out set and its mutants, confidence and calibration, token accounting, taxonomy and reports | teoguo | `nkibench.py`, `agent.py`, `scripts/`, [EVAL.md](projects/02-kernel-agent/EVAL.md), [CHECKER.md](projects/02-kernel-agent/CHECKER.md) |
| experiments A, E-F, E-div, v8 and v8.1, the ablation, this write-up | teoguo | `feedback_v8.py`, [V8.md](projects/02-kernel-agent/V8.md), [PLAN.md](PLAN.md), [NOTES.md](NOTES.md) |
| full trn2 builds, runs on a NeuronCore, the level-1 rule's live test (4090 stand-in), hand-in candidates | liuyq | [analysis/seat115_chip_and_l1rule.md](analysis/seat115_chip_and_l1rule.md), `check/compile_solves7.py`, `check/device_check.py` |
| seats 115–119, runs, re-audits | the team | [analysis/logs/](analysis/logs/README.md) |

Most code, analysis and text on teoguo's side was produced with Claude Code sessions that the team directed
and checked; each commit names its author. Every number in this document is computed by a script in this
repository from a logged run.

**Where a number comes from.**

- *Simulator* (`nki.simulate`, trn2 target): every score, solve rate and held-out result.
- *trn2 compiler*: v7's verdict lowers each solve for trn2; the level-1 solve lowered. A lowering is not a
  full build: every solve of liuyq's seat-115 runs and of the 4090 level-1 test was also built in full
  (neuronx-cc + birsim), which rejects forms the lowering passes (§5). All five level-1 solves of v8.2 were
  built in full the same way (`check/compile_solves7.py`, seat 117) on two shapes, (4,8,8)/2 and (8,12,12)/3:
  5 of 5 match in birsim, worst error 1.9e-7 of the output's RMS. So were all of v8.2's and v8.3's solves on
  levels 2–4, and every one matches: level 2 on 4 shapes (worst error 0), level 3 on its one shape (1.2e-6),
  level 4 on two shapes (2.9e-6). None was rejected. The compiler's own estimate also shows what correctness
  hides: one family of level-2 solves writes HBM one element at a time and is predicted at 44–169 µs, against
  about 2.6 µs for the others.
- *On a NeuronCore*: liuyq ran 63 kernel-shapes on seat-115's chip in three runs, the hand-in candidates
  for levels 1-4, 9 and 11 among them, with the organizers' references as controls; the full build
  predicted every result (§5, [analysis/seat115_chip_and_l1rule.md](analysis/seat115_chip_and_l1rule.md)).
  One of v8.2's level-1 solves (53900f4b) ran on a NeuronCore of seat 116 on 4 shapes and matched the reference
  on all four (device error 2.4e-7 to 4.8e-7 of the RMS; the simulator's was 1.7e-7 to 2.4e-7).
- *Measured or inferred.* Rates, counts and timings are measured. Explanations (why the skeleton failed, why
  level 2 repairs never succeed) are our reading of the logs, and each one points to the log it rests on.

## 11. Files

| what | where |
|---|---|
| attempt logs, every attempt with its score | earlier runs: [analysis/logs/](analysis/logs/README.md) (baseline, replication, experiments; what each is, its code version, md5, and whether it is comparable); final run: [analysis/logs/final/](analysis/logs/final/) |
| results tables | [analysis/final/](analysis/final/) (summary, checks, taxonomy, token chart and the version comparison, from `scripts/final_auto.py`) |
| checker, eval set, tolerance | [CHECKER.md](projects/02-kernel-agent/CHECKER.md), [EVAL.md](projects/02-kernel-agent/EVAL.md), `nkibench.py` |
| agent | `agent.py`, `feedback_v2.py` … `feedback_v7.py`, [V7.md](projects/02-kernel-agent/V7.md); `feedback_v8.py` (v8.2: commit `96a9fc9` and its switches), [V8.md](projects/02-kernel-agent/V8.md) (run command and every switch) |
| the solving kernels | every one is in the attempt logs (the `code` field of each 1.0 attempt); liuyq's hand-in candidates, built and run on the chip: [analysis/seat115_chip_and_l1rule.md](analysis/seat115_chip_and_l1rule.md) |
| full builds and chip runs, with the exact kernels | [analysis/logs/chip_seat115/](analysis/logs/chip_seat115/), `seat115_v7/compile.txt`, `task15_4090_l1rule/*_solves.log` |
| how we ran the day | [PLAN.md](PLAN.md), [NOTES.md](NOTES.md) |
