# The agent was stuck on the checker's words, not on tiling

**Team:** John (measurement, write-up), Achyuthan (checker), Perumal (agent and prompt).
**Long version, kept as the working record.** The final, shorter note is `NOTE.md` (`report/05-one-page-final.md`);
where the two differ, the final note has the later numbers (level 3 pooled to 16 of 17, level 8, `names`, `example`).

## Claim

The organisers' kernel agent sat at fixed scores on three levels: 0.30 (level 1), 0.30 (level 3) and 0.62
(level 4). Their reading was that the model does not understand tiling. Ours: **on levels 3 and 4 the model
was stuck because the checker's feedback described the simulator's crash instead of the kernel's mistake, and
in two places the feedback told it to do the wrong thing.** Changing only what the checker says solved level 3
in 6 of 7 completed runs on two chips. On level 4 the model learned to tile the contraction, and the harness
scores that 0.75 instead of 0.62, but 22 of the 23 kernels behind that 0.75 keep a tile taller than the chip's
128 rows, which the harness does not check. Run with that rule enforced, level 4 scored 0.62 in 5 of 6 runs
and 0.50 in one: no better than the baseline. Level 1 did not
move, because it fails on the algorithm, which no checker message can reach.

## What we ran

- Model: Qwen/Qwen3-8B on each seat's own server, thinking off. Simulator: nki 0.6.0, CPU (no device timing).
- Frozen settings for every scored run: `--rounds 8 --samples 4 --context 8192 --repeat 5`. Runs differ only
  in `--features` and level selection; levels are solved independently, so single-level runs compare with
  `--all` runs. Every run, its seat, commit and flags: `results/RUNS.tsv`.
- **Baseline: 15 runs on 3 seats.** Seat 152 ran the organisers' untouched code (5 runs); seats 154 and 158
  ran ours with every feature off (10 runs). They agree: levels 1, 3 and 4 never varied.
- Checker: about 20 runs over 10 seats, one feature set per run, so each change can be credited separately.

## Results (solved / runs; every run's best score in brackets where it varies)

The reported checker is the flag set `internal,origin`. Its two level 3 runs (`39b8e5a` on seat 153,
`ec306f1` on seat 157) are pooled because the code under `projects/` is byte-identical between those commits
(`git diff 39b8e5a ec306f1 -- projects` is empty). Replaying every logged kernel against the final code
(`results/replay-internal-origin-39b8e5a-vs-final.txt`) gives identical `internal,origin` feedback on levels 1
to 3, and different feedback on 7 level 4 kernels, from a wording change at 14:24 for too-tall tiles; no pooled
run crosses that change (the level 4 checker row is one run, at `39b8e5a`). The replays ran against the laptop
stand-in simulator, so they show the code takes the same path through the same errors, not that those errors
are the real simulator's.

| Level | Baseline (15 runs) | Checker `internal,origin` | + `ahead`, first version (`16dcf94`) | + `ahead`, fixed (`11fbb86`) | Prompt C, short (level 1) |
|---|---|---|---|---|---|
| 1 average pooling | 0/15, always 0.30 | 0/5, always 0.30 | not run | not run | 0/5, always 0.30, but **algorithm fixed**: `nc_matmul` 0/104 attempts (baseline 160/160), `nl.sum` 103/104 (0) |
| 2 transpose | 6/15 (1/5 to 3/5 per seat) | 2/5 | 3/5 (noise range) | 3/9 completed (noise range) | n/a |
| 3 matmul, one tile | 0/15, always 0.30 | **6/7 completed runs** on two chips [1, 1, 0.3, 1, 1 / 1, 1] | **10/10 on two chips; 8 in 4 rounds, 2 in 7** | 8/10 as scored, **6/10 chip-valid** (2 return the result from PSUM); run 1 failed on both chips | n/a |
| 4 matmul, tiled | 0/15, always 0.62 | 0/5, **always 0.75** as scored; **with the 128-row rule enforced (`strict`): 0.62 in 5 of 6 runs, 0.50 in 1** | not run alone; with `pieces`: **0.62** (1 run, regressed) | 0.75 in 4 of 5 completed runs, 0.62 in 1 (as scored) | n/a |

Other flag sets, all reported: `pieces` on level 4 at 8 rounds scored 0.75, 0.62 and 0.50, so it is not
adopted; `pieces,ahead` solves level 3 3/3. `names` (`internal,origin,ahead,names`, `2d87250`, stopped at 16:43): level 2 3/5;
level 3 3 solved in 8 runs, 2 chip-valid; level 4 0.75 ×2 as scored, 0.62 under `strict`. Not adopted. `example` (`32f6055`, a worked tiling example
in the repair prompt after a too-tall tile) is a prompt change, reported separately. Started 16:21 on `32f6055`,
all at **16 rounds** (outside the frozen 8, so comparable only with each other and with `base-r16-l4`):
`ex-all-l4` (154, 156: `internal,origin,ahead,names,strict,example`), `ex-min-l4` (159:
`internal,origin,strict,example`), and `hint-checker-l1` (151: the checker plus `ahead,names`, on code that
contains the detailed level 1 hint, so it is "checker + prompt C detailed", not a checker result).
`checker-l3c` (150, 158, `dae55d8`, from 16:50): 10 more level 3 runs of the reportable checker, **10 of 10,
every solve in round 6**, pooled with the 6/7 to **16 of 17** (identical level 3 feedback on all 93 logged
kernels; the only code change is the bytecode fix). Level 8, attention, untuned, 8 rounds: 0/5 (0.30) without
flags (157 `base-l8`) and 0/5 (0.30) with the reportable checker (153 `checker-l8`): no transfer. `example`
group: 2 of 14 runs reach a chip-valid 0.75 on level 4. `hint-checker-l1`: 0.30 in all 5 runs. `layout` (150, 158, started 16:43) was stopped
before finishing and is not scored. Scoring gaps found, each with a repro kernel run through
`nkibench.py --check` on the real simulator: `findings/`, output in `results/seat-150/findings.out`.
The fixed `ahead` replaces the checker only if level 3 holds, level 4 holds 0.75 and level 2 is no worse.
**Verdict: not adopted.** Level 3 held (8/10 against 6/7) and level 2 is no worse, but level 4 dropped to 0.62
in one run of five. The reported checker stays `internal,origin`. The first `ahead` (`16dcf94`) is the best
level 3 result we have, 10/10 and faster, and is reported as that; it was never run alone on level 4, so its
level 4 effect is known only in combination with `pieces` (0.62).

An eighth level 3 run (seat 157, run 3) stopped after round 3 with no error and no summary line; every solving
run was also still at 0.30 at round 3, so it says nothing either way. Counted as unsolved, the figure is 6 of 8.
**Every one of the 10 level 3 solves without `ahead` took exactly 6 rounds**, which makes rounds-to-solve a
low-noise measure for comparing checker variants. **With `ahead` added (`16dcf94`), 10 of 10 level 3 runs
solved, 8 of them in 4 rounds and 2 in 7** (seats 153 and 157). On level 4, `pieces,ahead` fell back to
0.62; `11fbb86` then narrowed `ahead` to count only later writes. That change alters `ahead`'s feedback on 77
of 303 logged kernels, so runs on `11fbb86` are a new condition, not more of the same. The plain
`internal,origin` feedback is identical between `16dcf94` and `11fbb86` on all 303 kernels (stand-in
simulator), so the checker column still pools. The narrowed `pieces` alone on level 4 neither helps nor regresses: it stays at 0.75, unlike
`pieces,shapes` (`d761cd0`), which fell back to 0.62. The solves came by **two different paths** (seat 153: runs 1, 2, 4
share one; run 5 starts from a different kernel). Level 2 moves between 1/5 and 3/5 in
every condition, baselines included: that is noise, not an effect.

**Which ingredient did it (level 3, 5 runs each unless noted):**

| Features | What the model is told, in addition | Level 3 | Level 4 |
|---|---|---|---|
| none (baseline) | the simulator's error plus advice keyed on its text | 0/15 | 0.62 |
| `locate` | the line of the kernel that failed | 0/4 | |
| `state` | plus the shape and buffer of every tile on that line, and the rule they break | 0/5 | **0.75** |
| `facts` | rule *replaces* the advice for every error | 0/4 | |
| `facts, origin` | plus the line that gave the wrong tile its shape | 0/5 | 0.62 (4 runs) |
| `internal, origin` | rule replaces the advice only for errors from *inside* the simulator | **4/5** | **0.75** |

Naming the line is not enough. What solved level 3 was tracing the wrong tile back to where it got its shape,
**and** dropping the advice only where it was keyed on a numpy error from inside the simulator. Dropping the
advice everywhere (`facts`) lost level 4, because for the simulator's own checks the advice carries the how-to.

## Why it was stuck: failure taxonomy of the baseline (416 attempts, every sample)

- **Level 3, 84 failed attempts, 0 contain a loop.** The common bug: the result tile given one dimension
  (`lhsT.shape[1:]`, in 62 of 84). The existing hint answered "give a vector the shape (1, N)"; the model
  followed it, and on another seat that produced 16 `cannot reshape` failures in kernels with **no reshape in
  them** (0 of 16), which were then told "Do not reshape". A second hint's example tile, (128, 512), was the
  exact wrong tile the model had allocated.
- **Level 4, 80 failed attempts: one kernel in 69 of them, never tiling.** It passes only the shape that fits a
  single tile, and was resent unchanged in 60 of 60 repair rounds. With `state` it starts tiling K, gets past
  the contraction limit (next attempt scored higher 12 of 12 times), and stalls on the next rule: M per tile
  at most 128.
- **Level 1, 160 failed attempts: the wrong algorithm in all 160.** The mean is computed as a matrix multiply
  of each window with itself, times 0.5, copied from the prompt's API card. It never reaches the numerical
  check, so the checker never sees the algorithm. All 10 checker runs on level 1 open with the same five API
  errors as the baseline.

**Under the checker the failures moved where its information points.** Level 3: the baseline never got past
loading the inputs (0 of 84 failing attempts); with `internal,origin` 64% of failing attempts got to the matmul
or later, 70% with `ahead`. Level 4: "never tiles" fell from 70 of 80 to 15 of 120 failing attempts, and 101 now
fail at the matmul on the next rule (M per tile <= 128). Level 1: unchanged; every attempt in every condition
still scales by 0.5 and none uses `nl.sum`.

Full taxonomy with examples: `report/01-baseline-and-taxonomy.md` (section 2b for the checker runs).

## Caveats we measured

- **The model is near-deterministic** on this server: on level 1 all five baseline runs were the same kernels
  character for character, and the four samples per round differed in only 12 of 104 rounds. So any change to
  the feedback changes the trajectory. We credit a change only where the failure type also moved toward what
  the new message said (level 4: from "never tiles" to "tiles K, stalls on M"; level 3: each round's message
  names a tile and the model fixes that tile next).
- **Adding flags can regress.** `internal,origin` plus `pieces,shapes` kept level 3 at 4/5 but returned level 4
  to 0.62: at the contraction step it named three limits at once and the model stopped moving. Adoption rule:
  a flag is adopted only if it is better on the level it targets and no worse on any level it can change (found
  by replay), and on level 3, where the checker already varies, "better" needs more than one run.
- **Every flag set we tried is reported**, including the ones that lost, so the winner is not cherry-picked.
- **Level 4's 0.75 passes through a gap in the rules.** 22 of the 23 distinct kernels that scored 0.75 allocate
  an SBUF or PSUM tile of 256 or 512 rows and hand each instruction only 128-row slices of it. The simulator
  checks the slice, not the tile, and the harness's rule scan only catches a literal number, so `(K, N)` gets
  through; Achyuthan confirmed on the real simulator that such a kernel runs. Replaying every log with the
  rule enforced (`--features strict`, `c349ec2`) turns 16 kernels from 0.75 into 0.62 and changes nothing on
  levels 1 to 3. Level 3 is clean: all 9 distinct solved kernels keep every tile within 128 rows. Measured
  with the rule enforced (`internal,origin,strict`, `c349ec2`): seat 154, 8 rounds, 0.50, 0.62, 0.62; seat 156,
  16 rounds, 0.62 three times, each stuck at round 4 on the same code. So level 4 is 0.75 as the harness
  scores it and 0.62 under its own rule; what moved is the failure (the model now tiles K), not a real score. `strict` is off in every reported run and leaves the `internal,origin` feedback
  unchanged (replay, 303 kernels, 0 differences).
- **Simulator only.** Every number is CPU simulation; no kernel ran on the device and no latency was measured.
  The simulator also accepts a kernel that returns its result from PSUM instead of HBM, which the chip would
  not. All 6 reportable-checker solves and all 10 first-`ahead` solves return from HBM; 2 of the 8 fixed-`ahead`
  solves (seat 150, runs 2 and 4) do not, so that row is 6/10 chip-valid. We checked every solve for this.
- **A grading bug in the agent, found and fixed** (`dae55d8`). The agent now writes each candidate to its own
  file; the original reused one path, and importlib's bytecode cache (which checks only size and modification
  time to the second) could grade one sample as another of the same length. Regrading the 3 suspicious records
  of 4,344 changed none, so no logged score is affected. The harness, `nkibench.py`, is byte-identical to the
  organisers' copy (`d11ccdf`).
- **Prompt change C** (level 1) is reported separately, no checker flags. The short hint measured (seat 155,
  `e29cc01` on `0278dc1`), verbatim: "Average pooling is a per-window sum followed by a scale; it needs no
  matrix multiplication. Use a strided view of pooling windows and nl.sum for the reduction." It changed the
  algorithm, not the score: `nc_matmul` in 0 of 104 attempts (baseline 160/160), `nl.sum` in 103, a `1/p^2`
  scale in 79 (baseline 0, always 0.5). It now fails on tile shapes (a fixed `(128, 128)` tile, 80 attempts:
  type 1a again) and on leaving out the `pool_size` argument (24). Round-0 prompt 2,484 characters against
  2,323 for the baseline on the same commit, so the hint was in every run. The log file also holds five
  earlier single-run tries (88 attempts, some with a different hint); only the final 104-attempt repeat is
  counted. The detailed version in `dev` (`ce61bac`) names the `.ap()` view, the reduction axes and the
  `1/(p*p)` scale, i.e. the reference algorithm; a solve with it would show the model can follow a recipe,
  not that it derived one. Result (seat 155, `l1-detailed-b`): followed in 100 of 100 attempts, still 0/5,
  failing on a fixed `(C, 128, 128)` tile, the same tile-sizing error as level 3.

## Conclusion

See `NOTE.md` for the final wording. In short: one wall fell cleanly and one moved, by changing only what
the checker tells the model, never what the answer is. Level 3 went from 0 of 15 to 6 of 7 completed runs, with
every solved kernel inside the chip's rules. Level 4's kernels went from never tiling to tiling the
contraction, but their 0.75 relies on a tile the harness fails to reject, so the honest level 4 score is
still 0.62. Level 1 did not move under any checker change, and the taxonomy says why: the model's algorithm
is wrong before any rule is broken; a one-sentence prompt hint fixed the algorithm but not the score. The largest single lesson is that the
existing feedback was not only vague but in two places misleading, and removing misleading advice mattered as
much as adding precise facts.
