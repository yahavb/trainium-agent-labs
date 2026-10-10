# H1: gate 1 asked one loop at a time. Staged 0/5; same wording unstaged 5/5, and end to end it SOLVES level 4

- **Who / seat:** Rishabh (+ Claude), seat 49, with its own vLLM server (Qwen3-8B, context 8192)
- **Commit measured:** uncommitted working tree on top of 8f1ca41. md5s: `agent.py` bf14b521,
  `plan_check.py` 181b7a09, `loop_tool.py` f4e419c2, `nkibench.py` 68e795d5. `plan_stage.py`
  4eb43e45 for the main staged run; f2e1d9d2 for all other runs. f2e1d9d2 adds `--unstaged` and
  `--acc-feedback`, and its default behaviour is identical.
- **Hypothesis:** "Qwen can plan the level-4 tiling when asked one loop at a time." Compared with gate 1
  diagnostic, 0/5 (results/2026-10-10-rishabh-plan-diagnostic.md).
- **Change:** new `projects/02-kernel-agent/plan_stage.py`. Stage m asks for the m loop's count plus
  out axis 0 and lhsT axis 1. Stage n asks for its count plus out axis 1 and rhs axis 1. Stage k asks
  for its count plus lhsT axis 0, rhs axis 0 and `accumulate_over`. Each stage is checked on its own,
  on Hypothesis contract shapes (exact coverage, tile limits, integer count, the two slices agree,
  names allowed), and is frozen and shown to the next stage once it passes. The frozen pieces are
  assembled and graded by the full `plan_check.search`. `--unstaged` uses the same header, wording
  and checks, but asks for all three loops in one JSON (ablation).
- **Settings (as for the 0/5):** `--samples 1`, 8 rounds per stage (8 total when unstaged),
  `--repeat 5`, context 8192, 800-token answer budget, temperature 0.6 (agent.ask).
- **Logs:** logs/attempts-rishabh-plan-stage.jsonl (main), `-v2`, `-unstaged`, `-loose`,
  `-unstaged-loose`, `-e2e`. Qwen's plan: logs/rishabh-plan-stage-qwen-plan.json. Kernel:
  logs/rishabh-plan-stage-e2e-kernel.py.

## Scores (plans passing the full plan_check.search, out of 5)

```
                          step-list wording                 loose wording
staged (H1)               0/5  m r0, n r0, stuck at k        0/5  stuck at m (slices ignore m)
staged, v2 acc feedback   0/5  m r0, n r0, stuck at k        -
unstaged (same checks)    5/5  all pass in round 1 (2nd)     0/5  stuck on m slices, all 8 rounds
gate 1 whole plan (ref)   0/5  (plan_prompt, diagnostic)
```

All 5 runs in each cell are byte-identical trajectories, so read 5/5 as "reproducible", not "robust".
Wall time: staged 489 s, unstaged 355 s for 5 runs, with three jobs sharing the server.

## Where it gets stuck, and on what

- **Staged, steps (H1 as specified): 0/5, all on `accumulate_over`.** Stage m and stage n pass on
  round 0 in every run: `"M // 128"`, `["m * 128", "(m + 1) * 128"]`, and the same for n with 512.
  Stage k has the count and both slices right in all 8 rounds of every run, but `accumulate_over`
  alternates between `["m", "n"]` and `["n"]`. The header states "the k steps' products are summed into
  the same out tile", yet with the frozen m and n loops shown above it, Qwen names the outer loops.
  - v1 feedback reports only the first problem ("lists ['n']. Each step of the n loop is a different out
    tile…"). The v2 feedback (`--acc-feedback all`) also says "does not list k…". With it the model
    moves to `["k", "m", "n"]` and back to `["m", "n"]`, still 0/5. Fixing the feedback doesn't unstick it.
- **Unstaged, steps: 5/5.** Round 0 has all three counts and all six slices right, and
  `accumulate_over: ["m", "n"]`. The feedback is "The k loop: … accumulate_over lists ['m', 'n']. Each
  step of the m loop is a different out tile…". In round 1 it answers `["k"]`, and the plan passes.
- **Loose wording, either arm: 0/5.** The counts are right (`M // 128`), but the slices never use the loop
  variable: `out ["0", "M"]`, `lhsT ["0", "K"]` (the wrong axis), then `["0", "128"]` and back. This is
  the gate-1 confusion again. Step 3 of the wording ("m is the tile NUMBER … the slice for tile m starts
  at m times the tile size") is what fixes the slices.

## End to end (seat 49, 1 run, clearly a single run)

Qwen's passing plan, the same in all 5 unstaged runs:
`{"loops": {"m": "M // 128", "n": "N // 512", "k": "K // 128"}, "accumulate_over": ["k"], "reads":
{"lhsT": [["k * 128", "(k + 1) * 128"], ["m * 128", "(m + 1) * 128"]], "rhs": [["k * 128", "(k + 1) * 128"],
["n * 512", "(n + 1) * 512"]]}, "writes": {"out": [["m * 128", "(m + 1) * 128"], ["n * 512", "(n + 1) * 512"]]}}`.
It is not byte-equal to `plan_check.REFERENCE` (it has spaces), but it is identical after normalising
whitespace. So Qwen derived the reference plan, without seeing it.

`python loop_tool.py --given-plan qwen-plan-unstaged.json --staged --wording steps --structure-check
--rounds 8 --samples 1 --context 8192 --repeat 1`: (A) passes in round 0, (B) in round 1, **reward 1.00,
SOLVED**. Stock checker on the kernel (`nkibench.py --level 4 --check`): **rules clean, numerics 4/4
shapes**. `stress.py --level 4 --check`: **PASSED, 50 generated inputs** (hostile values, within the
contract). HBM traffic is 1.6-2.0x the byte floor on the multi-tile shapes (the same as the earlier
staged kernel), and (B) reuses `sbuf_rhs` as the PSUM-to-SBUF staging tile.

The pipeline: Qwen writes the plan (gate 1, unstaged step-list wording, 2 rounds) -> the loop tool
expands the plan into loops and slices -> Qwen writes the tile body in two staged holes.

## Read this before quoting it

1. **The step-list wording is close to the method's formula.** It says: the tile size is the largest
   the limits allow (and the limits are listed), the count is how many tiles fit, and tile m starts at
   m times the tile size. It never writes `M // 128` or `m*128`, and the selftest scans every prompt
   for those strings and for "Fix:". Still, what's left for the model is arithmetic plus axis
   bookkeeping. The prompt also fixes the loop names to m, n, k and assigns each loop its axes, which
   gate 1's `plan_prompt` did not. So "unstaged 5/5 vs gate 1 0/5" changes wording, format and loop
   naming together. The loose arm (same format and names, 0/5) shows that the wording is necessary.
2. **Staging hurt here.** It is the opposite of the kernel-body result. The only failure point is
   `accumulate_over`, and it appears only when the m and n loops are shown as separate frozen
   pieces. A likely fix is to ask for `accumulate_over` outside stage k, or to drop it (plan_check
   defaults it to `["k"]`). That is untested.
3. One feedback round was needed in the unstaged arm, and that feedback is diagnostic: it names the
   loop that is wrong and why, not the answer. The "does not list k" line never fired in that arm.
4. The end-to-end run is one run. The earlier 5 staged loop-tool runs on the byte-equivalent reference
   plan were byte-identical, so it would likely repeat, but that is not measured.

## Exact wording (steps, stage m; n and k are the same with their own axes)

> Now plan the m loop. It tiles the M axis, which is out's axis 0 and lhsT's axis 1. Steps:
> 1. The tile size along M is the largest size both limits allow: out's axis 0 is a partition axis, and lhsT's axis 1 is lhsT's second axis.
> 2. The count is how many tiles of that size fit in M.
> 3. m is the tile NUMBER: it runs 0, 1, ..., count-1. It is not a size. The slice for tile m starts at m times the tile size and stops (exclusive) one tile size later.
> 4. out's axis 0 and lhsT's axis 1 are the same M rows of this tile, so give both the same slice.

Stage k adds: "5. accumulate_over lists the loops whose steps are summed into the same out tile." The
shared header: "The kernel has three nested loops: m steps over tiles of out's rows (the M axis), n
steps over tiles of out's columns (the N axis), and k steps over tiles of the K axis; the k steps'
products are summed into the same out tile. We plan one loop at a time." It is followed by the
hardware limits. Loose wording, stage m: "Now plan the m loop: how many times it runs, and the slice
of out's axis 0 and of lhsT's axis 1 that tile m covers." The full text is in `plan_stage.TASK` and
`HEADER`. The unstaged prompt joins the three step lists ("Plan the m loop…") under "Plan all three
loops in one answer." Repair prompts repeat the stage prompt, then add the last answer and the
checker's message.

## Verdict

H1 as stated (staged plus freeze) is **refuted: 0/5**, stuck at `accumulate_over` in stage k after m and
n pass on the first try. The thing that works is the **step-list wording with per-loop structure**:
asked for all three loops at once in that format, Qwen plans the tiling **5/5** in 2 rounds, against 0/5
for gate 1. Fed through the loop tool and the staged body, the kernel passes the stock checker and the
stress test. Level 4 is solved end to end, once, with Qwen choosing the plan and writing the code. The
caveat is point 1: the wording leaves the model mostly arithmetic.

## End-to-end rate: solved 5/5 (added 21:02 UTC)

Each run starts from scratch and makes a fresh plan (the saved plan is not reused). The steps:
1. `plan_stage.run_whole` (unstaged, step wording, 8 rounds) produces Qwen's plan.
2. `loop_tool.py --given-plan <that plan> --staged --wording steps --structure-check --rounds 8 --samples 1 --context 8192` writes the body.
3. The stock `nkibench.py --level 4 --check` grades the kernel.

Run 1 is the single run reported above. Runs 2-5 used the driver `e2e_rate.py` (in the pod at
/workspace/projects/02-kernel-agent/e2e_rate.py, md5 035ae452, not in the repo).

```
run  plan rounds  plan == REFERENCE*  body rounds (A, B)  loop_tool reward  nkibench (stock)
1    2            yes                 2 (1, 1)            1.00              rules clean, 4/4 shapes   (+ stress.py 50/50)
2    2            yes                 2 (1, 1)            1.00              rules clean, 4/4 shapes
3    2            yes                 2 (1, 1)            1.00              rules clean, 4/4 shapes
4    2            yes                 2 (1, 1)            1.00              rules clean, 4/4 shapes
5    2            yes                 2 (1, 1)            1.00              rules clean, 4/4 shapes
e2e solved 5/5
```

\* Equal after normalising whitespace (`"m * 128"` vs `"m*128"`). All 5 plan files have the same md5
(16695baf), and all 5 kernels have the same md5 (a8afecae). So this is one deterministic trajectory,
reproduced 5 times. That counts as reproducible, not robust: there is no sampling diversity at
`--samples 1` with these short prompts. Logs: logs/attempts-rishabh-plan-stage-e2e-rate-plans.jsonl,
logs/attempts-rishabh-plan-stage-e2e-rate-bodies.jsonl. The interpretation caveats above still apply,
especially point 1 (how much the step wording gives away) and the fact that the tool, not the model,
writes the loops.
