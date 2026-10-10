# Plan given + loop tool, staged: (A) alone, frozen, then (B) alone. Level 4 SOLVED 5/5

- **Who / seat:** Rishabh (+ Claude), seat 49, with its own vLLM server and no other jobs
- **Commit measured:** uncommitted working tree on top of 8f1ca41. md5s: `agent.py` 11917d5b (messages
  v2.1), `nkibench.py` 68e795d5, `plan_check.py` 181b7a09. `loop_tool.py` bb5606be (first run) and
  21c5085c (rerun with a harness indentation fix).
- **Hypothesis:** Qwen applies only the latest named fix and drops earlier ones
  (results/2026-10-10-rishabh-loop-tool-v21-struct.md). Checking hole (A) on its own, freezing it once it
  passes, and only then asking for (B) stops that.
- **Change:** `loop_tool.py --staged`. Stage A asks for (A) only; for the check, the harness fills
  (B) with the reference stub, which the model never sees. Stage B shows the model its frozen (A)
  and asks for (B) only. The kernel that counts is the model's (A) + the model's (B). The total
  budget is still 8 rounds.
- **Command:** `python loop_tool.py --given-plan reference --staged --structure-check --rounds 8 --samples 1 --context 8192 --repeat 5`
- **Logs:** logs/attempts-rishabh-loop-tool-staged.jsonl (first run), logs/attempts-rishabh-loop-tool-staged-fixed.jsonl
  (rerun), solved kernel: logs/rishabh-loop-tool-staged-solved-kernel.py

## Scores

```
not staged (v2.1 + structure check):  solved 0/5   all = [0.50, 0.30, 0.30, 0.50, 0.30]
staged, first run (bb5606be):         solved 5/5   all = [1.00, 1.00, 1.00, 1.00, 1.00]   4 rounds per run
staged, indent fix (21c5085c):        solved 5/5   all = [1.00, 1.00, 1.00, 1.00, 1.00]   2 rounds per run, ~13 s
```

The solved kernel was re-graded with the stock checker (`nkibench.py --level 4 --check`): rules
clean, 4/4 shapes. Both holes are the model's own code, and neither matches the reference body.

## Read this before quoting it

1. **Plan given, loops written by the tool.** This is not "Qwen solves level 4". It is "Qwen writes a
   correct tile body when the tiling is done for it and the body is asked for one hole at a time".
2. **Effectively one trajectory.** All 5 runs are byte-identical (greedy sampling, `--samples 1`).
   5/5 means it's reproducible, not that it's robust.
3. **The freeze was never exercised.** With the indent fix, each stage passes on its FIRST attempt,
   so no fix was ever at risk of being dropped. What made the difference is how the task is asked:
   one hole per prompt, and a stage-A task line that spells out the steps ("allocate SBUF tiles…,
   dma_copy each slice into its tile, then accumulate… with nisa.nc_matmul; each allocation directly
   above its load"). The non-staged prompt asked for both holes, more loosely worded. The two are
   confounded. Next ablation: non-staged with the stage-A wording, or staged with the old wording.
4. **The first run lost round 0 of each stage to a harness bug** (10/20 attempts "unexpected indent":
   the model sent its first line flush left and the rest indented, and `_clean` didn't normalise it).
   Earlier loop-tool runs had 0 such failures. Fixed in 21c5085c. In that first run, the model's
   round-0 (A) was already correct.
5. **(B) reuses `sbuf_rhs`**, a tile allocated inside the k loop, after the loop, as the PSUM→SBUF
   staging buffer. It passes `nki.simulate`. Whether that's valid on the device is unverified.
   HBM traffic is 1.6-2.0x the byte floor on the multi-tile shapes, which is level 5-7 territory.

## Verdict

Kept as the best loop-tool condition. Asked for one hole at a time, Qwen writes a correct body on the
first try. The earlier 0/5 conditions failed on how the task was asked, not on what the model knows.
Which part does the work (staging, the wording, or both) needs the ablation in point 3.

## Ablation: staging vs wording (loop_tool f4e419c2, same plan, v2.1 messages, structure check, --samples 1)

```
                 loose wording               step-list wording
not staged       0/5  [0.50,0.30,0.30,0.50,0.30]   2/5  [1.00,0.30,0.30,1.00,0.30]
staged           0/5  [0.50 x5]                    5/5  [1.00 x5]
```

Logs: logs/attempts-rishabh-ablation-staged-loose.jsonl, logs/attempts-rishabh-ablation-unstaged-steps.jsonl.
The "not staged, loose" cell is the earlier v2.1 + structure-check run (loop_tool fd282b9a,
logs/attempts-rishabh-loop-tool-given-v21-struct.jsonl). The later indentation fix doesn't affect it
(0 parse failures in that log), so it wasn't rerun. The "staged, steps" cell is the 21c5085c rerun.

- **The wording is necessary.** With the loose wording, neither arm solves.
- **Staging adds on top of it:** 2/5 unstaged becomes 5/5 staged with the same wording.
- **The freeze does work when exercised.** In staged + loose, (A) failed twice, passed in round 2, and
  stayed frozen in all 5 runs. The runs then fail in (B). Under the loose B line ("move acc out to
  out_slice") the model writes two *different* anonymous tiles,
  `tensor_copy(dst=nl.ndarray(...), src=acc)` and then `dma_copy(dst=out_slice, src=nl.ndarray(...))`,
  so it copies an uninitialised tile and gets NaN, four times, then stops. The NaN message doesn't name
  this ("allocate ONE tile, name it, and use it in both calls"). The step-list B line ("acc is in
  PSUM; out_slice is in HBM") avoids it.
- Unstaged + steps is not byte-identical across runs (2 solve, 3 don't), so sampling at
  `--samples 1` is not fully greedy for longer prompts.

Updated reading: the level-4 pass needs both (1) a task line that spells out the steps and where
the data lives, and (2) asking for one hole at a time with the passed part frozen. Either alone gives
0/5 or 2/5.
