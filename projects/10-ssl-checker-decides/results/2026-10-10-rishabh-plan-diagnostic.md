# Can Qwen plan the tiling? Gate 1 with diagnostic feedback, level 4

- **Who / seat:** Rishabh (+ Claude), seat 49, with its own vLLM server and no other jobs
- **Commit measured:** uncommitted working tree on top of 8f1ca41. md5s: `agent.py` 06805f2d,
  `plan_check.py` 181b7a09.
- **Hypothesis:** given honest feedback that names the error but not the fix, Qwen can reach a correct
  JSON tiling plan for the level-4 matmul within 8 rounds.
- **Change:** none. Uses `agent.py --plan-first --plan-feedback diagnostic`, which strips each
  plan_check message from " Fix:" on.
- **Command:** `python agent.py --level 4 --plan-first --plan-feedback diagnostic --plan-rounds 8 --rounds 8 --samples 1 --context 8192 --repeat 5`
- **Log:** logs/attempts-rishabh-plan-diagnostic.jsonl
- **Deviation:** `--samples 1` (sampling is effectively greedy here, see the loop-tool result).

## Scores

```
level 4: solved 0/5   plan passed 0/5   all = [0.00, 0.00, 0.00, 0.00, 0.00]
```

Every run used all 8 plan rounds, each run producing only 2 distinct plans. Wall time: 5 min 21 s.

## What it wrote (the same in all 5 runs)

- r0: loops `{"m": "M", "k": "K", "n": "N"}` (once per element) and whole-tensor slices.
  Feedback: "loop `m` runs `M` times, once per element of M; a loop should run once per TILE."
- r1: it renames the loops to `tile_m`, `tile_k`, `tile_n`, still running `M`, `K`, `N` times, and uses
  the loop variable as the slice stop: out = `[["0", "tile_m"], ...]`. So it reads "tile" as a size
  and confuses loop index with tile size. Run 5 did the same with `t`.
- r2-r7: the identical plan, every round. Feedback: "out axis 0: the slice from 0 to 0 is empty, at
  ... tile_m=0". It is accurate but names no change, and nothing moves for 6 rounds.

## Verdict

Qwen3-8B cannot plan this tiling from diagnostic feedback: 0/5, stuck at the same plan from round 1.
With prescriptive feedback (earlier probe), it applies each quoted "Fix:" line, one per round. So with
prescriptive feedback, a passing plan is the checker's work, not the model's. Taken together with the
loop-tool and plan-given runs: planning is a real wall, and NKI API usage is a second wall behind it.
