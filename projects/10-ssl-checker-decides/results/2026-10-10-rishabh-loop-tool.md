# Plan given + loop tool, level 4

- **Who / seat:** Rishabh (+ Claude), seat 49, with its own vLLM server and no other jobs
- **Commit measured:** uncommitted working tree on top of 8f1ca41. md5s: `loop_tool.py` c686ec70,
  `agent.py` 06805f2d, `plan_check.py` 181b7a09.
- **Hypothesis:** with the tiling taken off the model (the correct plan is given, and a tool writes
  the loops, slices and PSUM accumulator from it), Qwen only has to write the per-tile body and
  solves level 4.
- **Change:** new `projects/02-kernel-agent/loop_tool.py --given-plan reference`. The plan is
  `plan_check.REFERENCE`, the same one used by `agent.py --plan-file reference`. The model fills two
  holes: (A) load both slices into SBUF and matmul into `acc`; (B) copy `acc` out. Graded by the same
  `agent.grade` (private temp path) and today's unchanged `enrich()`.
- **Command:** `python loop_tool.py --given-plan reference --rounds 8 --samples 1 --context 8192 --repeat 5`
- **Log:** logs/attempts-rishabh-loop-tool-given.jsonl (screen run: ...-given-screen.jsonl)
- **Deviation:** `--samples 1`, not 4. Sampling on this server is effectively greedy (all probe rounds
  4/4 identical; 91/108 baseline rounds). The plan-given whole-kernel run on seat 48 uses 4.

## Scores

```
baseline (no plan, whole kernel):       level 4: solved 0/5   all = [0.62, 0.62, 0.62, 0.62]  (4 runs)
plan given + loop tool (this):          level 4: solved 0/5   all = [0.50, 0.50, 0.50, 0.50, 0.30]
```

The three-way comparison is in results/2026-10-10-rishabh-plan-given.md. Plan given with the whole
kernel written by the model scores 0.30 (0/1, --samples 4, seat 48), so the loop tool's 0.50 beats
handing over the plan as text, but neither solves.

Not a like-for-like comparison: this condition is handed the tiling. The point is where it gets stuck,
not the score. Wall time: 5 runs in 3 min 47 s (rounds 3-9 s on a server to itself).

## Where it fails (all 5 runs, gate 2)

- Round 0 is nearly right: dma_copy, dma_copy, nc_matmul(dst=acc), tensor_copy, dma_copy. The only bug
  is `nl.sbuf[0]` used as a tile instead of `nl.ndarray(..., buffer=nl.sbuf)`. The feedback is the raw
  `TypeError: 'MemoryRegion' object is not subscriptable`, which names no fix.
- It then drops the matmul, and the output is NaN. That scores 0.50, **higher** than the near-correct
  0.30 crash, because it runs.
- The NaN message says "uninitialised PSUM", so it invents ways to zero `acc`: `nisa.fill` (9 times),
  `acc.fill`, `nisa.psum_init`. It stops after 4 identical failures.

## Verdict

Not solved. With tiling removed, the wall is NKI API usage (allocation, invented functions), and two
of `enrich()`'s messages fail to name the fix. Next: one shared `enrich()` change in all conditions,
measured against a baseline with the same code: the MemoryRegion allocation hint, "output never
written" instead of "uninitialised PSUM" for NaN, and real-name mapping. Also a checker-design note:
the reward ranks a running wrong kernel above a nearly correct crashing one.
