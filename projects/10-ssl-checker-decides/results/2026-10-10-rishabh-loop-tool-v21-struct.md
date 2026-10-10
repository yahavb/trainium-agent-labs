# Plan given + loop tool: messages v2.1, then v2.1 + structural check, level 4

- **Who / seat:** Rishabh (+ Claude), seat 49, with its own vLLM server and no other jobs
- **Commit measured:** uncommitted working tree on top of 8f1ca41. md5s: `agent.py` 11917d5b (messages
  v2.1), `nkibench.py` 68e795d5, `plan_check.py` 181b7a09. `loop_tool.py` c686ec70 for step 1,
  fd282b9a for step 2.
- **Hypothesis:** (1) allocation hints written with the kernel's real names (v2.1, the other
  session's change) get the tile body past allocation. (2) Adding a structural check that names
  which hole the matmul belongs in (`--structure-check`, my file) gets it to a correct kernel.
- **Commands:** `python loop_tool.py --given-plan reference [--structure-check] --rounds 8 --samples 1 --context 8192 --repeat 5`
- **Logs:** logs/attempts-rishabh-loop-tool-given-v21.jsonl, logs/attempts-rishabh-loop-tool-given-v21-struct.jsonl

## Scores (same plan, same settings throughout)

```
old messages:             all = [0.50, 0.50, 0.50, 0.50, 0.30]  solved 0/5  any shape correct 0/39
messages v1:              all = [0.30, 0.30, 0.30, 0.30, 0.62]  solved 0/5  any shape correct 1/37
messages v2.1:            all = [0.50, 0.30, 0.30, 0.30, 0.50]  solved 0/5  any shape correct 0/40
v2.1 + structure check:   all = [0.50, 0.30, 0.30, 0.50, 0.30]  solved 0/5  any shape correct 0/40
```

## What happened

- **v2.1:** the allocation now comes out right (`lhsT_tile = nl.ndarray(lhsT_slice.shape, ...)`), but
  it is split across the holes: allocations in (A), loads/matmul in (B), or tile names used without
  allocating them (NameError 11x). "Do the same for out_slice" was copied literally: `out_tile =
  nl.ndarray(out_slice.shape)` in (A), before out_slice exists (UnboundLocalError 5x). The other
  session has both queued as v2.2 fixes.
- **+ structure check:** the hint fired on 20/40 attempts and **was followed 15/16 times**: the next
  round has `nc_matmul(dst=acc, ...)` in (A). But the 21 structurally correct rounds then fail on
  allocation again (NameError on lhsT_tile 6, MemoryRegion 5, invented sbuf_tile 3, ...). The model
  applies the fix it is given and drops a fix it had applied before.

## Verdict

Not solved. Each named fix works on its own: invented names are gone, allocation can be right, and
structure gets fixed 15/16 times. But Qwen applies only the latest message and doesn't keep earlier
fixes, so it cycles between two or three walls. More message tuning is unlikely to converge. What
should help is keeping the parts that are already right: grade (A) on its own (load + matmul into
acc), freeze it once it passes, then ask only for (B). That is the same decomposition idea one level
further down, and the next experiment.
