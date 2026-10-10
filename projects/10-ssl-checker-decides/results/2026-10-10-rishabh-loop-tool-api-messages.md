# Plan given + loop tool, rerun with the API messages, level 4

- **Who / seat:** Rishabh (+ Claude), seat 49, with its own vLLM server and no other jobs
- **Commit measured:** uncommitted working tree on top of 8f1ca41. md5s: `agent.py` 95739ffb,
  `nkibench.py` 68e795d5, `plan_check.py` 181b7a09, `loop_tool.py` c686ec70.
- **Hypothesis:** the shared message change (claims/rishabh-api-messages.md: allocation hint,
  "output never written" for NaN, invented-name intent map) gets the loop-tool condition past the
  NKI API wall.
- **Change:** none in loop_tool. Only the shared `enrich()` / `describe_mismatch` messages changed.
  The baseline rerun with the same messages is on seat 48 (the other session's result file).
- **Command:** `python loop_tool.py --given-plan reference --rounds 8 --samples 1 --context 8192 --repeat 5`
- **Log:** logs/attempts-rishabh-loop-tool-given-api.jsonl

## Scores

```
before (old messages):  solved 0/5   all = [0.50, 0.50, 0.50, 0.50, 0.30]   attempts that ran: 8/39, any shape correct: 0/39
after  (API messages):  solved 0/5   all = [0.30, 0.30, 0.30, 0.30, 0.62]   attempts that ran: 1/37, any shape correct: 1/37
```

The lower scores in runs 1-4 are not a regression. The old 0.50s were NaN kernels that ran
because the matmul had been dropped. The new NaN message no longer sends the model down that path,
so it keeps crashing on allocation instead. Because of the reward flaw, the mean score is the wrong
way to read this result; the per-attempt outcomes above are the right one.

## What changed in the failures

- **The allocation hint was copied literally.** The hint's example names a tile `t`, and six
  attempts wrote `dma_copy(dst=t, ...)` without ever allocating `t`, giving `NameError: name 't'`.
  This is the same pattern as plan_check's "add a loop, say `i`". Hints should use the kernel's own
  names (e.g. `lhsT_tile = nl.ndarray(lhsT_slice.shape, ...)`).
- **Other attempts called the buffer kind:** `nl.sbuf_tile(0)`, `nl.sbuf(0)` (6x "'MemoryRegion'
  object is not callable"), and `dma_copy` straight from PSUM (6x).
- **The first correct allocation (run 5, round 2) scored 0.62:** `t1/t2 = nl.ndarray(..., buffer=nl.sbuf)`,
  dma_copy, nc_matmul, tensor_copy, dma_copy. But the whole pipeline went into hole (B), after the
  k loop, with its own `t_acc`, so it multiplies only the last K tile. It is correct when K=128 and
  wrong otherwise. The feedback ("NUMERICAL MISMATCH … core arithmetic or the operand layout") does
  not name that fix.

## Verdict

Not solved, and still stuck on the NKI API. The messages moved the failures (allocation now appears
once, the NaN detour is gone), but two new things block it. Hint placeholders get copied literally.
And with the tool's two holes, the model can still put the matmul in the wrong one. Next candidates,
each measured on its own: (1) hints written with the kernel's actual names; (2) a loop-tool structural
check: "nc_matmul belongs in (A), inside the k loop, accumulating into acc".
