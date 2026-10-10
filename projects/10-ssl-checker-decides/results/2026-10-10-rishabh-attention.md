# Level 8 (single-head attention) solved by Qwen3-8B in three frozen stages

- **Who / seat:** Rishabh (+ Claude), seat 49 (idle)
- **Model:** Qwen/Qwen3-8B on the seat's vLLM, think off, max_tokens 2500, context 8192, 1 sample
- **Harness:** `projects/02-kernel-agent/attn_stage.py` (new, md5 27f7b7da…). It imports agent.py
  (pod copy bf14b521…) and nkibench.py (68e795d5…) and edits neither. Plus `check_l8.py`, which wraps
  the stock graders (see the bug below).
- **Command:** `python attn_stage.py --rounds 6 --runs 2`
- **Logs:** `logs/attempts-rishabh-attention.jsonl` (v2, 2 runs, 14 rows),
  `logs/attempts-rishabh-attention-v1.jsonl` (v1, stopped early, 4 rows). Kernel:
  `logs/rishabh-l8-kernel.py`

## Result

```
v1 (stock enrich() feedback only):  A passed round 2; B failed rounds 1-2 on the same error, run stopped
v2 (+ shape hints, its own names):  run 1: A round 2, B round 4, C round 1 -> SOLVED, reward 1.00
                                    run 2: identical (same code every round) -> SOLVED, reward 1.00
```

Stock `agent.grade(src, 8)`: 1.00, all four parts true. Stock `nkibench.verify(path, 8)` via
`check_l8.py`: rules clean, numerics 3/3 shapes, HBM traffic 1.00x the floor (4 transfers: q, k, v
in, result out), so the seq x seq scores never leave the chip. An extra hostile-input check
(inputs scaled by 10-100x, so exp() without the max subtraction overflows, plus seq=17 dim=5 and
seq=1 dim=1): 6/6 correct, no NaN. `stress.py` only supports levels 1-4, so this was an ad hoc script.

## Harness bug: the stock grader cannot pass level 8

`nkibench.py --level 8 --check` and `agent.grade(src, 8)` both raise `KeyError: 'M'` as soon as a
level-8 case is CORRECT. The roofline line reads `case["M"], case["K"], case["N"]`, and the level-8
shapes carry only `seq`/`dim`. A wrong kernel never gets that far, so nobody would have noticed.
My hand-written reference hit it on the first shape. `attn_stage.py` and `check_l8.py` add
`M=seq, K=dim, N=seq` (the QK^T matmul) to the shape dicts and then call the unchanged graders. The
inputs are built from seq/dim alone, so the kernel sees the same data. The proper fix is a level-8
flops function in nkibench.py and agent.py. Those are owned files, so I left them for their owner.

## What was built

Skeleton (harness): imports, `@nki.jit def nki_attention_(q, k, v):`, `seq, dim = q.shape`,
`# (A)`, `# (B)`, `# (C)`, `return result`. Each stage prompt shows the skeleton with the frozen
stages filled in. It gives one step list that names the method and the API (transposes via
nc_transpose into PSUM then tensor_copy to SBUF, because nc_matmul contracts over the partition axis;
the softmax_fused pattern for B; transposing probs for C) and the real ISA signatures from the pod.

- **(A) scores = q @ k.T into PSUM `scores`.** Checked by a harness-only tail that copies `scores` out,
  compared against NumPy on all 3 shapes.
- **(B) stable softmax of scores / sqrt(dim) into SBUF `probs`.** Checked the same way.
- **(C) probs @ v into HBM `result`.** The full kernel is graded by stock `agent.grade`.

Failure feedback is `agent.enrich(err, source)`. v2 adds `shape_hints()`, an AST pass over the
stage's own code that names inline `nl.ndarray(...)` keyword args and elementwise calls whose dst and
source tiles have different shapes. Every hint uses the kernel's real tile names and has no
placeholders.

## Per stage, and where it got stuck

- **A, round 1:** Qwen called nc_transpose into PSUM tiles and passed those PSUM tiles straight to
  nc_matmul. The error was "stationary must be in ['sbuf'], got psum", and the stock enrich message
  says to tensor_copy. **Round 2 passed:** Qwen added the two tensor_copy hops.
- **B, round 1:** two bugs. `reduce_res=nl.ndarray(...)` was written inline, so the row sums had no
  name, and `nisa.reciprocal(dst=row_sums, data=exp_scores)` read the (seq, seq) exp tile. The
  simulator said "could not be broadcast". The stock enrich text for that error is about slicing in
  loops, and **in v1 Qwen resent identical code twice.** I stopped the run and added `shape_hints`.
- **B in v2:** the hint named both bugs, and Qwen fixed both. In doing so it dropped some
  allocations, then fixed one NameError per round (`exp_scores`, then `probs`). **Round 4 passed.**
- **C, round 1 passed.** Qwen transposed probs, loaded v, ran nc_matmul, then PSUM -> SBUF -> HBM.

## Who did what

**Qwen wrote** every line of the kernel body: 37 lines between `seq, dim = q.shape` and `return
result`, with its own tile names. It also made each repair from the checker text.

**The harness did** the rest: the skeleton, the stage split, the intermediate checks, and the final
grading. The step text is very prescriptive. It names every ISA call, every tile shape and the key
arguments (`negate=True`, `reduce_cmd=...reset_reduce`, `operand0=1.0 / dim ** 0.5`, which operand
is stationary). It is a method spec close to pseudo-code, not code. I wrote and verified a reference
kernel first to confirm the transpose and softmax APIs. That reference was never shown to Qwen, but
the step wording follows it closely.

## Verdict

**Level 8 solved, 2/2 runs, stock grader reward 1.00** (with the M/K/N shape-key workaround, which
the stock grader needs for any correct level-8 kernel). The two runs produced identical code every
round, so this shows the result is reproducible, not robust; a temperature or seed sweep would test
robustness. The result says more about the harness than about the model. With staging, a
near-complete method spec and named fixes, an 8B model composes transpose, matmul, stable softmax,
transpose and matmul correctly in 7 calls. Without the shape hint it stalled on stage B, because the
stock message for a broadcast error points at loop slicing.

**Caveats:**
- Qwen hard-coded `dtype=nl.float32` for the output instead of the `q.dtype` the step asked for.
  That is correct for the float32 grader, but bfloat16 inputs would get a float32 result.
- The laptop agent.py (1c17bbaf…) has newer enrich() messages for levels 1/2 that the pod copy
  lacks. None of them apply to the errors seen here.

## Independent check (coordinating session, 2026-10-10 ~21:30 UTC, seat 48)

`logs/rishabh-l8-kernel.py` re-graded on a different seat: the stock `nkibench.py --level 8 --check`
reproduces the `KeyError: 'M'` crash; through `check_l8.py` (adds M/K/N to the shape dicts, nothing
else) `agent.grade` returns 1.00 with parses, rules, runs and correct all true, and the HBM traffic
is at the byte floor on every shape.
