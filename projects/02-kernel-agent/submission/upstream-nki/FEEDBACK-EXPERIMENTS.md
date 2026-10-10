# Feedback experiments on the upstream NKI kernel agent

Target: `trainium-agent-labs/projects/02-kernel-agent` (upstream commit `8f1ca41`), Qwen3-8B on
the seat pod's local vLLM, graded by `nkibench.py` on the CPU simulator. Upstream's own status
line: *no agent has solved any level; best 0.62 on level 4.*

Four sidecar scripts change only the text the model sees. None edits `agent.py` or `nkibench.py`;
each imports `agent`, rebinds `enrich` / `first_prompt` / `repair_prompt`, and calls `main()`, so
every version is graded by the unchanged checker and can be compared with the unchanged baseline.

    python agent_feedback_vN.py --check-feedback       # routing self-check, prints the new text, no model
    python agent_feedback_vN.py --level 4 --rounds 8 --samples 4 --context 8192 --repeat 5

## Setup

| | |
|---|---|
| model | Qwen/Qwen3-8B, thinking off, temperature 0.6, top_p 0.95 |
| rounds / samples / context | 8 / 4 / 8192 |
| repeats | 5 per level per version (solve rate, as upstream asks) |
| reward | 0.1 parses + 0.2 rules + 0.2 runs + 0.5 x (shapes passed / shapes) |
| pods | three pods, one level per pod, so runs never share `/tmp/_agent_levelN.py` |

## Results (interim, as of 17:55 UTC 2026-10-10; pending cells are still running)

| level | baseline | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| 1 avg pool | 0/5, 0.30 | 0/5, 0.30 | 0/2 so far, **0.50** every run | pending | - |
| 2 transpose | 5/5 | - | pending | - | - |
| 3 matmul single | 0/5, 0.30 | - | 0/5, 0.30 | 0/3 so far, 0.30 | pending |
| 4 matmul tiled | 0/5, 0.62 | - | **1/2 so far**: 0.75, **1.00 (SOLVED)** | pending | - |

Baseline and v1 have zero variance across runs: the model is effectively deterministic on an
unchanged prompt. Variance appears only once the prompt gives it room (v2, level 4).

### Where the 360 baseline attempts died

| level | attempts | failure | count |
|---|---|---|---|
| 1 | 160 | invented `nki.isa` name (`multiply`, `scalar_mul`) | 80 |
| 1 | | `dma_copy` element-count mismatch (one pool window per copy) | 40 |
| 1 | | `nc_matmul(... transpose_moving=)` | 20 |
| 1 | | `nc_matmul` dst in sbuf | 20 |
| 3 | 100 | `dma_copy` element-count mismatch | 39 |
| 3 | | reshape of result into `(1, 64)` | 32 |
| 3 | | 1-D tile (`out = lhsT.shape[1:]`) | 29 |
| 4 | 80 | `dma_copy` dst partition 256 > 128 (single-tile kernel, never tiled) | 72 |
| 4 | | psum `(M,N)` copied into the `(K,M)` operand tile | 8 |

## What each version changes, and the failure that motivated it

Every change came from the same loop: read the code the model actually wrote (`attempts-*.jsonl`),
find the line it is stuck on, classify the cause, change only the text that addresses it, rerun 5x.

**v1** (one hint). Error `nki.isa has no attribute multiply|scalar_mul` -> the exact replacement
call `nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=scale)` instead of a list of
"closest real names". *Result 0/5, identical to baseline.* The log shows the hint was delivered and
not acted on: the model's whole approach to pooling (nc_matmul on 2x2 windows) was wrong, so a
corrected function name changed nothing. Negative result, kept on purpose.

**v2** (three changes, v1 kept).
1. *Shape card.* Both prompts list every test shape and its output shape. The baseline never told
   the model that level 4 tests K=256, N=1024, so it wrote a single-tile kernel.
2. *Rewrite, not patch.* The stock `repair_prompt` says "change exactly what the checker names and
   keep everything else identical" while the checker's hint says "loop over the partition
   dimension in chunks". The two contradict; the 8B model obeyed the first (72 of 80 level-4
   attempts). When the feedback asks for a loop-structure change, v2 asks for a rewrite of the
   function body instead.
3. *Per-level direction*, prose only. Level 1: this is a reduction, no matmul, load once, how
   `.ap([[stride, count], ...])` works with a worked example on a different shape, then `nl.sum`
   and `tensor_scalar`. Level 3: the result is `(M, N)` and needs its own tiles. Level 4: which of
   M, N, K maps to which loop; PSUM accumulates across the K loop.

*Result:* level 4 0.62 -> 0.75 (run 1) and **1.00 (run 2, a genuine three-nested-loop tiled matmul,
roofline 36.6 Flops/Byte)**. Level 1 0.30 -> 0.50 every run: the kernel now has the right structure
(whole input in sbuf, 5-D `.ap()` view, `nl.sum(axis=[3,4])`, `tensor_scalar`, `dma_copy` out) and
fails only on the stride arithmetic. Level 3 stays 0.30 but the failure moved: from "output is
1-D" to `dma_copy(dst=nl.shared_hbm, ...)`, i.e. the memory region used as the output tensor,
caused by v2's own wording "dma_copy sbuf -> the shared_hbm output".

**v3** (v2 + three rules, each from a v2 failure).
- `.ap()` stride error -> the rule: a stride is how many elements one step along that axis skips
  in the row-major (C, H, W) tile; k rows = k*W, k columns = k; the partition pair is `[H*W, C]`,
  filled in with the shape's numbers. (The model had written `[[C, C], [H, H//p], ...]` and, told
  "must equal 1024", changed 32 to 2.)
- `'MemoryRegion' object ...` / operand `got private_hbm` -> `nl.shared_hbm` is a region, not a
  tensor; the allocation line `out = nl.ndarray((M, N), ..., buffer=nl.shared_hbm)`; and the
  matmul direction text is reworded to spell that out.
- Level-4 direction rewritten as THREE NESTED LOOPS with both slice axes written out, because v2's
  "each dimension is its own loop" was read as three sequential loops with `lhsT[k0:k0+128, :]`.
  `dma_copy` size mismatch with an integer src/dst ratio -> "you sliced only one axis".

*Interim:* level 3 still 0.30 after 3 runs, on a new line: `M, K = lhsT.shape` (K and M swapped).
Levels 1 and 4 pending.

**v4** (v3 + shape doctor, matmul levels). The checker's feedback carries `On K=.. M=.. N=..` and
`got src=.., dst=..`, so the sidecar can name which tensor each count is (`65536` = rhs, the whole
`(K, N)`; `32768` = `(M, N)`), and when the tile matches no legal shape or the wrong one, say that
a size variable was derived wrongly and give `K, M = lhsT.shape` (K first). Inference from
information the checker already prints; no reference code. *Pending.*

## Caveats

- Level 2's baseline solution indexes the HBM input element by element inside `nl.affine_range`.
  The simulator accepts it; not checked on hardware.
- v3's level-4 direction text describes the nesting of the reference tiled matmul closely. It is
  prose, not code, and upstream's README argues "one worked example beats a list of rules", but
  readers should judge for themselves how much was handed over.
- All numbers are simulator correctness (layer 1). No device timings.
