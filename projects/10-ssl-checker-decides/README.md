# Team 10 (SSL): the checker decides

**Project 2, the kernel agent.** Qwen3-8B on each seat's own Trainium2 chip writes NKI kernels; the
organizers' grader (`nkibench.py`, `nki.simulate`) checks them; the error message becomes the next
prompt. We kept the model and the grader fixed and changed only what the checker says back and how the
task is asked. Seats 45, 48, 49 · October 10, 2026.

**Presentation:** https://claude.ai/artifact/RwTLdortxpvZoZJR7m85WJ ·
**One-page note:** [NOTE.md](NOTE.md) · **One write-up per experiment:** [results/](results/)

## Results by level (stock grader, this folder's logs)

| Level | Original agent | End of day | How | Evidence |
|---|---|---|---|---|
| 1, average pooling | 0.30 | 0.30 | Not moved. Invented API calls cut from 50% to 12% of attempts; now blocked on `ap()` stride arithmetic | results/…-api-messages.md, …-step-list-l1-l2.md |
| 2, transpose | solved 2/4 | solved 1/3 (noisy; step list 0/5) | Not moved | results/…-baseline.md, …-step-list-l1-l2.md |
| **3, matmul, one tile** | 0.30, 0/4 | **solved 2/2** | Normal agent loop: step list + one named fix for PSUM→HBM DMA, written with the kernel's own variable names | results/…-level3-solved.md |
| **4, matmul, tiled** | 0.62, 0/4 | **solved 5/5** | Qwen plans the tiling (step wording, honest feedback: 0/5 → 5/5 plans) and writes the body; a tool expands the plan into loops | results/…-plan-stage.md, …-loop-tool-staged.md |
| 5, loads hoisted | not run | 0.88 | 3/4 shapes; 2.00x the byte floor against a 1.60x limit | results/…-roofline-optimize.md |
| 6–7, blocked | not run | 0.75 | Same kernel | — |
| **8, attention** | 0.30, 0/2 | **solved 2/2** | Three frozen stages (scores, softmax, P·V); Qwen wrote all 37 body lines | results/…-attention.md, …-l8-baseline.md |

Solved kernels passed the stock grader and `code/stress.py` (generated shapes and hostile values). Runs
were near-identical (sampling is effectively greedy on these seats: 91 of 108 baseline rounds returned
4 byte-identical samples), so the repeats show the results are **reproducible, not robust**.

## Measured on the chip (seat 49, model server stopped)

| Kernel | Compiler prediction | Measured on Trainium2 | Device output |
|---|---|---|---|
| Matmul, slow start (TILE_N=128), 512×256×1024 | 183.9 µs | **345.8 µs** | correct |
| Matmul, tutorial reference | 65.9 µs | **127.8 µs** | correct |
| **Matmul, Qwen's kernel** | 59.7 µs | **118.8 µs** | correct |
| Attention, 3 shapes | 8.4–8.8 µs | **19.6–20.5 µs** | correct |

Given a correct but slow matmul, our feedback loop got Qwen to make it **2.91x faster on the chip**, and
7% faster than the tutorial's kernel. Feedback naming one counted fix ("64 nc_matmul instructions where
16 would do: the moving tile is 128 wide, nc_matmul accepts 512") reached 3.02x (predicted); latency
numbers alone reached 1.03x, and a generic roofline diagnosis 1.00x (24/24 attempts unchanged). The
compiler under-predicts by about 2x but ranks every kernel in the same order as the chip.
(results/…-roofline-optimize.md, …-chip-timing.md)

## What we learned (each measured; details in NOTE.md)

1. One specific, named fix moves the model; a true but generic diagnosis freezes it.
2. Qwen applies only the latest fix. Freezing code that already passed works; a "keep these fixes" list
   in the prompt does not (recurrence unchanged).
3. Placeholder names in hints (`i`, `t`, fixed slices) get copied literally, so hints use the kernel's
   own names.
4. A recipe doesn't transfer by itself: the level-4 step list in the plain loop scored 0.30; giving the
   correct plan alone also scored 0.30.
5. The reward credits NaN output as "runs" (every loop-tool 0.50 was such a kernel); re-scored offline.

## The hand-in items

| Item asked for | Where |
|---|---|
| 1. The agent | `code/agent.py` (adds `--plan-first`, `--plan-file`, `--step-list`, `--keep-fixes`; all off by default, so the default behaviour is unchanged), `code/loop_tool.py`, `code/plan_stage.py`, `code/attn_stage.py`, `code/e2e_rate.py` |
| 2. Verification harness, with tolerance and reasoning | Stock `nkibench.py` (the tolerance is 2e-2 of the output's RMS, unchanged), plus `code/stress.py` (Hypothesis: shapes within each level's contract, shrunk to the smallest failure; fixed seed so scores are comparable) and `code/plan_check.py` (a tiling plan checked on every valid level-4 shape, ~75 ms). `code/nkibench.py` changes only the NaN message |
| 3. Eval set, including hostile values | `stress.py` `CONTRACTS` (per-level shapes, including sizes the pool doesn't divide) and `VALUE_KINDS` (zeros, constant rows, negative, 1e4, 1e-4, +100 offset); `plan_check.py` contract and ragged shapes |
| 4. Failure taxonomy with counts | The "failures" / "top failures" section of every file in `results/`; raw messages are the `feedback` field in `logs/*.jsonl` |
| 5. Token instrumentation | Every attempt in `logs/*.jsonl` records `prompt_chars` and `reply_chars` (about 4 chars per token; context 8192); gate-1 logs also keep the raw reply |
| 6. One-page reproduction note | [NOTE.md](NOTE.md) |

Other tools: `roofline/` (`roofline.py`, the lower-bound time per operation from cited hardware
numbers; `instcount.py`, the instruction counter behind the named fixes; `optimize.py`, the speed-up
loop; `predict.py`, compiler-predicted latency), and `code/chip_time.py` (on-device benchmark).

## Reproduce

In a seat pod, copy `code/*.py` into `/workspace/projects/02-kernel-agent/` (`pip install hypothesis`),
then for example:

```bash
python plan_check.py --selftest && python stress.py --selftest
python agent.py --level 3 --step-list --rounds 8 --samples 1 --context 8192 --repeat 2      # level 3
python e2e_rate.py                                                                          # level 4 end to end
python attn_stage.py --help                                                                 # level 8, staged
python check_l8.py <kernel.py>        # stock level-8 grader + the M/K/N shape-key fix (see below)
LNC=1 NEURON_LOGICAL_NC_CONFIG=1 python chip_time.py    # on-chip timing; needs the vLLM server stopped
```

`roofline/` runs from any folder (`python roofline.py`).

## Issues found in the stock harness

- **Level 8 can't be passed as shipped:** `nkibench.py --level 8 --check` and `agent.grade(src, 8)`
  raise `KeyError: 'M'` on any correct kernel, because the roofline line reads `case["M"/"K"/"N"]` and
  the level-8 shapes have only `seq`/`dim`. `code/check_l8.py` adds those keys and changes nothing else.
- **`serve.sh` doesn't pin cores:** vLLM claims all 4 logical NeuronCores while computing on 2, so no
  kernel can run on the chip while it's up (README Part 3 says cores 0-1 are free; they aren't).
- **Sampling is effectively greedy**, so `--samples 4` mostly adds load (README Part 2 assumes Qwen's
  samples differ).

## Limits

Correctness is `nki.simulate`; speed was optimized on the compiler's prediction and checked on the chip
at the end. Level 4 relies on a tool to expand Qwen's plan into loops. The repeated runs were
near-identical.
