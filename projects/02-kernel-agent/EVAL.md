# Held-out eval set, tolerance, and token accounting

Everything here runs in the NKI 0.6.0 CPU simulator (`nki.simulate`), on a seat pod or in the
`python:3.12-slim` container from `SETUP_PYTHON.md`. No number below comes from the device.

## Two sets of cases, kept apart

| | shapes | values | who sees the result |
|---|---|---|---|
| **loop set** (`LEVELS[n]["shapes"]`) | 4 per level (1 for level 3) | standard normal, seed 0 | the model, as feedback every round |
| **held-out set** (`EVAL_SHAPES`, `VALUE_KINDS`) | 4–5 new shapes per level (level 3: the same one) | 4 kinds, seed 1000 | **nobody during the loop**; run on a finished kernel with `--eval` |

The held-out result is never put in a prompt. If it were, it would turn into part of the loop set
and stop telling us anything. `agent.py` and the loop's grading do not touch it.

```bash
python nkibench.py --level 4 --eval my_kernel.py      # 16 cases: 4 shapes x 4 value kinds
```

### Held-out shapes

All of them are inside each level's contract: the shipped reference kernel passes every case
(`--eval reference_levelN.py`: 20/20, 16/16, 4/4, 16/16).

| level | shapes | what is new compared with the loop set |
|---|---|---|
| 1 avgpool | (128,32,32)/2, (16,30,30)/5, (1,64,64)/8, (96,20,12)/4, (7,10,11)/3 | full 128 partitions, one partition, H ≠ W, H and W not divisible by the pool size, pool sizes 5 and 8 |
| 2 transpose | (128,128) as 16×8, (1,24) as 4×6, (100,60) as 6×10, (17,77) as 7×11 | 128 rows, 1 row, rows not a power of 2 |
| 3 matmul | K=128 M=64 N=512 | none: the reference asserts this exact shape, so only the values change |
| 4–7 matmul | K384 M256 N512, K128 M384 N1536, K640 M128 N1024, K128 M128 N2048 | K of 5 tiles, M of 3 tiles, N of 4 tiles; the loop set never goes past K=512 |

Not tested: shapes the reference itself rejects (on level 4, M not a multiple of 128, for example).
A failure there would tell us nothing about the agent.

### Value kinds

| kind | what it catches |
|---|---|
| `normal` | a different random draw from a seed the loop never used |
| `ramp` | `linspace(-1, 1)`: every element is distinct and ordered, so an element read from or written to the wrong place cannot cancel out |
| `large` | normal × 1e4: a kernel that computes or stores in float16 overflows to Inf |
| `float16` | the inputs arrive as float16. The simulator's `dma_copy` casts silently, so only the eval's **dtype check** (output dtype must equal the reference's) catches a kernel that hard-codes float32 |

NaN and Inf inputs are left out. The reference returns non-finite values for them, and the
checker counts any non-finite output as a failure.

### Proof that it catches things: mutants

These are deliberately wrong kernels in `mutants/`. Each one passes **every** loop shape.

| mutant | loop set | held-out | caught by |
|---|---|---|---|
| `l1_square.py`: assumes H == W | 4/4 | **12/20** | the H ≠ W shapes |
| `l4_k512.py`: stops after 4 K tiles | 4/4 | **12/16** | K=640 (error 0.39–1.96 of the output RMS) |
| `l4_fp16out.py`: writes float16 | 4/4 | **4/16** | `large` (Inf) and the dtype check |
| `l4_fp32.py`: hard-codes float32 | 4/4 | **12/16** | `float16` (dtype check only) |

## Tolerance: max |error| ≤ 2e-2 × RMS(reference output)

The check is relative to the output's RMS, not to each element. A relative per-element test blows
up near zero, and matmul and avgpool outputs cross zero all the time.

**Why 2e-2. Measured:** we rounded the inputs to bf16 and ran the NumPy reference, over every loop
and held-out shape and every value kind. The worst error was **0.0134** of the RMS (fp16: 0.0016).
The smallest error any mutant bug produced was **0.393**. So 2e-2 lets through a kernel that
legitimately uses bf16 (1.5× margin) and rejects real bugs by a factor of 20 or more. The margin
above bf16 is thin: a kernel that also *accumulates* in bf16 instead of float32 PSUM would land
close to the limit.


## Token accounting

`attempts.jsonl` now records for every attempt: the endpoint's `prompt_tokens` and
`completion_tokens`, and `prompt_split`, which is the prompt cut into `instructions`, `reference`,
`api_card`, `prev_code`, `feedback`, `ledger` (and `chat_template`). The split is read off the
prompt text, so it also holds for the repair prompts in `feedback_v2.py` and `feedback_v3.py`. If
the served model's tokenizer is installed locally, it counts each piece; otherwise the endpoint's
exact total is shared out in proportion to characters (`count_method` says which).

```bash
python scripts/token_budget.py attempts.jsonl -o analysis/token_budget   # .png + .csv
```
