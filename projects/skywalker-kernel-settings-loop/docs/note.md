# Hand-in note: what decides whether a small-model loop can tune a kernel

Hack the Chip, NYU × Annapurna Labs. Track: kernel optimisation. Seats 205–209 (loop) and 170–171
(model inference).

**The authoritative numbers are the files in `results/` and `runs/`.** If this note and a JSON file
disagree, the JSON is right.

## Summary

**What we ran.** A loop of two Qwen3-8B agents that tunes the three block-size settings of AWS's
blocked matmul kernel, with every attempt compiled, run and profiled on a real Trainium2 core. A
checker agent reads the chip's measurements and gives one instruction; an optimiser agent changes
the settings. No model larger than 8B is in the loop. Claude wrote the harness only.

**On what.** Two shapes with full ground truth, every legal setting measured on the chip:
2048 × 2048 (75 settings) and a rank-256 adapter-style shape sized to Qwen3-8B's layer widths (96 settings).
Plus one loop run on each of Qwen3-8B's four linear-layer shapes, and small traced models to
measure real inference time.

**What came out.**

1. **The checker's design decides the outcome, not the model.** With the same 8B model and a redesigned checker, the share of runs ending within 10% of the best went from 3 of 30 to 29 of 30, and attempts that produced no measurement went from 217 of 300 to 0.
2. **The model is not what adds value.** A fixed one-line rule choosing from the same measured menu, with no model, finds the best setting in 71% of runs. The model manages 50%, and random choice 38%.
3. **Tuning the settings reduces real inference time, up to the level of AWS's own compiler and no further.** In a model built from Qwen3-8B's feed-forward block, the loop's settings cut inference time by 13.5% at 32-bit, level with plain PyTorch. Run directly at 16-bit and scored on the block's own inference time, the loop took it from as much as 8.63 ms down to 2.86–2.89 ms in all five runs, about 3x faster and within 5% of plain PyTorch at 16-bit. Random choice does almost as well on that problem.
4. **Building the on-chip checker exposed things the documentation and the simulator get wrong** (section 4).

**How many runs, and the spread.** Timings repeat to under 1% across three captures. Model arms:
10 runs of 10 attempts at each shape, then 30 runs at the adapter shape. No-model arms: 200 runs.
Full tables are in section 5.

**Simulator or chip.** Every time, busy percentage and byte count in this note is from the real
chip, except the one column labelled "Simulator" in section 4.

## 1. The three roles

| Role | What it is | Where it runs |
|---|---|---|
| Workload | AWS's blocked tutorial matmul kernel, float32, with three block-size settings | Trainium2 chip, seats 206–209 |
| Checker | Code measures the kernel on the chip; Qwen3-8B turns the measurements into one instruction | Model on seat 205 |
| Optimiser | Qwen3-8B reads the instruction and sets the three values | Model on seat 205 |

**This is settings tuning, not free-form code editing.** The optimiser outputs three numbers
(`TILES_IN_BLOCK_M`, `_N`, `_K`) and code writes them into the kernel. The kernel is AWS's.

The model server takes the whole chip, so the model and the kernels run on separate seats.

## 2. The checker and its reasoning

Every attempt is compiled, run on a NeuronCore at LNC=1 and profiled. Nothing is scored on the
simulator. An attempt stops at the first stage it fails.

| Stage in the log | What it means | Why it is there |
|---|---|---|
| `unreadable` | The optimiser's reply did not contain three usable settings | A small model does not always answer in the format asked |
| `duplicate` | The setting was already tried in this run | A repeat teaches nothing and wastes an attempt |
| `compile` | The kernel did not compile: an illegal value, or blocks too big for the on-chip buffer | The feedback is shortened to the one reason a model can act on |
| `numerics` | Output differs from the NumPy reference by more than 2e-2 | A wrong kernel does not crash; it returns plausible numbers |
| `timed` | Correct on the chip; median time of three profile captures, plus engine busy, bytes and transfers | This is the score |

`hwcheck.py` packages the same measurement for the hackathon's own ladder and adds the rules scan
and a check that the kernel did not write into its inputs.

**Five designs of the checker's message.** Each fixes a failure read from the previous design's log.
The model is Qwen3-8B throughout.

| Design | What the checker model is shown | Failure found in its log |
|---|---|---|
| v1 | Raw profile lines of every attempt | Argues from a rule of thumb against its own measurements; proposes illegal values; repeats |
| v2 | Comparisons worked out in code ("raising N from 2 to 24: 1572 -> 4112 µs, 2.62x SLOWER"), plus the legal values | Direction fixed, but it garbles the instruction and re-proposes tried settings |
| v3 | A menu of untried single-setting moves from the best so far, each with its measured record; the model picks one | The optimiser changed a line it was not asked to touch, landing on a tried setting |
| v4 | v3, and the optimiser's edit is checked against the instruction before any chip time | Walks one step at a time; sometimes picks the option its own reasoning argued against |
| v5 | v4, plus the furthest untried value in each direction, and the model answers by naming the move | No wasted attempts |

Two exchanges that show the change:

- **v1:** start at N=2, 1,572 µs. Checker: "Change N to 24 ... reduces data movement busy time". Result: 4,112 µs. It then proposed N=12, two illegal values, and N=24 again.
- **v4:** "Lowering N from 3 to 2 has shown consistent improvement in speed (faster 6 of 6 times)". 1,651 → 1,432 → 1,361 µs, the grid best, at attempt 6.

## 3. Headroom: is there anything to win on the chip?

Measured before building the loop. AWS's four tutorial matmul versions at 2048, three captures
each, all correct (`results/tutorial_matmul_sweep_2048.json`).

| Version | Time (µs) | Tensor engine busy | Data movement busy | Bytes moved | Transfers |
|---|---|---|---|---|---|
| Tiled (naive) | 1258.6 | 80.6% | 90.7% | 318,767,104 | 1600 |
| Loads hoisted | 1093.2 | 87.3% | 92.4% | 301,989,888 | 1344 |
| Blocked | 1019.6 | 90.6% | 62.3% | 167,772,160 | 832 |
| Fully optimised | 905.8 | 97.5% | 21.7% | 67,108,864 | 96 |

Naive to best is 1.39x faster with 4.75x fewer bytes. Run-to-run spread is under 1%.

## 4. What the checker caught

**The simulator and the chip disagree at small shapes.** Level 4 reference kernel, bytes moved:

| Shape | Simulator | Chip | Minimum |
|---|---|---|---|
| K=128 M=128 N=512 | 589,824 | 589,824 | 589,824 |
| K=256 M=256 N=1024 | 3,670,016 | 2,359,296 | 2,359,296 |
| K=512 M=128 N=512 | 1,572,864 | 1,572,864 | 1,572,864 |
| K=256 M=512 N=1024 | 7,340,032 | 3,670,016 | 3,670,016 |

On two shapes the simulator counts re-reads that the chip does not perform (mechanism not
confirmed). The hackathon's levels 5–7 are scored on the simulator, so at toy sizes that score can
reward changes that make no difference on hardware.

**The pod's default core setting doubles every profile count.** At LNC=2 a kernel runs on both
physical cores. Our first profile said the kernel moved twice the data it needed. We measure at
LNC=1.

**The README's "two cores stay free beside the model server" does not hold on our pod.** On seat
205 with the server up, every core request was refused with "cores busy", at LNC=2 and LNC=1.

**AWS's tutorial default settings (16/2/8) do not run at any of Qwen3-8B's layer shapes at 512
tokens** (section 6). They are illegal by the kernel's own rules.

**An untuned kernel can silently fail accuracy at 16-bit.** In a 36-layer stack, the unblocked
kernel at 16-bit came out 2.4% off, past the 2% bar, with no error raised (section 7).

**How this differs from AWS's own recommendation tool.** `neuron-explorer recommend` exists on the
pod. In print-only mode its banner says "AWS Bedrock-powered", its prompt is 155,604 characters,
and it is told "Please do not give code or pseudocode". That is a hosted large model advising a
person. Our checker's prompt is a few thousand characters for an 8B model, and the loop changes the
kernel and measures again. We did not run theirs end to end, because it needs a Bedrock call.

## 5. Results of the loop

### 5.1 Size 2048: a forgiving shape, where nothing separates

All 75 settings measured (`results/grid_2048.jsonl`): slowest 1,209.3 µs (1/1/1), best 904.7 µs
(4/2/4), median 919.7 µs, 39 of 75 within 2% of the best. AWS's tutorial default measures 905.5 µs.
So the 1.34x gain is from the worst setting, not from what AWS ships.

10 runs per model arm, each from a different one of the 10 slowest settings; 200 runs for the
no-model arms. "Within 2%" means 922.8 µs or faster. Source: `results/summary_2048.json`.

| Arm | Runs | Median final time (µs) | Worst (µs) | Runs within 2% of best |
|---|---|---|---|---|
| Random (no model) | 200 | 905.2 | 927.3 | 199 of 200 |
| Menu + fixed rule (no model) | 200 | 905.2 | 911.2 | 200 of 200 |
| Optimiser on raw profile | 10 | 905.2 | 938.8 | 9 of 10 |
| Optimiser + fixed sentence | 10 | 905.2 | 938.8 | 9 of 10 |
| Checker v1 | 10 | 905.3 | 947.7 | 9 of 10 |
| Checker v2 | 10 | 912.1 | 1,036.8 | 7 of 10 |
| Checker v3 | 10 | 906.7 | 1,004.0 | 7 of 10 |
| Checker v4 | 10 | 907.0 | 907.0 | 10 of 10 |
| Checker v5 | 10 | 907.0 | 907.0 | 10 of 10 |

Every arm's median is within 1% of the best. v4 and v5 are the only model arms with no bad run. The
raw and fixed-sentence arms made the same first move in all 10 runs, so they are not independent
confirmations of each other.

### 5.2 Adapter shape: where the designs separate

K=256, M=4096, N=12288: a rank-256 adapter-style matmul onto Qwen3-8B's 12288-wide feed-forward
dimension. All 96 settings measured (`results/grid_K256_M4096_N12288.jsonl`): best 1,360.9 µs
(4/1/2), unblocked 1,629.2 µs, median 2,147.6 µs, worst 9,716.6 µs (4/24/1), which is 7.14x slower
than the best. Only 4 of 96 settings are within 2% of the best. "Bigger blocks are better" held at
2048 and is wrong here.

**10 starts** spread through the slowest 90% of the grid, the same for every arm; 10 attempts per
run. "Within 2%" means 1,388.1 µs or faster. Source: `results/summary_K256_M4096_N12288.json`.

| Arm | Runs | Median final time (µs) | Worst (µs) | Runs within 2% of best | Attempts with no measurement |
|---|---|---|---|---|---|
| Random (no model) | 200 | 1,467.0 | 2,097.3 | 70 of 200 (35%) | 0 |
| Menu + fixed rule (no model) | 200 | 1,361.0 | 2,044.9 | 155 of 200 (78%) | 0 |
| v5's menu + fixed rule (no model) | 200 | 1,361.0 | 2,971.7 | 132 of 200 (66%) | 0 |
| Optimiser on raw profile | 10 | 1,906.9 | 2,535.2 | 0 of 10 | 64 of 100 |
| Optimiser + fixed sentence | 10 | 1,906.9 | 2,958.4 | 0 of 10 | 77 of 100 |
| Checker v1 | 10 | 2,284.8 | 3,114.8 | 0 of 10 | 66 of 100 |
| Checker v2 | 10 | 1,553.3 | 2,843.4 | 1 of 10 | 70 of 100 |
| Checker v3 | 10 | 1,491.6 | 3,775.8 | 4 of 10 | 25 of 100 |
| Checker v4 | 10 | 1,494.9 | 1,937.6 | 4 of 10 | 2 of 100 |
| Checker v5 | 10 | 1,494.2 | 1,494.9 | 4 of 10 | 0 of 100 |

**30 starts**, the larger-sample check. Same design; each model arm on its own Qwen3-8B server; the
no-model arms run 200 times over the same 30 starts. v2 was not rerun. Source:
`results/summary_s30_K256_M4096_N12288.json`.

| Arm | Runs | Median final time (µs) | Worst (µs) | Within 2% of best | Within 10% of best | Attempts with no measurement |
|---|---|---|---|---|---|---|
| Random (no model) | 200 | 1,460.8 | 1,694.2 | 77 of 200 (38%) | 145 of 200 (73%) | 0 |
| Menu + fixed rule (no model) | 200 | 1,361.0 | 2,173.1 | 142 of 200 (71%) | 176 of 200 (88%) | 0 |
| v5's menu + fixed rule (no model) | 200 | 1,360.9 | 2,971.7 | 136 of 200 (68%) | 196 of 200 (98%) | 0 |
| Optimiser on raw profile | 30 | 1,735.2 | 3,114.8 | 0 of 30 | 3 of 30 | 178 of 300 |
| Checker v1 | 30 | 2,135.2 | 3,114.8 | 1 of 30 | 3 of 30 | 217 of 300 |
| Checker v3 | 30 | 1,488.7 | 3,775.8 | 10 of 30 (33%) | 19 of 30 | 83 of 300 |
| Checker v4 | 30 | 1,406.2 | 2,698.6 | 15 of 30 (50%) | 26 of 30 (87%) | 5 of 300 |
| Checker v5 | 30 | 1,431.6 | 2,971.7 | 15 of 30 (50%) | 29 of 30 (97%) | 0 of 300 |

**What these tables support.**
- **Supported:** from v1 to v5, on the same model, runs ending within 10% of the best went from 3 of 30 to 29 of 30, and attempts with no measurement from 217 of 300 to 0. Random manages 73% within 10%, so v5 is more reliable than random.
- **Not supported:** that the model beats random at finding the best. 15 of 30 against random's 38% is ahead, but not conclusively at this sample size.
- **Also true:** the fixed rule on the same menu, with no model, finds the best in 71% of runs. The model choosing from the menu does not beat the rule.

Charts: `results/curve_2048.png`, `results/curve_K256_M4096_N12288.png`,
`results/curve_s30_K256_M4096_N12288.png`. Logs: `runs/*.jsonl`.

## 6. From a kernel to a model's layers

A linear layer `y = x W` maps onto this kernel as K = input width, M = token count, N = output
width. Qwen3-8B's config (hidden 4096, intermediate 12288, 32 heads and 8 key-value heads of width
128, 36 layers) gives four shapes at 512 tokens. One checker (v1) run of 10 attempts from 1/1/1 on
each, measured live on the chip (`runs/layer_*.jsonl`).

| Layer | Shape (K / M / N) | Per layer × layers | Legal settings | AWS default 16/2/8 | Loop result | AWS compiler, no hand-written kernel |
|---|---|---|---|---|---|---|
| Attention query and output projections | 4096 / 512 / 4096 | 2 × 36 | 72 | Illegal | 1,075.3 → 906.2 µs at 4/4/8 (1.19x) | 908.1 µs |
| Attention key and value projections | 4096 / 512 / 1024 | 2 × 36 | 36 | Illegal | 321.4 → 245.0 µs at 2/2/16 (1.31x) | 248.0 µs |
| Feed-forward gate and up projections | 4096 / 512 / 12288 | 2 × 36 | 144 | Illegal | 3,117.5 → 2,667.2 µs at 1/8/2 (1.17x) | 2,665.9 µs |
| Feed-forward down projection | 12288 / 512 / 4096 | 1 × 36 | 144 | Illegal | 3,029.1 → 2,664.7 µs at 1/4/1 (1.14x) | 2,706.3 µs |

- The settings-count and legality columns are computed from the config and the kernel's rules, not measured. "Illegal" is specific to 512 tokens: the default needs M divisible by 2048.
- The loop improved all four shapes. Only gate/up has a grid to compare against (partial; best 2,663.6 µs, so 2,667.2 is within 0.2%). For the others we do not know how close to the best these are.
- **Against AWS's compiler the loop's settings are level on every layer**: between 1.5% faster and 0.05% slower. The compiler column is a plain matmul traced with `torch_neuronx`, on-chip time, median of three captures, measured on seats 170–173 (`results/compiler_default_layers.json`); the kernel columns were measured on seats 206–209, and timings on different seats have agreed to about 0.3%. A gap of 1.5% is small enough that we do not claim to beat the compiler.
- Summed over the model's 252 projections: 434.1 ms unblocked, 370.9 ms with the loop's settings, 14.6% less; the compiler's own matmuls sum to 372.6 ms. This is a sum of standalone float32 kernel times, not a model run.
- The output head (151,936 wide) is not a multiple of 512, so this kernel cannot run it. It is excluded.

## 7. Real inference time, like for like

No tokens-per-second number for Qwen3-8B is possible: the served model's matmuls come from AWS's
compiler, so these settings have nothing to plug into. Instead we put the kernel inside small
models traced with `torch_neuronx` and compared them with plain PyTorch **at the same precision**.

Inference time is wall clock per call, median of 50 after 5 warm-ups, LNC=1. Every output was
checked against CPU. The models have random weights at Qwen3-8B's layer sizes; they are not
Qwen3-8B. Scripts, raw results and setup: `inference/`. Do not compare numbers across the two
seats; each table below comes from one seat.

**Two Qwen3-8B feed-forward blocks, 512 tokens (seat 171).**

| Precision | Version | Inference time (ms) | On chip (µs) | Compute utilisation |
|---|---|---|---|---|
| 32-bit | Plain PyTorch | 17.26 | 16,320 | 24% |
| 32-bit | Our kernel, unblocked 1/1/1 | 20.12 | 20,858 | 19% |
| 32-bit | Our kernel, the loop's per-layer settings | 17.40 | 16,490 | 24% |
| 16-bit | Plain PyTorch | 4.77 | 4,106 | 96% |
| 16-bit | Our kernel, unblocked 1/1/1 | 16.53 | 18,718 | 21% |
| 16-bit | Our kernel, the 32-bit settings | 8.45 | 7,797 | 50% |
| 16-bit | Our kernel, retuned at 16-bit (4/4/4) | 5.02 | 4,364 | 90% |

- **At 32-bit the loop's settings cut inference time by 13.5%** (20.12 → 17.40 ms), level with plain PyTorch (17.26 ms). This agrees with the 14.6% from the standalone sum in section 6.
- **At 16-bit the spread between settings is 3.3x**, and the settings found at 32-bit are 1.7x off the retuned ones. The right settings change with precision as well as shape, which is the case for doing this automatically.
- The 16-bit retune in this table (4/4/4) was a hand search over seven candidates on a one-block model. The loop was then run at 16-bit itself; see the next table.

**The loop itself at 16-bit, scored on in-model inference time (seats 170–174).**

The team's loop code is used unchanged (checker v5, the menu, the optimiser edit check, the
logging). Only the measurement is replaced: each attempt builds one Qwen3-8B feed-forward block
(hidden 4096, intermediate 12288, 512 tokens) with all three matmuls using the kernel at the
proposed setting in 16-bit, traces it, checks it against CPU, and scores it by inference time on
one core (median of 30 calls). One setting is shared by the three matmuls, which leaves 72 legal
settings. Code and logs: `inference/kopt16.py`, `inference/loop16/`.

*Ground truth:* all 72 settings measured on the chip. 69 ran correctly and 3 failed to compile (the
ones with N=8 and K=32). Best 2,860 µs (2/4/8), median 3,746 µs, worst 8,635 µs (1/1/1), a 3.02x
spread. 10 of 72 are within 2% of the best. None failed the accuracy bar.

*Five runs of the loop, 10 attempts each, one per seat, each with its own Qwen3-8B server:*

| Start | Start time (µs) | Best found (µs) | Setting found |
|---|---|---|---|
| 2/2/2 | 3,828 | 2,894 | 4/4/32 |
| 1/1/1 | 8,628 | 2,867 | 4/4/8 |
| 1/8/1 | 6,415 | 2,862 | 2/4/8 |
| 4/1/1 | 6,640 | 2,879 | 2/8/4 |
| 1/1/8 | 8,465 | 2,891 | 2/8/8 |

| Arm | Runs | Median final time (µs) | Worst (µs) | Runs within 2% of best | Attempts with no measurement |
|---|---|---|---|---|---|
| Checker v5 + optimiser (Qwen3-8B) | 5 | 2,879 | 2,894 | 5 of 5 | 18 of 50 |
| Random (no model), same five starts | 200 | 2,891 | 3,826 | 166 of 200 | 89 of 2,000 |
| v5's menu + fixed rule (no model), same starts | 200 | 2,894 | 4,000 | 163 of 200 | 304 of 2,000 |

For comparison, the same block in plain PyTorch at 16-bit takes 2,745 µs and the hand-found
setting 4/4/4 takes 2,877 µs.

- **Every loop run ended within 2% of the best setting**, 1.3x to 3.0x faster than its own start, and 4% to 5% behind plain PyTorch.
- **This does not show the model beating random.** Random reaches within 2% in 83% of runs here, so five out of five is what random would do about 4 times in 10. It is five runs; no percentage should be quoted for the loop.
- **Wasted attempts came back in a new form.** 9 attempts hit the buffer limit at compile, and in two runs the checker spent its last four or five attempts naming a move that was not on the menu ("Set K to 32" when K was already 32).
- **Timing agrees across seats.** The two reference settings measured on all five seats agree to 0.15% and 0.17%, and 21 of 23 settings measured on more than one seat agree to 0.31% or better. Two settings did not repeat: 4/8/16 gave 3,057, 3,912 and 3,000 µs, and 4/4/8 gave 2,867 and 3,188 µs. The ground truth uses the median; we do not know the cause.

**36-layer stack of 2048 × 2048 matmuls (seat 170).**

| Precision | Version | Inference time (ms) |
|---|---|---|
| 32-bit | Plain PyTorch | 33.00 |
| 32-bit | Our kernel, unblocked 1/1/1 | 35.56 |
| 32-bit | Our kernel, the loop's setting 4/2/4 | 33.01 |
| 16-bit | Plain PyTorch | 9.28 |
| 16-bit | Our kernel, 4/2/4 | 9.81 |
| 16-bit | Our kernel, 16/2/8 | 9.60 |

The unblocked kernel at 16-bit is left out of this table because it fails the accuracy bar (2.4%
relative error against 2%).

**Where the compiler is clearly ahead.** One adapter-shape layer followed by a mean, 32-bit: our
best setting 2.47 ms, plain PyTorch 1.77 ms. The profile shows the compiler fuses the matmul with
the mean and moves 16.9 MB, where our kernel writes its full result out and moves 516 MB. A
hand-written kernel is a boundary the compiler cannot fuse across.

**The limit.** Plain PyTorch through AWS's compiler is as fast as or faster than our tuned kernel
in every case measured. Lowering the compiler's optimisation level made no difference to the
PyTorch baseline.

## 8. What went wrong along the way

- The first pass of arm runs was cut short when the model server died. The cause was ours: a request that included a `seed` parameter. Those runs are not quoted.
- Repeating a run from the same start is not independent, because the served model answers the same prompt almost identically each time. Every run therefore starts from a different setting.
- The first 200-run baselines did not use the same starts as the model arms. They were rerun from the same starts, and the tables above use the rerun.
- A fresh `torch_neuronx` environment could not compile anything until one dependency was pinned (`inference/README.md`).
- In the 16-bit in-model experiment, two of 72 settings gave times that differed by 11% and 30% between repeats.
- The hardware profile of the 36-layer stack with the unblocked kernel would not capture, so that row has a wall-clock time only.

## 9. What we do not claim

- We do not beat AWS's engineers or AWS's compiler. The kernel is theirs; at best, tuned settings tie the compiler's plain matmul.
- The models do not write kernel code. They choose three numbers.
- The 8B model does not beat a one-line rule on the same menu, and is not conclusively better than random at finding the best setting.
- One operation, float32 in the loop, two shapes with full ground truth. Nothing here shows the loop transfers to other kernels.
- No energy figures. The pod exposes no power reading.
- The tutorial kernel is public, so Qwen3-8B may have seen its default settings in training. Comparisons between model arms are still fair, because they share the model.
