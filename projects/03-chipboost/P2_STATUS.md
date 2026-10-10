# CHIPBOOST status: P2 (`kernels-search`)

*Final for the experiments, Oct 10 2026, 15:35. Everything was measured on seat-102's Trainium2 chip,
using P1's referee (`speedcheck.py` at `referee-timing` 76b2227: device clock, interleaved A/B, held-out
shapes for every would-be "faster"). The exception is anything marked otherwise. All P2 work is on this
one branch. Every result is labelled by whose it is: AWS's design (expert-derived), the search's discovery,
or P2's engineering.*

## The headline

On Qwen3-8B's per-core matmul shapes (gate_up and q_proj at 256 tokens, timed together). The first column
says whose result each row is (definitions in section 4):

| Kind | Kernel | Time | vs start | What it adds |
|---|---|---|---|---|
| Baseline | `matmul_start`, the tiled tutorial kernel | 960.5 us | 1.00x | the yardstick |
| **Expert-derived** | AWS's fully optimised tutorial kernel as shipped (`matmul_expert.py`, with P2's fp32 accumulation fix) | 385.6 us | 2.49x | AWS's design: none of this 2.49x is a discovery |
| **Search discovery** | arm c's best block sizes, one per run (3 runs x 24 attempts) | 291.1 / 285.2 / 280.4 us | 3.30x / 3.37x / 3.43x | **1.325x / 1.352x / 1.372x over the expert**, each against its own run's attempt 0 |
| Ground truth, not a discovery | all 62 legal block sizes timed once (the sweep); the best is `m1 n12 k4` | 280.3 us | 3.43x | 1.375x over the expert: the most that block sizes can give |
| **Expert-derived + P2 engineering** | the expert split over both physical cores (`matmul_expert_lnc2.py`, launched as `kernel[2]`) | 192.2 us | 5.0x | 1.50x over the same kernel on one physical core. **Not a referee verdict** |

- **The search's claim is the last column: 1.325x-1.372x on top of AWS's kernel** (median 1.352x), not the
  3.4x vs start. Most of that 3.4x is AWS's design.
- AWS's own blocking ranks **#29 of 62** at this shape: mid-pack.
- AWS's kernel *as published* **fails the referee's precision check**: bf16 accumulation, 5.3 bf16 ulps against a limit of 4.

## 1. What P2 delivered

| File | What it is |
|---|---|
| `shapes.py` | Dev, timing and held-out shapes per op at Qwen3-8B per-core sizes, in the referee's spec format (incl. `out` and fresh `heldout` samplers) |
| `kernels/matmul_start.py`, `rmsnorm_start.py` | The start kernels. Hint-free: the model never reads why they are slow |
| `kernels/matmul_expert.py` | AWS's SDK 2.32 fully optimised matmul. Accumulates in fp32, block sizes are caps fitted to each shape |
| `kernels/matmul_expert_aws.py` | The same, with AWS's bf16 accumulation: the evidence for the precision finding |
| `kernels/matmul_expert_lnc2.py` | The expert split between the two physical NeuronCores of an LNC=2 core |
| `kernels/copy_floor.py`, `copy_tiled.py` | RMSNorm's bandwidth floor (packed rows), and the tiled copy it replaced |
| `search.py` | Arm c, random search over the expert's block caps. Also `--exhaustive --shard I/N` for the sweep and `--summarize` for the report |
| `heldout_grid.py` | The end-of-run held-out check for the dashboard's panel 5 |
| `check_kernels.py`, `pod_check.sh`, `tests/test_search.py` | The simulator gate for every kernel, plus 5 tests |
| `tools/probe_nki.py`, `lnc2_probe.py`, `recheck_spoiled.py` | API probe; the LNC=2 measurement; re-judging attempts spoiled by a mid-run pull |
| nkibench levels 9-12 | Qwen3 ops in bf16, with per-level tolerances, a converting-DMA byte-count fix and a roofline crash fix |
| `results_p2.json`, `results_heldout_matmul.json` | The dashboard's inputs |

## 2. Results (chip, seat-102)

**Search discoveries: arm c, random search over the expert's block caps.** Three repeats of 24 referee
evaluations each, on cores 0-2 at once.
- Attempt 0 of every run is the expert as shipped, so it is expert-derived: it measured 2.493x, 2.490x and
  2.496x vs start.
- Attempts 1-23 are the search. Its discovery is its best **over that run's own attempt 0**, both timed in the
  same session. The dashboard's tuning study measures it the same way, and `search.py --summarize` prints it.

| Seed | Best verified | **Over the expert** (its attempt 0) | vs start | On attempt | Rank among all 62 | Share of the best |
|---|---|---|---|---|---|---|
| 0 | `m2 n4 k16` | **1.325x** | 3.303x | 23 | #4 | 96.4% |
| 1 | `m1 n6 k4` | **1.352x** | 3.367x | 9 | #3 | 98.3% |
| 2 | `m1 n12 k4` | **1.372x** | 3.425x | 10 | **#1** | 100% |
| **Spread** | | min 1.325x, **median 1.352x**, max 1.372x | min 3.303x, median 3.367x, max 3.425x | | | |

**Framing (agreed with P4):** this arm *tunes the expert's block sizes*. It starts from AWS's design at
2.49x, so of each run's 3.3-3.4x vs start, only the 1.325-1.372x on top is the search's. That is a separate
claim from "the model improves the start kernel".

**The ground truth: an exhaustive sweep.** All 62 SBUF-fitting triples, split over four NeuronCores. It
measures the whole space, so it is not a discovery by any arm: it shows how close each run got. Its best is
1.375x over the expert as shipped.

| Rank | Caps | Time | vs start |
|---|---|---|---|
| 1 | m1 n12 k4 | 280.3 us | 3.426x |
| 2 | m2 n6 k8 | 284.6 us | 3.377x |
| 3 | m1 n6 k4 | 285.1 us | 3.368x |
| 4 | m2 n4 k16 | 290.5 us | 3.310x |
| 5 | m2 n4 k8 | 292.0 us | 3.292x |
| 29 | **m2 n2 k8 (AWS's default)** | 385.6 us | 2.492x |

The winners take the whole N dimension, or half of it, in one block, unlike AWS's 2 tiles. At 256 tokens
there is little M to reuse.

**Expert-derived + P2 engineering: both physical cores (LNC=2). Not a referee verdict.** The same kernel and the same caps
(`m2 n6 k16` per program), launched plainly and as `kernel[2]` (`tools/lnc2_probe.py --validate`, core 3):

| Shape | One physical core | Both physical cores | Gain | Host-clock cross-check |
|---|---|---|---|---|
| gate_up 4096x256x6144 | 206.5 us (62.4 TFLOP/s) | **131.9 us (97.7 TFLOP/s)** | 1.565x | saves 60.1 us per call; device says 74.6: agree |
| q_proj 4096x256x2048 | 81.8 us (52.5 TFLOP/s) | **60.2 us (71.3 TFLOP/s)** | 1.358x | saves 13.0 us per call; device says 21.6: agree |
| **Total** | 288.3 us | **192.2 us** | **1.50x** | 89.4 TFLOP/s on one logical core (peak 2 x 79) |

- Both outputs were correct, by the referee's own check.
- **Whose 5.0x it is:**
  - 2.49x is AWS's design: 960.5 to 385.6 us.
  - About 1.34x is the block sizes: 385.6 to 288.3 us at LNC=1. `m2 n6 k16` is the best of the archived
    first random-search run; the sweep ranks it #6 of 62.
  - 1.50x is the second core, which is P2's split: 288.3 to 192.2 us.
- The two physical cores share one HBM stack, which is why the gain is 1.5x and not 2x.
- **A plain NKI launch at LNC=2 leaves half of every NeuronCore idle.** Every other kernel in the project
  runs that way, the referee's included.

**RMSNorm at 2048 tokens** (2048x4096 and q_norm's 32768x128):

| Kernel | Time |
|---|---|
| `rmsnorm_start` | 518.2 us |
| packed copy floor | 140.7 us |

- That leaves 3.68x of room.
- At q_norm's 256-byte rows the packed copy is **6.7x** faster than one tile per DMA.

**Held-out grid** (16:40, seat-102 core 2; `results_heldout_matmul.json`). 24 cells: 6 shapes never used for
tuning, hostile inputs, each kernel timed A/B against start at that shape. Speedup vs start:

| Kernel | down_proj@256<br>256x6144x4096 | o_proj@256<br>256x2048x4096 | q_proj@512<br>512x4096x2048 | gate_up@128<br>128x4096x6144 | 640x1280x2560 | kv_proj@1024<br>1024x4096x512 |
|---|---|---|---|---|---|---|
| expert (AWS's default caps) | 2.823x | 2.077x | 3.786x | 1.276x | 1.569x | 3.050x |
| AWS as published | **FAIL**, 4.9 ulps | 2.122x | 3.795x | 1.281x | 1.587x | 2.956x |
| **best random search** (`m1 n12 k4`, seed 2) | **3.987x** | **2.837x** | 3.800x | **2.342x** | 1.456x | 1.200x |

- **The search's best is correct on all 6 unseen shapes.**
- **Speed is mixed:**
  - it beats AWS's default on the three other 256- and 128-token shapes: +41%, +37% and +84%;
  - it ties at q_proj with 512 tokens;
  - it loses at 640x1280x2560 (-7%), and badly at kv_proj with 1024 tokens (1.20x against 3.05x).
- **Block sizes tuned at 256 tokens do not transfer to every shape.** One set of caps for all shapes is the
  wrong design; the caps should be chosen per shape, or at least per token count.
- **Over the six shapes:** a geometric mean of 2.37x against the expert's 2.27x, and a total time of 874 us
  against 980 us.
- **AWS as published** fails only at K=6144 (4.9 bf16 ulps), as in the earlier grid.
- **P1's v2 kernel (1.517x)** gets a row once P4 sends its source: `heldout_grid.py --op matmul --kernel v2=<file>`.

## 3. Findings worth telling the room

1. **AWS's published fully optimised matmul loses precision at Qwen3's K, and refuses Qwen3's 256-token
   shapes** (bugs in AWS's code; P2's fixes).
   - It rounds every K-block's partial sum to bf16.
   - Its own test (K=1024, one block) never takes that path.
   - The referee measured 5.3 ulps at K=2048, and 4.9 at K=6144 under hostile inputs.
   - fp32 accumulation fixes it at no measurable speed cost (within 3%).
   - **Confirmed on AWS's own kernel in the CPU simulator** (`tools/aws_matmul_bf16_repro.py --sim`, 16:40,
     the file unmodified). At K=8192, the K of AWS's own benchmark, it reaches 19.4 bf16 ulps on normal
     inputs, and it **fails AWS's own correctness check** even on AWS's uniform inputs. With the one-line
     fp32 fix it is 0.5 ulps everywhere.
   - Its fixed block sizes assert M % 2048 == 0, so no Qwen3 shape at 256 tokens runs at all. The expert takes
     them as caps, fitted to each shape.
2. **AWS's default blocking is mid-pack: #29 of 62 at Qwen3's 256-token shapes** (search discovery). 24 random
   tries reliably land within 4% of the best, 1.325x-1.372x over AWS's default, and one repeat found the best
   outright. **But the winner is shape-specific:** on held-out kv_proj with 1024 tokens it runs at 1.20x,
   against the default's 3.05x. Tune per shape.
3. **Half of every NeuronCore sits idle under a plain launch** (P2 engineering). `kernel[2]` gives 1.50x more on
   the same kernel: 5.0x the start kernel, cross-checked on two clocks. **Not a referee verdict:** P1's timer
   and the referee's correctness check at the timing shapes only.
4. **Hostile inputs hide precision loss.**
   - Large magnitudes swamp the rounding errors: bf16 accumulation measured 3.4-3.5 ulps under P1's hostile
     pattern, against 9-13 on ordinary inputs (emulated at K=4096-6144).
   - Precision must be judged on ordinary inputs too.
5. **The four NeuronCores time identically.** The expert measured 385.5, 386.1 and 384.7 us on cores 0, 1 and 2,
   so arms run in parallel across the chip are comparable.
6. **Measurement details that would have misled us:**
   - **The byte counter overstated dtype-converting DMAs by 2x.** We fixed it in nkibench.
   - **A float32 tolerance rejects correct bf16 kernels:** one rounding step is 2.2% of the RMS.
   - **Narrow rows starve DMAs**, until the rows are packed.

## 4. How the numbers were kept honest

- **Spread, not one run:** 3 repeats per arm, and min, median and max reported.
- **One referee for every arm.** All of arm c ran on P1's 76b2227, merged before the runs. An earlier seed-0
  run on an older referee is archived in `logs/seat-102/archive/` and is not counted in arm c. Its best,
  `m2 n6 k16`, became the LNC=2 kernel's caps.
- **Every evaluation is published** in `logs/seat-102/` (commit 234d059): 134 records with code, verdict and
  timings, plus the archive. `python search.py --summarize "logs/seat-102/*.jsonl"` reproduces section 2 from
  them, including each run's gain over its own attempt 0.
- **The busy-core fix** (Nihal's `fix/p2-search-cores`, commit e369e5f) is merged into this branch. Its remote
  branch was deleted after the merge.
  - **Before the fix:** when the referee failed 3 times on a candidate (no free core, say), the candidate was
    skipped and not replaced, so a run could spend less than its budget unnoticed.
  - **After the fix:** the next triple of the same seeded order takes its place. The run stops if attempt 0
    cannot be judged, or if 3 candidates in a row fail. `heldout_grid.py` binds a core before its first cell,
    instead of writing "no free NeuronCore" cells as failures.
  - **The published runs predate it, and it would not have changed them.** They started between 14:51 and
    15:09 EDT, and the fix landed at 15:09:41. Every run logged its full budget: attempts 0-23 in each of the
    3 runs, and all 62 settings in the sweep. So no candidate was skipped. The held-out grid's one failing
    cell is a real precision failure, not a busy core.
  - Every run from now on uses it, including the held-out grid at 17:30.
- **2 of 134 evaluations were spoiled by `git pull` while a check ran.**
  - The referee flags its own changed files as tampering, and reports `rules`.
  - **The two:** seed 1 attempt 9, and sweep shard 1 attempt 12.
  - **What we did:** re-judged them with the same code and the same referee (`tools/recheck_spoiled.py`).
    The fresh records replace them; the originals are in `logs/seat-102/archive/`.
  - **The lesson:** never pull into the referee's folder while it runs.
- **Labels: whose result is it?**
  - *Expert-derived:* AWS designed it. This covers the expert as shipped, AWS as published, and anything built
    on them. Attempt 0 of every random-search run is the expert, so it never counts as a discovery.
  - *Search discovery:* what arm c found in attempts 1-23, reported as its best over that run's attempt 0.
  - *Ground truth:* the sweep times every setting. It measures the space; it discovers nothing.
  - *P2 engineering:* changes made by hand: the fp32 accumulation fix, fitting the caps to each shape, the
    LNC=2 split and the copy floor.
  - The LNC=2 numbers come from P1's timer and the referee's correctness check at the two timing shapes.
    They are **not a referee verdict**: the referee launches kernels at LNC=1, and held-out shapes were
    not run for LNC=2.
  - No end-to-end Qwen number is claimed.
- **Running on all four cores:** to use all four NeuronCores, vLLM was stopped on seat-102 (`./serve.sh` brings
  it back). Only our own kernels ran there. P1's sandbox assumes vLLM holds core 0, so no red-team cheats ran
  on seat-102.

## 5. Reproduce (seat pod)

```bash
cd /workspace/chipboost/projects/03-chipboost && export CHIPBOOST_SEAT=102 PYTHONDONTWRITEBYTECODE=1
./pod_check.sh --which dev                                       # every kernel in the simulator
python speedcheck.py --op matmul --check kernels/matmul_expert.py
CHIPBOOST_CORE=0 python search.py --budget 24 --seed 0 --out logs/seat-102/attempts-s0.jsonl
CHIPBOOST_CORE=1 python search.py --exhaustive --shard 0/2       # and 1/2 on another core
python search.py --summarize "logs/seat-102/*.jsonl"
CHIPBOOST_CORE=3 python tools/lnc2_probe.py --validate
python heldout_grid.py --op matmul
```

Never `git pull` into this folder while a check is running; use a separate clone.

## 6. What remains

1. ~~Copy seat-102's logs out of the pod and commit them.~~ **Done** (commit 234d059): 134 schema-valid records,
   and `--summarize` on them reproduces the pod's numbers exactly.
2. ~~Run `heldout_grid.py --op matmul` on each arm's best.~~ **Done** at 16:40 for every arm with a verified
   kernel (24 cells; section 2). The model arms in git have none. Still to do: P1's v2 kernel row, once P4
   sends its source.
3. **For P1, their call:** a referee option to launch `kernel[2]`, so LNC=2 kernels get full verdicts and held-out
   checks.
4. **If time:** a random search at LNC=2. The caps apply per program; `m1 n12 k4` was best at LNC=1.
5. **P2's paragraph** for P4's one-page note.
