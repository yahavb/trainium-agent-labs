# Chip and Guac (#26) - Feedback-Driven NKI Agent

**Chip and Guac (#26)**

**Historical Best Scores: Level 2 (Transpose) 1.000 | Level 4 (Tiled Matrix Multiplication) 1.000**

**Model:** Qwen3-8B

**Hardware:** AWS Trainium2-hosted model inference

**Kernel validation:** Original NKI CPU simulator and checker

**Contribution:** Shape-Curriculum Feedback plus targeted SDK-verified repair guidance for Level 4

## Overview

We enabled Qwen3-8B to generate a Level 4 tiled matrix multiplication kernel
that passed all four official shapes under the unchanged original checker.
Our innovation is in the feedback loop: prioritize simpler failing shapes, then
translate selected NKI errors into actionable repairs. We did not fine-tune the
model, modify evaluation rules, or insert a reference answer.

Our earlier original-agent baseline experiments also achieved **1.000 on
Level 2 (Transpose)**. That historical result is separate from Arm C's Level 4
achievement; Arm C was not separately evaluated on Level 2.

## Key Results

### Historical Best Scores Across Two Levels

| Level | Best official reward | Provenance |
|---|---:|---|
| Level 2 (Transpose) | **1.000** | Earlier unchanged original-agent baselines; solved in 3/5 runs |
| Level 4 (Tiled Matrix Multiplication) | **1.000** | Enhanced Arm C; independently validated with the unchanged original checker; 4/4 official shapes passed |

These are historical best scores from different experiments, not results from
one matched two-level evaluation. The earlier baseline Level 2 rewards were
0.300, 0.300, 1.000, 1.000 and 1.000; the archived 404-candidate audit reproduced
the original rewards, scoring components and feedback exactly. Arm C's
contribution and the final matched B/C experiment below concern **Level 4 only**.

### Final Matched B/C Experiment: Level 4 Only

| Metric | Shape Curriculum (B) | Enhanced Agent (C) |
|---|---:|---:|
| Official Level 4 reward | 0.625 | **1.000** |
| Official shapes passed | 1/4 | **4/4** |
| Generation rounds | 8 | **5** |
| Model calls | 32 | **20** |
| Total tokens | 35,413 | **26,238** |
| Wall time | 646.71 s | **484.84 s** |
| Final outcome | Round cap | **Solved** |

C used **37.5% fewer calls, 25.9% fewer tokens and 25.0% less wall time** in this
one matched comparison. These are measured observations, not statistically
established average improvements. Wall time includes generation and CPU grading,
not device kernel latency.

| Official shape (K, M, N) | Control B | Enhanced C |
|---|---|---|
| (128, 128, 512) | Pass | **Pass** |
| (256, 256, 1024) | Fail | **Pass** |
| (512, 128, 512) | Fail | **Pass** |
| (256, 512, 1024) | Fail | **Pass** |

## How We Made the Agent Succeed

### Shape-Curriculum Feedback

The original checker reports its first collected failing shape. We select the
failure with the fewest dimensions exceeding nominal tile limits: K > 128,
M > 128 and N > 512; ties retain official shape order. This prioritizes K-only
tiling before combined K/M/N tiling. Tests still execute in their original
order, and rewards, references, mutation checks, numerical validation, traffic
checks and hardware-hazard gates are unchanged. Early checker exits are retained.

### Targeted NKI Repair Guidance

The installed SDK (`0.6.0+31049202112.g85070674`) can report:

```text
Matmul stationary free dimension 256 exceeds gemm_stationary_fmax=128
```

The original agent lacks a dedicated handler for that free-dimension error.
Our conditional guidance explains K chunks <=128, stationary free extent <=128,
moving free extent <=512, M/N output tiling, a separate FP32 PSUM accumulator
per output tile, first-write overwrite followed by K accumulation, and completed
output copies through SBUF to the appropriate HBM row and column offsets.

Only an assertion from the actual SDK validator during nc_matmul, belonging to
the selected failure, triggers guidance. The moving matcher uses the verified
gen2/gen3 wording `exceeds max ... for nc_version=...`, not a guessed legacy
message. Other errors are untouched. No complete tiled implementation or
reference kernel is supplied to the model.

## How the Successful Repair Happened

1. Qwen initially handled only the smallest shape.
2. Curriculum feedback directed repairs toward K tiling.
3. The model added K-chunked DMA and matmul accumulation.
4. It reached 0.750, then encountered the stationary free-dimension error.
5. Targeted guidance described the required multidimensional repair structure.
6. The next response added M/N output loops, per-output-tile PSUM,
   `accumulate=(k0 > 0)`, and correctly offset writeback.
7. The original checker accepted every official shape at reward **1.000**.

C reached 0.750 **before the new targeted guidance was delivered**. The next
guided repair increased reward to 1.000. At 20 calls per arm, B remained at 0.625
and C reached 1.000. This trace supports feasibility, not a causal solve-rate
estimate. See [the actual repair diff](results/guided_repair.diff).

## Experimental Development

| Approach | Best demonstrated Level 4 reward |
|---|---:|
| Original agent baseline | 0.625 |
| Shape-Curriculum Feedback | 0.750 |
| Curriculum + targeted repair guidance | **1.000** |

These are historical best observations from separate runs, not one matched
three-arm ablation. We began with repeated baselines, analyzed recurring NKI
errors, and improved feedback ordering and actionability. The fresh matched B
control scored 0.625, illustrating variation between runs.

## Validation and Evidence

| Evidence | Result |
|---|---|
| Original-agent baseline | Five complete runs |
| Baseline independent regrading | 404/404 reward, parts and feedback parity |
| Arm C archived replay | 448/448 reward and scoring-component parity |
| Disabled guidance reproduces B | 448/448 feedback parity |
| Targeted feedback delta | 15 historical selected stationary errors enriched; 433 other messages unchanged relative to B |
| Moving handler | Verified with an isolated installed-SDK test |
| Winning kernel | Original agent.grade called directly in a fresh process; 4/4 shapes passed |
| Portable submission tests | 17/17 passed in the Neuron environment |
| Canonical benchmark self-test | Passed |

**Official scoring, reference kernels, numerical tolerances, evaluation shapes
and model-server settings were not changed.**

The agent reuses the sibling [original project](../02-kernel-agent/), not a copied
checker. [source_manifest.json](source_manifest.json) pins its seven runtime files
with LF-normalized hashes. The evaluated upstream revision is
`8f1ca418827a3b4adc88a766262d065b639b7d46`.

The compact [summary](results/summary.json), [52 sanitized attempt records](results/attempts.jsonl)
and [one-page findings](results/findings.md) preserve the observed experiment.
The [winning example](examples/winning_kernel.py) is exported model output, never
injected as an agent answer. No raw HTTP bodies, credentials or SDK installation
files are committed.

## Running the Agent

Run from the repository root on Linux with NumPy, httpx and the installed Neuron
NKI SDK. The verified SDK environment used Python 3.13 and the NKI version above.
The SDK CPU simulator is required for grading; a Trainium-hosted Qwen service or
another explicitly supplied compatible endpoint is required for live generation.
Installing the Python dependencies alone does not install Neuron.

```bash
python -m pip install numpy httpx
python projects/02-kernel-agent/nkibench.py --selftest
python -m unittest discover -s projects/chip-and-guac-nki-agent/tests -v
```

Without the SDK, the 15 pure tests run and the two integration tests are explicitly
skipped. Full simulation tests require the hackathon/Neuron environment.

Connect to an already healthy Qwen server when available. On a newly prepared
seat only, the organizers' actual server script accepts:

```bash
MAX_MODEL_LEN=8192 TP=2 ./serve.sh
```

Do not restart an existing server. The evaluated endpoint was localhost port
8000, with four simultaneous sequences and thinking disabled.

Curriculum-only B, in a fresh output directory:

```bash
python projects/chip-and-guac-nki-agent/agent.py \
  --approved-live --output-dir projects/chip-and-guac-nki-agent/runs/B \
  --feedback-policy curriculum \
  --level 4 --rounds 8 --samples 4 --context 8192 --max-tokens 2500 \
  --give-up-after 4 --terse 0 --repeat 1 \
  --base http://localhost:8000/v1 --model Qwen/Qwen3-8B
```

Enhanced C, after B has ended:

```bash
python projects/chip-and-guac-nki-agent/agent.py \
  --approved-live --output-dir projects/chip-and-guac-nki-agent/runs/C \
  --feedback-policy curriculum --free-dimension-guidance \
  --level 4 --rounds 8 --samples 4 --context 8192 --max-tokens 2500 \
  --give-up-after 4 --terse 0 --repeat 1 \
  --base http://localhost:8000/v1 --model Qwen/Qwen3-8B
```

The guidance flag defaults off. `--feedback-policy official` restores original
first-failure selection. Original agent flags are forwarded unchanged; no seed,
temperature or retry modification is introduced. The canonical request uses
temperature=0.6 and top_p=0.95. The canonical context-based token-budget rule is
retained. `--approved-live` acknowledges live inference; `--offline` reference
replay is intentionally not supported by this submission wrapper.

Directories are exclusive: choose new names for subsequent runs. Inspect
`attempts.jsonl` for code/rewards/feedback, `grades.jsonl` for shape/gate outcomes,
and `requests.jsonl` for full runtime request hashes, response metadata, usage
and timestamps. Keep runtime logs private and out of Git. A process guard and
shared advisory lock reject other known agents/graders; the model server may
remain running. Candidates use private temporary paths.

Independently call the original checker, with only candidate-file I/O isolated:

```bash
python projects/chip-and-guac-nki-agent/verify_kernel.py \
  projects/chip-and-guac-nki-agent/examples/winning_kernel.py --level 4
```

This verified command reports reward 1.0 and all scoring parts true; it exits
nonzero if correctness fails. It does not run inference or use feedback wrappers.

## Limitations and Future Work

The 1.000 result is **official CPU-simulator correctness**, not physical-device
compilation or measured Trainium2 kernel latency. The winning example retains
full-K SBUF backing allocations with suspicious partition extents. Device
legality and efficiency need further validation; CPU acceptance is not a
hardware certification. A single matched B/C pair cannot estimate success rates;
repeated controlled trials would be needed.

## Team and Credits

**Chip and Guac (#26)**

**NYU x Amazon Annapurna Labs Hack the Chip 2026**

| Member | Email | Resume |
|---|---|---|
| Sai Rithwik Amajala | sa9880@nyu.edu | [PDF](resumes/Sai_Rithwik_Amajala_Resume.pdf) |
| Saishruti Sairam Vedha | ss20513@nyu.edu | [PDF](resumes/Saishruti_Sairam_Vedha_Resume.pdf) |
| Akhil Vardan Mallipeddu | am16362@nyu.edu | [PDF](resumes/Akhil_Vardan_Mallipeddu_Resume.pdf) |
| FNU Karuna Venkatesh | fk2496@nyu.edu | [PDF](resumes/Karuna_Venkatesh_Resume.pdf) |
| Vinay Kumar | vk2771@nyu.edu | - |

Thanks to the organizers for the original benchmark and agent framework.
Sample code is reused under the repository's event reuse terms; Qwen retains
its own model license.

Official references: [AWS NKI tiling constraints](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/get-started/about/tiling-overview.html),
[current ISA contracts](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_modules/nki/isa.html),
[tile-size constants](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/nki.language.tile_size.html)
and [current loop APIs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_modules/nki/language.html).
