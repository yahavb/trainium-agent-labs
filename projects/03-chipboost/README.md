# CHIPBOOST: Qwen3 speeds up its own kernels, and we prove every speedup is real

*Hack the Chip, NYU x Annapurna Labs, Oct 10 2026*

> Qwen3-8B, running on a Trainium chip, rewrites the kernels it is built from (matmul, RMSNorm) to run faster
> on that same chip. A strict referee proves every version is still correct on shapes the agent never saw,
> times it on the real silicon, and names one change. And because the event's whole thesis is "the checker
> decides whether the loop works", we attack our own referee to prove it can't be fooled.

---

## 1. Why this, out of everything we considered

| Idea | Verdict | What we keep |
|---|---|---|
| Fix crash walls in Project 2 (feedback rules) | Solid but small; "just Project 2" | Verdict -> instruction feedback, `enrich()` |
| Qwen3-layer levels | Engineer: "that's Project 2" | Real Qwen3 ops at real Qwen3 sizes |
| FORGE v1 (end-to-end Qwen +10%) | Integration into served Qwen unlikely today; 10% target risky | Amdahl estimate, "correctness before speed" |
| FORGE v2 (performance cliffs) | Best original idea; full scope too big | Held-out and tile-edge shapes; speedups must survive them |
| FORGE-lite (on-chip timing) | The missing piece in the repo | On-chip timing as referee layer 2 |
| **CHIPBOOST** | Best structured, most complete | The backbone of this plan |

**What we checked before deciding (facts, with sources):**

- **"Can an LLM agent speed up Trainium kernels?" is already answered yes.** Amazon's AccelOpt (ICLR/MLSys
  2026) raised average peak throughput on NKIBench kernels from 45% to 59% on Trainium 2, and open-source
  models matched Claude Sonnet 4's gains at ~26x lower cost
  ([arXiv:2511.15915](https://arxiv.org/abs/2511.15915)). So "AI makes kernels faster" alone is not novel.
  Our uniqueness has to come from elsewhere (section 3).
- **The pods run Neuron SDK 2.32, which ships NKI 0.6.0** (container
  `pytorch-inference-vllm-neuronx:0.24.0.1.1.0-neuronx-py313-sdk2.32.0`, in the repo's `k8s/` manifests).
- **NKI 0.6.0 has no public timing API.** `nki.benchmark` is gone; a plain `kernel(*args)` call recompiles
  each time (~1.5 s, i.e. you'd be timing the compiler, not the kernel). The internal route is
  `ParserFrontend().compile()` -> `CompiledKernel.from_frontend()` -> `.benchmark()`, which returns
  mean/min/max/std (nki-samples issue #134, PR #133). **On-chip timing is the #1 risk and the first thing to
  ask the engineers.**
- **Measurement can lie, silently.** On SDK 2.32, the legacy `baremetal()` harness returns all-zero output
  with exit code 0 for correct kernels (issue #134). A referee that times a kernel without re-checking its
  on-chip output could "measure" a speedup of a kernel that computes nothing.
- **The repo's byte counter only sees `nisa.dma_copy`** (`simulate_and_count`, `nkibench.py:611`). NKI also
  has `dma_transpose` and `dma_compute`; a kernel moving data another way would look like it moves fewer bytes
  than it does.
- **Every repo helper CHIPBOOST relies on exists:** `check_rules` (l.358), `describe_mismatch` (l.421),
  `reuse_report` (l.548), `check_traffic_bar` (l.570), `check_inputs_untouched` (l.596),
  `simulate_and_count` (l.611), `explain_with_ceiling` (l.72), `level(...)` (l.229), level 8 example
  (~l.326). The repo's own `STATE.md` lists "layer 2: real latency" and `NEURON_RT_VISIBLE_CORES=0,1` as the
  next things to build.

## 2. What we build

```
Qwen3-8B (cores 2-3) --> candidate kernel --> REFEREE --> score + ONE named change --+
       ^                                                                             |
       +-------------------------------- next attempt <------------------------------+
```

```
REFEREE (speedcheck.py), in order, stops at first failure:
  1. Rules        static scan (repo's check_rules)
  2. Correct      simulator vs NumPy on dev shapes, incl. ragged + hostile values
  3. Correct      ON THE CHIP, same check (catches silent-zero harness failures)
  4. Fast         on-chip timing on cores 0-1, interleaved A/B with the baseline, noise threshold
  5. Robust       held-out shapes the loop never saw (tile edges: 127/128/129, 511/512/513...)
  6. Why slow     bytes vs floor, transfer count, intensity vs ridge -> ONE instruction
```

**Score:** 0 if wrong anywhere (including held-out shapes). Otherwise speedup over the start kernel and share
of the expert kernel's speed reached.

**Kernels** (Qwen3-8B sizes; verify hidden 4096 / intermediate 12288 / eps 1e-6 in the pod's `config.json`):

- **Matmul** (start: `reference_level4.py`; expert ceiling: the NKI tutorial's fully optimized matmul)
- **RMSNorm** (memory-bound; the lesson is "read each byte once")
- **Stretch: SwiGLU** (the fusion move is the win to find)

## 3. Our unique qualities (achievable today)

- **Self-referential and small.** An 8B model optimizing the kernels it is made of, at its own real sizes, on
  the chip it is served from. AccelOpt's abstract describes larger setups optimizing a benchmark suite; check
  the full paper before claiming more.
- **A referee that is attacked, not just trusted.** A referee red-team suite of planted cheats it must catch
  (section 4). Reward hacking in kernel generation is a known industry problem; showing a Trainium referee
  that catches each known trick, with a pass/fail table, is the artifact the organizers say they want to keep.
- **Equal-budget, three-arm comparison on matmul:** (a) model + referee, (b) model alone ("make it faster"),
  (c) random/grid search over tile sizes, no AI. If random search wins anywhere, we say so. (AccelOpt's
  abstract compares models, not against non-AI search.)
- **Speedups must survive held-out and tile-edge shapes.** Borrowed from FORGE v2: a speedup that only holds
  at friendly shapes, or breaks at 129 rows, doesn't count. We show a small before/after map.
- **Honesty by construction.** Every number labelled simulator / chip / projection; rates over repeats with
  spread; the agent reports when it failed.

## 4. The referee red-team suite (built in the morning, before any agent run)

Each is a tiny hand-written "cheating" kernel. The referee must reject every one with a clear message.

| # | Cheat | Which referee check catches it |
|---|---|---|
| 1 | Returns zeros / uninitialized output | correctness (sim + chip) |
| 2 | Writes into its input and returns it | `check_inputs_untouched` |
| 3 | Correct at dev shapes, wrong at a ragged last tile | held-out tile-edge shapes |
| 4 | Hands the whole op to NumPy/framework | rules scan |
| 5 | Moves data with a DMA call the byte counter doesn't hook | extended byte counting (all DMA ops) |
| 6 | Accumulates in lower precision to look faster | per-dtype tolerance, stated and justified |
| 7 | "Faster" only by timing noise | interleaved A/B + noise threshold |
| 8 | Timing includes compile time (the 1.5 s trap) | compile excluded, warm-up, sanity check vs empty kernel |

**Report:** "referee catches 8/8" (or honestly which it misses, and why).

## 5. Who runs what (5 seats, each self-contained; share via git)

| Seat | Morning | Afternoon |
|---|---|---|
| **104** (infra) | Gate: get on-chip timing working on cores 0-1 (ask engineer: `CompiledKernel.benchmark()` vs `neuron-profile`); measure noise on `reference_level4.py`. Push `speedcheck.py`. | Quiet timing chip if noise is high; repeat runs for spread |
| **103** (referee) | Write the 8 cheat kernels; make referee catch all; held-out shape sets | Failure taxonomy from all `attempts.jsonl`; cliff map before/after |
| **100** | Matmul at Qwen3 sizes passes the referee as a start kernel | Main loop: model + referee, `--repeat` |
| **101** | Random/grid tile search script (no AI) | Arms (b) and (c) on the same budget |
| **102** | Register RMSNorm level, correct start kernel | Main loop + model-alone arm; SwiGLU if time |

## 6. Timeline

| Time | Milestone |
|---|---|
| 12:00 | Pitch + timing question to engineer; all seats `./serve.sh`; fork repo |
| 13:30 | Gate: on-chip timing works? Red-team cheats written; start kernels pass |
| 15:00 | All loops running; first real speedup or honest "none yet" |
| 17:30 | Repeats done; three-arm table; red-team table; held-out map |
| 18:30 | Stop building. Note, demo, PR |

## 7. Risks and fallbacks

- **On-chip timing doesn't work by 13:30** -> referee layer 4 uses the simulator's bytes/intensity (labelled
  "simulator estimate"). Everything else stands: red-team suite, three arms, held-out shapes.
- **Timing too noisy beside vLLM** -> seat 104 stops its model server and becomes the quiet timing chip;
  others push candidates via git for batch timing.
- **Agent never beats the start kernel** -> the three-arm comparison and failure taxonomy are the result.
- **Engineer says "still Project 2"** -> fine: our contributions (red-team referee, on-chip layer, fair
  baselines, held-out robustness) are exactly what Project 2 is missing.

## 8. Deliverables

- **Checker:** `speedcheck.py` + red-team suite, every accept/reject and tolerance justified
- **Attempt logs:** `attempts.jsonl` from every seat (prompt, code, correctness, timing)
- **One-page note:** what ran where, runs and spread, simulator vs chip vs projection
- **Demo:** a fast-but-wrong kernel rejected, then a real speedup accepted, and the one-line instruction that
  caused it; the red-team table; the three-arm chart
- **PR** from our fork

## 9. Pitch (30 seconds)

> "AI can already speed up Trainium kernels; AccelOpt showed that. What isn't solved is trusting the result.
> On our chips, Qwen3-8B speeds up its own matmul and RMSNorm. Our referee checks correctness on the chip and
> on shapes the agent never saw, times on the silicon, and we red-team it with eight cheating kernels to
> prove it can't be fooled. We compare against the model alone and random search on an equal budget."

**Ask the engineer:** (1) the supported way to time an NKI kernel on device in SDK 2.32 / NKI 0.6.0; (2) does
running on cores 0-1 beside vLLM disturb timing; (3) how projects are judged.
