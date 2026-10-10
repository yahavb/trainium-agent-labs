# 03 — attention kernels on Trainium2

Two pieces of work on the same NKI attention kernels, merged into one folder without duplicates:

- **03-attention-kernel** (seat-65): the single-head versions V0–V4, profiled with `compare_all.py`,
  and the tools around them. Everything from it is here unchanged, except `nkibench.py`, which gains
  one level.
- **trainium-agent-labs** `projects/02-kernel-agent` (seat-68): only what 03 did not already have.
  That is the hardware-DMA-descriptor versions V5 and V6, multi-head attention, the Qwen3-8B prefill
  kernel patch, the matmul comparison, and all of seat-68's results.

The results from both are summarised in **[COMBINED.md](COMBINED.md)**, generated from
`results/combined.json`. **[RESULTS.md](RESULTS.md)** is 03's own seat-65 write-up and is left as it
was.

## Kernels

| Version | File | What it changes | Measured on |
|---|---|---|---|
| V0 | `my_attention_v0.py` | Baseline. | both |
| V1 | `my_attention.py` | V0 rescheduled, same maths. | both |
| V2 | `my_attention_v2_bf16.py` | V1 with P transpose and P @ V in bf16. | both |
| V3 | `my_attention_v3_dmaT.py` | V1 with Q and K transposed by the DMA engine. | seat-65 |
| V4 | `my_attention_v4_kchunk.py` | V1 with the back half in 32-key chunks. | seat-65 |
| V5 | `my_attention_v5_hwdge.py` | V1 with every DMA on hardware descriptors. seat-68 called it V3. | seat-68 |
| V6 | `my_attention_v6_hwdge_bf16.py` | V5 plus V2's bf16 P @ V. seat-68 called it V4. | seat-68 |
| M0 | `mha_v0_loop.py` | Multi-head (level 9): V0's algorithm once per head, one core. | seat-68 |
| M6 | `mha_v6_loop.py` | Multi-head: V6 once per head, one core. seat-68 called it M4. | seat-68 |
| fast | `mha_fast.py` | Multi-head across both cores of an LNC 2 launch, DMAs in chunks of heads, batched reductions. Its other three entry points are the experiments tried on top of it. | seat-68 |
| — | `attention_cte_hwdge.py` | nkilib's `attention_cte` (the kernel vLLM-neuron runs for Qwen3-8B prefill) with hardware descriptors on its Q, K and V loads. | seat-68 |
| — | `matmul_parallel.py`, `reference_level4.py` | Tiled matmul: an "all inputs resident" version against the level-4 reference. | seat-68 |

**How overlap was removed:**
- **Same code in both folders, kept once:** V0, V1, V2 and `matmul_parallel.py`. Each 03 file is
  identical in code to seat-68's copy, or differs only in comments and line order. Where seat-68 used
  a reconstruction of V0 rebuilt from V1's docstring, the original from 03 is kept.
- **Near-copies folded into one file:** the two experiment variants of the fast kernel were separate
  files differing by a few lines. They are now entry points of `mha_fast.py`, switches on one shared
  body.
- **Renumbered, not renamed away:** seat-68's V3 and V4 used different techniques from 03's V3 and V4,
  so they became V5 and V6.
- **Code otherwise unchanged:** V5, V6, M0 and M6 are byte-for-byte the code that was measured; only
  their docstrings changed. `mha_fast.py` changed shape to hold the variants, so it was re-run on
  seat-68 after the merge: correct on all four shapes, 2.188x over M0 at LNC 2, inside the earlier
  runs' 2.187–2.193x.

## Tools

| Tool | From | Covers |
|---|---|---|
| `bench_device.py` | 03 | Single-head kernels on the device: correctness against float64, then the runtime's device benchmark. Any `file[:entry]`. |
| `compare_all.py` | 03 | V0–V4 on every level-8 shape, 5 profiled runs each; writes RESULTS.md. |
| `profile_device.py`, `profile_steps.py`, `calibration.py`, `compare_torch.py`, `verify_sdk.py`, `probe_nki.py`, `identify_neffs.py`, `qwen3/` | 03 | Device and simulator profiles, fixed-cost calibration, torch comparison, SDK checks, Qwen3-8B serving benchmark and stack probe. |
| `nkibench.py` | 03, plus level 9 | The CPU-simulator checker. Level 9 (multi-head attention, `[seq, heads, dim]` inputs) comes from seat-68. Nothing else differs, and `--selftest` passes. |
| `bench_suites.py` | seat-68 (was its `bench_device.py`) | The suites `bench_device.py` here does not cover: `matmul` (level 4) and `mha` (level 9, where a version can pin LNC 1 or 2). Its single-head suite was left out. |
| `bench_qwen_attention.py` | seat-68 | nkilib `attention_cte` against `attention_cte_hwdge.py` at Qwen3-8B's prefill shapes. |
| `combine_results.py` | new | Reads every results file below and writes `results/combined.json` and COMBINED.md. Measures nothing. |

Run in a seat pod from this folder. The model server holds NeuronCores 0–1, so pass `--lnc 1` for
single-head kernels:

```bash
python nkibench.py --level 9 --check mha_fast.py                     # simulator, any machine with nki
python bench_device.py my_attention.py my_attention_v5_hwdge.py my_attention_v6_hwdge_bf16.py --all-shapes --lnc 1
python bench_suites.py mha                                           # multi-head, timed and profiled
python bench_suites.py matmul
python bench_qwen_attention.py                                       # Qwen3 prefill kernel, 512 / 2048 / 8192 tokens
python combine_results.py                                            # refresh combined.json and COMBINED.md
```

## Results

```
results/
  compare_20261010_210528.json      seat-65, compare_all.py (unchanged; RESULTS.md is rendered from it)
  seat68/
    attention/run-*/results.json    single-head V0-V2, V5, V6; run-20261010-204410 and -204707 are the full runs
    mha/run-*/results.json          multi-head; run-20261010-213836 is the last complete run
    matmul/run-*/results.json
    qwen3_attention/run-*/results.json
    */run-*/<version>_case<n>/metrics.json   neuron-explorer summary per version and shape, for profiled runs
    logs/                           the console output of each run, including bench_device-V1-V5-V6.log,
                                    03's bench_device.py re-timing V1, V5 and V6 on seat-68 after the merge
    traces/                         Perfetto traces (open at ui.perfetto.dev), V0 reconstruction and V6 at seq 128 dim 64
  combined.json                     all 232 records, one schema
```

**The two seats were measured differently, so their numbers are never compared with each other.**
- **seat-65** reports neuron-explorer's `total_time` for one profiled execution, the median of 5
  captures.
- **seat-68** reports the runtime's device-benchmark mean over 100–2,000 iterations.

The profiled single execution runs several microseconds higher, and it ranks hardware-descriptor
kernels wrongly. That is why V5 and V6 are timed with `bench_device.py` or `bench_suites.py`, not
`compare_all.py`. Every speedup in COMBINED.md divides by a baseline from the same seat and the same
run, and every record in `combined.json` carries its metric.

seat-68's logs use the labels from when they were written ("V3 hwdge", "V4 hwdge+bf16", "M4 V4 per
head"). `combined.json` maps each to the version here.

## What did not work

These were measured and are kept as records:
- **bf16 P @ V (V2, V6)** misses nkibench's tolerance at 16 heads: the worst error is 0.0245 of the
  output RMS against 0.02. This shows in `mha_v6_loop.py`; `mha_fast.py` keeps P @ V in float32.
- **TF32 matmuls** (`mha_fast.py` entry points `nki_mha_tf32_*`) pass the CPU simulator but do not
  compile for the device with nki 0.6.0 / neuronx-cc 2.27.
- **Moving work onto GpSimd** fails: it cannot read PSUM. Moving the same work to Vector instead
  (`nki_mha_balanced_`) gives no gain.
- **Hardware descriptors in Qwen3-8B's prefill kernel** give 1.028x at 512 tokens and 1.014x at 2,048,
  but 0.987x at 8,192, with one run per length. That kernel is not swapped into the model server.
- **Per-head DMAs on hardware descriptors** (M6 at 32 heads) are slower than V0's loop: each trigger
  costs about 0.6 µs on Sync. Chunking the DMAs, as `mha_fast.py` does, is what makes them pay.
