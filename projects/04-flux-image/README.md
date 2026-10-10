# 04 — FLUX.1-dev image generation on Trainium

Generates images with FLUX.1-dev (1024×1024, 25 steps) on a seat pod's Trainium chip using AWS's NxD Inference Flux code.

```
./run.sh --prompt "A robot named trn2" --num 4
```

`generate.py` prints latency, ms/step and images/s and writes `out/metrics.json`. While it runs, `neuron-monitor` records `out/neuron-monitor.jsonl`, and `neuron_summary.py` summarises it.

Flags that define an experiment: `--tp` (NeuronCores), `--cp` (context parallel), `--cc-opt` (neuronx-cc `-O` level for the transformer), `--nki` (fused QK-RMSNorm + RoPE + flash-attention NKI kernel, in [nki_kernels/](nki_kernels/)), `--warmup`. Each recompiles into its own folder under `/workspace/flux-compiled/`.

## Metrics so far

### Baseline: stock NxDI path, TP=4, 1024², 25 steps, 4 images

Source: `out/metrics.json` and `out/neuron-monitor.jsonl` (same files in `flux-out/` locally and on seat-185) (the first image is excluded from the steady-state numbers).

| Metric | Value |
|---|---|
| Config | `--tp 4`, no CP, `-O1`, no NKI |
| Latency, per image | 7.30 / 7.49 / 7.56 / 7.65 s |
| Latency, steady-state mean (images 2–4) | 7.57 s |
| Latency p50 / max | 7.56 s / 7.65 s |
| Time per denoising step | 303 ms |
| Throughput | 0.132 images/s (7.9 images/min) |
| Compile (cached) | 1.1 s |
| Model load | 50.3 s |
| NeuronCore utilisation, mean (all 4 cores) | 35.3 % (peak 97.9 %) |
| NeuronCore utilisation while busy (>0 %) | 54.6 % |
| Device memory, peak | 47.7 GiB |
| Per-execution latency, p50 / p99 (mean over samples) | 210.1 ms / 226.5 ms |

### Experiments (seat-185): same prompt, 1024², 25 steps, 4 images

Source: `exp/*.json` and `exp/*.log` on seat-185 (`/workspace/04-flux-image/`), run by `exp/run_exps.sh`. Steady state excludes the first image. Neuron-monitor was off for the O2 and CP runs (`NO_MONITOR=1`); the NKI run was launched separately, with the monitor on.

| Run | Flags | Mean latency | p50 | ms/step | Images/min | Compile | Load |
|---|---|---|---|---|---|---|---|
| Baseline | `--tp 4` | 7.57 s | 7.56 s | 303 | 7.93 | 1.1 s (cached) | 50.3 s |
| O2 | `--tp 4 --cc-opt 2` | 7.61 s | 7.60 s | 304 | 7.88 | 771 s | 48.2 s |
| Context parallel | `--tp 2 --cp` (4 cores, CP=2) | 7.38 s | 7.36 s | 295 | 8.13 | 846 s | 91.3 s |
| Fused NKI attention | `--tp 4 --nki --warmup 1 --num 3` | 8.38 s | 8.38 s | 335 | 7.16 | 276 s | 45.3 s |

- `-O2` gives no speedup (+0.6% latency, within run-to-run noise) for about 13 minutes of extra compile.
- TP=2 with context parallel is about 2.5% faster than the TP=4 baseline. That is a small gain, and it is a single 4-image run.
- The fused NKI attention is about 11% slower end to end (8.38 s vs 7.57 s, +32 ms per step). That matches the microbenchmark, which predicted about +37 ms per step. Its three images were timed after one untimed warmup generation, so all three count toward the mean.

### NeuronCore utilisation and memory, baseline vs NKI

Both from neuron-monitor (output pasted from the NKI run's terminal; the baseline is `out/neuron-monitor.jsonl`).

| Metric | Baseline | Fused NKI |
|---|---|---|
| NeuronCore utilisation, mean (all 4 cores) | 35.3 % | 35.0 % |
| Utilisation peak | 97.9 % | 100.0 % |
| Utilisation while busy (>0 %) | 54.6 % | 52.6 % |
| Device memory, peak | 47.7 GiB | 46.8 GiB |
| Per-execution latency p50 (mean over samples) | 210.1 ms | 230.5 ms |
| Per-execution latency p99 (mean over samples) | 226.5 ms | 235.1 ms |

Utilisation and memory are essentially unchanged. Only the per-execution latency moved, up about 10%.

### NKI attention microbenchmark

[nki_kernels/bench_attention.py](nki_kernels/bench_attention.py), per-rank attention shape (S=4608, 6 heads, D=128, LNC=2), from `/tmp/nki_flux/bench.log` on seat-185. "Per step" is the per-call time times 57 blocks.

| Variant | ms / attention call | ms / step (57 blocks) |
|---|---|---|
| baseline (NxDI glue + nkilib `attention_cte`) | 1.840 | 104.9 |
| fused (`flux_qknorm_rope_attention`) | 2.491 | 142.0 |
| `attention_cte` core only | 1.341 | 76.4 |

The fused kernel is about 35% slower than the baseline path in this microbenchmark. The kernel core alone (1.34 ms) is faster than both, so the gap comes from the fused kernel's own QK-norm and RoPE work.

### Device profile (neuron-explorer, transformer NEFF)

Source: `/workspace/ne-profile/profiles/global/flux-transformer@latest` on seat-185, queried with the scripts in `/workspace/analysis/` (`te.py`, `te2.py`, `dma.py`, `s5.py`, `spill.py`, `spill2.py`, `w.py`). One transformer execution on the two physical cores of one logical NeuronCore, window 273 ms. Both cores behave the same, so numbers are per core. Peak figures come from the profile's metadata: 78.6 TFLOPS on the tensor engine, 435 GB/s DMA, 716 GB/s HBM.

| Metric | Value |
|---|---|
| Instructions traced | 6.33 M (tensor 4.66 M, vector 0.60 M, scalar 0.50 M, sync 0.42 M, gpsimd 0.15 M) |
| Tensor-engine active time | 183 ms of 273 ms (67%) |
| Achieved tensor throughput | 55.5 TFLOPS = 70.5% of peak (ideal 129 ms, gap 54 ms) |
| Transposes | 5% of tensor FLOPs, 6.6 ms |
| Tensor-engine matmul issue interval | 221 ms measured vs 129 ms ideal, so 92 ms of excess per core |
| DMA bytes moved | 45.3 GB per core |
| DMA achieved bandwidth | 234 GB/s (54% of 435 GB/s) |
| DMA memory-bound time | 194 ms (ideal at peak 104 ms, gap about 90 ms) |
| Spill traffic (Scalar + GpSimd + Sync DMA) | save about 10.1 GB, reload about 15.6 GB |

What it shows:

- **Matmul size matters.** Matmuls with N=512 run at 74% of ideal rate, N=128 at 39% and N=64 at 36%. The N=128 and N=64 matmuls add about 34 ms of the 92 ms excess.
- **Attention is a big share of the tensor work.** `attention_cte.py:3887` alone is 141 ms of tensor-engine time, all N=128 matmuls at 41% efficiency. `attention_cte.py:3611` is N=512 at 52%.
- **Weights are re-read from HBM.** 16.2 GB per core is loaded from the inputs into SBUF, against 5.35 GB of actual tensor data. The worst inputs (18.9 MB each, 50 of them) are reloaded 16 times, about 15 GB in total.
- **Many small DMAs.** Hardware-dynamic DMAs under 2 KB (about 23.5 M packets) move 11.4 GB at 12 to 18 B/ns per engine, against 21 to 22 B/ns for 4 to 16 KB packets.
- **Transpose DMAs.** 14.5 GB of the traffic is SBUF-to-SBUF transpose-mode DMA.

Caveats:

- The profile has three warnings: DMA block notifications were dropped (some DMA data may be wrong), the NEFF lacks compiler metrics, and HLO FLOP stats are missing, so MFU from HLO is unavailable. DMA packet coverage is 79% (71.6 of 91.1 GB).
- Ingest logged "DGE packet count exceeds number of DMA trace entries" for several DMA engines.
- A separate profiling run in `prof/run.log` hit `NRT_EXEC_SW_NQ_OVERFLOW`.
- The spill and reload totals are my sums from `s5.py`; they do not separate spills from ordinary intermediate traffic.

### In progress / not yet recorded

| Experiment | Status |
|---|---|
| Neuron-monitor utilisation for O2 and CP | Not captured (only the baseline and NKI runs have it). |
| Earlier 4-image run before the baseline | Metrics overwritten. |
