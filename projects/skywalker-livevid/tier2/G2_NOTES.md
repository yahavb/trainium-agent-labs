# G2 notes: quick facts

Paths are relative to `repos/StreamDiffusionV2` (commit `6961a5c`). `pipe` = `models/wan/causal_stream_inference.py`,
`inf` = `streamv2v/inference.py`, `cm` = `models/wan/causal_model.py`, `tw` = `models/wan/taehv_wrapper.py`,
`ww` = `models/wan/wan_wrapper.py`, `vae` = `models/wan/wan_base/modules/vae.py`.

## a) DiT forward passes per chunk: 2, and no extra cache pass

- Per chunk the demo calls `pipe.inference_stream` once (`inf:238`), which makes **one** generator call
  (`pipe:281-289`) with batch = number of denoising steps = 2 (`pipe:206`). Row 0 is the newest latent frame at the
  first step, row 1 is the previous frame at step 500 (`pipe:269-279`). So **2 DiT passes per chunk**, batched.
- There is **no clean / t=0 context pass**. The only generator calls are `pipe:173`, `pipe:192` (session start),
  `pipe:281` (stream), `pipe:323` / `pipe:342` (no-batch mode) and `pipe:385` / `pipe:410` (multi-GPU split). The
  KV cache is written inside those same passes (`cm:369-370`, `cm:388-389`) from the **noisy** input: row 0's cache
  holds K/V at the first step's noise level, row 1's at step 500. Unlike Self-Forcing / CausVid, nothing is re-run
  on the clean output.
- Each row keeps its own 30 caches (`pipe:224-229`), so there are 60 cache pairs in total.
- First step is adaptive: `current_step = int(1000 * noise_scale) - 100` (`inf:51-57`); on the test clip it was
  610, 601, 600, 600, 600, 600 rather than 700.
- Session start is different: 5 pixel frames = 2 latent frames in one 2048-token chunk, 2 sequential passes at
  batch 1 (`pipe:167-200`, `inf:188-202`).
- Quirk: on the first stream call row 1 denoises an all-zero latent (`pipe:241-243`, `pipe:269-270`); its output is
  never decoded (`inf:248`) and its cache slot is overwritten on the next call.

**FPS estimate with the true pass count:** unchanged, because the G1 estimate already used 2 passes:
11.41 ms x 30 layers x 2 passes / 4 frames = 0.171 s per frame = **5.8 FPS** (DiT only, one core). Measured in the
full 30-layer run below: 0.65 s per chunk = **6.1 FPS** (DiT only).

## b) How the v2v input is built, and which VAE

1. **Encode**: `vae.stream_encode(images, is_scale=False)` (`inf:148`, `ww:134-141`, `vae:546-583`): the Wan 2.1
   VAE encoder, causal, with a feature cache carried between calls. First call takes 1 + 4 frames, later calls 4
   frames each. By default the latents are **not** mean/std normalised (`--normalize_latents` is off, `inf:456`).
2. **Noise**: `noise * noise_scale + latents * (1 - noise_scale)` (`inf:173-177`), `noise_scale` 0.8 at start
   (`demo/config.py:85`), then adapted to frame-to-frame motion (`inf:51-57`; 0.70 on the test clip).
3. **Denoise**: the 2 batched passes above; between them `scheduler.add_noise` to step 500 (`pipe:291-298`).
4. **Decode**: only the finished row is decoded (`inf:179-186`, `inf:248-249`) with
   `vae.stream_decode_to_pixel` (`ww:143-151`, `vae:611-648`), which does apply the mean/std scale.

`--use_taehv` swaps in `TAEHVWanVAEWrapper` (`pipe:27-33`). It covers **decode only**: encoding still goes through
the full Wan VAE encoder (`tw:390`, `tw:476-480`); decoding uses TAEHV in fp16 with a 3-latent context window
(`tw:482-505`). Expected weights: `ckpts/taew2_1.pth` from `github.com/madebyollin/taehv` (`tw:45-46`).
Downloaded to `seat-234:/workspace/livevid/sdv2_root/ckpts/taew2_1.pth`.

## c) Where one block call's 11.4 ms goes

`neuron-explorer capture` on the G1 block NEFF, core 0 (`tier2/results/g2_profile_block0_summary.txt`).

| | ms | share of 11.41 ms |
|---|---|---|
| Wall time per call from the host (G1 bench, mean) | 11.41 | 100% |
| Device execution (`total_exec_time`) | 4.21 | 37% |
| Everything else: host<->device copies of inputs/outputs + runtime call overhead | 7.20 | 63% |

Inside the 4.21 ms of device execution (engines overlap, so these do not add up):

| Engine | Active time | Share of device time |
|---|---|---|
| Tensor engine (matmuls) | 2.87 ms | 68% |
| DMA (moving data between HBM and on-chip buffers) | 2.09 ms | 49% |
| Vector engine | 3.00 ms | 71% |
| Scalar engine | 3.00 ms | 71% |

Data crossing the host per call: about 42.6 MB in (two 18.9 MB caches, x, context, RoPE, bias) and 40.9 MB out.
The two caches are 89% of it.

**FLOPs per block call: 132.1 GFLOP** (`torch.utils.flop_counter` on the CPU port, 2 FLOPs per multiply-add:
90.2 G in Linear layers + 41.9 G in attention matmuls; `tier2/results/g2_flops.json`). The profiler's own count,
including transposes, is 161.3 G (`hardware_flops`).

**Peak used: 130.7 TFLOPS per logical NeuronCore** (LNC = 2). This is the peak implied by the profiler itself:
`hardware_flops` 161.3 G / 4.21 ms / `mfu_inst_estimated_percent` 0.2934. It is not a published spec number.

| Basis | Throughput | % of peak |
|---|---|---|
| Wall clock, 11.41 ms per call | 11.6 TFLOPS | **8.9%** |
| Device execution only, 4.21 ms | 31.4 TFLOPS | 24% |
| Profiler's own MFU (its 161.3 G count over device time) | 38.3 TFLOPS | 29.3% |

## Steps 1-3 (done on seat-234 before the re-scope)

- CPU reference: the original `CausalStreamInferencePipeline` on CPU fp32, 29 frames of `tier1/test_clip.mp4`
  (every 2nd frame): 5-frame session start + 6 stream chunks. Adaptive sink off, steps [700, 500], seed 1234.
  T5 run once, embedding saved to `artifacts/tier2/g2/prompt_anime.pt` (29 real tokens).
- 30 layers traced one graph per layer, 25 to 37 s each, 3 in parallel.
- Session start (2 latent frames) runs on CPU with the fp32 port; all 6 stream chunks run on Neuron.

| Stream chunk | Row 0 (first step) cos | Row 1 (final output) cos | Decoded PSNR vs CPU ref (Wan VAE) |
|---|---|---|---|
| 0 | 0.999978 | not run (see quirk above) | - |
| 1 | 0.999973 | 0.999981 | 50.8 dB |
| 2 | 0.999975 | 0.999977 | 50.4 dB |
| 3 | 0.999970 | 0.999972 | 49.9 dB |
| 4 | 0.999974 | 0.999963 | 47.9 dB |
| 5 | 0.999975 | 0.999960 | 47.1 dB |

Gate 0.999: **PASS** (minimum 0.999960). The same host code with the fp32 CPU port instead of Neuron gives 1.000000
on every chunk, so the remaining error is bf16 / Neuron, not host logic. Cosine drifts down slowly with chunk
count; worth watching on longer runs. Video: `tier2/out/g2_side_by_side.mp4` (input | CPU ref | Neuron), TAEHV
version alongside, stills in `tier2/out/g2_contact.png`.

Timing per chunk on one core (2 passes = 60 block calls), steady state:

| Part | Seconds per chunk |
|---|---|
| 60 Neuron block calls (10.6 ms each) | 0.64 |
| of which device execution (60 x 4.21 ms, from the profile) | 0.25 |
| of which host<->device transfer and call overhead | 0.39 |
| CPU-side ops (patch embed, time embed, head, scheduler, cache bookkeeping) | 0.015 |
| **DiT total** | **0.65 -> 6.1 FPS** |
| Wan VAE encode on CPU | 12.1 |
| Wan VAE decode on CPU | 15.8 |
| TAEHV decode on CPU (alternative to Wan decode) | 0.66 |

The VAE numbers are CPU, 8 threads, measured while other jobs shared the 11-CPU quota: indicative only. End to end
with the VAE on CPU this is about 0.14 FPS (Wan decode) or 0.30 FPS (TAEHV decode); the VAE is not on Neuron yet.
