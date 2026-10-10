# End to end on Trainium2: TAEHV encoder -> causal Wan DiT -> TAEHV decoder

**Result: MEASURED 5.64 FPS on one NeuronCore**, with the encoder, all 30 DiT layers and the decoder running as
traced Neuron graphs for every stream chunk. One 4-frame chunk at 512 x 512 takes 0.709 s. The output is the same
stylised video as Track 2's G2 run (which used the Wan VAE on CPU): 26.2 dB PSNR against it with identical noise.
Measured the same way as Track 2's tuned run (clip looped to 25 chunks, first 5 excluded) the untuned baseline is
**5.62 FPS**, p99 chunk time 0.729 s.

Run on seat-230 (trn2.48xlarge, `NEURON_RT_VISIBLE_CORES=0`), 2026-10-10. Code: `tier2/e2e.py`. Raw numbers:
`tier2/results/e2e.json`. Video: `tier2/out/e2e_trainium.mp4` (input | output). Still: `tier2/out/e2e_compare.png`.

One stage is not on Trainium: the **session start** (the first 2 latent frames, a one-off 2048-token chunk) runs
the DiT on CPU with the fp32 port, exactly as in Track 2's G2 run, because the traced block graph is fixed at one
frame. It took 15.6 s once. Its encode and decode are on Neuron. All 6 stream chunks are Neuron only.

## Measured timing per chunk (one core)

Per stream chunk: 1 encoder call, 2 DiT passes (60 block calls), 1 decoder call, 4 output frames.

| Chunk | Wall | TAEHV encode | DiT (60 block calls) | TAEHV decode | Host overhead | FPS |
|---|---|---|---|---|---|---|
| 0 | 0.414 s | 18.4 ms | 0.339 s (1 pass) | none | 57 ms | no output yet |
| 1 | 0.893 s | 18.8 ms | 0.775 s | 38.0 ms | 61 ms | 4.48 |
| 2 | 0.795 s | 23.1 ms | 0.718 s | 38.0 ms | 16 ms | 5.03 |
| 3 | 0.705 s | 18.7 ms | 0.633 s | 38.0 ms | 15 ms | 5.67 |
| 4 | 0.708 s | 18.7 ms | 0.633 s | 38.0 ms | 19 ms | 5.65 |
| 5 | 0.714 s | 18.7 ms | 0.642 s | 38.0 ms | 16 ms | 5.60 |
| **Mean of chunks 3-5** | **0.709 s** | **18.7 ms (2.6%)** | **0.636 s (89.7%)** | **38.0 ms (5.4%)** | **16.6 ms (2.3%)** | **5.64** |

- FPS = 4 output frames / wall seconds per chunk. The headline uses chunks 3 to 5; chunks 1 and 2 are still
  warming up. Over all five chunks that emit frames the mean is 0.763 s, 5.24 FPS.
- Chunk 0 runs one DiT pass and emits nothing: a frame's final latent comes out one chunk after it goes in.
- "DiT" is host call to host return for the 60 block graphs, so it includes moving each block's KV cache both ways.
- "Host overhead" is everything else inside the chunk: noise-scale computation, latent rescale, noise mixing,
  patch embed, time embed, head, scheduler math, and converting the decoded frames to uint8.
- A separate run with freshly seeded noise instead of replayed noise gave the same timing: 0.711 s per chunk.
- Only 3 steady-state chunks were measured; the clip is short. Not measured: frame capture, video writing, a
  longer run (the `t_refresh` position rewind after about 45 chunks and adaptive sink refresh are still not
  implemented in the host code).

## Untuned baseline measured Track 2's way (looped clip, first 5 chunks excluded)

`python e2e.py --chunks 25`: the 29-frame clip is looped to 25 stream chunks and the first 5 are excluded, the same
accounting as Track 2's tuned 38.4 FPS run (`tier2/g4_e2e.py` on `tier2-g1`). Nothing else changed: one graph per
layer, KV caches through the host on every block call, TAEHV state through the host, one core, one chunk at a time.
Raw numbers: `tier2/results/e2e_loop25.json`.

| Steady state, chunks 5-24 (20 chunks) | |
|---|---|
| **FPS, MEASURED** | **5.62** |
| Chunk time, mean | 0.7117 s |
| Chunk time, p50 | 0.7107 s |
| Chunk time, p99 | 0.7287 s |
| Chunk time, min / max | 0.7041 s / 0.7303 s |
| of which TAEHV encode | 18.6 ms |
| of which DiT (60 block calls) | 0.635 s |
| of which TAEHV decode | 38.0 ms |
| of which host overhead | 19.7 ms |

With 20 samples the p99 is an interpolation just below the maximum. The number agrees with the 3-chunk figure
above (5.64), so the short clip was not flattering the baseline. On the same accounting the tuned 4-core pipeline
is 38.4 / 5.62 = 6.8x this baseline.

### Config differences against `g4_e2e.py`

Same in both: TAEHV graphs (the identical `taehv_enc/dec_512x512_bf16.pt` shapes and code from `vae_neuron.py`),
rescale `lat * std + mean`, TAEHV chunk f = pixel frames 4f..4f+3 with the decoder's first 3 frames dropped, clip
looped by wrapping the frame index, 2 DiT passes per chunk at [adaptive first step, 500], fresh seeded noise
(seed 1234), TAEHV state carried on the host side, first 5 chunks excluded.

| | This baseline (`e2e.py --chunks 25`) | Track 2 tuned (`g4_e2e.py`) |
|---|---|---|
| Cores | 1 | 4 |
| DiT graphs | 30, one per layer | 4 stages of 8, 8, 7, 7 layers |
| KV caches | cross the host on every block call | resident on the device |
| Concurrency | one chunk at a time, one process | 6 chunks in flight across 4 stage processes |
| FPS definition | 4 x chunks / sum of per-chunk wall times | 4 x chunks / elapsed time between completions (throughput); the two agree for a sequential loop |
| Tail metric | p99 of chunk time | mean / p50 / max of submit-to-done latency (no p99) |
| Chunks | 25, skip 5 | 120, skip 5 |
| Session start | in the run but before the timed chunks: 2-frame start on CPU (15.4 s) | not run: KV caches pre-warmed by `g3_pipeline.py warm`, encoder warmed on pixel chunks 0-2, decoder on 3 warm latents |
| First step and noise scale | recomputed from the pixels every chunk (`noise_scale_and_step`), inside the timed chunk | replayed from the 5 values saved in the CPU reference, cycled; no per-chunk computation |
| Row schedule | staggered as in the repo: row 1 denoises the previous frame, output one chunk later | both rows on the same frame back to back, no one-chunk delay; same number of passes |
| uint8 conversion of output frames | inside the timed chunk | not timed (only the first 5 chunks' frames are kept) |
| Looped RoPE frame index | 2 to 26 | 3 to 122 (neither does the `t_refresh` rewind) |

The first-step computation and uint8 conversion are part of this baseline's 19.7 ms of host overhead, so they cost
it at most 3% relative to Track 2's accounting. The rest of the gap is the tuning itself.

Compared with the VAE on CPU (Track 2's figures: Wan encode 12.1 s, Wan decode 15.8 s per chunk), the same DiT
went from about 0.14 FPS to 5.64 FPS end to end. The DiT is now 90% of the chunk.

## How it is wired

- **DiT:** the 13 layers missing on seat-230 were traced with the G1 method (one graph per layer, 3 at a time,
  about 33 s each); all 30 are in `/workspace/livevid/artifacts/tier2/`. The host code is Track 2's
  `g2_neuron.Runner`, unchanged: steps [700, 500] with the adaptive first step (610, 601, then 600), one KV cache
  set per denoising row, 3 sink frames, cached T5 embedding for the "anime" prompt.
- **Lookahead fix:** TAEHV latent k is pixel frames 4k..4k+3, while Wan latent k is frames 4k-3..4k. The clip is
  padded with 3 copies of its last frame at the end, and every encoder call is one plain 4-frame chunk. In a live
  stream this means the output runs 3 frames (about 0.2 s at 16 fps) later than with the Wan encoder.
- **Scale fix:** TAEHV latents are mean/std normalised and the pipeline encodes un-normalised
  (`streamv2v/inference.py:148`), so latents are mapped with `lat * std + mean` before noise is added.
- **Decode:** the DiT output is already in the normalised space TAEHV decodes; the first 3 decoded frames are
  dropped. Output is 25 frames (1 + 4 x 6), the same 25 as Track 2's video.
- The encoder fallback (Wan-encoded latents) was not needed.

## Quality against Track 2's G2 output

Same clip, prompt and schedule. To make the comparison meaningful the noise tensors of a CPU reference run
(`g2_ref.py` on seat-230, same seed as Track 2's) were replayed, so the only differences are the TAEHV encoder,
the TAEHV decoder and bf16.

Encoder check inside the run: rescaled TAEHV latents vs the Wan encoder's latents, cosine 0.9980 to 0.9985 on all
7 encode calls (RMS difference 0.12 to 0.20 on latents of RMS 2.3 to 3.2).

PSNR of the all-Trainium output, per chunk of output frames:

| Compared with | Frames 0-4 | 5-8 | 9-12 | 13-16 | 17-20 | 21-24 | All |
|---|---|---|---|---|---|---|---|
| Track 2 G2 video, Neuron DiT + Wan VAE (`g2_side_by_side.mp4`, right panel) | 26.3 | 25.6 | 25.6 | 26.1 | 26.8 | 27.1 | **26.2 dB** |
| Track 2 G2 video, CPU reference + Wan VAE (middle panel) | 26.3 | 25.6 | 25.5 | 26.1 | 26.8 | 27.1 | 26.2 dB |
| CPU reference latents from seat-230, decoded by the same Neuron TAEHV decoder | 27.1 | 26.2 | 26.0 | 26.7 | 27.5 | 28.1 | 26.9 dB |

- No drift over the 6 chunks; the last chunks are the closest.
- The third row removes the decoder difference and still shows 26.9 dB, so most of the gap comes from the TAEHV
  encoder's latents being amplified by the DiT, not from the TAEHV decoder.
- Track 2's frames were read back from an mp4, so those two rows include video compression error.
- Visually (`tier2/out/e2e_compare.png`, frames 0, 8, 16, 24): same character, pose, colours and ball position;
  small differences in fine detail such as the eyes. With different noise the style holds but the content differs
  much more (18.1 dB), which is sampling variation, not an error.

## Resident TAEHV state (step 5)

`tier2/vae_resident.py`, raw numbers in `tier2/results/vae_resident.json`. The 9 memories are parameters of the
traced module, aliased to its outputs with `input_output_aliases` and moved to the device with
`move_trace_to_device` (the mechanism Track 2 used for the KV cache). Only the frames or the latent cross the host.

| Per chunk, one core, 100 calls | State through the host (mean / p50 / p99) | State resident (mean / p50 / p99) | Parity vs CPU TAEHV, min cos over 8 chunks |
|---|---|---|---|
| TAEHV encode | 17.91 / 17.92 / 17.95 ms | 13.24 / 13.30 / 13.32 ms | 0.999953 |
| TAEHV decode | 37.82 / 37.80 / 38.19 ms | 31.22 / 31.22 / 31.29 ms | 0.999996 |

- Parity is the same as with host-side state on every chunk, so the on-device state does advance correctly.
- This saves 11.3 ms per chunk. It was measured standalone and is **not wired into the end-to-end run above**: the
  5.64 FPS figure uses host-side state. By arithmetic it would give 0.698 s per chunk, about 5.73 FPS (estimate,
  not measured).
- Compile: 28 s encoder, 43 s decoder. These traced modules were used in-process and not saved as artifacts.

## Not done

- Wiring the resident TAEHV graphs into `e2e.py` and re-measuring end to end.
- Session start on Neuron (needs a 2-frame block graph).
- seat-231 was not used; seat-232 and seat-234 were not touched.

## Reproduce (seat-230)

```
tier2/remote.sh push
tier2/remote.sh run "./g2_compile_all.sh"                       # skips layers that already exist
tier2/remote.sh run "python -u vae_neuron.py compile enc && python -u vae_neuron.py compile dec"
tier2/remote.sh run "python -u g2_ref.py run"                   # optional: CPU reference whose noise is replayed
tier2/remote.sh run "NEURON_RT_VISIBLE_CORES=0 python -u e2e.py"
```
