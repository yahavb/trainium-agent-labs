# TAEHV encoder and decoder on Neuron

**Result: PASS, and the TAEHV encoder looks usable.** Both halves of TAEHV (`taew2_1.pth`) run on one NeuronCore
as fixed-shape bf16 graphs with explicit temporal state. Against the repo's TAEHV on CPU fp32 the minimum cosine
over 8 chunks is 0.999953 (gate 0.999). One chunk costs **17.9 ms to encode and 37.8 ms to decode**, against
8.9 s and 15.1 s for the Wan VAE on CPU.

Run on seat-230 (trn2.48xlarge, NeuronCore 0), 2026-10-10. torch 2.9.1, torch-neuronx 2.9.0.2.15.32035. Code:
`tier2/vae_neuron.py`. Raw numbers: `tier2/results/vae_neuron.json`, `tier2/results/enc_probe.json`.

Not tested: TAEHV-encoded latents have not been run through the DiT, so "usable" rests on latent and pixel
agreement with the Wan encoder, not on a stylised output.

## What was traced

| | Encoder | Decoder |
|---|---|---|
| Input | 4 frames `[4, 3, 512, 512]` in [0, 1] | 1 latent `[1, 16, 64, 64]` |
| Output | 1 latent `[1, 16, 64, 64]` | 4 frames `[4, 3, 512, 512]` in [0, 1] |
| State in and out | 9 tensors: 3 x `[1, 64, 256, 256]`, 3 x `[1, 64, 128, 128]`, 3 x `[1, 64, 64, 64]` | 9 tensors: 3 x `[1, 256, 64, 64]`, 3 x `[1, 128, 128, 128]`, 3 x `[1, 64, 256, 256]` |
| State size, bf16 | 33 MB each way | 44 MB each way |
| Compile time | 36.3 s | 53.9 s |
| Compiler args | `--auto-cast=none` | `--auto-cast=none` |
| Artifact | `artifacts/tier2/vae/taehv_enc_512x512_bf16.pt` | `artifacts/tier2/vae/taehv_dec_512x512_bf16.pt` |

Artifacts are under `seat-230:/workspace/livevid/`. Weights were fetched the way the repo does, from
`github.com/madebyollin/taehv` into `sdv2_root/ckpts/taew2_1.pth`.

**Temporal state.** TAEHV does carry state: each of its `MemBlock`s takes the previous timestep's input as a second
input (`taehv_wrapper.py:57-72`). The graphs take those 9 memories as inputs and return the updated ones. Starting
from zeros and calling chunk by chunk reproduces the repo's sequential pass over the whole video: in fp32 on CPU
the chunked port and `encode_video` / `decode_video(parallel=False)` agree with max abs error 0.

**Frame alignment (this matters for the encoder).** TAEHV chunk i is pixel frames 4i..4i+3, and its latent lines up
with Wan latent i, which covers frames 4i-3..4i. The decoder drops its first 3 output frames to match. So the
encoder needs 3 frames of lookahead: a live stream using it runs 3 frames behind the Wan encoder's timing. My first
run padded at the front instead and got a 3-frame shift (latent cosine vs Wan 0.92, round trip 14 dB); padding at
the end gives the numbers below. `tier2/results/enc_probe.json` has the four paddings and the shift sweep.

## Parity: Neuron bf16 vs CPU TAEHV fp32

29 frames of `tier2/data/test_clip.mp4` (every 2nd frame) plus 3 repeats of the last frame = 8 chunks. Neuron feeds
its own bf16 state back; the CPU reference is one sequential pass. Decoder parity uses the CPU latents as input.

| Chunk | Encoder latent cos | Encoder max abs | Decoder frames cos | Decoder PSNR vs CPU |
|---|---|---|---|---|
| 0 | 0.999954 | 0.035 | 0.999997 | 54.0 dB |
| 1 | 0.999953 | 0.062 | 0.999996 | 54.8 dB |
| 2 | 0.999959 | 0.075 | 0.999997 | 55.1 dB |
| 3 | 0.999963 | 0.078 | 0.999996 | 55.4 dB |
| 4 | 0.999965 | 0.074 | 0.999997 | 55.5 dB |
| 5 | 0.999967 | 0.074 | 0.999997 | 55.5 dB |
| 6 | 0.999968 | 0.070 | 0.999996 | 55.4 dB |
| 7 | 0.999967 | 0.071 | 0.999996 | 54.9 dB |

**Minimum cosine 0.999953, gate 0.999: PASS.** No drift over the 8 chunks. Latent RMS is about 1.3.

## Quality: encode -> decode round trips, PSNR vs the input clip

Wan VAE on CPU fp32 (`stream_encode` with mean/std scaling, `stream_decode`), TAEHV on Neuron.

| Chunk (frames) | Wan enc + Wan dec (CPU) | TAEHV enc + TAEHV dec (Neuron) | TAEHV enc (Neuron) + Wan dec | Wan enc + TAEHV dec (Neuron) | Latent cos, TAEHV vs Wan |
|---|---|---|---|---|---|
| 0 (0-4) | 37.4 dB | 31.4 dB | 34.6 dB | 32.4 dB | 0.9976 |
| 1 (5-8) | 37.8 dB | 32.0 dB | 35.2 dB | 32.9 dB | 0.9975 |
| 2 (9-12) | 38.7 dB | 33.1 dB | 35.9 dB | 34.2 dB | 0.9973 |
| 3 (13-16) | 39.4 dB | 33.8 dB | 36.5 dB | 35.0 dB | 0.9974 |
| 4 (17-20) | 38.0 dB | 32.5 dB | 35.2 dB | 33.5 dB | 0.9975 |
| 5 (21-24) | 36.9 dB | 31.4 dB | 33.9 dB | 32.4 dB | 0.9971 |
| 6 (25-28) | 36.0 dB | 30.5 dB | 33.1 dB | 31.4 dB | 0.9970 |
| **All 29 frames** | **37.6 dB** | **32.0 dB** | **34.8 dB** | **32.9 dB** | **0.9973** |

- TAEHV on Neuron and TAEHV on CPU give the same round trip (31.96 vs 31.99 dB), so bf16 costs nothing visible.
- **Can TAEHV encode replace the Wan encode?** Probably yes. Its latents match the Wan encoder's with cosine 0.997
  (RMS difference about 0.10 on latents of RMS 1.3), and decoding them with the Wan decoder loses 2.8 dB against
  a pure Wan round trip. In v2v the latent is then mixed with noise at scale 0.7 to 0.8, which is far larger than
  that difference. This needs one end-to-end run through the DiT to confirm.
- The decoder is the weaker half: swapping only the decoder costs 4.7 dB, swapping only the encoder 2.8 dB.
- Two things to handle when wiring the encoder in: the 3-frame lookahead above, and scaling. TAEHV produces
  mean/std-normalised latents, while the pipeline's default encode call uses `is_scale=False`
  (`streamv2v/inference.py:148`), so the latents need `lat * std + mean` to be a drop-in replacement.

Still: `tier2/out/vae_compare.png` (frames 2, 14, 26; input and the four round trips).

## Latency on one core

`NEURON_RT_VISIBLE_CORES=0`, 5 warm-up calls then 100 timed calls, state fed back every call, host call to host
return (so it includes moving the state both ways).

| Per chunk (4 frames) | mean | p50 | p99 |
|---|---|---|---|
| TAEHV encode, Neuron | 17.91 ms | 17.92 ms | 17.95 ms |
| TAEHV decode, Neuron | 37.82 ms | 37.80 ms | 38.19 ms |

For comparison, CPU fp32 on the same pod (11-CPU quota, 8 threads):

| Per chunk | CPU | Neuron | Speed-up |
|---|---|---|---|
| Wan VAE encode | 8.9 s | - | - |
| Wan VAE decode | 15.1 s | - | - |
| TAEHV encode | 0.44 s | 17.9 ms | 25x |
| TAEHV decode | 0.64 s | 37.8 ms | 17x |

The Wan numbers were taken while the two compiles shared the CPU quota for part of the run, so treat them as
indicative; Track 2 measured 12.1 s and 15.8 s on seat-234.

**What this does to end-to-end FPS.** Using Track 2's DiT figure of 0.65 s per chunk (`tier2/G2_NOTES.md`):
0.65 + 0.018 + 0.038 = 0.71 s per 4-frame chunk, about **5.7 FPS** on one core with all three stages on Neuron,
against about 0.14 FPS with the Wan VAE on CPU. This is arithmetic from separately measured stages, not a measured
end-to-end run. The VAE is now 8% of the chunk time; the DiT is the bottleneck again.

## Errors and open items

- No compiler or runtime errors.
- Not done: an end-to-end v2v run with the TAEHV encoder feeding the DiT; keeping the 33 / 44 MB of state on the
  device instead of crossing the host each call (the same lever Track 2 identified for the KV cache).
- seat-231 was not needed; seat-232 was not used.

## Reproduce (seat-230)

```
tier2/remote.sh push
tier2/remote.sh run "python -u vae_neuron.py compile enc"
tier2/remote.sh run "python -u vae_neuron.py compile dec"
tier2/remote.sh run "python -u vae_neuron.py ref"
tier2/remote.sh run "NEURON_RT_VISIBLE_CORES=0 python -u vae_neuron.py eval"
tier2/remote.sh run "python -u vae_enc_probe.py"
```
