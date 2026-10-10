# Tier 1 benchmark: sd-turbo img2img on Neuron

stabilityai/sd-turbo, 512x512, 1 UNet step, TAESD encode/decode, bf16, batch 1. seat-233, 200 frames from `test_clip.mp4`.

## Compile

| component | compile s | compiler args | cosine vs CPU fp32 (random input) |
|---|---|---|---|
| taesd_enc | 256.1 | `--auto-cast=none` | 0.999941 |
| unet | 466.2 | `--auto-cast=none --model-type=unet-inference` | 0.999953 |
| taesd_dec | 271.1 | `--auto-cast=none` | 0.999858 |

## Single core (core 0, in-process)

| stage | mean ms | p50 ms | p99 ms |
|---|---|---|---|
| pre | 0.395 | 0.388 | 0.455 |
| encode | 14.703 | 14.7 | 14.74 |
| unet | 56.238 | 56.238 | 56.34 |
| decode | 17.182 | 17.179 | 17.211 |
| post | 0.913 | 0.904 | 1.011 |
| total | 89.431 | 89.422 | 89.66 |

End-to-end: **11.18 FPS** (200 frames in 17.8913 s). `pre`/`post` are the host uint8<->float conversions.

## Worker pool (round-robin dispatcher, reordered outputs)

| cores | in flight per worker | FPS | latency p50 ms | latency p99 ms | worker service p50 ms |
|---|---|---|---|---|---|
| 1 | 1 | **10.94** | 91.353 | 91.737 | 89.386 |
| 1 | 2 | **11.08** | 180.572 | 181.899 | 89.418 |
| 4 | 1 | **43.24** | 92.218 | 94.113 | 90.269 |
| 4 | 2 | **43.78** | 182.405 | 184.789 | 90.449 |

Latency is dispatcher submit -> result received (queue wait + worker + IPC).

## Similarity filter

Mean abs pixel difference (0..255, every 4th pixel) against the last processed frame, on `test_clip.mp4` (300 frames).

| threshold | skip rate |
|---|---|
| 1.0 | 20.0% |
| 2.0 | 20.0% |
| 3.0 | 20.7% |
| 4.0 | 23.0% |
| 6.0 | 34.3% |
| 8.0 | 56.7% |
| 12.0 | 73.7% |

4 cores with threshold 3.0: skipped 62/300 (20.7%), delivered **54.76 FPS** (43.44 FPS actually processed).
