# Tier 1 quick profile

sd-turbo img2img, 512x512, 1 step, TAESD encode/decode, bf16. Frames from `webcam_clip.mp4`.

## Per-core utilization during the 4-core run

Pool running flat out (2 frames in flight per worker) for 20 s: **43.5 FPS**, latency p50 182.591 ms. neuron-monitor, 1 s periods, 20 samples. Each logical core is two physical cores (logical-neuroncore-config 2); the logical figure is their mean.

| visible core | monitor index | logical util mean % | min % | max % | physical cores mean % |
|---|---|---|---|---|---|
| 0 | 2 | 78.3 | 71.7 | 83.7 | nc_v3.4 95.0, nc_v3.5 61.5 |
| 1 | 3 | 78.0 | 72.4 | 79.3 | nc_v3.6 94.7, nc_v3.7 61.2 |
| 2 | 0 | 78.3 | 73.8 | 79.5 | nc_v3.0 95.1, nc_v3.1 61.6 |
| 3 | 1 | 78.3 | 73.7 | 79.6 | nc_v3.2 95.0, nc_v3.3 61.5 |

## One UNet call (neuron-explorer device profile)

Total time 56.2 ms, of which some engine is active 96.7%.

| engine | active % of the call | instructions |
|---|---|---|
| tensor (matmul) | 71.1 | 598613 |
| vector | 66.1 | 145735 |
| scalar | 59.7 | 103219 |
| sync | 20.0 | 41466 |
| gpsimd | 0.8 | 15732 |

| % of peak | value |
|---|---|
| model FLOPs utilization (MFU) | 19.0% |
| max achievable MFU for this instruction mix | 73.0% |
| hardware FLOPs utilization (HFU, includes transposes) | 25.0% |
| memory bandwidth utilization (MBU) | 20.1% |

The profile also reports a utilization limit on the two physical cores: the 50% limit was active 71.6% of the call, average limit 64.0%. Spill traffic: 2.0 GB saved, 3.2 GB reloaded per call; weights 1.78 GB.
