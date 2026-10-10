<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Samudra forward progress with latitude tiling

This experiment adds 30-row tiling to the existing BF16 matmult-autocast and
BatchNorm-folded benchmark. It uses the same real one-degree checkpoint and
prepared eight-year inputs, 299 autoregressive calls, eight CPU threads,
visible Neuron core 0, and LNC2. The two full-resolution ConvNeXt blocks are
tiled; other blocks keep their existing forward path. No band graph cuts or
segmentation are added.

This is a timing-only experiment. No prediction comparisons, forecast metrics,
or precision checker run. Compilation and warmup are outside the measured
forward time. The report records the tile size and selected blocks.

The manifest includes the existing three measured experiments and the new
full-range tiling report. Plot with:

```bash
/amogh/samudra-one-degree/.venv/bin/python \
  projects/03-mechanical-sympathy/runners/plot_forward_progress.py \
  projects/03-mechanical-sympathy/results/tiling-forward-2026-10-10/experiments.json \
  --output projects/03-mechanical-sympathy/results/tiling-forward-2026-10-10/progress
```

![Forward progress](progress.png)

Tiling: 42.5839 s for 299 calls versus 73.2887 s without tiling
(1.721x speedup; -41.90% change in forward time).
Warmup including compilation: 366.52 s.
