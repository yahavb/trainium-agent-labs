# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""One real Samudra block, spatially divided across Neuron cores; timing only."""

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--samudra-root", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--repeats", type=int, default=50)
a = p.parse_args()
lib = (
    Path(sys.base_prefix)
    / "lib"
    / f"libpython{sys.version_info.major}.{sys.version_info.minor}.so.1.0"
)
ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
os.environ["LD_LIBRARY_PATH"] = (
    str(lib.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
)
os.environ["PATH"] = str(Path(sys.executable).parent) + ":" + os.environ["PATH"]
sys.path.insert(0, str(a.samudra_root / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runners"))
import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
from samudra.models.modules.blocks import ConvNeXtBlock
from samudra.utils.inference_fusion import fold_batch_norm
from samudra_tiled import _is_spatial_conv, tile_block, wrap_lon

torch.set_num_threads(2)
rank = int(os.environ.get("RANK", "0"))
world = int(os.environ.get("WORLD_SIZE", "1"))
if world > 1:
    dist.init_process_group("xla")
dev = torch_xla.device()
torch.manual_seed(7)
block = ConvNeXtBlock(
    in_channels=162, out_channels=280, upscale_factor=2, norm="batch"
).eval()
ckpt = torch.load(
    a.samudra_root / "checkpoints/onedeg/ema_ckpt.pt",
    map_location="cpu",
    weights_only=False,
)
prefix = "unet.layers.0."
sd = {
    k.removeprefix("module.")[len(prefix) :]: v
    for k, v in ckpt["model"].items()
    if k.removeprefix("module.").startswith(prefix)
}
block.load_state_dict(sd)
del ckpt
fold_batch_norm(block)
tile_block(block, 30)
block.requires_grad_(False).to(dev)
x = torch.randn(1, 162, 180, 360).to(dev)


def sync():
    torch_xla.sync(wait=True)
    xm.wait_device_ops()


def distributed_block(x):
    H = x.shape[2]
    if H % world:
        raise ValueError(f"Grid height {H} must divide evenly across {world} ranks")
    local_h = H // world
    R = 30
    T = block.halo
    n = block.N_pad
    padded = torch.nn.functional.pad(wrap_lon(x, T), (0, 0, T, T))
    bands = []
    for r0 in range(rank * local_h, (rank + 1) * local_h, R):
        end = min(r0 + R, (rank + 1) * local_h)
        h = padded[:, :, r0 : end + 2 * T, :]
        rem = T
        for layer in block.convblock:
            if _is_spatial_conv(layer):
                rows = torch.arange(r0 - rem, end + rem, device=h.device)
                h = torch.where(((rows >= 0) & (rows < H)).view(1, 1, -1, 1), h, 0.0)
                h = layer(h)
                rem -= n
            else:
                h = layer(h)
        bands.append(block.skip_module(x[:, :, r0:end, :]) + h)
    local = torch.cat(bands, dim=2)
    return (
        xm.all_gather(local, dim=2, groups=[list(range(world))]) if world > 1 else local
    )


results = {}
with torch.no_grad():
    for label, fn in [
        ("replicated_full_block", lambda: block(x)),
        ("spatial_parallel_block", lambda: distributed_block(x)),
    ]:
        t = time.perf_counter()
        for _ in range(3):
            y = fn()
            sync()
        warm = time.perf_counter() - t
        samples = []
        for _ in range(a.repeats):
            t = time.perf_counter()
            y = fn()
            sync()
            samples.append(time.perf_counter() - t)
        results[label] = {
            "median_seconds": statistics.median(samples),
            "samples_seconds": samples,
            "warmup_seconds": warm,
        }
        print(rank, label, results[label]["median_seconds"], flush=True)
a.output.mkdir(parents=True, exist_ok=True)
(a.output / f"rank{rank}.json").write_text(
    json.dumps(
        {
            "rank": rank,
            "world_size": world,
            "block": "unet.layers.0",
            "real_checkpoint": True,
            "input_shape": list(x.shape),
            "output_shape": list(y.shape),
            "input_values": "seeded random",
            "forecast_metrics_computed": False,
            "results": results,
        },
        indent=2,
    )
    + "\n"
)
