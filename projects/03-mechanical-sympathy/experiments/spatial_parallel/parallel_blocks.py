# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import os
from types import MethodType

import torch
import torch_xla.core.xla_model as xm
from samudra_tiled import _is_spatial_conv, wrap_lon


def parallel_forward(block, x):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    H = x.shape[2]
    if H % world:
        raise ValueError(f"Grid height {H} must divide evenly across {world} ranks")
    local_h = H // world
    R = block.band_rows
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


def install(model):
    names = []
    for name, block in model.named_modules():
        if hasattr(block, "band_rows"):
            block.forward = MethodType(parallel_forward, block)
            names.append(name)
    return names
