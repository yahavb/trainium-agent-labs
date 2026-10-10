"""Latitude-band tiling for Samudra ConvNeXt blocks (inference only).

At 1 degree the full-resolution blocks have intermediate tensors of 84-145 MB,
far more than a NeuronCore's on-chip memory, so the compiler spills them to HBM
(about 55 GB of spill traffic per forward pass). Tiling runs each large block on
bands of latitude rows, so one band's whole block chain is small enough to stay
on chip.

How one block is computed on a band of output rows [r0, r0 + R):

1. Pad the full input once: wrap T columns in longitude and add T zero rows at
   each pole, where T is the sum of the 3x3 convolutions' paddings (2 * N_pad).
2. Take rows [r0 - T, r0 + R + T) of the padded input and run the block's
   layers with no padding of their own. Each 3x3 convolution trims N_pad rows
   and columns from each side, so the band ends at exactly R rows and the full
   width.
3. Before each 3x3 convolution, zero the rows that lie beyond a pole. The
   original pads its intermediate tensor with zeros at the poles, not with the
   convolution of zeros, so this keeps the result the same.
4. Concatenate the bands.

Longitude needs no masking: the convolution of wrapped input equals the wrapped
convolution output. BatchNorm in eval mode, GELU and clamp act per pixel, so
they are unaffected. InstanceNorm averages over the whole grid, so blocks that
use it are refused.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from samudra_core import ConvNeXtBlock, wrap_lon


def _is_spatial_conv(layer: nn.Module) -> bool:
    return isinstance(layer, nn.Conv2d) and layer.kernel_size[0] != 1


class TiledConvNeXtBlock(ConvNeXtBlock):
    """A ConvNeXtBlock that computes its output in latitude bands.

    Made by converting an existing block in place (see ``tile_block``), so the
    weights, their names and ``isinstance(layer, ConvNeXtBlock)`` checks in
    UNetBackbone all stay the same.
    """

    band_rows: int
    halo: int
    band_cut = None  # optional callable run after each band, e.g. torch_xla.sync

    def forward(self, x):
        if self.training:
            raise RuntimeError("tiling is inference-only (BatchNorm must use running stats)")
        H = x.shape[-2]
        R, T, n = self.band_rows, self.halo, self.N_pad
        if H % R != 0:
            raise ValueError(f"band_rows={R} must divide the grid height {H}")
        padded = F.pad(wrap_lon(x, T), (0, 0, T, T), mode="constant")
        bands = []
        for r0 in range(0, H, R):
            h = padded[:, :, r0 : r0 + R + 2 * T, :]
            rem = T  # halo rows/cols still attached to h on each side
            for layer in self.convblock:
                if _is_spatial_conv(layer):
                    # Rows of h cover global rows [r0 - rem, r0 + R + rem).
                    rows = torch.arange(r0 - rem, r0 + R + rem, device=h.device)
                    inside = ((rows >= 0) & (rows < H)).view(1, 1, -1, 1)
                    h = torch.where(inside, h, 0.0)
                    h = layer(h)
                    rem -= n
                else:
                    h = layer(h)
            skip = self.skip_module(x[:, :, r0 : r0 + R, :])
            bands.append(skip + h)
            if self.band_cut is not None:
                # A graph boundary: the device finishes this band before the next,
                # so the compiler cannot interleave bands. The math is unchanged.
                self.band_cut()
        return torch.cat(bands, dim=2)


def tile_block(block: ConvNeXtBlock, band_rows: int) -> ConvNeXtBlock:
    """Convert ``block`` in place to compute in bands of ``band_rows`` rows."""
    if block.pad != "circular":
        raise ValueError("tiling assumes circular longitude padding")
    for layer in block.convblock:
        if isinstance(layer, (nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm)):
            raise ValueError(f"{type(layer).__name__} is not row-local; cannot tile")
        if _is_spatial_conv(layer) and layer.padding != (0, 0):
            raise ValueError("expected unpadded 3x3 convolutions")
    n_spatial = sum(_is_spatial_conv(layer) for layer in block.convblock)
    block.__class__ = TiledConvNeXtBlock
    block.band_rows = band_rows
    block.halo = n_spatial * block.N_pad  # rows/cols consumed by the chain
    return block


def tile_large_blocks(model: nn.Module, grid_hw: tuple[int, int], band_rows: int,
                      min_pixels: int, band_cut=None) -> list[str]:
    """Replace UNet blocks that run on at least ``min_pixels`` grid cells.

    Only levels big enough to spill are tiled; deeper levels already fit on chip
    and their large dilations would make the halo expensive. Returns the names of
    the tiled layers. Call after loading the checkpoint and calling eval().
    """
    H, W = grid_hw
    layers = model.unet.layers
    tiled = []
    level = 0
    for i, layer in enumerate(layers):
        h, w = H >> level, W >> level
        if isinstance(layer, ConvNeXtBlock) and h * w >= min_pixels and h % band_rows == 0:
            tile_block(layer, band_rows)
            layer.band_cut = band_cut
            tiled.append(f"unet.layers.{i} (level {level}, {h}x{w}, {layer.in_channels}->"
                         f"{layer.out_channels})")
        # Track the resolution: AvgPool halves it, the upsample doubles it.
        name = type(layer).__name__
        if name == "AvgPool":
            level += 1
        elif name == "ZonallyPeriodicBilinearUpsample":
            level -= 1
    return tiled
