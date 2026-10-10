# Standalone, inference-only copy of the Samudra UNet model (the `samudra_om4` architecture).
#
# Adapted from https://github.com/m2lines/Samudra (src/samudra/models/samudra.py,
# models/modules/{blocks,activations,unet_backbone}.py), Apache-2.0 / MIT, (c) Samudra Authors,
# Matthias Karlbauer, Nathaniel Cresswell-Clay, Thorsten Kurth.
#
# Why a copy: the samudra package pulls in microsoft-aurora, cartopy, dask, ... and can drag in a
# different torch, which would break torch-neuronx in the pod. This file needs only torch.
# The module tree and parameter names match the original, so a real Samudra checkpoint's
# state_dict loads into it unchanged (strict=True). Training-only paths (activation
# checkpointing, drop-path) are removed; they are identity at inference.

import torch
import torch.nn as nn
import torch.nn.functional as F


class CappedGELU(nn.Module):
    def __init__(self, cap_value: float = 10.0):
        super().__init__()
        self.gelu = nn.GELU()
        self.cap = nn.Buffer(torch.tensor(cap_value, dtype=torch.float32))

    def forward(self, x):
        return torch.clamp(self.gelu(x), max=self.cap)


class AvgPool(nn.Module):
    def __init__(self, pooling: int = 2):
        super().__init__()
        self.avgpool = nn.AvgPool2d(pooling)

    def forward(self, x):
        return self.avgpool(x)


def wrap_lon(x, n: int):
    """Circular pad of n columns in longitude (last axis), written as slice + concat.

    Bit-identical to F.pad(x, (n, n, 0, 0), mode="circular") for n <= W, but torch-xla lowers
    F.pad(mode="circular") with a full-size all-true bool (i1) mask constant, and neuronx-cc 2.27
    fails to parse i1 constants with >= 2**17 elements (NCC_EMOD021). Slices + concat need none.
    """
    if n == 0:
        return x
    return torch.cat((x[..., -n:], x, x[..., :n]), dim=-1)


class ZonallyPeriodicBilinearUpsample(nn.Module):
    """2x bilinear upsampling that wraps around in longitude (the last axis)."""

    def forward(self, x):
        width = x.shape[-1]
        padded = wrap_lon(x, 1)
        up = F.interpolate(padded, scale_factor=(2, 2), mode="bilinear", align_corners=False)
        return up[..., 2 : 2 + width * 2]


def globe_pad(x, n: int, mode: str):
    """Wrap around in longitude, zero-pad in latitude (the poles)."""
    x = wrap_lon(x, n) if mode == "circular" else F.pad(x, (n, n, 0, 0), mode=mode)
    return F.pad(x, (0, 0, n, n), mode="constant")


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 pad="circular", upscale_factor=2, norm="instance"):
        super().__init__()
        assert kernel_size % 2 == 1
        self.in_channels, self.out_channels = in_channels, out_channels
        self.N_pad = int((kernel_size + (kernel_size - 1) * (dilation - 1) - 1) / 2)
        self.pad = pad
        mid = int(in_channels * upscale_factor)
        if in_channels == out_channels:
            self.skip_module = lambda x: x
        else:
            self.skip_module = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding="same")

        def norm_layer():
            return {"batch": lambda: nn.BatchNorm2d(mid),
                    "instance": lambda: nn.InstanceNorm2d(mid),
                    "nonorm": lambda: None}[norm]()

        layers = [nn.Conv2d(in_channels, mid, kernel_size, dilation=dilation)]
        if (n := norm_layer()) is not None:
            layers.append(n)
        layers += [CappedGELU(), nn.Conv2d(mid, mid, kernel_size, dilation=dilation)]
        if (n := norm_layer()) is not None:
            layers.append(n)
        layers += [CappedGELU(), nn.Conv2d(mid, out_channels, kernel_size=1, padding="same")]
        self.convblock = nn.Sequential(*layers)

    def forward(self, x):
        skip = self.skip_module(x)
        for layer in self.convblock:
            if isinstance(layer, nn.Conv2d) and layer.kernel_size[0] != 1:
                x = globe_pad(x, self.N_pad, self.pad)
            x = layer(x)
        return skip + x


class UNetBackbone(nn.Module):
    def __init__(self, in_channels, ch_width, dilation, n_layers, pad, upscale_factor, norm,
                 kernel_size=3):
        super().__init__()
        assert all(n == 1 for n in n_layers)
        self.out_channels = ch_width[0]
        widths = [in_channels] + list(ch_width)
        dil = list(dilation)

        def block(a, b, d):
            return ConvNeXtBlock(a, b, kernel_size, d, pad, upscale_factor, norm)

        down = AvgPool()  # one shared instance, as in the original
        layers = []
        pairs = list(zip(widths[:-1], widths[1:]))
        for i, (a, b) in enumerate(pairs):
            layers += [block(a, b, dil[i]), down]
        layers.append(block(b, b, dil[i]))                 # middle
        layers.append(ZonallyPeriodicBilinearUpsample())   # first upsample
        widths.reverse(); dil.reverse()
        up_pairs = list(zip(widths[:-2], widths[1:-1]))
        for i, (a, b) in enumerate(up_pairs):
            layers += [block(a, b, dil[i]), ZonallyPeriodicBilinearUpsample()]
        layers.append(block(b, b, dil[i]))                 # final
        self.layers = nn.ModuleList(layers)
        self.num_steps = len(ch_width)

    def forward(self, fts, cut=None):
        """cut: optional callable run after every layer (e.g. torch_xla.sync) so a device
        compiles each layer as its own graph; the math is unchanged."""
        skips = []
        count = 0
        for layer in self.layers:
            fts = layer(fts)
            if cut is not None:
                cut()
            if count < self.num_steps:
                if isinstance(layer, ConvNeXtBlock):
                    skips.append(fts)
                    count += 1
            elif isinstance(layer, ZonallyPeriodicBilinearUpsample):
                skip = skips[2 * self.num_steps - count - 1]
                dh = skip.shape[2] - fts.shape[2]
                dw = skip.shape[3] - fts.shape[3]
                fts = F.pad(fts, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
                fts = fts + skip
                count += 1
        return fts


class SamudraNet(nn.Module):
    """forward(prognostic, boundary, mask) -> next prognostic state (one model call).

    prognostic: [B, prog_ch, H, W] float32   (77 variables x input_steps)
    boundary:   [B, bnd_ch, H, W] float32    (surface forcing x input_steps)
    mask:       [B or 1, out_ch, H, W] bool  (True = ocean)
    """

    def __init__(self, prog_channels=154, boundary_channels=8, out_channels=154,
                 ch_width=(280, 380, 480, 520), dilation=(1, 2, 4, 8), n_layers=(1, 1, 1, 1),
                 upscale_factor=2, norm="instance", last_kernel_size=3, pad="circular",
                 use_bfloat16=True, pred_residuals=False):
        super().__init__()
        in_ch = prog_channels + boundary_channels
        self.out_channels = out_channels
        self.N_pad = (last_kernel_size - 1) // 2
        self.pad = pad
        self.use_bfloat16 = use_bfloat16
        self.pred_residuals = pred_residuals
        self.register_parameter("positional_params", None)
        self.unet = UNetBackbone(in_ch, list(ch_width), list(dilation), list(n_layers), pad,
                                 upscale_factor, norm)
        self.decoder = nn.Conv2d(self.unet.out_channels, out_channels, last_kernel_size)

    def forward(self, prognostic, boundary, mask, cut=None):
        fts = torch.cat((prognostic, boundary), dim=1)
        if self.use_bfloat16:  # CPU reference path, like the original
            with torch.autocast("cpu", dtype=torch.bfloat16):
                fts = self.unet(fts, cut)
                fts = globe_pad(fts, self.N_pad, self.pad)
        else:  # plain graph: what gets traced for Trainium (the compiler casts matmuls itself)
            fts = self.unet(fts, cut)
            fts = globe_pad(fts, self.N_pad, self.pad)
        out = self.decoder(fts.to(self.decoder.weight.dtype))  # fp32 unless the whole model is bf16
        out = torch.where(mask > 0, out, 0.0)  # mask may be bool or 0/1 float
        if self.pred_residuals:
            out = prognostic[:, -self.out_channels:] + out
        return out


def read_state_dict(path: str) -> dict:
    """Find the model weights inside a Samudra checkpoint, whatever wrapper it uses."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        print("checkpoint top-level keys:", list(ck)[:10])

    def is_weights(d):
        return isinstance(d, dict) and any(str(k).endswith("decoder.weight") for k in d)

    sd = ck
    if not is_weights(sd):
        # prefer EMA weights when both are present, as the Samudra eval does
        for key in ("ema", "ema_state_dict", "ema_model", "model", "model_state_dict", "state_dict"):
            cand = ck.get(key) if isinstance(ck, dict) else None
            if isinstance(cand, dict) and not is_weights(cand):
                cand = next((v for v in cand.values() if is_weights(v)), cand)
            if is_weights(cand):
                print("using weights from key:", key)
                sd = cand
                break
    if not is_weights(sd):
        raise SystemExit(f"no model weights found in {path}; top-level keys: {list(ck)[:20]}")
    out = {}
    for k, v in sd.items():
        for p in ("module.", "_orig_mod.", "model.", "ema_model.", "module."):
            k = k.removeprefix(p)
        out[k] = v
    return out


def load_checkpoint(model: nn.Module, path: str):
    """Load a Samudra checkpoint into SamudraNet. Returns (missing, unexpected) key lists."""
    sd = read_state_dict(path)
    sd = {k: v for k, v in sd.items() if k != "n_averaged"}  # EMA bookkeeping, not a weight
    return model.load_state_dict(sd, strict=False)
