"""Probe: why is the first UNet block slow on Trainium, and does reshaping its input help?

Builds the real 1 degree model, takes UNet layer 0 (162 -> 280 channels, 180 x 360) and runs
exact variants of it on one NeuronCore. Each variant is checked against the original block on
the CPU and timed (forward + sync + wait, median of --iters).

    PYTHONPATH=/workspace/xla_plugin PJRT_DEVICE=NEURON NEURON_RT_VISIBLE_CORES=3 \
        python probe_block0.py --ckpt /workspace/samudra/onedeg/ema_ckpt.pt
"""
import argparse
import copy
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from samudra_core import SamudraNet, load_checkpoint  # noqa: E402


def pad_input_channels(block: nn.Module, total: int) -> nn.Module:
    """Return a copy of ``block`` that accepts ``total`` input channels.

    The extra input channels get zero weights in the first 3x3 convolution and in the 1x1 skip
    convolution, so with zero-filled extra inputs the output is unchanged.
    """
    block = copy.deepcopy(block)
    first = block.convblock[0]
    w = first.weight.data
    extra = total - w.shape[1]
    new = nn.Conv2d(total, first.out_channels, first.kernel_size, dilation=first.dilation,
                    bias=first.bias is not None)
    new.weight.data = torch.cat([w, w.new_zeros(w.shape[0], extra, *w.shape[2:])], dim=1)
    if first.bias is not None:
        new.bias.data = first.bias.data.clone()
    block.convblock[0] = new
    skip = block.skip_module
    ws = skip.weight.data
    new_skip = nn.Conv2d(total, skip.out_channels, 1, padding="same", bias=skip.bias is not None)
    new_skip.weight.data = torch.cat([ws, ws.new_zeros(ws.shape[0], extra, 1, 1)], dim=1)
    if skip.bias is not None:
        new_skip.bias.data = skip.bias.data.clone()
    block.skip_module = new_skip
    block.in_channels = total
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--variants", default="A,B,C,D")
    args = ap.parse_args()
    torch.set_num_threads(4)

    model = SamudraNet(154, 8, 154, norm="batch", use_bfloat16=False).eval()
    missing, unexpected = load_checkpoint(model, args.ckpt)
    assert not missing and not unexpected
    block0 = model.unet.layers[0]

    g = torch.Generator().manual_seed(1)
    prog = torch.randn(1, 154, 180, 360, generator=g)
    bnd = torch.randn(1, 8, 180, 360, generator=g)
    with torch.inference_mode():
        ref = block0(torch.cat((prog, bnd), dim=1))
    ref_rms = ref.pow(2).mean().sqrt().item()

    plugin = "/workspace/xla_plugin"
    if os.path.isdir(plugin) and plugin not in sys.path:
        sys.path.insert(0, plugin)
    import torch_xla
    import torch_xla.core.xla_model as xm

    dev = torch_xla.device()
    p_d, b_d = prog.to(dev), bnd.to(dev)
    x162 = torch.cat((p_d, b_d), dim=1)
    torch_xla.sync()  # materialized 162-channel input, for B
    zeros = {}
    for total in (192, 256):
        zeros[total] = torch.zeros(1, total - 162, 180, 360, device=dev)
    xpad = {t: torch.cat((x162, zeros[t]), dim=1) for t in (192, 256)}
    torch_xla.sync()  # materialized padded inputs, for C and D

    blk = copy.deepcopy(block0).to(dev)
    blk192 = pad_input_channels(block0, 192).to(dev)
    blk256 = pad_input_channels(block0, 256).to(dev)
    variants = {
        "A": ("concat inside the program (as in --segments 1)",
              lambda: blk(torch.cat((p_d, b_d), dim=1))),
        "B": ("input already concatenated (162 ch)", lambda: blk(x162)),
        "C": ("input zero-padded to 256 ch", lambda: blk256(xpad[256])),
        "D": ("input zero-padded to 192 ch", lambda: blk192(xpad[192])),
    }

    for key in args.variants.split(","):
        desc, fn = variants[key]

        def step():
            y = fn()
            torch_xla.sync()
            xm.wait_device_ops()
            return y

        with torch.no_grad():
            t0 = time.perf_counter()
            out = step().cpu()
            compile_s = time.perf_counter() - t0
            step()
            times = []
            for _ in range(args.iters):
                t = time.perf_counter()
                step()
                times.append(1000 * (time.perf_counter() - t))
        rel = (out - ref).pow(2).mean().sqrt().item() / ref_rms
        print(f"{key}: {desc:46s} median {statistics.median(times):7.2f} ms  "
              f"(min {min(times):6.2f})  compile+first {compile_s:5.1f} s  rel_err {rel:.1e}",
              flush=True)


if __name__ == "__main__":
    main()
