"""Trainium adapter for runners/trainium_runner.py (project 03, mechanical sympathy).

Runs Samudra2 one-degree forward_once on one Trainium2 NeuronCore through torch-xla, on the
exact inputs stored in the CPU fixture (prognostic, boundary, label_mask).

    python runners/trainium_runner.py --adapter trainium/candidate_adapter.py \
        --fixture fixtures/cpu_reference.npz --candidate-output runs/trainium/candidate.npz \
        --metrics-json runs/trainium/metrics.json --precision float32

Precision (--precision):
  float32          plain fp32 on the chip (rel. RMS error vs CPU ~1e-6 on our tests)
  bfloat16         fp32 model, compiler auto-casts matmuls/convs to bf16
                   (NEURON_CC_FLAGS=--auto-cast=matmult --auto-cast-type=bf16; ~1% rel. RMS).
                   Whole-model bf16 (model.to(bfloat16)) crashes neuronx-cc 2.27 on the 1-degree
                   grid with NCC_IBIR229, so it is not offered here.

Model code: samudra_core.py next to this file, a standalone inference copy of Samudra's
samudra_om4_v2 UNet (same parameter names; verified bit-identical to the original modules).
One change makes it compile on Trainium: circular padding is written as slice + concat
(wrap_lon), because torch-xla lowers F.pad(mode="circular") with a large boolean constant that
neuronx-cc 2.27 cannot parse (NCC_EMOD021). The change is bit-identical on CPU.

samudra_om4_v2 settings that this matches: pred_residuals=false (so _assemble_prediction returns
the decoding unchanged), use_bfloat16=false (the CPU reference is fp32), norm="batch",
pad="circular", no positional parameters, no 3-D coordinates.

Environment (optional): SAMUDRA_CKPT (checkpoint path; default = manifest checkpoint.path_on_seat
relative to the project folder), NEURON_RT_VISIBLE_CORES (default 2; vLLM usually holds 0-1),
XLA_PLUGIN_DIR (default /workspace/xla_plugin, where libneuronxla was pip-installed).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

AUTOCAST_BF16 = "--auto-cast=matmult --auto-cast-type=bf16"


class Prepared:
    def __init__(self, model, args, torch_xla, xm, info):
        self.model, self.args, self.torch_xla, self.xm, self.info = model, args, torch_xla, xm, info
        self.out = None

    def run(self):
        import torch
        with torch.no_grad():
            self.out = self.model(*self.args)
        self.torch_xla.sync()          # cut the graph and launch it
        return self.out

    def synchronize(self):
        self.xm.wait_device_ops()      # wait until the chip has finished

    def metrics(self):
        return self.info


def prepare(inputs, context):
    precision = str(context.get("precision", "float32")).lower()
    if precision in ("float32", "fp32"):
        cc_flags = ""
    elif precision in ("bfloat16", "bf16", "autocast-bf16"):
        cc_flags = AUTOCAST_BF16
    else:
        raise ValueError(f"unsupported precision {precision!r}: use float32 or bfloat16")

    # Device setup must happen before torch_xla is imported.
    os.environ.setdefault("PJRT_DEVICE", "NEURON")
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "2")
    if cc_flags:
        os.environ["NEURON_CC_FLAGS"] = (os.environ.get("NEURON_CC_FLAGS", "") + " " + cc_flags).strip()
    plugin = os.environ.get("XLA_PLUGIN_DIR", "/workspace/xla_plugin")
    if os.path.isdir(plugin) and plugin not in sys.path:
        sys.path.insert(0, plugin)
    sys.path.insert(0, str(HERE))

    import numpy as np
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    from samudra_core import SamudraNet, load_checkpoint, read_state_dict

    for key in ("prognostic", "boundary", "label_mask"):
        if key not in inputs:
            raise ValueError(f"fixture is missing {key!r}; found {sorted(inputs)}")

    ckpt = os.environ.get("SAMUDRA_CKPT") or str(
        PROJECT / context["manifest"]["checkpoint"]["path_on_seat"])
    sd = read_state_dict(ckpt)
    prog_ch = sd["decoder.weight"].shape[0]
    in_ch = sd["unet.layers.0.convblock.0.weight"].shape[1]
    norm = "batch" if any("running_mean" in k for k in sd) else "instance"
    model = SamudraNet(prog_ch, in_ch - prog_ch, prog_ch, norm=norm, use_bfloat16=False).eval()
    missing, unexpected = load_checkpoint(model, ckpt)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")

    prog = torch.from_numpy(np.ascontiguousarray(inputs["prognostic"])).float()
    bnd = torch.from_numpy(np.ascontiguousarray(inputs["boundary"])).float()
    mask = torch.from_numpy(np.ascontiguousarray(inputs["label_mask"]))
    if prog.shape[1] != prog_ch or bnd.shape[1] != in_ch - prog_ch:
        raise ValueError(f"input channels {prog.shape[1]}+{bnd.shape[1]} do not match the "
                         f"checkpoint's {prog_ch}+{in_ch - prog_ch}")

    dev = torch_xla.device()
    model = model.to(dev)
    args = (prog.to(dev), bnd.to(dev), mask.to(dev))
    prepared = Prepared(model, args, torch_xla, xm, {
        "adapter": "samudra_core torch-xla",
        "neuron_rt_visible_cores": os.environ.get("NEURON_RT_VISIBLE_CORES"),
        "neuron_cc_flags": os.environ.get("NEURON_CC_FLAGS", ""),
        "checkpoint": ckpt,
        "norm": norm,
        "channels": [prog_ch, in_ch - prog_ch],
    })
    # Compile inside prepare, as the runner contract requires: the first call traces the graph
    # and neuronx-cc compiles it (minutes the first time, cached afterwards).
    prepared.run()
    prepared.synchronize()
    return prepared
