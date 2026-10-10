"""Find which Samudra op neuronx-cc rejects: compile each op alone (seconds each).

  python probe_ops.py            # all cases, each in its own process
  python probe_ops.py upsample   # one case
Prints one line per case: PASS / FAIL <error code>. Also a reusable checker for the agent loop:
any op an NKI kernel replaces can be probed here before a full-model compile.
"""
import os, sys, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def tiny_net(sc, **kw):
    return sc.SamudraNet(6, 2, 6, ch_width=(8, 8, 8, 8), use_bfloat16=False, **kw).eval()


def unet_prefix(sc, F, unet, n):
    """Run only the first n layers of the UNet, with the same skip logic as UNetBackbone.forward."""
    def fwd(fts):
        skips, count = [], 0
        for layer in list(unet.layers)[:n]:
            fts = layer(fts)
            if count < unet.num_steps:
                if isinstance(layer, sc.ConvNeXtBlock):
                    skips.append(fts); count += 1
            elif isinstance(layer, sc.ZonallyPeriodicBilinearUpsample):
                skip = skips[2 * unet.num_steps - count - 1]
                dh, dw = skip.shape[2] - fts.shape[2], skip.shape[3] - fts.shape[3]
                fts = F.pad(fts, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2]) + skip
                count += 1
        return fts
    return fwd


def cases(torch, nn, F, sc):
    x = lambda c=8, h=12, w=24: torch.randn(1, c, h, w)
    P, B = torch.randn(1, 6, 90, 180), torch.randn(1, 2, 90, 180)
    Kb = torch.rand(1, 1, 90, 180).expand(1, 6, 90, 180) > 0.3      # bool mask
    Kf = Kb.float()                                                   # same mask as 0/1 float
    net = tiny_net(sc)
    return {
        "conv3x3":        (nn.Conv2d(8, 8, 3), x()),
        "conv_dilated4":  (nn.Conv2d(8, 8, 3, dilation=4), x(h=20, w=24)),
        "conv1x1_same":   (nn.Conv2d(8, 4, 1, padding="same"), x()),
        "circular_pad":   (lambda t: F.pad(t, (2, 2, 0, 0), mode="circular"), x()),
        "globe_pad":      (lambda t: sc.globe_pad(t, 2, "circular"), x()),
        "avgpool":        (sc.AvgPool(), x()),
        "interp_bilinear":(lambda t: F.interpolate(t, scale_factor=(2, 2), mode="bilinear",
                                                   align_corners=False), x()),
        "upsample":       (sc.ZonallyPeriodicBilinearUpsample(), x()),
        "capped_gelu":    (sc.CappedGELU(), x()),
        "instancenorm":   (nn.InstanceNorm2d(8), x()),
        "where_mask":     (lambda t: torch.where(t > 0, t, 0.0), x()),
        "convnext_d2":    (sc.ConvNeXtBlock(8, 8, dilation=2), x()),
        "convnext_8to16": (sc.ConvNeXtBlock(8, 16, dilation=1), x()),
        # --- pieces that only appear in the full model ---
        "bool_input":     (lambda t, k: torch.where(k > 0, t, 0.0), (x(6, 90, 180), Kb)),
        "bool_input_nocmp":(lambda t, k: torch.where(k, t, 0.0), (x(6, 90, 180), Kb)),
        "cat":            (lambda a, b: torch.cat((a, b), 1), (P, B)),
        "avgpool_odd":    (sc.AvgPool(), x(8, 45, 90)),
        "skip_pad_add":   (lambda a, s: F.pad(a, [0, 1, 0, 1]) + s, (x(8, 22, 44), x(8, 23, 45))),
        "unet_only_2deg": (net.unet, torch.cat((P, B), 1)),
        "tiny_floatmask": (net, (P, B, Kf)),
        "tiny_samudra":   (net, (P, B, Kb)),
        # middle block alone at its real 5x11 size (dilation 8, pad 8 > height 5)
        "mid_block_5x11": (sc.ConvNeXtBlock(8, 8, dilation=8), x(8, 5, 11)),
        "circ_pad8_w11":  (lambda t: sc.globe_pad(t, 8, "circular"), x(8, 5, 11)),
        # size sweep: which ops break as the grid grows to 90x180?
        **{f"sz_{op}_{h}x{w}": (m, x(8, h, w))
           for (h, w) in [(24, 48), (45, 90), (64, 128), (90, 180)]
           for op, m in [("conv3", nn.Conv2d(8, 16, 3)), ("inorm", nn.InstanceNorm2d(8)),
                         ("gelu", sc.CappedGELU()), ("block", sc.ConvNeXtBlock(8, 8))]},
        **{f"unet_L{n:02d}": (unet_prefix(sc, F, net.unet, n), torch.cat((P, B), 1))
           for n in range(1, len(net.unet.layers) + 1)},
    }


def run_one(name):
    sys.path.insert(0, os.environ.get("XLA_PLUGIN_DIR", "/workspace/xla_plugin"))
    sys.path.insert(0, str(HERE))
    os.environ.setdefault("PJRT_DEVICE", "NEURON")
    import torch, torch.nn as nn, torch.nn.functional as F, torch_xla
    import samudra_core as sc
    torch.manual_seed(0)
    allc = cases(torch, nn, F, sc)
    mod, inp = allc[name]
    net = allc["tiny_samudra"][0]
    inp = inp if isinstance(inp, tuple) else (inp,)
    d = torch_xla.device()
    with torch.no_grad():
        ref = mod(*inp)
        if isinstance(mod, nn.Module):
            mod = mod.to(d)
        elif name.startswith("unet_L"):
            net.unet.to(d)
        y = mod(*[t.to(d) for t in inp])
    torch_xla.sync()
    err = ((y.cpu().float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)).item()
    print(f"RESULT PASS rel_err={err:.2e} shape={tuple(y.shape)}")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        return run_one(sys.argv[2])
    import torch, torch.nn as nn, torch.nn.functional as F  # noqa: just for the case list
    sys.path.insert(0, str(HERE)); import samudra_core as sc
    allnames = list(cases(torch, nn, F, sc))
    names = [n for a in (sys.argv[1:] or allnames) for n in allnames if n == a or
             (a.endswith("*") and n.startswith(a[:-1]))]
    env = dict(os.environ, PJRT_DEVICE="NEURON")
    env.setdefault("NEURON_RT_VISIBLE_CORES", "2")
    for n in names:
        try:
            r = subprocess.run([sys.executable, __file__, "--one", n], env=env,
                               capture_output=True, text=True, timeout=600)
            out = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            out = "TIMEOUT"
        line = next((l for l in out.splitlines() if l.startswith("RESULT")), None)
        if line:
            print(f"{n:16s} {line[7:]}", flush=True)
        else:
            import re
            code = re.findall(r"NCC_\w+|not available|TIMEOUT|\w+Error: [^\n]{0,120}", out)
            print(f"{n:16s} FAIL {code[-1] if code else out.strip().splitlines()[-1:]}", flush=True)
            Path("probe_logs").mkdir(exist_ok=True)
            Path(f"probe_logs/{n}.log").write_text(out)


if __name__ == "__main__":
    main()
