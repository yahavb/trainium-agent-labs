"""TAEHV with its 9 memories resident on the NeuronCore (same mechanism as g2_resident.py).

    python vae_resident.py

The memories are nn.Parameters of the traced module; `input_output_aliases` marks graph output 1 + i as the new
value of memory i, and `move_trace_to_device` keeps them on the device. Each call then only moves the 4 frames
or the 1 latent across the host boundary. Checks parity chunk by chunk against CPU TAEHV (a stale state would show
up from chunk 1 on), then latency over 100 calls including bringing the result back to the host.
"""
import json
import statistics
import time

import torch
import torch.nn as nn

from vae_neuron import BF16, COMPILER_ARGS, VAE_DIR, StreamTAE, cos, example_inputs, load_taehv


class ResidentTAE(nn.Module):
    def __init__(self, inner, mems):
        super().__init__()
        self.inner = inner
        self.mems = nn.ParameterList([nn.Parameter(m.clone(), requires_grad=False) for m in mems])

    def forward(self, x):
        return self.inner(x, *self.mems)


def main():
    torch.set_grad_enabled(False)
    import torch_neuronx
    tae = load_taehv()
    r = torch.load(VAE_DIR / "ref.pt")
    chunks = {"enc": (r["x32"].split(4), [l for l in r["lat_tae"][0].split(1)]),
              "dec": ([l for l in r["lat_tae"][0].split(1)], r["dec_tae"][0].split(4))}
    res = {}
    for which, seq in (("enc", tae.encoder), ("dec", tae.decoder)):
        inner = StreamTAE(seq).eval()
        ex = example_inputs(inner, which)
        module = ResidentTAE(inner, ex[1:]).to(BF16)
        aliases = {p: 1 + i for i, p in enumerate(module.mems)}
        work = VAE_DIR / "work" / f"resident_{which}"
        work.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        traced = torch_neuronx.trace(module, (ex[0].to(BF16),), input_output_aliases=aliases,
                                     compiler_workdir=str(work), compiler_args=COMPILER_ARGS)
        compile_s = time.perf_counter() - t0
        torch_neuronx.move_trace_to_device(traced, 0)
        devices = sorted({str(p.device) for p in traced.states._parameters.values()})
        print(f"{which}: COMPILE OK {compile_s:.1f}s, states on {devices}", flush=True)
        xs, refs = chunks[which]
        cosines = [cos(traced(x.to(BF16))[0].float().cpu(), y) for x, y in zip(xs, refs)]
        print(f"{which}: per-chunk cos vs CPU TAEHV {[round(c, 6) for c in cosines]}", flush=True)
        x = xs[1].to(BF16)
        ms = []
        for i in range(105):
            t = time.perf_counter()
            _ = traced(x)[0].cpu()
            if i >= 5:
                ms.append((time.perf_counter() - t) * 1e3)
        ms.sort()
        res[which] = {"compile_s": round(compile_s, 1), "state_devices": devices, "cos_per_chunk": cosines,
                      "min_cos": min(cosines), "mean_ms": statistics.mean(ms), "p50_ms": ms[50], "p99_ms": ms[98]}
        print(which, json.dumps(res[which]), flush=True)
    (VAE_DIR / "vae_resident.json").write_text(json.dumps(res, indent=2) + "\n")
    print("RESIDENT OK", flush=True)


if __name__ == "__main__":
    main()
