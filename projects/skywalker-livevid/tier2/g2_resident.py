"""Step 4 prototype: KV caches resident on the NeuronCore instead of crossing the host every call.

    python g2_resident.py --layers 1     one block, K/V as aliased state
    python g2_resident.py --layers 8     8 blocks chained in one graph, 16 aliased state tensors

Mechanism (torch_neuronx 2.9, xla_impl/trace.py): the caches are nn.Parameters of the traced module and
`torch_neuronx.trace(..., input_output_aliases={param: output_index})` marks graph output i as the new value of
that parameter. The traced NeuronModule keeps them in `.states`; `torch_neuronx.move_trace_to_device(traced, core)`
moves those states to the device, after which each call updates them in place there. This is the same
input/output aliasing NxD Inference's ModelBuilder uses for LLM KV caches.

Checks: 3 consecutive chunks vs the CPU fp32 port (same protocol as G1 parity: warmed full cache, each side feeds
its own state), then latency over 50 calls.
"""
import argparse
import json
import statistics
import time

import torch
import torch.nn as nn

from block_port import ART_DIR, BlockCfg, CacheState, CausalBlockPort, cosine, make_inputs, max_abs, rope_cos_sin, extract_weights

BF16 = torch.bfloat16
COMPILER_ARGS = ["--auto-cast=none", "--model-type=transformer"]


class ResidentBlocks(nn.Module):
    """N chained blocks whose caches are module state. Outputs: (y, k_0, v_0, k_1, v_1, ...)."""

    def __init__(self, ports, caches):
        super().__init__()
        self.ports = nn.ModuleList(ports)
        self.k = nn.ParameterList([nn.Parameter(k.clone(), requires_grad=False) for k, _ in caches])
        self.v = nn.ParameterList([nn.Parameter(v.clone(), requires_grad=False) for _, v in caches])

    def forward(self, x, e, context, rope_cos, rope_sin, attn_bias):
        out = []
        for port, k, v in zip(self.ports, self.k, self.v):
            x, nk, nv = port(x, e, context, rope_cos, rope_sin, k, v, attn_bias)
            out += [nk, nv]
        return (x, *out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    import torch_neuronx

    cfg, n = BlockCfg(), args.layers
    embed = extract_weights(0)["embed"]
    ports = []
    for i in range(n):
        m = CausalBlockPort(cfg)
        m.load_state_dict(torch.load(ART_DIR / f"block{i}_fp32.pt")["block"], strict=True)
        ports.append(m.eval().requires_grad_(False))

    # warm every layer's cache on CPU (6 cold-start chunks through the chain)
    states = [CacheState(cfg) for _ in range(n)]
    for f in range(cfg.num_kv_cache):
        x, e, ctx = make_inputs(cfg, embed, seed=100 + f)
        cos_, sin_ = rope_cos_sin(cfg, f)
        for port, st in zip(ports, states):
            x, nk, nv = port(x, e, ctx, cos_, sin_, st.k, st.v, st.bias())
            st.commit(nk, nv)
    bias = states[0].bias()
    cpu_kv = [(st.k, st.v) for st in states]

    x, e, ctx = make_inputs(cfg, embed, seed=1)
    cos_, sin_ = rope_cos_sin(cfg, 6)
    example = tuple(t.to(BF16) for t in (x, e, ctx, cos_, sin_, bias))
    module = ResidentBlocks(ports, cpu_kv).to(BF16)
    aliases = {}
    for i in range(n):
        aliases[module.k[i]] = 1 + 2 * i
        aliases[module.v[i]] = 2 + 2 * i
    work = ART_DIR / "work" / f"resident_{n}layers_{cfg.tag}"
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    traced = torch_neuronx.trace(module, example, input_output_aliases=aliases, compiler_workdir=str(work),
                                 compiler_args=COMPILER_ARGS)
    compile_s = time.perf_counter() - t0
    print(f"COMPILE OK {compile_s:.1f}s", flush=True)
    torch_neuronx.move_trace_to_device(traced, 0)
    print("states on:", sorted({str(p.device) for p in traced.states._parameters.values()}), flush=True)

    # CPU ports back to fp32 for the reference (module.to(BF16) converted them in place)
    for i, m in enumerate(ports):
        m.float()
    rows = []
    for c in range(args.chunks):
        frame = cfg.num_kv_cache + c
        x, e, ctx = make_inputs(cfg, embed, seed=100 + frame, timestep=(700.0, 500.0)[c % 2])
        cos_, sin_ = rope_cos_sin(cfg, frame)
        y_cpu, y_stale = x, x
        new_kv = []
        for port, (k, v), st in zip(ports, cpu_kv, states):
            y_stale = port(y_stale, e, ctx, cos_, sin_, st.k, st.v, bias)[0]  # cache frozen at its warm-up value
            y_cpu, nk, nv = port(y_cpu, e, ctx, cos_, sin_, k, v, bias)
            new_kv.append((nk, nv))
        cpu_kv = new_kv
        out = traced(*(t.to(BF16) for t in (x, e, ctx, cos_, sin_, bias)))
        y_dev = out[0].float().cpu()
        row = {"chunk": c, "y_cos": cosine(y_dev, y_cpu), "y_max_abs": max_abs(y_dev, y_cpu),
               "ref_abs_max": y_cpu.abs().max().item(), "outputs": len(out),
               "aliased_output_device": str(out[1].device), "aliased_output_shape": list(out[1].shape),
               # did the on-device state really advance? compare the state's effect against a frozen cache
               "state_effect_cos": cosine(y_dev - y_stale, y_cpu - y_stale) if c else None,
               "state_effect_ref_rms": (y_cpu - y_stale).pow(2).mean().sqrt().item()}
        try:
            k_last = out[-2].float().cpu()
            row["k_cache_last_layer_cos"] = cosine(k_last, cpu_kv[-1][0])
        except Exception as exc:  # aliased outputs may not be readable from the host
            row["k_cache_read_error"] = repr(exc)[:200]
        rows.append(row)
        print("chunk", json.dumps(row), flush=True)

    inp = tuple(t.to(BF16) for t in (x, e, ctx, cos_, sin_, bias))
    for _ in range(5):
        traced(*inp)
    lat, lat_y = [], []
    for _ in range(args.iters):
        t = time.perf_counter()
        out = traced(*inp)
        lat.append((time.perf_counter() - t) * 1e3)
    for _ in range(args.iters):
        t = time.perf_counter()
        out = traced(*inp)
        _ = out[0].cpu()
        lat_y.append((time.perf_counter() - t) * 1e3)

    def stats(xs):
        s = sorted(xs)
        return {"mean": round(statistics.mean(xs), 3), "p50": round(statistics.median(xs), 3),
                "p99": round(s[min(len(s) - 1, int(round(0.99 * (len(s) - 1))))], 3)}

    res = {"layers": n, "cfg": cfg.tag, "compile_s": round(compile_s, 1), "chunks": rows,
           "latency_ms_call": stats(lat), "latency_ms_call_plus_y_to_host": stats(lat_y),
           "latency_ms_per_layer": round(statistics.mean(lat_y) / n, 3)}
    (ART_DIR / f"g2_resident_{n}layers.json").write_text(json.dumps(res, indent=2) + "\n")
    print("RESIDENT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
