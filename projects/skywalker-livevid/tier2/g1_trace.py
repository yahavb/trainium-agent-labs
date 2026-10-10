"""Gate G1: trace the rewritten causal block on Neuron (bf16, fixed shapes, fixed-size KV cache) and check it.

    python g1_trace.py compile [--height 512 --width 512]   torch_neuronx.trace -> artifacts/tier2/g1_block0_<tag>_bf16.pt
    python g1_trace.py parity  [...]                        3 consecutive chunks, Neuron bf16 vs CPU fp32, caches fed forward
    python g1_trace.py parity --cold --chunks 8             same from an empty cache (attn_bias masking on Neuron)
    python g1_trace.py cacheprobe                           effect of the cache contents on y, Neuron vs CPU
    python g1_trace.py bench   [...]                        latency over 50 calls on one core + FPS estimate

Each stage writes artifacts/tier2/g1_<tag>_<stage>.json. Run with NEURON_RT_VISIBLE_CORES=0 (one core).
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from block_port import ART_DIR, BlockCfg, CacheState, cosine, load_port, make_inputs, max_abs, rope_cos_sin

BF16 = torch.bfloat16
GATE = 0.999
NUM_LAYERS = 30            # Wan2.1-T2V-1.3B
DENOISE_STEPS = 2          # demo: --step 2 -> denoising_step_list [700, 500]
PIXEL_FRAMES_PER_CHUNK = 4  # one latent frame = 4 output frames (VAE 4x temporal)


def paths(cfg, tag):
    stem = f"g1_block0_{cfg.tag}{tag}"
    return {s: ART_DIR / f"{stem}_{s}.json" for s in ("compile", "parity", "bench")} | {
        "artifact": ART_DIR / f"{stem}_bf16.pt", "workdir": ART_DIR / "work" / stem}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


def to_bf16(*ts):
    return tuple(t.to(BF16) for t in ts)


def warm_cache(port, cfg, embed, chunks):
    """Fill the cache on CPU fp32 from a cold start so the Neuron run starts from real K/V values."""
    st = CacheState(cfg)
    for i in range(chunks):
        x, e, ctx = make_inputs(cfg, embed, seed=100 + i)
        cos_, sin_ = rope_cos_sin(cfg, start_frame=i)
        _, nk, nv = port(x, e, ctx, cos_, sin_, st.k, st.v, st.bias())
        st.commit(nk, nv)
    return st


def do_compile(cfg, args, p):
    import torch_neuronx
    port, data = load_port(cfg, manual_attn=args.manual_attn)
    st = warm_cache(port, cfg, data["embed"], 1)
    x, e, ctx = make_inputs(cfg, data["embed"], seed=1)
    cos_, sin_ = rope_cos_sin(cfg, start_frame=1)
    example = to_bf16(x, e, ctx, cos_, sin_, st.k, st.v, st.bias())
    model = port.to(BF16)
    compiler_args = args.compiler_args.split()
    p["workdir"].mkdir(parents=True, exist_ok=True)
    print(f"compiling {cfg.tag} args={compiler_args} manual_attn={args.manual_attn}", flush=True)
    print("input shapes:", [tuple(t.shape) for t in example], flush=True)
    t0 = time.perf_counter()
    traced = torch_neuronx.trace(model, example, compiler_workdir=str(p["workdir"]), compiler_args=compiler_args)
    compile_s = time.perf_counter() - t0
    torch.jit.save(traced, str(p["artifact"]))
    meta = {"cfg": cfg.tag, "compile_s": round(compile_s, 1), "compiler_args": compiler_args,
            "manual_attn": args.manual_attn, "artifact": str(p["artifact"]),
            "artifact_bytes": os.path.getsize(p["artifact"]),
            "input_shapes": [list(t.shape) for t in example], "dtype": "bf16",
            "torch": torch.__version__, "torch_neuronx": torch_neuronx.__version__}
    write_json(p["compile"], meta)
    print(f"COMPILE OK compile_s={compile_s:.1f}", flush=True)


def do_parity(cfg, args, p):
    """Steady state by default (CPU-warmed full cache, zero bias). --cold starts both sides from an empty cache,
    which exercises attn_bias masking and the host sink bookkeeping on Neuron as well."""
    import torch_neuronx  # noqa: F401  (registers the Neuron ops torch.jit.load needs)
    port, data = load_port(cfg)
    neuron = torch.jit.load(str(p["artifact"]))
    n, l = cfg.num_kv_cache, cfg.frame_tokens
    st_cpu = CacheState(cfg) if args.cold else warm_cache(port, cfg, data["embed"], n)
    st_dev = CacheState(cfg, dtype=BF16)
    st_dev.k, st_dev.v = to_bf16(st_cpu.k, st_cpu.v)
    st_dev.frames, st_dev.seen = list(st_cpu.frames), st_cpu.seen
    rows = []
    for c in range(args.chunks):
        frame = st_cpu.seen
        x, e, ctx = make_inputs(cfg, data["embed"], seed=100 + frame, timestep=(700.0, 500.0)[c % 2])
        cos_, sin_ = rope_cos_sin(cfg, start_frame=frame)
        bias = st_cpu.bias()
        y_cpu, k_cpu, v_cpu = port(x, e, ctx, cos_, sin_, st_cpu.k, st_cpu.v, bias)
        y_dev, k_dev, v_dev = neuron(*to_bf16(x, e, ctx, cos_, sin_), st_dev.k, st_dev.v, st_dev.bias())
        row = {"chunk": c, "frame": frame, "masked_slots": int((bias < 0).sum().item()) // l}
        for name, a, b in (("y", y_dev, y_cpu), ("k_cache", k_dev, k_cpu), ("v_cache", v_dev, v_cpu),
                           ("k_new", k_dev[:, -l:], k_cpu[:, -l:]), ("v_new", v_dev[:, -l:], v_cpu[:, -l:]),
                           ("y_minus_x", y_dev.float() - x, y_cpu - x)):
            a = a.float()
            row[name] = {"cos": cosine(a, b), "max_abs": max_abs(a, b), "ref_abs_max": b.abs().max().item(),
                         "finite": bool(torch.isfinite(a).all())}
        rows.append(row)
        st_cpu.commit(k_cpu, v_cpu)
        st_dev.commit(k_dev, v_dev)
        print(f"chunk {c} (frame {frame}, masked slots {row['masked_slots']}): " + "  ".join(
            f"{k} cos={v['cos']:.6f} max_abs={v['max_abs']:.4f}" for k, v in row.items() if isinstance(v, dict)),
            flush=True)
    gated = [r[k]["cos"] for r in rows for k in ("y", "k_cache", "v_cache")]
    finite = all(r[k]["finite"] for r in rows for k in ("y", "k_cache", "v_cache"))
    res = {"cfg": cfg.tag, "gate": GATE, "cold_start": args.cold, "chunks": rows, "min_cosine": min(gated),
           "pass": bool(min(gated) >= GATE and finite),
           "note": "Neuron feeds its own bf16 caches forward; CPU feeds its own fp32 caches. Both start from the "
                   "same cache (CPU-warmed and full, or empty with --cold). k_new/v_new = the slice written by "
                   "this chunk only. y_minus_x = what the block adds to the residual stream (stricter than y; "
                   "not gated). k_cache/v_cache are the graph outputs before the host sink copy."}
    out = p["parity"].with_name(p["parity"].stem + ("_cold" if args.cold else "") + ".json")
    write_json(out, res)
    print(f"G1 PARITY{' (cold start)' if args.cold else ''} min_cosine={res['min_cosine']:.6f} gate={GATE} -> "
          f"{'PASS' if res['pass'] else 'FAIL'}", flush=True)


def do_cacheprobe(cfg, args, p):
    """Does the Neuron graph actually READ the cache? y barely moves with cache contents in layer 0, so compare the
    cache's effect instead: (y with the real cache) - (y with an all-zero cache), Neuron vs CPU."""
    import torch_neuronx  # noqa: F401
    port, data = load_port(cfg)
    neuron = torch.jit.load(str(p["artifact"]))
    n = cfg.num_kv_cache
    st = warm_cache(port, cfg, data["embed"], n)
    x, e, ctx = make_inputs(cfg, data["embed"], seed=100 + n)
    cos_, sin_ = rope_cos_sin(cfg, start_frame=n)
    bias, zk = st.bias(), torch.zeros_like(st.k)
    d_cpu = port(x, e, ctx, cos_, sin_, st.k, st.v, bias)[0] - port(x, e, ctx, cos_, sin_, zk, zk, bias)[0]
    small = to_bf16(x, e, ctx, cos_, sin_)
    d_dev = (neuron(*small, *to_bf16(st.k, st.v, bias))[0].float()
             - neuron(*small, *to_bf16(zk, zk, bias))[0].float())
    res = {"cfg": cfg.tag, "cache_effect_cos": cosine(d_dev, d_cpu), "cache_effect_max_abs_err": max_abs(d_dev, d_cpu),
           "cache_effect_ref_abs_max": d_cpu.abs().max().item(), "cache_effect_ref_rms": d_cpu.pow(2).mean().sqrt().item(),
           "cache_effect_err_rms": (d_dev - d_cpu).pow(2).mean().sqrt().item()}
    write_json(p["parity"].with_name(p["parity"].stem.replace("_parity", "_cacheprobe") + ".json"), res)
    print("CACHEPROBE " + json.dumps(res), flush=True)


def do_bench(cfg, args, p):
    import torch_neuronx  # noqa: F401
    port, data = load_port(cfg)
    neuron = torch.jit.load(str(p["artifact"]))
    st = warm_cache(port, cfg, data["embed"], 1)
    x, e, ctx = make_inputs(cfg, data["embed"], seed=5)
    cos_, sin_ = rope_cos_sin(cfg, start_frame=7)
    inp = to_bf16(x, e, ctx, cos_, sin_, st.k, st.v, torch.zeros_like(st.bias()))
    for _ in range(args.warmup):
        neuron(*inp)
    lat = []
    for _ in range(args.iters):
        t = time.perf_counter()
        neuron(*inp)
        lat.append((time.perf_counter() - t) * 1e3)
    # same loop but feeding the returned caches back in, as the real pipeline does
    lat_fb = []
    k, v = inp[5], inp[6]
    for _ in range(args.iters):
        t = time.perf_counter()
        _, k, v = neuron(*inp[:5], k, v, inp[7])
        lat_fb.append((time.perf_counter() - t) * 1e3)

    def stats(xs):
        s = sorted(xs)
        return {"mean": round(statistics.mean(xs), 3), "p50": round(statistics.median(xs), 3),
                "p99": round(s[min(len(s) - 1, int(round(0.99 * (len(s) - 1))))], 3),
                "min": round(s[0], 3), "max": round(s[-1], 3), "iters": len(xs)}

    st_ = stats(lat)
    sec_per_frame = st_["mean"] / 1e3 * NUM_LAYERS * DENOISE_STEPS / PIXEL_FRAMES_PER_CHUNK
    res = {"cfg": cfg.tag, "visible_cores": os.environ.get("NEURON_RT_VISIBLE_CORES"),
           "latency_ms": st_, "latency_ms_cache_feedback": stats(lat_fb),
           "estimate": {"formula": "block_latency * layers * denoising_steps_per_chunk / frames_per_chunk",
                        "layers": NUM_LAYERS, "denoising_steps_per_chunk": DENOISE_STEPS,
                        "frames_per_chunk": PIXEL_FRAMES_PER_CHUNK,
                        "sec_per_chunk": round(st_["mean"] / 1e3 * NUM_LAYERS * DENOISE_STEPS, 4),
                        "sec_per_output_frame": round(sec_per_frame, 4),
                        "fps_one_core": round(1.0 / sec_per_frame, 3)}}
    write_json(p["bench"], res)
    print("BENCH " + json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["compile", "parity", "cacheprobe", "bench"])
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--text-len", type=int, default=512)
    ap.add_argument("--manual-attn", action="store_true", help="matmul+softmax instead of F.scaled_dot_product_attention")
    ap.add_argument("--compiler-args", default="--auto-cast=none --model-type=transformer")
    ap.add_argument("--tag", default="", help="suffix for artifact names (e.g. _O1)")
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--cold", action="store_true", help="parity from an empty cache (exercises attn_bias masking)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    cfg = BlockCfg(height=args.height, width=args.width, text_len=args.text_len)
    p = paths(cfg, args.tag)
    {"compile": do_compile, "parity": do_parity, "cacheprobe": do_cacheprobe, "bench": do_bench}[args.stage](cfg, args, p)


if __name__ == "__main__":
    main()
