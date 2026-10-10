"""Benchmark: Qwen3 decode layers, megakernel (one NKI launch for L layers) vs per-op torch.compile.

Both run on the same free logical NeuronCore with weights resident on the device, batch=1, one new
token. Weights are random (timing does not depend on values). Shapes default to Qwen3-8B.

  NEURON_RT_VISIBLE_CORES=3 python kernels/bench_mega.py --layers 1 2 4
  NEURON_RT_VISIBLE_CORES=3 python kernels/bench_mega.py --tiny --layers 2
"""
import argparse, json, os, sys, time
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "3")
sys.path.insert(0, os.path.dirname(__file__))
import torch
import libtorch_neuronx_lite  # registers the neuron device + torch.compile backends
import torch.distributed as dist
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from qwen3_ref import decode_layers_ref, rope_tables, decode_mask
import qwen3_megakernel

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4])
ap.add_argument("--tiny", action="store_true")
ap.add_argument("--S_ctx", type=int, default=1024)
ap.add_argument("--S_max", type=int, default=2048)
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--no-baseline", action="store_true")
ap.add_argument("--no-mega", action="store_true")
ap.add_argument("--hwdge-oproj", action="store_true", help="hardware-DGE weight loads in output_projection_tkg (o_proj and LM head)")
ap.add_argument("--out", default=None, help="append JSON lines here")
ap.add_argument("--kernel", default="qwen3_decode_layers", help="kernel function in qwen3_megakernel.py")
ap.add_argument("--kw", default="{}", help="extra compile-time kwargs for the megakernel, as JSON")
args = ap.parse_args()
if args.hwdge_oproj:
    # libtorch's NKI trace cache keys on the top-level kernel's source only, so a replaced nkilib helper
    # would silently reuse the unpatched trace (and a patched trace would poison later unpatched runs)
    os.environ["NEURON_LIBTORCH_DISABLE_COMPILE_CACHE"] = "1"
    from qwen3_decode_step import use_hwdge_output_projection
    use_hwdge_output_projection()

if args.tiny:
    H, I, q, kv, d = 512, 1024, 4, 2, 128
else:  # Qwen3-8B
    H, I, q, kv, d = 4096, 12288, 32, 8, 128
B, t, eps = 1, args.S_ctx - 64, 1e-6
S_ctx, S_max = args.S_ctx, args.S_max
lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
dev = "neuron:0"

if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29581")
    dist.init_process_group("gloo", rank=0, world_size=1)


def make_inputs(L, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(0)
    def rn(*shape, scale=0.02): return (torch.randn(*shape, generator=g) * scale).to(dtype)
    W = dict(
        W_qkv=rn(L, H, (q + 2 * kv) * d), W_out=rn(L, q * d, H), W_gate=rn(L, H, I), W_up=rn(L, H, I),
        W_down=rn(L, I, H), g_attn=torch.ones(L, 1, H, dtype=dtype), g_mlp=torch.ones(L, 1, H, dtype=dtype),
        g_q=torch.ones(L, 1, d, dtype=dtype), g_k=torch.ones(L, 1, d, dtype=dtype),
    )
    K_cache = rn(L, B, kv, d, S_max, scale=1.0); V_cache = rn(L, B, kv, S_max, d, scale=1.0)
    X = rn(B, 1, H, scale=1.0)
    pos = torch.full((B, 1), t, dtype=torch.int64)
    cos, sin = rope_tables(torch.full((B,), t), d, 1e6)
    mask = decode_mask(pos, S_ctx, q)
    return W, K_cache, V_cache, X, pos, cos.to(dtype), sin.to(dtype), mask


def bytes_weights(L):
    return 2 * L * (H * (q + 2 * kv) * d + q * d * H + 3 * H * I)


def timeit(fn, iters):
    for _ in range(3):
        fn()[0].reshape(-1)[0].cpu()        # warm
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()[0].reshape(-1)[0].cpu()    # the executor is async and its queue is shallow: sync every call
    return (time.perf_counter() - t0) / iters * 1000.0


mega_kw = json.loads(args.kw)
results = []
for L in args.layers:
    W, K_cache, V_cache, X, pos, cos, sin, mask = make_inputs(L)
    row = dict(layers=L, hwdge_oproj=args.hwdge_oproj, kernel=args.kernel, mega_kw=mega_kw, H=H, I=I, q=q, kv=kv, d=d, S_ctx=S_ctx, S_max=S_max,
               weight_bytes=bytes_weights(L), kv_read_bytes=2 * L * B * kv * S_ctx * d * 2)
    floor_ms = (row["weight_bytes"] + row["kv_read_bytes"]) / 736e9 * 1000  # 2 x 368 GB/s per logical core (LNC=2)
    row["hbm_floor_ms_at_736GBps"] = round(floor_ms, 3)

    if not args.no_baseline:
        dW = {k: v.to(dev) for k, v in W.items()}
        dK, dV, dX = K_cache.to(dev), V_cache.to(dev), X.to(dev)
        dpos, dcos, dsin, dmask = pos.to(dev), cos.to(dev), sin.to(dev), mask.to(dev)

        def base_fn(X=dX):
            out, _, _ = decode_layers_ref(X, dW["W_qkv"], dW["W_out"], dW["W_gate"], dW["W_up"], dW["W_down"],
                                          dW["g_attn"], dW["g_mlp"], dW["g_q"], dW["g_k"], dK, dV, dcos, dsin,
                                          dmask, dpos, L, eps, q, kv, d, compute_dtype=torch.bfloat16)
            return (out,)
        base_c = torch.compile(base_fn, backend="neuron_libtorch", fullgraph=True)
        t0 = time.time(); base_c(); base_c()[0].reshape(-1)[0].cpu(); row["baseline_compile_s"] = round(time.time() - t0, 1)
        row["baseline_ms"] = round(timeit(base_c, args.iters), 3)
        print(f"L={L}: per-op torch.compile(neuron_libtorch): {row['baseline_ms']:.3f} ms  (compile {row['baseline_compile_s']}s)", flush=True)
        del dW, dK, dV, base_c

    if not args.no_mega:
        mW = {k: v.to(dev) for k, v in W.items()}
        mK, mV, mX = K_cache.to(dev), V_cache.to(dev), X.to(dev)
        mpos = pos.to(torch.int32).to(dev)  # uint32 is not a torch dtype on this path; int32 bits are identical
        mcos, msin, mmask = cos.to(dev), sin.to(dev), mask.to(dev)
        kern = wrap_nki(getattr(qwen3_megakernel, args.kernel))

        def mega_fn(X=mX):
            out = kern[lnc](X=X, W_qkv=mW["W_qkv"], W_out=mW["W_out"], W_gate=mW["W_gate"], W_up=mW["W_up"],
                            W_down=mW["W_down"], g_attn=mW["g_attn"], g_mlp=mW["g_mlp"], g_q=mW["g_q"], g_k=mW["g_k"],
                            K_cache=mK, V_cache=mV, cos=mcos, sin=msin, mask=mmask, pos_ids=mpos,
                            num_layers=L, eps=eps, **mega_kw)
            return (out,)
        mega_c = torch.compile(mega_fn, backend="neuron_libtorch", fullgraph=True)
        t0 = time.time(); mega_c(); mega_c()[0].reshape(-1)[0].cpu(); row["mega_compile_s"] = round(time.time() - t0, 1)
        row["mega_ms"] = round(timeit(mega_c, args.iters), 3)
        print(f"L={L}: megakernel (one launch): {row['mega_ms']:.3f} ms  (compile {row['mega_compile_s']}s)", flush=True)
        del mW, mK, mV, mega_c

    print(f"L={L}: HBM floor at 736 GB/s = {floor_ms:.3f} ms;  weights {row['weight_bytes']/1e9:.2f} GB", flush=True)
    results.append(row)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")

print(json.dumps(results, indent=1))
