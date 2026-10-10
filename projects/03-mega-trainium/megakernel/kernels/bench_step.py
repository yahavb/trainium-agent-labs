"""Benchmark the FULL decode step: one-launch qwen3_decode_step vs per-op torch.compile of the same math.

Both sides: token id -> embedding -> L layers -> final RMSNorm -> LM head -> argmax, batch 1, weights
resident on the same free logical core, random weights at Qwen3-8B shapes (timing does not depend on
values). Same timing method as bench_mega.py (sync every call).

  NEURON_RT_VISIBLE_CORES=3 python kernels/bench_step.py --layers 36 --out results/bench_step.jsonl
"""
import argparse, json, os, sys, time
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "3")
sys.path.insert(0, os.path.dirname(__file__))
import torch
import libtorch_neuronx_lite  # registers the neuron device + torch.compile backends
import torch.distributed as dist
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from qwen3_ref import decode_layers_ref, rope_tables, decode_mask, rmsnorm
from qwen3_decode_step import (qwen3_decode_step, qwen3_decode_step_resident, qwen3_decode_step_tiled, pack_lm_head,
                               pack_lm_head_tiled, rope_tables_all)

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=36)
ap.add_argument("--S_ctx", type=int, default=1024)
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--no-baseline", action="store_true")
ap.add_argument("--no-mega", action="store_true")
ap.add_argument("--resident", action="store_true", help="time qwen3_decode_step_resident (no per-step inputs)")
ap.add_argument("--tiled-head", action="store_true", help="with --resident: qwen3_decode_step_tiled (our own LM head)")
ap.add_argument("--hwdge-oproj", action="store_true", help="hardware-DGE weight loads in output_projection_tkg (o_proj and LM head)")
ap.add_argument("--out", default=None)
args = ap.parse_args()
if args.hwdge_oproj:
    # libtorch's NKI trace cache keys on the top-level kernel's source only, so a replaced nkilib helper
    # would silently reuse the unpatched trace (and a patched trace would poison later unpatched runs)
    os.environ["NEURON_LIBTORCH_DISABLE_COMPILE_CACHE"] = "1"
    from qwen3_decode_step import use_hwdge_output_projection
    use_hwdge_output_projection()
H, I, q, kv, d, V = 4096, 12288, 32, 8, 128, 151936
L, B, S_ctx, eps = args.layers, 1, args.S_ctx, 1e-6
t = S_ctx - 64
lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
dev = "neuron:0"
if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29621")
    dist.init_process_group("gloo", rank=0, world_size=1)

g = torch.Generator().manual_seed(0)
def rn(*shape, scale=0.02): return (torch.randn(*shape, generator=g) * scale).bfloat16()
W = dict(W_qkv=rn(L, H, (q + 2 * kv) * d), W_out=rn(L, q * d, H), W_gate=rn(L, H, I), W_up=rn(L, H, I),
         W_down=rn(L, I, H), g_attn=torch.ones(L, 1, H).bfloat16(), g_mlp=torch.ones(L, 1, H).bfloat16(),
         g_q=torch.ones(L, 1, d).bfloat16(), g_k=torch.ones(L, 1, d).bfloat16())
embed, lm, g_final = rn(V, H), rn(V, H), torch.ones(1, H).bfloat16()
K_cache, V_cache = rn(L, B, kv, d, S_ctx, scale=1.0), rn(L, B, kv, S_ctx, d, scale=1.0)
pos = torch.full((B, 1), t, dtype=torch.int64)
cos, sin = rope_tables(torch.full((B,), t), d, 1e6)
mask = decode_mask(pos, S_ctx, q)
token = torch.tensor([[1234]], dtype=torch.int32)

weight_bytes = 2 * (L * (H * (q + 2 * kv) * d + q * d * H + 3 * H * I) + V * H)  # + LM head; embedding is one row
kv_bytes = 2 * L * B * kv * S_ctx * d * 2
row = dict(layers=L, hwdge_oproj=args.hwdge_oproj, resident=args.resident, tiled_head=args.tiled_head, S_ctx=S_ctx, weight_bytes=weight_bytes, kv_read_bytes=kv_bytes,
           hbm_floor_ms_at_736GBps=round((weight_bytes + kv_bytes) / 736e9 * 1e3, 3))
print(f"HBM floor (layers + LM head + KV, at 736 GB/s): {row['hbm_floor_ms_at_736GBps']:.2f} ms", flush=True)


def timeit(fn, iters):
    for _ in range(3):
        fn()[0].reshape(-1)[0].cpu()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()[0].reshape(-1)[0].cpu()
    return (time.perf_counter() - t0) / iters * 1e3


dW = {k: v.to(dev) for k, v in W.items()}
dK, dV = K_cache.to(dev), V_cache.to(dev)
dcos, dsin, dmask = cos.bfloat16().to(dev), sin.bfloat16().to(dev), mask.to(dev)

if not args.no_baseline:
    dE, dLM, dG = embed.to(dev), lm.to(dev), g_final.to(dev)
    dtok, dpos = token.to(torch.int64).to(dev), pos.to(dev)

    def base_fn():
        X = dE[dtok.view(-1)].view(1, 1, H)
        h, _, _ = decode_layers_ref(X, dW["W_qkv"], dW["W_out"], dW["W_gate"], dW["W_up"], dW["W_down"], dW["g_attn"],
                                    dW["g_mlp"], dW["g_q"], dW["g_k"], dK, dV, dcos, dsin, dmask, dpos, L, eps, q, kv, d,
                                    compute_dtype=torch.bfloat16)
        logits = rmsnorm(h, dG, eps).bfloat16() @ dLM.t()
        return (logits.argmax(-1), logits)
    base_c = torch.compile(base_fn, backend="neuron_libtorch", fullgraph=True)
    t0 = time.time(); base_c()[0].cpu(); row["baseline_compile_s"] = round(time.time() - t0, 1)
    row["baseline_ms"] = round(timeit(base_c, args.iters), 3)
    print(f"per-op torch.compile full step: {row['baseline_ms']:.3f} ms (compile {row['baseline_compile_s']}s)", flush=True)
    del dE, dLM, base_c

if not args.no_mega:
    W_lm, lm_bias, _ = pack_lm_head(lm, lnc)
    dev_w = dict(embed=embed.to(dev), g_final=g_final.to(dev), W_lm=W_lm.to(dev), **dW)
    mtok, mpos = token.to(dev), pos.to(torch.int32).to(dev)
    if not args.resident:
        kern = wrap_nki(qwen3_decode_step)

        def mega_fn():
            tok, logits, _ = kern[lnc](token_ids=mtok, K_cache=dK, V_cache=dV, cos=dcos, sin=dsin, mask=dmask,
                                       pos_ids=mpos, lm_bias=None, num_layers=L, eps=eps, **dev_w)
            return (tok, logits)
    else:
        cos_t, sin_t = rope_tables_all(S_ctx, d, 1e6)
        dev_w.update(cos_table=cos_t.bfloat16().to(dev), sin_table=sin_t.bfloat16().to(dev))
        if args.tiled_head:
            kern = wrap_nki(qwen3_decode_step_tiled)
            dev_w.pop("W_lm")
            dev_w["W_tiled"] = pack_lm_head_tiled(lm, lnc)[0].to(dev)
            head_kw = dict(V=V)
        else:
            kern = wrap_nki(qwen3_decode_step_resident)
            head_kw = dict(lm_bias=None)

        def mega_fn():
            tok, next_pos, logits = kern[lnc](token_ids=mtok, pos_ids=mpos, K_cache=dK, V_cache=dV, num_layers=L,
                                              S_ctx=S_ctx, eps=eps, **head_kw, **dev_w)
            return (tok, logits)
    mega_c = torch.compile(mega_fn, backend="neuron_libtorch", fullgraph=True)
    t0 = time.time(); mega_c()[0].cpu(); row["mega_compile_s"] = round(time.time() - t0, 1)
    row["mega_ms"] = round(timeit(mega_c, args.iters), 3)
    print(f"one-launch decode step: {row['mega_ms']:.3f} ms (compile {row['mega_compile_s']}s)", flush=True)

print(json.dumps(row))
if args.out:
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
