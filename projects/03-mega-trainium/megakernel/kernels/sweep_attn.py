"""Attention-only shape sweep: which dimension breaks the fused attention block?

Random weights (no HF needed: the fp32 reference is validated against HF in test_tiny.py), MLP
down-projection zeroed so only the attention block contributes. One layer. Reports rel-L2 of the
kernel vs the fp32 reference for each (H, q, kv, S_ctx) config.

  NEURON_RT_VISIBLE_CORES=2 python kernels/sweep_attn.py
"""
import os, sys, time, json
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, ml_dtypes
from qwen3_ref import decode_layers_ref, rope_tables, decode_mask
from qwen3_megakernel import qwen3_decode_layers_jit

lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
bf = ml_dtypes.bfloat16
def np_bf16(x): return x.detach().contiguous().float().numpy().astype(bf)

def run(H, q, kv, S_ctx=128, S_max=256, t=100, I=1024, d=128, L=1, seed=0, kv_scale=1.0):
    g = torch.Generator().manual_seed(seed)
    def rn(*shape, scale): return torch.randn(*shape, generator=g) * scale
    B = 1
    W = dict(W_qkv=rn(L, H, (q + 2 * kv) * d, scale=H ** -0.5), W_out=rn(L, q * d, H, scale=(q * d) ** -0.5),
             W_gate=rn(L, H, I, scale=H ** -0.5), W_up=rn(L, H, I, scale=H ** -0.5), W_down=torch.zeros(L, I, H),
             g_attn=torch.rand(L, 1, H, generator=g) + 0.5, g_mlp=torch.ones(L, 1, H),
             g_q=torch.rand(L, 1, d, generator=g) + 0.5, g_k=torch.rand(L, 1, d, generator=g) + 0.5)
    K_cache = rn(L, B, kv, d, S_max, scale=kv_scale); V_cache = rn(L, B, kv, S_max, d, scale=kv_scale)
    X = rn(B, 1, H, scale=1.0)
    pos = torch.full((B, 1), t, dtype=torch.int64)
    cos, sin = rope_tables(torch.full((B,), t), d, 1e6)
    mask = decode_mask(pos, S_ctx, q)
    Wb = {k: v.bfloat16().float() for k, v in W.items()}   # reference sees the bf16-rounded weights
    ref, _, _ = decode_layers_ref(X.bfloat16().float(), *[Wb[k] for k in ("W_qkv", "W_out", "W_gate", "W_up", "W_down", "g_attn", "g_mlp", "g_q", "g_k")],
                                  K_cache.bfloat16().float(), V_cache.bfloat16().float(), cos.bfloat16().float(), sin.bfloat16().float(),
                                  mask, pos, L, 1e-6, q, kv, d)
    inputs = dict(X=np_bf16(X), **{k: np_bf16(v) for k, v in W.items()}, K_cache=np_bf16(K_cache), V_cache=np_bf16(V_cache),
                  cos=np_bf16(cos), sin=np_bf16(sin), mask=mask.numpy().astype(np.uint8), pos_ids=pos.numpy().astype(np.uint32))
    t0 = time.time()
    out = qwen3_decode_layers_jit[lnc](**inputs, num_layers=L, eps=1e-6)
    out_t = torch.from_numpy(np.asarray(out).astype(np.float32)).reshape(B, 1, H)
    attn_ref = ref - X.bfloat16().float()            # the attention contribution alone (MLP is zero)
    attn_k = out_t - X.bfloat16().float()
    e = ((attn_k - attn_ref).norm() / (attn_ref.norm() + 1e-9)).item()
    return e, time.time() - t0

if __name__ == "__main__":
  configs = [
    dict(H=512, q=4, kv=2),            # the tiny test config
    dict(H=512, q=8, kv=2),            # q_per_group 4
    dict(H=512, q=8, kv=8),            # MHA, 8 kv heads
    dict(H=512, q=32, kv=8),           # real head config, small H
    dict(H=4096, q=4, kv=2),           # real H, tiny heads
    dict(H=4096, q=32, kv=8),          # real config
    dict(H=4096, q=32, kv=8, S_ctx=256, S_max=512, t=200),
    dict(H=512, q=4, kv=2, kv_scale=4.0),   # larger logits -> peakier softmax
]
  for c in configs:
    try:
        e, dt = run(**c)
        print(f"{json.dumps(c):70s} attention rel-L2 = {e:.3e}   ({dt:.0f}s)", flush=True)
    except Exception as ex:
        print(f"{json.dumps(c):70s} FAILED: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
