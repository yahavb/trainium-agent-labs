"""Fit the kernel's active-token semantics: score the kernel against several attention references."""
import os, sys, math, json
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, ml_dtypes
from qwen3_ref import rmsnorm, rope_rotate_half, rope_tables, decode_mask
from qwen3_megakernel import qwen3_decode_layers_jit

H, q, kv, d, I, L, B = 256, 2, 1, 128, 512, 1, 1
t, S_ctx, S_max = int(sys.argv[1]) if len(sys.argv) > 1 else 100, 128, 128
g = torch.Generator().manual_seed(0)
rn = lambda *s, scale: torch.randn(*s, generator=g) * scale
W = dict(W_qkv=rn(L, H, (q + 2 * kv) * d, scale=H ** -0.5), W_out=rn(L, q * d, H, scale=(q * d) ** -0.5),
         W_gate=rn(L, H, I, scale=H ** -0.5), W_up=rn(L, H, I, scale=H ** -0.5), W_down=torch.zeros(L, I, H),
         g_attn=torch.rand(L, 1, H, generator=g) + 0.5, g_mlp=torch.ones(L, 1, H),
         g_q=torch.rand(L, 1, d, generator=g) + 0.5, g_k=torch.rand(L, 1, d, generator=g) + 0.5)
K_cache = rn(L, B, kv, d, S_max, scale=1.0); V_cache = rn(L, B, kv, S_max, d, scale=1.0)
K_cache[..., t:] = 0; V_cache[:, :, :, t:, :] = 0
X = rn(B, 1, H, scale=1.0)
pos = torch.full((B, 1), t, dtype=torch.int64)
cos, sin = rope_tables(torch.full((B,), t), d, 1e6)
mask = decode_mask(pos, S_ctx, q)
b16 = lambda x: x.bfloat16().float()
Wb = {k: b16(v) for k, v in W.items()}; Xb, Kb, Vb, cb, sb = b16(X), b16(K_cache), b16(V_cache), b16(cos), b16(sin)

def attn_ref(active="normal", rope_k_active=True, norm="full", qk_norm=True):
    h = rmsnorm(Xb, Wb["g_attn"][0, 0], 1e-6)
    qkv = h @ Wb["W_qkv"][0]
    qp = qkv[..., :q * d].reshape(B, 1, q, d); kp = qkv[..., q * d:(q + kv) * d].reshape(B, 1, kv, d); v = qkv[..., (q + kv) * d:].reshape(B, 1, kv, d)
    qn = rmsnorm(qp, Wb["g_q"][0, 0], 1e-6) if qk_norm else qp
    kn = rmsnorm(kp, Wb["g_k"][0, 0], 1e-6) if qk_norm else kp
    qr = rope_rotate_half(qn, cb, sb); kr = rope_rotate_half(kn, cb, sb) if rope_k_active else kn
    scale = 1 / math.sqrt(d)
    Kp = Kb[0, :, :, :, :S_ctx]; Vp = Vb[0, :, :, :S_ctx, :]
    m = mask.view(S_ctx, B, kv, q // kv, 1).permute(1, 4, 2, 3, 0).bool()
    qg = qr.reshape(B, 1, kv, q // kv, d)
    sp = torch.einsum("bskgd,bkdc->bskgc", qg, Kp) * scale
    sp = torch.where(m, sp, torch.full_like(sp, float("-inf")))
    sa = (qg * kr.reshape(B, 1, kv, 1, d)).sum(-1, keepdim=True) * scale
    if active == "excluded":
        pr = torch.softmax(sp, -1); out = torch.einsum("bskgc,bkcd->bskgd", pr, Vp)
    elif active == "double":   # also written into slot t and attended there
        Kp2 = Kp.clone(); Vp2 = Vp.clone(); Kp2[:, :, :, t] = kr[:, 0]; Vp2[:, :, t, :] = v[:, 0]
        m2 = m.clone(); m2[..., t] = True
        sp2 = torch.where(m2, torch.einsum("bskgd,bkdc->bskgc", qg, Kp2) * scale, torch.full_like(sp, float("-inf")))
        pr = torch.softmax(torch.cat([sp2, sa], -1), -1); out = torch.einsum("bskgc,bkcd->bskgd", pr[..., :S_ctx], Vp2) + pr[..., S_ctx:] * v.reshape(B, 1, kv, 1, d)
    else:
        allp = torch.cat([sp, sa], -1)
        if norm == "prior_only":
            e = torch.exp(allp - allp.max(-1, keepdim=True).values); pr = e / e[..., :S_ctx].sum(-1, keepdim=True)
        else:
            pr = torch.softmax(allp, -1)
        out = torch.einsum("bskgc,bkcd->bskgd", pr[..., :S_ctx], Vp) + pr[..., S_ctx:] * v.reshape(B, 1, kv, 1, d)
    return out.reshape(B, 1, q * d) @ Wb["W_out"][0]

bf = ml_dtypes.bfloat16
npb = lambda x: x.detach().contiguous().float().numpy().astype(bf)
inputs = dict(X=npb(X), **{k: npb(v) for k, v in W.items()}, K_cache=npb(K_cache), V_cache=npb(V_cache), cos=npb(cos), sin=npb(sin),
              mask=mask.numpy().astype(np.uint8), pos_ids=pos.numpy().astype(np.uint32))
out = qwen3_decode_layers_jit[int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))](**inputs, num_layers=L, eps=1e-6)
attn_k = torch.from_numpy(np.asarray(out).astype(np.float32)).reshape(B, 1, H) - Xb
rel = lambda a, b: ((a - b).norm() / (b.norm() + 1e-9)).item()
print(f"t={t} S_ctx={S_ctx}")
for name, kw in {"normal": {}, "active_excluded": dict(active="excluded"), "active_double_counted": dict(active="double"),
                 "k_active_not_roped": dict(rope_k_active=False), "normalizer_prior_only": dict(norm="prior_only"),
                 "no_qk_norm": dict(qk_norm=False)}.items():
    print(f"  kernel vs ref[{name:22s}]: rel-L2 {rel(attn_k, attn_ref(**kw)):.3e}")
