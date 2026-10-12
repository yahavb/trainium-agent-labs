"""Device microbenchmark at the FLUX-1024 per-rank attention shape (S=4608, 6 heads, D=128, LNC=2).

  baseline : NxDI path  -> view/transpose, RMSNorm (fp32), apply_rotary_emb (fp32), cat-free,
                           nkilib attention_cte[2](tp_q, tp_k), transpose back      (XLA glue + NKI)
  fused    : flux_qknorm_rope_attention[2] on [1, S, H*D] (one NKI kernel)
  cte_only : nkilib attention_cte[2] alone on pre-normalised [H, S, D] (the kernel core only)

Each graph chains N_CHAIN calls (output feeds the next call's v) so launch overhead is amortised.
    source env.sh && python bench_attention.py
"""
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch_xla.core.xla_model as xm  # noqa: E402
from nkilib.core.attention.attention_cte import attention_cte  # noqa: E402

from flux_attention_nki import flux_qknorm_rope_attention  # noqa: E402
from test_flux_attention import apply_rotary_emb_nxdi, flux_rope_tables  # noqa: E402

S, H, D, N_TXT = 4608, 6, 128, 512
N_CHAIN, ITERS = 8, 10
EPS = 1e-6


def rms(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w).to(x.dtype)


def baseline(q, k, v, cos, sin, w):
    B = q.shape[0]
    qh = q.view(B, S, H, D).transpose(1, 2)
    kh = k.view(B, S, H, D).transpose(1, 2)
    vh = v.view(B, S, H, D).transpose(1, 2)
    qh, kh = rms(qh, w), rms(kh, w)
    qh, kh = apply_rotary_emb_nxdi(qh, cos, sin), apply_rotary_emb_nxdi(kh, cos, sin)
    o = attention_cte[2](qh.reshape(B * H, S, D), kh.reshape(B * H, S, D), vh.reshape(B * H, S, D),
                         1 / math.sqrt(D), causal_mask=False, tp_q=True, tp_k=True, tp_out=False)
    return o.reshape(B, H, S, D).transpose(1, 2).reshape(B, S, H * D)


def fused(q, k, v, cos_h, sin_h, w2):
    return flux_qknorm_rope_attention[2](q, k, v, cos_h, sin_h, w2, w2, w2, w2, n_txt=N_TXT, eps=EPS)


def cte_only(q, k, v):
    return attention_cte[2](q, k, v, 1 / math.sqrt(D), causal_mask=False, tp_q=True, tp_k=True, tp_out=False)


def timeit(name, fn, *args):
    def graph(*a):
        a = list(a)
        out = None
        for _ in range(N_CHAIN):
            out = fn(*a)
            a[2] = out  # chain through v
        return out
    o = graph(*args); xm.mark_step(); xm.wait_device_ops()  # compile + warmup
    o = graph(*args); xm.mark_step(); xm.wait_device_ops()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        o = graph(*args)
        xm.mark_step()
    xm.wait_device_ops()
    ms = (time.perf_counter() - t0) / (ITERS * N_CHAIN) * 1e3
    print(f"{name:10s}: {ms:7.3f} ms / attention call  (x57 blocks = {ms * 57:6.1f} ms / step)", flush=True)
    return o


def main():
    dev = xm.xla_device()
    g = torch.Generator().manual_seed(0)
    q = (torch.randn(1, S, H * D, generator=g) * 2).bfloat16()
    k = (torch.randn(1, S, H * D, generator=g) * 2).bfloat16()
    v = torch.randn(1, S, H * D, generator=g).bfloat16()
    cos, sin = flux_rope_tables(N_TXT, 64, 64)
    w = torch.ones(D)
    which = sys.argv[1:] or ["baseline", "fused", "cte_only"]
    if "baseline" in which:
        timeit("baseline", baseline, *[t.to(dev) for t in (q, k, v, cos, sin, w)])
    if "fused" in which:
        timeit("fused", fused, *[t.to(dev) for t in (q, k, v, cos[:, 0::2].contiguous(),
                                                     sin[:, 0::2].contiguous(), w[None])])
    if "cte_only" in which:
        qh = q.view(1, S, H, D).transpose(1, 2).reshape(H, S, D).contiguous()
        kh = k.view(1, S, H, D).transpose(1, 2).reshape(H, S, D).contiguous()
        vh = v.view(1, S, H, D).transpose(1, 2).reshape(H, S, D).contiguous()
        timeit("cte_only", cte_only, *[t.to(dev) for t in (qh, kh, vh)])


if __name__ == "__main__":
    main()
