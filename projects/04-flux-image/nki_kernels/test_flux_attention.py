"""Numerics test: fused NKI QK-norm + RoPE + flash attention vs. a PyTorch reference of the NxDI Flux path.

The reference reproduces NeuronFluxAttention exactly as NxDI runs it (CustomRMSNorm in fp32 -> bf16,
apply_rotary_emb in fp32 -> bf16 on [B,H,S,D], then attention), computed in fp32 on CPU.
We also report a "bf16 baseline" (same path with bf16 softmax/PV, which is what the device does today)
so the NKI error can be judged relative to the error the existing pipeline already has.

    # CPU simulator (no Neuron device used; works anywhere nki is installed, e.g. the flux venv):
    python test_flux_attention.py                      # small shapes, all stages
    python test_flux_attention.py --full               # + FLUX-1024 shape: S=4608 (512 txt), 6 heads
    # On a free Trainium core (compiles a NEFF; needs torch_xla):
    python test_flux_attention.py --device [--full]
"""

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

D = 128


# ----------------------------------------------------------------------------------------------
# Reference (mirrors neuronx_distributed_inference flux modeling + embeddings)
# ----------------------------------------------------------------------------------------------
def flux_rope_tables(n_txt, grid_h, grid_w, theta=10000, axes_dim=(16, 56, 56)):
    """FluxPosEmbed(theta=10000, axes_dim=[16,56,56]) for txt ids (zeros) + image ids (0, row, col).
    Returns cos, sin: [S, D] fp32, repeat-interleaved (as NxDI/diffusers produce them)."""
    txt_ids = torch.zeros(n_txt, 3)
    img_ids = torch.zeros(grid_h, grid_w, 3)
    img_ids[..., 1] = torch.arange(grid_h)[:, None]
    img_ids[..., 2] = torch.arange(grid_w)[None, :]
    ids = torch.cat([txt_ids, img_ids.reshape(-1, 3)], dim=0).double()
    cos_out, sin_out = [], []
    for i, dim in enumerate(axes_dim):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        f = torch.outer(ids[:, i], freqs)
        cos_out.append(f.cos().repeat_interleave(2, dim=1).float())
        sin_out.append(f.sin().repeat_interleave(2, dim=1).float())
    return torch.cat(cos_out, -1), torch.cat(sin_out, -1)


def rms_norm_nxdi(x, w, eps):
    """CustomRMSNorm: fp32 compute, cast back to input dtype."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    return y.to(x.dtype)


def apply_rotary_emb_nxdi(x, cos, sin):
    """diffusers/NxDI apply_rotary_emb(use_real=True, use_real_unbind_dim=-1); x: [B,H,S,D]."""
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rot = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos[None, None] + x_rot.float() * sin[None, None]).to(x.dtype)


def reference(q, k, v, cos, sin, qw_img, kw_img, qw_txt, kw_txt, n_txt, eps,
              apply_norm=True, apply_rope=True, attn_dtype=torch.float32):
    """q,k,v: [B,S,H*D] bf16 -> [B,S,H*D]. attn_dtype=float32 is the 'exact' reference;
    attn_dtype=bfloat16 approximates today's on-device bf16 attention."""
    B, S, HD = q.shape
    H = HD // D
    def heads(x):
        return x.view(B, S, H, D).transpose(1, 2)  # [B,H,S,D]
    qh, kh, vh = heads(q), heads(k), heads(v)
    if apply_norm:
        qh = torch.cat([rms_norm_nxdi(qh[:, :, :n_txt], qw_txt, eps), rms_norm_nxdi(qh[:, :, n_txt:], qw_img, eps)], 2)
        kh = torch.cat([rms_norm_nxdi(kh[:, :, :n_txt], kw_txt, eps), rms_norm_nxdi(kh[:, :, n_txt:], kw_img, eps)], 2)
    if apply_rope:
        qh = apply_rotary_emb_nxdi(qh, cos, sin)
        kh = apply_rotary_emb_nxdi(kh, cos, sin)
    s = (qh.to(attn_dtype) @ kh.to(attn_dtype).transpose(-1, -2)) * (D ** -0.5)
    p = torch.softmax(s.float(), dim=-1).to(attn_dtype)
    o = (p @ vh.to(attn_dtype)).float()
    return o.transpose(1, 2).reshape(B, S, HD)


# ----------------------------------------------------------------------------------------------
def make_inputs(B, S, H, n_txt, seed, w_scale):
    g = torch.Generator().manual_seed(seed)
    grid = int(math.isqrt(S - n_txt))
    assert grid * grid == S - n_txt, "image tokens must form a square grid for the rope table"
    q = (torch.randn(B, S, H * D, generator=g) * 2.0).bfloat16()
    k = (torch.randn(B, S, H * D, generator=g) * 2.0).bfloat16()
    v = torch.randn(B, S, H * D, generator=g).bfloat16()
    # RMSNorm weights: FLUX's learned qk-norm weights are O(1) with spread; w_scale makes softmax peaky.
    ws = [(1.0 + w_scale * torch.randn(1, D, generator=g)).bfloat16().float() for _ in range(4)]
    cos, sin = flux_rope_tables(n_txt, grid, grid)
    return q, k, v, cos, sin, ws


def metrics(out, ref):
    out, ref = out.double().flatten(), ref.double().flatten()
    diff = out - ref
    return dict(max_abs=diff.abs().max().item(), mean_abs=diff.abs().mean().item(),
                rel_l2=(diff.norm() / ref.norm()).item(),
                cos=torch.nn.functional.cosine_similarity(out, ref, dim=0).item())


def run_kernel(mode, args_t, n_txt, eps, apply_norm, apply_rope, lnc=1):
    import nki
    from flux_attention_nki import flux_qknorm_rope_attention as kern
    if lnc > 1:
        kern = kern[lnc]  # SPMD grid == LNC degree: heads split across the physical cores
    q, k, v, cos_h, sin_h, qwi, kwi, qwt, kwt = args_t
    kw = dict(n_txt=n_txt, eps=eps, apply_norm=apply_norm, apply_rope=apply_rope)
    if mode == "sim":
        return nki.simulate(kern)(q, k, v, cos_h, sin_h, qwi, kwi, qwt, kwt, **kw)
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    o = kern(*[t.to(dev) for t in (q, k, v, cos_h, sin_h, qwi, kwi, qwt, kwt)], **kw)
    return o.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", action="store_true", help="run on a Neuron core instead of the CPU simulator")
    ap.add_argument("--full", action="store_true", help="also run the FLUX 1024x1024 per-core shape (slow in sim)")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--lnc", type=int, default=2, help="SPMD grid (2 = FLUX's LNC=2 config); cases with odd H use 1")
    a = ap.parse_args()
    mode = "device" if a.device else "sim"
    if a.device:
        os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

    # (name, B, S, H, n_txt, apply_norm, apply_rope, w_scale)
    cases = [
        ("attn-only     S=512  H=1",         1, 512, 1, 256, False, False, 0.0),
        ("norm+rope     S=384  H=2 txt=128", 1, 384, 2, 128, True, True, 0.3),
        ("joint         S=640  H=2 txt=384", 1, 640, 2, 384, True, True, 0.3),  # 3 chunks, partial last
        ("single(txt=0) S=1024 H=2 B=2",     2, 1024, 2, 0, True, True, 0.3),
        ("peaky softmax S=640  H=1 txt=384", 1, 640, 1, 384, True, True, 1.5),
    ]
    if a.full:
        cases.append(("FLUX-1024 per-core S=4608 H=6 txt=512", 1, 4608, 6, 512, True, True, 0.3))

    all_ok = True
    for name, B, S, H, n_txt, an, ar, wsc in cases:
        q, k, v, cos, sin, (qwi, kwi, qwt, kwt) = make_inputs(B, S, H, n_txt, seed=S + H, w_scale=wsc)
        ref = reference(q, k, v, cos, sin, qwi, kwi, qwt, kwt, n_txt, a.eps, an, ar)
        base = reference(q, k, v, cos, sin, qwi, kwi, qwt, kwt, n_txt, a.eps, an, ar, attn_dtype=torch.bfloat16)
        cos_h = cos[:, 0::2].contiguous()
        sin_h = sin[:, 0::2].contiguous()
        t0 = time.time()
        lnc = a.lnc if H % a.lnc == 0 else 1
        out = run_kernel(mode, (q, k, v, cos_h, sin_h, qwi, kwi, qwt, kwt), n_txt, a.eps, an, ar, lnc)
        dt = time.time() - t0
        m = metrics(out, ref)
        mb = metrics(base, ref)
        # bf16 tolerance: output is bf16 and P is bf16 for the PV matmul (same as attention_cte).
        # Global error must be small; the worst element may not exceed max(2e-2, 1.25x the error the
        # bf16 baseline already has) -- with a peaky softmax a fixed atol is not meaningful for bf16 P.
        ok = (m["cos"] > 0.9995 and m["rel_l2"] < 1.5e-2
              and m["max_abs"] <= max(2e-2, 1.25 * mb["max_abs"]))
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:40s} {mode} lnc{lnc} {dt:6.1f}s | "
              f"nki: max {m['max_abs']:.2e} mean {m['mean_abs']:.2e} relL2 {m['rel_l2']:.2e} cos {m['cos']:.6f} | "
              f"bf16-baseline relL2 {mb['rel_l2']:.2e} max {mb['max_abs']:.2e}", flush=True)
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
