"""CPU test of the generate.py --nki integration glue (flux_nki_patch), no Neuron / nki / NxDI needed.

The NKI kernel's numerics are covered by test_flux_attention.py (nki.simulate or device). This test checks
the *wiring*: that the patched NeuronFluxAttention.forward feeds the kernel the right tensors in the right
layout (token-major [B,S,H*D], text tokens first, per-segment norm weights, de-duplicated rope tables from
NxDI's [S,D,2] bf16 image_rotary_emb), splits/projects the output like the stock forward, and falls back
to the stock forward for unsupported configs.

The kernel is replaced by an independent torch emulation of its documented contract, and the stock forward
is a transcription of NxDI's NeuronFluxAttention.forward non-CP path (RMSNorm -> bf16, rope fp32 -> bf16,
attention_cte replaced by fp32 SDPA).

    python test_flux_nki_patch.py
"""
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flux_nki_patch  # noqa: E402
from test_flux_attention import flux_rope_tables  # noqa: E402

D = 128
EPS = 1e-6


# ---------------------------------------------------------------- stock NxDI path (non-CP, Trn2) -------------
class RMS(nn.Module):  # CustomRMSNorm semantics
    def __init__(self, n, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.hidden_size, self.variance_epsilon = n, eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon) * self.weight.float()).to(x.dtype)


def apply_rotary_emb(x, freqs_cis):  # NxDI embeddings.apply_rotary_emb, use_real=True, unbind_dim=-1
    cos, sin = freqs_cis.unbind(-1)
    cos, sin = cos[None, None], sin[None, None]
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rot = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rot.float() * sin).to(x.dtype)


def stock_forward(self, hidden_states, image_rotary_emb, attention_mask=None, encoder_hidden_states=None,
                  rotary_emb_text=None, rotary_emb_image=None):
    self.stock_calls += 1
    batch_size = (hidden_states if encoder_hidden_states is None else encoder_hidden_states).shape[0]
    query, key, value = self.to_q(hidden_states), self.to_k(hidden_states), self.to_v(hidden_states)
    head_dim = key.shape[-1] // self.heads
    t = lambda x: x.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)  # noqa: E731
    query, key, value = t(query), t(key), t(value)
    query, key = self.norm_q(query), self.norm_k(key)
    if encoder_hidden_states is not None:
        eq = self.norm_added_q(t(self.add_q_proj(encoder_hidden_states)))
        ek = self.norm_added_k(t(self.add_k_proj(encoder_hidden_states)))
        ev = t(self.add_v_proj(encoder_hidden_states))
        query, key, value = torch.cat([eq, query], 2), torch.cat([ek, key], 2), torch.cat([ev, value], 2)
    query, key = apply_rotary_emb(query, image_rotary_emb), apply_rotary_emb(key, image_rotary_emb)
    hs = torch.nn.functional.scaled_dot_product_attention(query.float(), key.float(), value.float(),
                                                          attn_mask=attention_mask).to(query.dtype)
    hs = hs.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
    if encoder_hidden_states is not None:
        n = encoder_hidden_states.shape[1]
        e, hs = hs[:, :n], hs[:, n:]
        return self.to_out[1](self.to_out[0](hs)), self.to_add_out(e)
    return hs[..., : self.out_dim] if self.padded_inner_dim != self.out_dim else hs


class FakeAttn(nn.Module):
    forward = stock_forward

    def __init__(self, dim, heads, joint, out_dim=None, g=None):
        super().__init__()
        inner = heads * D
        self.heads, self.padded_inner_dim = heads, inner
        self.out_dim = out_dim or inner
        self.context_parallel_enabled = False
        self.stock_calls = 0
        lin = lambda i, o: nn.Linear(i, o, bias=True)  # noqa: E731
        self.to_q, self.to_k, self.to_v = lin(dim, inner), lin(dim, inner), lin(dim, inner)
        self.norm_q, self.norm_k = RMS(D, EPS), RMS(D, EPS)
        self.add_q_proj = self.add_k_proj = self.add_v_proj = self.norm_added_q = self.norm_added_k = None
        if joint:
            self.add_q_proj, self.add_k_proj, self.add_v_proj = lin(dim, inner), lin(dim, inner), lin(dim, inner)
            self.norm_added_q, self.norm_added_k = RMS(D, EPS), RMS(D, EPS)
            self.to_out = nn.ModuleList([lin(inner, dim), nn.Dropout(0.0)])
            self.to_add_out = lin(inner, dim)
        for p in self.parameters():  # learned-looking qk-norm weights, small projections
            with torch.no_grad():
                p.copy_(1 + 0.3 * torch.randn(p.shape, generator=g) if p.dim() == 1 and p.numel() == D
                        else torch.randn(p.shape, generator=g) / math.sqrt(p.shape[-1]))
        self.to(torch.bfloat16)


# ---------------------------------------------------------------- kernel contract emulation -----------------
def kernel_emulator(grid):
    def kern(q, k, v, cos, sin, qwi, kwi, qwt, kwt, n_txt=0, eps=1e-6):
        assert q.dtype == torch.bfloat16 and cos.dtype == torch.float32 and cos.shape[1] == D // 2
        B, S, HD = q.shape
        H = HD // D
        assert H % grid == 0
        txt = (torch.arange(S) < n_txt)[:, None]

        def prep(x, wi, wt):
            x = x.float()
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * torch.where(txt, wt, wi)
            e, o = x[..., 0::2], x[..., 1::2]
            r = torch.stack([e * cos - o * sin, o * cos + e * sin], -1).flatten(-2)  # interleaved back
            return r.bfloat16().float()

        out = torch.empty(B, S, HD, dtype=torch.bfloat16)
        for b in range(B):
            for h in range(H):
                sl = slice(h * D, (h + 1) * D)
                Q, K, V = prep(q[b, :, sl], qwi, qwt), prep(k[b, :, sl], kwi, kwt), v[b, :, sl].float()
                out[b, :, sl] = (torch.softmax(Q @ K.T * D ** -0.5, -1) @ V).bfloat16()
        return out
    return kern


# ---------------------------------------------------------------- test ---------------------------------------
def cmp(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return (a - b).norm().item() / b.norm().item(), torch.nn.functional.cosine_similarity(a, b, 0).item()


def main():
    g = torch.Generator().manual_seed(0)
    flux_nki_patch.kernel_factory = kernel_emulator
    n_txt, grid_hw, dim, heads, B = 128, 16, 256, 2, 1
    S_img = grid_hw * grid_hw
    cos, sin = flux_rope_tables(n_txt, grid_hw, grid_hw)
    rot = torch.stack([cos, sin], dim=2).to(torch.bfloat16)  # what ModelWrapperFluxBackbone feeds the model
    img = torch.randn(B, S_img, dim, generator=g).bfloat16()
    txt = torch.randn(B, n_txt, dim, generator=g).bfloat16()
    joint = FakeAttn(dim, heads, joint=True, g=g)
    single = FakeAttn(dim, heads, joint=False, g=g)
    single_pad = FakeAttn(dim, heads, joint=False, out_dim=D + 64, g=g)  # padded-heads slice path
    full = torch.cat([txt, img], 1)

    with torch.no_grad():
        ref_joint = joint(img, rot, encoder_hidden_states=txt)
        ref_single = single(full, rot)
        ref_pad = single_pad(full, rot)

        flux_nki_patch.enable(FakeAttn, verbose=False)
        ok = True
        for name, got, ref in [("joint img", joint(img, rot, encoder_hidden_states=txt)[0], ref_joint[0]),
                               ("joint txt", joint(img, rot, encoder_hidden_states=txt)[1], ref_joint[1]),
                               ("single", single(full, rot), ref_single),
                               ("single padded", single_pad(full, rot), ref_pad)]:
            r, c = cmp(got, ref)
            good = got.shape == ref.shape and r < 2e-2 and c > 0.9998
            ok &= good
            print(f"[{'PASS' if good else 'FAIL'}] {name:14s} shape {tuple(got.shape)} relL2 {r:.2e} cos {c:.6f}")

        # fallbacks must take the stock path (stock_calls counts entries into stock_forward)
        before = single.stock_calls
        mask = torch.zeros(1, 1, n_txt + S_img, n_txt + S_img)
        single(full, rot, attention_mask=mask)
        single.context_parallel_enabled = True
        single(full, rot)
        single.context_parallel_enabled = False
        odd = FakeAttn(dim, 3, joint=False, g=g)
        os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
        odd(torch.cat([txt, img], 1), rot)          # 3 heads, LNC=2 grid -> fallback
        single(full, rot)                           # 2 heads, LNC=2 grid -> fused
        os.environ.pop("NEURON_RT_VIRTUAL_CORE_SIZE")
        ragged = full[:, :-64]
        single(ragged, rot[:-64])                    # S % 128 != 0 -> fallback
        fell = (single.stock_calls - before, odd.stock_calls)
        good = fell == (3, 1)
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] fallbacks      stock calls (mask+cp+ragged, odd heads@lnc2) = {fell}")
        print("stats:", flux_nki_patch.stats())
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
