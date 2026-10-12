"""Opt-in monkeypatch: route NxDI FLUX attention through the fused NKI kernel.

    from flux_nki_patch import enable
    enable()            # before app.compile(); patches NeuronFluxAttention.forward

What changes, per attention call (19 joint + 38 single-stream blocks per FLUX step):

    stock NxDI                                         patched
    ----------                                         -------
    q/k/v .view(B,S,H,D).transpose(1,2)   (XLA)        -
    norm_q/k, norm_added_q/k (fp32 RMSNorm, XLA)       \
    torch.cat([txt, img], dim=2)   [B,H,S,D]   (XLA)    |  cat on dim 1 ([B,S,H*D], same copy volume)
    apply_rotary_emb(q), (k)  (fp32, XLA)               |  flux_qknorm_rope_attention (one NKI kernel)
    attention_cte[2]                       (NKI)        |
    out.transpose(1,2).reshape(B,S,H*D)   (XLA)        /

Anything the kernel does not support falls back to the original forward, unchanged:
context parallel (--cp), an attention mask, Trn1, head_dim != 128, non-bf16, sequence or text length
not a multiple of 128, heads per rank not divisible by the LNC grid (e.g. --tp 8 -> 3 heads at LNC=2),
or S too large for the kernel's SBUF-resident Q^T/K^T/V.
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_P = 128
_MAX_S = 12288          # qT + kT + V + staging + cos/sin ~ 12 B/token/partition; 192 KB SBUF/partition on trn2
_orig_forward = None
_stats = {"fused": 0, "fallback": 0}


def _default_kernel(grid):
    from flux_attention_nki import flux_qknorm_rope_attention
    return flux_qknorm_rope_attention[grid] if grid > 1 else flux_qknorm_rope_attention


# Tests swap this for a pure-torch emulator so the glue can be checked on a CPU without nki.
kernel_factory = _default_kernel


def rope_half_tables(image_rotary_emb):
    """NxDI image_rotary_emb [S, D, 2] (cos/sin stacked, repeat-interleaved, bf16) -> cos, sin [S, D//2] fp32."""
    cos = image_rotary_emb[:, 0::2, 0].float().contiguous()
    sin = image_rotary_emb[:, 0::2, 1].float().contiguous()
    return cos, sin


def _w(norm, D):
    return norm.weight.float().reshape(1, D)


def lnc_grid():
    # Same signal NxDI's attention_wrapper_sharded_without_swap uses; NeuronFluxBackboneApplication.get_compiler_args
    # sets it to 2 on Trn2 before tracing.
    return 2 if int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1")) == 2 else 1


def fused_attention(q, k, v, image_rotary_emb, w_q, w_k, w_q_txt, w_k_txt, n_txt, eps, grid):
    """q/k/v [B, S, H*D] bf16 (text tokens first when n_txt > 0) -> [B, S, H*D] bf16."""
    cos, sin = rope_half_tables(image_rotary_emb)
    kern = kernel_factory(grid)
    return kern(q, k, v, cos, sin, w_q, w_k, w_q_txt, w_k_txt, n_txt=n_txt, eps=eps)


def _eligible(self, hidden_states, image_rotary_emb, attention_mask, encoder_hidden_states):
    if getattr(self, "context_parallel_enabled", False) or attention_mask is not None:
        return False
    if image_rotary_emb is None or self.norm_q is None or self.norm_k is None:
        return False
    if self.norm_q.weight is None or self.norm_k.weight is None:
        return False
    try:
        from neuronx_distributed_inference.models.diffusers.flux import modeling_flux as mf
        from neuronx_distributed.utils.utils import hardware
        if mf._HARDWARE == hardware.TRN1:
            return False
    except Exception:  # noqa: BLE001 - tests run without NxDI
        pass
    D = self.norm_q.weight.numel()      # CustomRMSNorm(dim_head)
    if D != _P or hidden_states.dtype != torch.bfloat16:
        return False
    n_txt = 0
    if encoder_hidden_states is not None:
        if self.add_q_proj is None or self.norm_added_q is None or self.norm_added_k is None:
            return False
        if self.norm_added_q.weight is None or self.norm_added_k.weight is None:
            return False
        n_txt = encoder_hidden_states.shape[1]
    S = hidden_states.shape[1] + n_txt
    if S % _P or n_txt % _P or S > _MAX_S:
        return False
    if tuple(image_rotary_emb.shape) != (S, D, 2):
        return False
    return self.heads % lnc_grid() == 0


def _nki_forward(self, hidden_states, image_rotary_emb, attention_mask=None, encoder_hidden_states=None,
                 rotary_emb_text=None, rotary_emb_image=None):
    if not _eligible(self, hidden_states, image_rotary_emb, attention_mask, encoder_hidden_states):
        _stats["fallback"] += 1
        return _orig_forward(self, hidden_states, image_rotary_emb, attention_mask=attention_mask,
                             encoder_hidden_states=encoder_hidden_states, rotary_emb_text=rotary_emb_text,
                             rotary_emb_image=rotary_emb_image)
    _stats["fused"] += 1
    query = self.to_q(hidden_states)          # [B, S_img, H*D] per rank (ColumnParallel, gather_output=False)
    key = self.to_k(hidden_states)
    value = self.to_v(hidden_states)
    D = _P
    eps = self.norm_q.variance_epsilon
    w_q, w_k = _w(self.norm_q, D), _w(self.norm_k, D)

    if encoder_hidden_states is not None:
        n_txt = encoder_hidden_states.shape[1]
        query = torch.cat([self.add_q_proj(encoder_hidden_states), query], dim=1)
        key = torch.cat([self.add_k_proj(encoder_hidden_states), key], dim=1)
        value = torch.cat([self.add_v_proj(encoder_hidden_states), value], dim=1)
        w_q_txt, w_k_txt = _w(self.norm_added_q, D), _w(self.norm_added_k, D)
    else:
        n_txt = 0
        w_q_txt, w_k_txt = w_q, w_k

    out = fused_attention(query, key, value, image_rotary_emb, w_q, w_k, w_q_txt, w_k_txt,
                          n_txt, eps, lnc_grid()).to(query.dtype)

    if encoder_hidden_states is not None:
        encoder_out, out = out[:, :n_txt], out[:, n_txt:]
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        encoder_out = self.to_add_out(encoder_out)
        return out, encoder_out
    if self.padded_inner_dim != self.out_dim:
        return out[..., : self.out_dim]
    return out


def enable(attention_cls=None, verbose=True):
    """Patch NeuronFluxAttention.forward (idempotent). Call before compiling the backbone."""
    global _orig_forward
    if attention_cls is None:
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import NeuronFluxAttention
        attention_cls = NeuronFluxAttention
    if getattr(attention_cls.forward, "_flux_nki_patched", False):
        return
    _orig_forward = attention_cls.forward
    _nki_forward._flux_nki_patched = True
    attention_cls.forward = _nki_forward
    # neuronx-cc / any spawned tracer process must be able to import the kernel module by name.
    os.environ["PYTHONPATH"] = _HERE + os.pathsep + os.environ.get("PYTHONPATH", "")
    if verbose:
        print(f"[nki] NeuronFluxAttention.forward -> fused QK-norm + RoPE + flash attention ({_HERE})", flush=True)


def stats():
    """How many attention calls were traced through the kernel vs. the stock path (check after compile)."""
    return dict(_stats)
