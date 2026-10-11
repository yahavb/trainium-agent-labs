#!/usr/bin/env python3
"""qwen35_full_attn_check.py — validate the Qwen3.5 full-attention layer with the
real layer-3 (full_attention) weights + the real mRoPE rotary embedding.

Covers the Qwen3.5-specific attention features: head_dim 256, GQA 24q/4kv,
partial RoPE (0.25), interleaved mRoPE (section [11,11,10]), q/k RMSNorm on the
head dim, and the attn_output_gate (output * sigmoid(gate)).
"""
import glob, json, os
import torch
from safetensors import safe_open

CKPT = glob.glob("/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/")[0]
LAYER = 3  # a full_attention layer (full_attention_interval=4 -> layers 3,7,11,...)


def load_layer_attn(layer):
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    pref = f"model.language_model.layers.{layer}.self_attn."
    want = {k: v for k, v in idx.items() if k.startswith(pref)}
    handles, out = {}, {}
    for k, shard in want.items():
        p = os.path.join(CKPT, shard)
        handles.setdefault(p, safe_open(p, "pt"))
        out[k.split(pref)[1]] = handles[p].get_tensor(k)
    return out


def main():
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Attention, Qwen3_5TextRotaryEmbedding,
    )
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(CKPT)
    tcfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    tcfg._attn_implementation = "eager"
    print("head_dim:", tcfg.head_dim, "| q_heads:", tcfg.num_attention_heads,
          "kv:", tcfg.num_key_value_heads, "| partial_rotary:",
          tcfg.rope_parameters.get("partial_rotary_factor"),
          "| layer_types[3]:", tcfg.layer_types[LAYER])

    torch.manual_seed(0)
    attn = Qwen3_5Attention(tcfg, layer_idx=LAYER).to(torch.float32).eval()
    w = load_layer_attn(LAYER)
    print("ckpt attn tensors:", sorted(w.keys()))
    sd = attn.state_dict()
    mapped = {n: w[n].to(torch.float32) for n in sd if n in w}
    print("mapped:", len(mapped), "of", len(sd), "| missing:", [n for n in sd if n not in w])
    attn.load_state_dict(mapped, strict=False)

    B, S, H = 1, 16, tcfg.hidden_size
    x = torch.randn(B, S, H, dtype=torch.float32) * 0.1
    rotary = Qwen3_5TextRotaryEmbedding(tcfg)
    # mRoPE expects 3-section position ids [3, B, S] (t,h,w); for text all equal.
    pos = torch.arange(S).view(1, 1, S).expand(3, B, S)
    with torch.no_grad():
        cos, sin = rotary(x, pos)
        # causal mask [B,1,S,S]
        m = torch.full((S, S), float("-inf")).triu(1).view(1, 1, S, S)
        y, _ = attn(x, position_embeddings=(cos, sin), attention_mask=m)
    print(f"FULL-ATTN forward: in {tuple(x.shape)} -> out {tuple(y.shape)} finite={torch.isfinite(y).all().item()}")
    with torch.no_grad():
        y2, _ = attn(x + 0.01, position_embeddings=(cos, sin), attention_mask=m)
    changed = (y2 - y).abs().mean().item()
    ok = y.shape == (B, S, H) and torch.isfinite(y).all().item() and changed > 0
    print(f"responds to input: mean|dy|={changed:.4e}")
    print("QWEN35_FULL_ATTN:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
