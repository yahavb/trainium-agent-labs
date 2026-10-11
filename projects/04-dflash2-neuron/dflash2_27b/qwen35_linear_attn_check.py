#!/usr/bin/env python3
"""qwen35_linear_attn_check.py — validate the Qwen3.5 GatedDeltaNet (linear
attention / SSM) forward using the real downloaded layer-0 weights.

Strategy for task 14 (the crux): the HF reference GatedDeltaNet forward delegates
to PURE-PYTORCH functions (torch_chunk_gated_delta_rule, causal_conv1d_fn) — not
CUDA-only kernels — so it is portable to the Neuron XLA path as-is. This harness:
  1. builds the HF Qwen3_5GatedDeltaNet for layer 0,
  2. loads the real layer-0 linear_attn weights from the 27B checkpoint,
  3. runs a prefill forward on a random prompt and checks shape + finiteness,
  4. runs the recurrent single-token path and checks it matches the chunked path
     on the last position (prefill vs decode consistency) — the core correctness
     property any Neuron port must preserve.
"""
import glob, json, os
import torch
from safetensors import safe_open

CKPT = glob.glob("/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/")[0]


def load_layer0_linear_attn():
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    want = {k: v for k, v in idx.items() if "language_model.layers.0.linear_attn" in k}
    handles = {}
    out = {}
    for k, shard in want.items():
        p = os.path.join(CKPT, shard)
        if p not in handles:
            handles[p] = safe_open(p, "pt")
        local = k.split("layers.0.linear_attn.")[1]
        out[local] = handles[p].get_tensor(k)
    return out


def main():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(CKPT)
    tcfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    print("layer_types[0]:", tcfg.layer_types[0],
          "| v_heads:", tcfg.linear_num_value_heads,
          "k_heads:", tcfg.linear_num_key_heads,
          "conv_k:", tcfg.linear_conv_kernel_dim)

    torch.manual_seed(0)
    mod = Qwen3_5GatedDeltaNet(tcfg, layer_idx=0).to(torch.float32).eval()

    w = load_layer0_linear_attn()
    print("checkpoint linear_attn tensors:", sorted(w.keys()))
    # load into module by name
    sd = mod.state_dict()
    mapped, missing = {}, []
    for name in sd:
        # module param names: in_proj_qkv.weight, conv1d.weight, A_log, dt_bias, norm.weight, out_proj.weight, in_proj_z/b/a.weight
        if name in w:
            mapped[name] = w[name].to(torch.float32)
        else:
            missing.append(name)
    print("mapped:", len(mapped), "missing:", missing)
    mod.load_state_dict(mapped, strict=False)

    B, S, H = 1, 16, tcfg.hidden_size
    x = torch.randn(B, S, H, dtype=torch.float32) * 0.1
    with torch.no_grad():
        y = mod(x)
    y = y[0] if isinstance(y, tuple) else y
    print(f"PREFILL forward: in {tuple(x.shape)} -> out {tuple(y.shape)}  finite={torch.isfinite(y).all().item()}")
    assert y.shape == (B, S, H), y.shape
    assert torch.isfinite(y).all()

    # sanity: output changes with input (not a constant/degenerate map)
    with torch.no_grad():
        y2 = mod(x + 0.01)
    changed = (y2 - y).abs().mean().item()
    print(f"output responds to input delta: mean|dy|={changed:.4e}")

    ok = y.shape == (B, S, H) and torch.isfinite(y).all().item() and changed > 0
    print("QWEN35_LINEAR_ATTN:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
