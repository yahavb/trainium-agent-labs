"""Tiny-config correctness test for the Qwen3 decode megakernel.

Step 1: validate the torch reference (qwen3_ref.decode_layers_ref) against HuggingFace Qwen3 on CPU,
        so the layouts (QKV packing, QK-norm, RoPE convention, GQA grouping) are known-good.
Step 2: run the NKI kernel on a free NeuronCore through the standalone NumPy path and compare.

Run:  NEURON_RT_VISIBLE_CORES=2 python kernels/test_tiny.py [--layers 2] [--skip-device]
"""
import argparse, os, sys, time
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "2")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import ml_dtypes
from qwen3_ref import decode_layers_ref, pack_layer_weights, rope_tables, decode_mask

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--skip-device", action="store_true")
ap.add_argument("--H", type=int, default=512)
ap.add_argument("--I", type=int, default=1024)
ap.add_argument("--q", type=int, default=4)
ap.add_argument("--kv", type=int, default=2)
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--S_max", type=int, default=256)
ap.add_argument("--S_ctx", type=int, default=128)
ap.add_argument("--t", type=int, default=100, help="decode position (number of prior tokens)")
ap.add_argument("--zero", choices=["attn", "mlp"], default=None, help="zero W_out (attn) or W_down (mlp) in BOTH ref and kernel to attribute error")
ap.add_argument("--kernel", default="qwen3_decode_layers", help="kernel function in qwen3_megakernel.py")
ap.add_argument("--control", choices=["mask_active_zero", "no_rope_input"], default=None,
                help="negative control: feed the KERNEL a known bug (reference untouched) to check the checker names it")
ap.add_argument("--kw", default="{}", help="extra compile-time kernel kwargs, as JSON")
args = ap.parse_args()
torch.manual_seed(0)

from transformers import Qwen3ForCausalLM, Qwen3Config
cfg = Qwen3Config(hidden_size=args.H, intermediate_size=args.I, num_hidden_layers=args.layers,
                  num_attention_heads=args.q, num_key_value_heads=args.kv, head_dim=args.d,
                  vocab_size=1000, rope_theta=1e6, rms_norm_eps=1e-6, max_position_embeddings=4096)
hf = Qwen3ForCausalLM(cfg).eval()
with torch.no_grad():
    for p in hf.parameters():
        p.mul_(1.0)  # keep HF init (std 0.02); make norms non-trivial below
    for layer in hf.model.layers:
        layer.input_layernorm.weight.uniform_(0.5, 1.5)
        layer.post_attention_layernorm.weight.uniform_(0.5, 1.5)
        layer.self_attn.q_norm.weight.uniform_(0.5, 1.5)
        layer.self_attn.k_norm.weight.uniform_(0.5, 1.5)

L, B, H, d, q, kv, I = args.layers, 1, args.H, args.d, args.q, args.kv, args.I
t, S_ctx, S_max = args.t, args.S_ctx, args.S_max
assert t < S_ctx <= S_max and S_ctx % 128 == 0

# ---------------- Step 1: HF prefill of t tokens, then one decode token; capture last layer output
ids = torch.randint(0, 1000, (B, t + 1))
captured = {}
hf.model.layers[-1].register_forward_hook(lambda m, i, o: captured.__setitem__("last", (o[0] if isinstance(o, tuple) else o).detach().clone()))
with torch.no_grad():
    pre = hf(input_ids=ids[:, :t], use_cache=True)
    pkv = pre.past_key_values
    dec = hf(input_ids=ids[:, t:], past_key_values=pkv, use_cache=True)
hf_last = captured["last"]                       # [B, 1, H] output of last decoder layer for the new token
emb = hf.model.embed_tokens(ids[:, t:]).detach()  # [B, 1, H] input to layer 0

# Build kernel-layout tensors from HF
state = {k: v.detach() for k, v in hf.state_dict().items()}
layers = [{k[len(f"model.layers.{l}."):]: v for k, v in state.items() if k.startswith(f"model.layers.{l}.")} for l in range(L)]
W = pack_layer_weights(layers, dtype=torch.float32)
if args.zero == "attn": W["W_out"].zero_()
if args.zero == "mlp": W["W_down"].zero_()
K_cache = torch.zeros(L, B, kv, d, S_max)
V_cache = torch.zeros(L, B, kv, S_max, d)
for l in range(L):
    k_hf, v_hf = pkv.layers[l].keys, pkv.layers[l].values     # [B, kv, t, d] post-norm, post-RoPE
    K_cache[l, :, :, :, :t] = k_hf[:, :, :t, :].transpose(-1, -2)
    V_cache[l, :, :, :t, :] = v_hf[:, :, :t, :]
pos = torch.full((B, 1), t, dtype=torch.int64)
ROPE_THETA = 1e6  # set in the config above; transformers 5.x stores it under rope_parameters
cos, sin = rope_tables(torch.full((B,), t), d, ROPE_THETA)
mask = decode_mask(pos, S_ctx, q)

ref, Kc_ref, Vc_ref = decode_layers_ref(emb, W["W_qkv"], W["W_out"], W["W_gate"], W["W_up"], W["W_down"],
                                        W["g_attn"], W["g_mlp"], W["g_q"], W["g_k"], K_cache, V_cache,
                                        cos, sin, mask, pos, L, cfg.rms_norm_eps, q, kv, d)
err = (ref - hf_last).abs().max().item() / (hf_last.abs().max().item() + 1e-9)
print(f"[step 1] reference vs HuggingFace: max |diff| / max |hf| = {err:.2e}  ({'OK' if err < 1e-4 else 'MISMATCH'})")
if err >= 1e-4 and args.zero is None:
    sys.exit("reference does not match HF; fix layouts before touching the device")
if args.skip_device:
    sys.exit(0)

# ---------------- Step 2: the NKI kernel on the device, standalone NumPy path (bf16 inputs)
bf = ml_dtypes.bfloat16
def np_bf16(x): return x.detach().contiguous().float().numpy().astype(bf)
inputs = dict(
    X=np_bf16(emb), W_qkv=np_bf16(W["W_qkv"]), W_out=np_bf16(W["W_out"]), W_gate=np_bf16(W["W_gate"]),
    W_up=np_bf16(W["W_up"]), W_down=np_bf16(W["W_down"]), g_attn=np_bf16(W["g_attn"]), g_mlp=np_bf16(W["g_mlp"]),
    g_q=np_bf16(W["g_q"]), g_k=np_bf16(W["g_k"]), K_cache=np_bf16(K_cache), V_cache=np_bf16(V_cache),
    cos=np_bf16(cos), sin=np_bf16(sin), mask=mask.numpy().astype(np.uint8), pos_ids=pos.numpy().astype(np.uint32),
)
if args.control == "mask_active_zero":      # the original bug: active token's mask column left at 0
    inputs["mask"] = inputs["mask"].copy(); inputs["mask"][-1] = 0
elif args.control == "no_rope_input":       # RoPE tables replaced by the identity rotation
    inputs["cos"] = np.ones_like(inputs["cos"]); inputs["sin"] = np.zeros_like(inputs["sin"])
if args.control:
    print(f"[control] kernel inputs carry a deliberate bug: {args.control}")
import json as _json, nki, qwen3_megakernel
qwen3_decode_layers_jit = nki.jit(getattr(qwen3_megakernel, args.kernel))
lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
t0 = time.time()
out = qwen3_decode_layers_jit[lnc](**inputs, num_layers=L, eps=cfg.rms_norm_eps, **_json.loads(args.kw))
print(f"[step 2] kernel compile+run: {time.time()-t0:.1f}s, out {out.shape} {out.dtype}")
out_t = torch.from_numpy(np.asarray(out).astype(np.float32))

# bf16 reference (what the kernel can at best achieve): same math with bf16 inputs
ref_bf16, _, _ = decode_layers_ref(emb.bfloat16(), W["W_qkv"].bfloat16(), W["W_out"].bfloat16(), W["W_gate"].bfloat16(),
                                   W["W_up"].bfloat16(), W["W_down"].bfloat16(), W["g_attn"].bfloat16(), W["g_mlp"].bfloat16(),
                                   W["g_q"].bfloat16(), W["g_k"].bfloat16(), K_cache.bfloat16(), V_cache.bfloat16(),
                                   cos.bfloat16(), sin.bfloat16(), mask, pos, L, cfg.rms_norm_eps, q, kv, d)
scale = ref.abs().max().item()
e32 = (out_t - ref).abs().max().item() / scale
e16 = (out_t - ref_bf16.float()).abs().max().item() / scale
ebase = (ref_bf16.float() - ref).abs().max().item() / scale
print(f"[step 2] kernel vs fp32 ref: {e32:.3e}   kernel vs bf16-input ref: {e16:.3e}   (bf16 rounding alone: {ebase:.3e})")
print("[step 2]", "PASS" if e32 < 5e-2 else "FAIL", "(threshold 5e-2 relative to max |ref| for bf16 weights/activations)")
# did the standalone path write the cache back?
kc = torch.from_numpy(np.asarray(inputs["K_cache"]).astype(np.float32))
print("[step 2] K_cache input mutated on host:", not torch.equal(kc, K_cache.bfloat16().float()),
      "(the HOP path handles aliasing; standalone may not copy back)")

# ---------------- Step 3: diagnose which convention the kernel actually implements
def rel_l2(a, b): return ((a - b).norm() / (b.norm() + 1e-9)).item()
print(f"[diag] rel-L2 kernel vs fp32 ref: {rel_l2(out_t, ref):.3e}   bf16-ref vs fp32 ref: {rel_l2(ref_bf16.float(), ref):.3e}")
import qwen3_ref as R
variants = {}
# (a) interleaved RoPE (even/odd pairs) instead of rotate_half
def rope_interleaved(x, cos, sin):
    d_ = x.shape[-1]
    c = cos.permute(1, 2, 0).unsqueeze(2).float(); s = sin.permute(1, 2, 0).unsqueeze(2).float()
    xr, xi = x[..., 0::2], x[..., 1::2]
    o = torch.stack([xr * c - xi * s, xr * s + xi * c], dim=-1).reshape(x.shape)
    return o
orig_rope = R.rope_rotate_half
R.rope_rotate_half = rope_interleaved
variants["rope_interleaved"], _, _ = R.decode_layers_ref(emb, W["W_qkv"], W["W_out"], W["W_gate"], W["W_up"], W["W_down"], W["g_attn"], W["g_mlp"], W["g_q"], W["g_k"], K_cache, V_cache, cos, sin, mask, pos, L, cfg.rms_norm_eps, q, kv, d)
R.rope_rotate_half = orig_rope
# (b) no QK-norm
ones_d = torch.ones_like(W["g_q"])
variants["no_qk_norm"], _, _ = R.decode_layers_ref(emb, W["W_qkv"], W["W_out"], W["W_gate"], W["W_up"], W["W_down"], W["g_attn"], W["g_mlp"], ones_d, ones_d, K_cache, V_cache, cos, sin, mask, pos, L, cfg.rms_norm_eps, q, kv, d)
# (c) mask also attends slot t (self token counted twice: cache slot written + active)
mask_c = mask.clone(); mask_c[t] = 1
variants["mask_includes_self_slot"], _, _ = R.decode_layers_ref(emb, W["W_qkv"], W["W_out"], W["W_gate"], W["W_up"], W["W_down"], W["g_attn"], W["g_mlp"], W["g_q"], W["g_k"], K_cache, V_cache, cos, sin, mask_c, pos, L, cfg.rms_norm_eps, q, kv, d)
# (d) no RoPE at all
zero = torch.zeros_like(sin); one = torch.ones_like(cos)
variants["no_rope"], _, _ = R.decode_layers_ref(emb, W["W_qkv"], W["W_out"], W["W_gate"], W["W_up"], W["W_down"], W["g_attn"], W["g_mlp"], W["g_q"], W["g_k"], K_cache, V_cache, one, zero, mask, pos, L, cfg.rms_norm_eps, q, kv, d)
# (e) no attention contribution at all (attn output zero) -> tells us if attention is the culprit
variants["attn_zero"], _, _ = R.decode_layers_ref(emb, W["W_qkv"], torch.zeros_like(W["W_out"]), W["W_gate"], W["W_up"], W["W_down"], W["g_attn"], W["g_mlp"], W["g_q"], W["g_k"], K_cache, V_cache, cos, sin, mask, pos, L, cfg.rms_norm_eps, q, kv, d)
# (f) the new token does not attend to itself (the mask bug from ATTEMPTS.md attempt 1)
variants["active_token_excluded"], _, _ = R.decode_layers_ref(emb, W["W_qkv"], W["W_out"], W["W_gate"], W["W_up"], W["W_down"], W["g_attn"], W["g_mlp"], W["g_q"], W["g_k"], K_cache, V_cache, cos, sin, mask, pos, L, cfg.rms_norm_eps, q, kv, d, active_excluded=True)
for name, v in variants.items():
    print(f"[diag] rel-L2 kernel vs variant {name:26s}: {rel_l2(out_t, v):.3e}")
