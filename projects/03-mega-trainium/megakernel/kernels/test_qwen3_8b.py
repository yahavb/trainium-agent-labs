"""Real-weights correctness test: Qwen3-8B decode step, megakernel vs HuggingFace, layer by layer.

Loads Qwen/Qwen3-8B from the local HF cache (CPU, float32 for the reference), prefills a real
prompt of t tokens, then decodes one token. The NKI kernel runs the first L layers for that token
on a free NeuronCore (standalone NumPy path, bf16) and is compared with the HF hidden state after
layer L-1. With --layers 36 it also applies the final norm + lm_head to both and reports next-token
and top-k agreement.

  NEURON_RT_VISIBLE_CORES=2 python kernels/test_qwen3_8b.py --layers 4
  NEURON_RT_VISIBLE_CORES=2 python kernels/test_qwen3_8b.py --layers 36
"""
import argparse, os, sys, time, json
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "2")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, ml_dtypes
from qwen3_ref import pack_layer_weights, rope_tables, decode_mask, decode_layers_ref

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--t", type=int, default=100)
ap.add_argument("--S_ctx", type=int, default=128)
ap.add_argument("--S_max", type=int, default=256)
ap.add_argument("--prompt", default="The Trainium2 chip has eight physical NeuronCores, and the Neuron Kernel Interface lets you")
ap.add_argument("--ref-layers", action="store_true", help="also run the torch fp32 reference for the same L (slow for large L)")
ap.add_argument("--out", default=None)
ap.add_argument("--zero", choices=["attn", "mlp"], default=None, help="zero W_out or W_down (in kernel AND reference) to attribute error")
ap.add_argument("--kernel", default="qwen3_decode_layers", help="kernel function in qwen3_megakernel.py")
ap.add_argument("--kw", default="{}", help="extra compile-time kernel kwargs, as JSON")
args = ap.parse_args()
L, t, S_ctx, S_max = args.layers, args.t, args.S_ctx, args.S_max
assert t < S_ctx <= S_max and S_ctx % 128 == 0

from transformers import AutoTokenizer, AutoModelForCausalLM
name = "Qwen/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(name)
t0 = time.time()
hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
cfg = hf.config
print(f"loaded {name} in {time.time()-t0:.0f}s: L={cfg.num_hidden_layers} H={cfg.hidden_size} I={cfg.intermediate_size} q={cfg.num_attention_heads} kv={cfg.num_key_value_heads} d={cfg.head_dim}")
H, I, q, kv, d = cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
rope_theta = getattr(cfg, "rope_theta", None) or cfg.rope_parameters["rope_theta"]
eps = cfg.rms_norm_eps

ids = tok(args.prompt, return_tensors="pt").input_ids
if ids.shape[1] < t + 1:   # pad the prompt by repeating it
    reps = (t + 1) // ids.shape[1] + 1
    ids = tok(" ".join([args.prompt] * reps), return_tensors="pt").input_ids
ids = ids[:, : t + 1]
B = 1

layer_out = {}
hooks = [hf.model.layers[l].register_forward_hook(
    (lambda l: lambda m, i, o: layer_out.__setitem__(l, (o[0] if isinstance(o, tuple) else o).detach().clone()))(l))
    for l in range(cfg.num_hidden_layers)]
t0 = time.time()
with torch.no_grad():
    pre = hf(input_ids=ids[:, :t], use_cache=True)
    pkv = pre.past_key_values
    layer_out.clear()
    dec = hf(input_ids=ids[:, t:], past_key_values=pkv, use_cache=True)
print(f"HF prefill({t}) + decode(1) on CPU fp32: {time.time()-t0:.0f}s; HF next token: {tok.decode(dec.logits[0, -1].argmax())!r}")
emb = hf.model.embed_tokens(ids[:, t:]).detach()

# Pack weights for the first L layers (bf16, the dtype the model ships in)
state = hf.state_dict()
layers = [{k[len(f"model.layers.{l}."):]: v for k, v in state.items() if k.startswith(f"model.layers.{l}.")} for l in range(L)]
W = pack_layer_weights(layers, dtype=torch.bfloat16)
if args.zero == "attn": W["W_out"].zero_()
if args.zero == "mlp": W["W_down"].zero_()
K_cache = torch.zeros(L, B, kv, d, S_max, dtype=torch.bfloat16)
V_cache = torch.zeros(L, B, kv, S_max, d, dtype=torch.bfloat16)
for l in range(L):
    K_cache[l, :, :, :, :t] = pkv.layers[l].keys[:, :, :t, :].transpose(-1, -2).bfloat16()
    V_cache[l, :, :, :t, :] = pkv.layers[l].values[:, :, :t, :].bfloat16()
pos = torch.full((B, 1), t, dtype=torch.int64)
cos, sin = rope_tables(torch.full((B,), t), d, rope_theta)
mask = decode_mask(pos, S_ctx, q)

bf = ml_dtypes.bfloat16
def np_bf16(x): return x.detach().contiguous().to(torch.float32).numpy().astype(bf)
inputs = dict(X=np_bf16(emb), **{k: np_bf16(v) for k, v in W.items()}, K_cache=np_bf16(K_cache), V_cache=np_bf16(V_cache),
              cos=np_bf16(cos), sin=np_bf16(sin), mask=mask.numpy().astype(np.uint8), pos_ids=pos.numpy().astype(np.uint32))
print(f"kernel inputs: {sum(v.nbytes for v in inputs.values())/1e9:.2f} GB (host->device copied once for this standalone call)")

import json as _json, nki, qwen3_megakernel
qwen3_decode_layers_jit = nki.jit(getattr(qwen3_megakernel, args.kernel))
lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
t0 = time.time()
out = qwen3_decode_layers_jit[lnc](**inputs, num_layers=L, eps=eps, **_json.loads(args.kw))
print(f"kernel compile+H2D+run for L={L}: {time.time()-t0:.0f}s")
out_t = torch.from_numpy(np.asarray(out).astype(np.float32)).reshape(B, 1, H)

hf_L = layer_out[L - 1]                       # HF hidden after layer L-1, fp32
if args.zero is not None:   # HF hidden no longer applies; use the fp32 torch reference with the same zeroed weights
    Wf = {k: v.float() for k, v in W.items()}
    hf_L, _, _ = decode_layers_ref(emb, *[Wf[k] for k in ("W_qkv", "W_out", "W_gate", "W_up", "W_down", "g_attn", "g_mlp", "g_q", "g_k")],
                                   K_cache.float(), V_cache.float(), cos, sin, mask, pos, L, eps, q, kv, d)
    print(f"(--zero {args.zero}: comparing against fp32 torch reference with the same zeroed weights)")
def rel_l2(a, b): return ((a - b).norm() / (b.norm() + 1e-9)).item()
res = dict(layers=L, t=t, rel_l2_vs_hf_fp32=rel_l2(out_t, hf_L), max_rel=((out_t - hf_L).abs().max() / hf_L.abs().max()).item())
print(f"[L={L}] kernel vs HF fp32 hidden after layer {L-1}: rel-L2 {res['rel_l2_vs_hf_fp32']:.3e}, max|diff|/max|hf| {res['max_rel']:.3e}")
# what pure bf16 weights do to HF itself (upper bound on "acceptable")
if args.ref_layers:
    Wf = {k: v.float() for k, v in W.items()}
    ref_bf16, _, _ = decode_layers_ref(emb.bfloat16(), *[W[k] for k in ("W_qkv", "W_out", "W_gate", "W_up", "W_down", "g_attn", "g_mlp", "g_q", "g_k")],
                                       K_cache, V_cache, cos.bfloat16(), sin.bfloat16(), mask, pos, L, eps, q, kv, d, compute_dtype=torch.bfloat16)
    res["rel_l2_torch_bf16_ref_vs_hf"] = rel_l2(ref_bf16.float(), hf_L)
    res["rel_l2_kernel_vs_torch_bf16_ref"] = rel_l2(out_t, ref_bf16.float())
    print(f"[L={L}] torch bf16 reference vs HF fp32: {res['rel_l2_torch_bf16_ref_vs_hf']:.3e};  kernel vs torch bf16 ref: {res['rel_l2_kernel_vs_torch_bf16_ref']:.3e}")

if L == cfg.num_hidden_layers:
    with torch.no_grad():
        logits_k = hf.lm_head(hf.model.norm(out_t))[0, -1]
        logits_h = dec.logits[0, -1]
    topk_h = logits_h.topk(5).indices.tolist(); topk_k = logits_k.topk(5).indices.tolist()
    res.update(next_token_hf=tok.decode(topk_h[0]), next_token_kernel=tok.decode(topk_k[0]),
               argmax_match=topk_h[0] == topk_k[0], top5_overlap=len(set(topk_h) & set(topk_k)),
               logits_rel_l2=rel_l2(logits_k, logits_h))
    print(f"[full model] HF next token {res['next_token_hf']!r} vs kernel {res['next_token_kernel']!r}; argmax match {res['argmax_match']}; top-5 overlap {res['top5_overlap']}/5; logits rel-L2 {res['logits_rel_l2']:.3e}")
if args.out:
    with open(args.out, "a") as f:
        f.write(json.dumps(res) + "\n")
