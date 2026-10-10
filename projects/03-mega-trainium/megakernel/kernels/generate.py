"""Greedy text generation with the one-launch Qwen3 decode step, checked token-for-token against HF.

Every token (prompt tokens included: the prompt is prefilled by running the same decode step once per
prompt token) is one launch of qwen3_decode_step: token id -> embedding -> L layers -> final norm ->
LM head -> argmax -> next token id. Weights and the KV cache stay resident on the device (torch HOP
path); per step the host uploads only the token id, RoPE cos/sin, the mask and the position, and reads
back the next token id.

The reference is HuggingFace Qwen3-8B in float32 on the CPU, truncated to the same number of layers
when --layers < 36 (fast compile for iterating; the full model is --layers 36).

  NEURON_RT_VISIBLE_CORES=2 python kernels/generate.py --layers 2 --new 16
  NEURON_RT_VISIBLE_CORES=2 python kernels/generate.py --layers 36 --new 32 --out results/generate_L36.jsonl
"""
import argparse, json, os, sys, time
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "2")
sys.path.insert(0, os.path.dirname(__file__))
import torch
import libtorch_neuronx_lite  # registers the neuron device + torch.compile backends
import torch.distributed as dist
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from qwen3_ref import pack_layer_weights, rope_tables, decode_mask
from qwen3_decode_step import (qwen3_decode_step, qwen3_decode_step_resident, qwen3_decode_step_tiled,
                               qwen3_decode_loop, pack_lm_head, pack_lm_head_tiled, rope_tables_all)

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=36)
ap.add_argument("--new", type=int, default=32, help="tokens to generate after the prompt")
ap.add_argument("--S_ctx", type=int, default=256)
ap.add_argument("--prompt", default="The Trainium2 chip has eight physical NeuronCores, and the Neuron Kernel Interface lets you")
ap.add_argument("--timing-iters", type=int, default=30, help="device-only timing of one step after generation")
ap.add_argument("--teacher-force", action="store_true",
                help="feed HF's greedy tokens instead of the kernel's own, so every step is compared on the same context")
ap.add_argument("--resident", action="store_true",
                help="qwen3_decode_step_resident: no per-step uploads; (token, pos) outputs feed the next step on the device")
ap.add_argument("--tiled-head", action="store_true", help="with --resident: qwen3_decode_step_tiled (our own LM head)")
ap.add_argument("--feed", choices=["host", "device"], default="host",
                help="resident mode: feed the next step from host copies of (token, pos) (re-uploaded, ~0.02 ms) "
                     "or from the kernel's device outputs (measured 6.4 ms/step slower at 36 layers through torch.compile)")
ap.add_argument("--loop", type=int, default=0,
                help="persistent mode: after prefill (resident step kernel), generate --loop tokens per launch with "
                     "qwen3_decode_loop; needs --resident and --new divisible by --loop")
ap.add_argument("--no-logits", action="store_true", help="do not read logits back each step (timing runs)")
ap.add_argument("--out", default=None)
args = ap.parse_args()
L, S_ctx = args.layers, args.S_ctx
lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
dev = "neuron:0"
if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29611")
    dist.init_process_group("gloo", rank=0, world_size=1)

from transformers import AutoTokenizer, AutoModelForCausalLM
name = "Qwen/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(name)
t0 = time.time()
hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
cfg = hf.config
H, q, kv, d = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
eps, theta = cfg.rms_norm_eps, (getattr(cfg, "rope_theta", None) or cfg.rope_parameters["rope_theta"])
if L < cfg.num_hidden_layers:
    hf.model.layers = hf.model.layers[:L]
    hf.config.num_hidden_layers = L
print(f"loaded {name} in {time.time() - t0:.0f}s; using {L} of {cfg.num_hidden_layers if L == 36 else 36} layers")

ids = tok(args.prompt, return_tensors="pt").input_ids
P = ids.shape[1]
assert P + args.new < S_ctx - 1, "write positions must stay below S_ctx-1 (see decode_mask)"

# ---- HF float32 greedy reference
t0 = time.time()
with torch.no_grad():
    ref = hf.generate(ids, max_new_tokens=args.new, do_sample=False, output_scores=True,
                      return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
ref_new = ref.sequences[0, P:].tolist()
ref_scores = torch.stack(ref.scores)[:, 0]          # [new, V] fp32 logits HF picked from
print(f"HF fp32 greedy ({time.time() - t0:.0f}s): {tok.decode(ref_new)!r}")

# ---- pack weights (bf16, the dtype the model ships in) and move them to the device once
state = hf.state_dict()
layers = [{k[len(f"model.layers.{l}."):]: v for k, v in state.items() if k.startswith(f"model.layers.{l}.")} for l in range(L)]
W = {k: v.to(dev) for k, v in pack_layer_weights(layers, dtype=torch.bfloat16).items()}
W_lm, lm_bias, V = pack_lm_head(hf.lm_head.weight.detach().bfloat16(), lnc)
assert lm_bias is None
dev_w = dict(embed=hf.model.embed_tokens.weight.detach().bfloat16().contiguous().to(dev),
             g_final=hf.model.norm.weight.detach().view(1, H).bfloat16().to(dev), W_lm=W_lm.to(dev), **W)
del state, layers
K_cache = torch.zeros(L, 1, kv, d, S_ctx, dtype=torch.bfloat16).to(dev)
V_cache = torch.zeros(L, 1, kv, S_ctx, d, dtype=torch.bfloat16).to(dev)

# uint32 is not a torch dtype on this path; int32 bits are identical for these values
def dev_int(v): return torch.tensor([[v]], dtype=torch.int32).to(dev)


if not args.resident:
    kern = wrap_nki(qwen3_decode_step)

    def step(token_ids, cos, sin, mask, pos):
        token, logits, _ = kern[lnc](token_ids=token_ids, K_cache=K_cache, V_cache=V_cache, cos=cos, sin=sin,
                                     mask=mask, pos_ids=pos, lm_bias=None, num_layers=L, eps=eps, **dev_w)
        return token, logits

    step_c = torch.compile(step, backend="neuron_libtorch", fullgraph=True)

    def step_inputs(token_id, t):
        pos = torch.full((1, 1), t, dtype=torch.int64)
        cos, sin = rope_tables(torch.full((1,), t), d, theta)
        return (dev_int(token_id), cos.bfloat16().to(dev), sin.bfloat16().to(dev),
                decode_mask(pos, S_ctx, q).to(dev), pos.to(torch.int32).to(dev))

    def run_step(token_id, t, state):
        """token_id: host int to feed, or None to feed the previous step's output. Returns (token, logits, state)."""
        token, logits = step_c(*step_inputs(token_id if token_id is not None else state, t))
        nxt = int(token.cpu().reshape(-1)[0])
        return nxt, logits, nxt
else:
    cos_t, sin_t = rope_tables_all(S_ctx, d, theta)
    dev_w.update(cos_table=cos_t.bfloat16().to(dev), sin_table=sin_t.bfloat16().to(dev))
    if args.tiled_head:
        kern = wrap_nki(qwen3_decode_step_tiled)
        dev_w.pop("W_lm")
        dev_w["W_tiled"] = pack_lm_head_tiled(hf.lm_head.weight.detach().bfloat16(), lnc)[0].to(dev)
        head_kw = dict(V=V)
    else:
        kern = wrap_nki(qwen3_decode_step_resident)
        head_kw = dict(lm_bias=None)

    def step(token_ids, pos):
        token, next_pos, logits = kern[lnc](token_ids=token_ids, pos_ids=pos, K_cache=K_cache, V_cache=V_cache,
                                            num_layers=L, S_ctx=S_ctx, eps=eps, **head_kw, **dev_w)
        return token, next_pos, logits

    step_c = torch.compile(step, backend="neuron_libtorch", fullgraph=True)

    def run_step(token_id, t, state):
        """state = previous step's (device token, device pos, host token)."""
        if args.feed == "host" or state is None:
            feed_tok = token_id if token_id is not None else state[2]
            token, next_pos, logits = step_c(dev_int(feed_tok), dev_int(t))
        else:
            dtok, dpos, _ = state
            token, next_pos, logits = step_c(dev_int(token_id) if token_id is not None else dtok, dpos)
        nxt = int(token.cpu().reshape(-1)[0])
        if args.feed == "host":
            # keep no device outputs alive across calls: holding them costs ~6 ms per call at 36 layers
            return nxt, logits, (None, None, nxt)
        return nxt, logits, (token, next_pos, nxt)


# ---- prefill (one launch per prompt token) + greedy decode, all through the same kernel. With
# --teacher-force the kernel is fed HF's tokens, so gen[i] is its argmax on HF's context at step i.
t0 = time.time()
_, _, state = run_step(ids[0, 0].item(), 0, None)      # compile + first prompt token
compile_s = time.time() - t0
print(f"compile + first launch: {compile_s:.0f}s")
gen, step_ms, logits_at = [], [], {}
loop_ms, loop_compile_s = [], None
if args.loop:
    assert args.resident and not args.teacher_force, "--loop needs --resident and free-running generation"
    lkern = wrap_nki(qwen3_decode_loop)

    def loop_fn(token_ids, pos):
        return lkern[lnc](token_ids=token_ids, pos_ids=pos, K_cache=K_cache, V_cache=V_cache, lm_bias=None,
                          num_layers=L, S_ctx=S_ctx, num_steps=args.loop, eps=eps, **dev_w)

    loop_c = torch.compile(loop_fn, backend="neuron_libtorch", fullgraph=True)
    for t in range(1, P):                                  # prefill with the step kernel; gen[0] comes out of it
        nxt, logits, state = run_step(ids[0, t].item(), t, state)
    gen.append(nxt)
    t = P
    while len(gen) < args.new:                             # then num_steps tokens per launch
        t1 = time.perf_counter()
        out_tokens, _ = loop_c(dev_int(gen[-1]), dev_int(t))
        chunk = out_tokens.cpu().reshape(-1).tolist()
        dt = (time.perf_counter() - t1) * 1e3
        if loop_compile_s is None:
            loop_compile_s = dt / 1e3
        else:
            loop_ms.append(dt)
        gen.extend(int(x) for x in chunk)
        t += args.loop
    print(f"loop kernel compile + first launch {loop_compile_s:.0f}s; per launch of {args.loop} tokens: "
          f"{[round(x, 2) for x in loop_ms]} ms")
for t in range(1, P + args.new) if not args.loop else []:
    feed = ids[0, t].item() if t < P else (ref_new[t - P] if args.teacher_force else None)
    t1 = time.perf_counter()
    nxt, logits, state = run_step(feed, t, state)
    step_ms.append((time.perf_counter() - t1) * 1e3)
    if t >= P - 1:
        gen.append(nxt)
        if len(gen) <= args.new and not args.no_logits:
            logits_at[len(gen) - 1] = logits.cpu().float().reshape(-1)[:V]
    del logits                                         # see run_step: do not hold device outputs into the next call
    if len(gen) == args.new:
        break
# the token predicted after the LAST prompt token is generation step 0; the loop above starts
# collecting at t = P - 1, so gen[0] is the prediction from position P - 1
gen = gen[: args.new]
print(f"kernel greedy: {tok.decode(gen)!r}")

match = [a == b for a, b in zip(gen, ref_new)]
if args.teacher_force:
    miss = [i for i, m in enumerate(match) if not m]
    # A disagreement is "explained" when HF's own fp32 margin between its pick and the kernel's pick is
    # below 3 sigma of the kernel's logit error at that step: then bf16 noise alone can flip the argmax.
    def sigma(i): return float((logits_at[i] - ref_scores[i]).std())
    res_tf = {i: dict(hf=tok.decode(ref_new[i]), kernel=tok.decode(gen[i]),
                      hf_margin=round(float(ref_scores[i].max() - ref_scores[i][gen[i]]), 4),
                      three_sigma=round(3 * sigma(i), 4),
                      explained_by_bf16_noise=bool(float(ref_scores[i].max() - ref_scores[i][gen[i]]) < 3 * sigma(i)))
              for i in miss}
    print(f"teacher-forced argmax agreement: {sum(match)}/{len(match)}; disagreements: {res_tf}")
first_div = match.index(False) if False in match else None
res = dict(layers=L, resident=args.resident, tiled_head=args.tiled_head, feed=args.feed, logits_read=not args.no_logits, teacher_forced=args.teacher_force, prompt_tokens=P, new_tokens=args.new, S_ctx=S_ctx, compile_s=round(compile_s, 1),
           tokens_match=sum(match), first_divergence=first_div,
           kernel_text=tok.decode(gen), hf_text=tok.decode(ref_new),
           host_loop_ms_per_step_median=round(sorted(step_ms)[len(step_ms) // 2], 3) if step_ms else None,
           loop_tokens_per_launch=args.loop or None,
           loop_ms_per_token=round(sorted(loop_ms)[len(loop_ms) // 2] / args.loop, 3) if loop_ms else None)
if first_div is not None:
    s = ref_scores[first_div]
    top2 = s.topk(2)
    res["hf_top2_margin_at_divergence"] = round(float(top2.values[0] - top2.values[1]), 4)
    res["hf_logit_of_kernel_choice_gap"] = round(float(s.max() - s[gen[first_div]]), 4)
n_cmp = args.new if args.teacher_force else (first_div if first_div is not None else args.new)
rels = [float((logits_at[i] - ref_scores[i]).norm() / ref_scores[i].norm()) for i in range(min(len(logits_at), n_cmp))]
res["logits_rel_l2_vs_hf_mean_before_divergence"] = round(sum(rels) / max(len(rels), 1), 5)
res["logit_err_sigma_mean"] = round(sum(float((logits_at[i] - ref_scores[i]).std()) for i in range(min(len(logits_at), n_cmp)))
                                    / max(min(len(logits_at), n_cmp), 1), 5)
print(f"tokens matching HF: {sum(match)}/{args.new}; first divergence: {first_div}; "
      f"logits rel-L2 vs HF (mean, matched prefix): {res['logits_rel_l2_vs_hf_mean_before_divergence']:.3e}")
if first_div is not None:
    print(f"  at divergence HF top-2 margin {res['hf_top2_margin_at_divergence']} (fp32 logits); "
          f"HF logit gap to kernel's choice {res['hf_logit_of_kernel_choice_gap']}")
if step_ms:
    print(f"host loop per step (upload inputs + one launch + read token): median {res['host_loop_ms_per_step_median']:.2f} ms")
if loop_ms:
    print(f"persistent loop: {res['loop_ms_per_token']:.2f} ms per token ({args.loop} tokens per launch, wall clock incl. host)")

# ---- device-only time of one full step (same method as bench_mega: sync each call, inputs resident)
if args.timing_iters:
    inp = step_inputs(gen[-1], P + args.new - 1) if not args.resident else (dev_int(gen[-1]), dev_int(P + args.new - 1))
    for _ in range(3):
        step_c(*inp)[0].cpu()
    t1 = time.perf_counter()
    for _ in range(args.timing_iters):
        step_c(*inp)[0].cpu()
    res["device_ms_per_step"] = round((time.perf_counter() - t1) / args.timing_iters * 1e3, 3)
    print(f"one full decode step (inputs resident, sync each call): {res['device_ms_per_step']:.3f} ms")
if args.teacher_force:
    res["teacher_forced_disagreements"] = res_tf
if args.out:
    with open(args.out, "a") as f:
        f.write(json.dumps(res) + "\n")
