"""CPU fp32 parity: tier2/block_port.py vs the ORIGINAL CausalWanAttentionBlock from repos/StreamDiffusionV2.

The original block is imported unmodified and loaded with the same layer-0 weights. flash_attn is not installed
here, so the repo's own scaled_dot_product_attention fallbacks run (causal_model.py:126, attention.py:80); nothing
is monkeypatched.

Test A  cold start, N consecutive chunks: the original fills and then rotates its ring buffer; the port runs the
        fixed-size graph with host CacheState. Compares y every chunk and the final caches frame by frame.
Test B  one call on a fully random K/V cache (steady state: the ring is full and evicts its oldest non-sink slot).

Gate: every cosine >= 0.9999.
"""
import argparse
import json
import sys
import types
import warnings
from pathlib import Path

import torch

from block_port import BlockCfg, CacheState, ART_DIR, cosine, load_port, make_inputs, max_abs, rope_cos_sin

REPO = Path(__file__).resolve().parent.parent / "repos" / "StreamDiffusionV2"
GATE = 0.9999


def import_original():
    """Import models.wan.causal_model without running the package __init__ files (they pull in T5, VAE, av, ...)."""
    for name, rel in (("models", "models"), ("models.wan", "models/wan"),
                      ("models.wan.wan_base", "models/wan/wan_base"),
                      ("models.wan.wan_base.modules", "models/wan/wan_base/modules")):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(REPO / rel)]
        sys.modules[name] = pkg
    import importlib
    return importlib.import_module("models.wan.causal_model"), importlib.import_module(
        "models.wan.wan_base.modules.model")


def build_original(cfg, block_sd):
    cm, base = import_original()
    blk = cm.CausalWanAttentionBlock("t2v_cross_attn", cfg.dim, cfg.ffn_dim, cfg.num_heads, (-1, -1), True, True,
                                     cfg.eps)
    blk.load_state_dict(block_sd, strict=True)
    blk.eval().requires_grad_(False)
    blk.self_attn.sink_size = cfg.num_sink
    # adapt_sink_threshold (0.2 in the yaml) is data-dependent control flow that the port leaves to the host.
    blk.self_attn.adapt_sink_thr = -1
    d = cfg.head_dim
    freqs = torch.cat([base.rope_params(1024, d - 4 * (d // 6)), base.rope_params(1024, 2 * (d // 6)),
                       base.rope_params(1024, 2 * (d // 6))], dim=1)
    return cm, blk, freqs


def new_orig_cache(cm, cfg):
    shape = (1, cfg.l_cache, cfg.num_heads, cfg.head_dim)
    return {
        "k": torch.zeros(shape), "v": torch.zeros(shape),
        "global_end_index": torch.tensor([0], dtype=torch.long),
        "local_end_index": torch.tensor([0], dtype=torch.long),
        "pos": torch.full((1, cfg.num_kv_cache), cm.KV_POS_EMPTY, dtype=torch.long),
        # one denoising stream per cache, as in the demo's batched mode (every call advances the ring)
        "total_steps": 1, "current_step": 1,
    }


def call_original(blk, cfg, freqs, kv, x, e, context, frame):
    start = torch.tensor([frame * cfg.frame_tokens], dtype=torch.long)
    return blk(x, e=e, seq_lens=torch.tensor([cfg.l_chunk]),
               grid_sizes=torch.tensor([[cfg.frames_per_chunk, cfg.grid_h, cfg.grid_w]]), freqs=freqs,
               context=context, context_lens=None, block_mask=None, kv_cache=kv, crossattn_cache=None,
               current_start=start, current_end=start + cfg.l_chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=9)
    ap.add_argument("--out", default=str(ART_DIR / "check_cpu.json"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*flash_attn.*")
    torch.manual_seed(0)

    cfg = BlockCfg(height=args.height, width=args.width)
    port, data = load_port(cfg)
    cm, orig, freqs = build_original(cfg, data["block"])
    l = cfg.frame_tokens
    result = {"cfg": cfg.tag, "gate": GATE, "cold_start": [], "manual_attn": None}

    # ---- Test A: cold start ----
    kv = new_orig_cache(cm, cfg)
    st = CacheState(cfg)
    for i in range(args.chunks):
        x, e, ctx = make_inputs(cfg, data["embed"], seed=100 + i, timestep=700.0)
        y_ref = call_original(orig, cfg, freqs, kv, x, e, ctx, frame=i)
        cos_, sin_ = rope_cos_sin(cfg, start_frame=i)
        y, nk, nv = port(x, e, ctx, cos_, sin_, st.k, st.v, st.bias())
        st.commit(nk, nv)
        row = {"chunk": i, "y_cos": cosine(y, y_ref), "y_max_abs": max_abs(y, y_ref),
               "y_ref_abs_max": y_ref.abs().max().item(), "port_slots": list(st.frames),
               "orig_slots": [p if p != cm.KV_POS_EMPTY else None for p in kv["pos"][0].tolist()]}
        result["cold_start"].append(row)
        print(f"A chunk {i}: y cos={row['y_cos']:.8f} max_abs={row['y_max_abs']:.3e} "
              f"port_slots={row['port_slots']} orig_slots={row['orig_slots']}", flush=True)

    # final caches, compared frame by frame (slot order differs by design)
    orig_slot = {p: s for s, p in enumerate(kv["pos"][0].tolist()) if p != cm.KV_POS_EMPTY}
    port_slot = {f: s for s, f in enumerate(st.frames) if f is not None}
    same_set = sorted(orig_slot) == sorted(port_slot)
    k_ref = torch.cat([kv["k"][:, orig_slot[f] * l:(orig_slot[f] + 1) * l] for f in sorted(orig_slot)], dim=1)
    v_ref = torch.cat([kv["v"][:, orig_slot[f] * l:(orig_slot[f] + 1) * l] for f in sorted(orig_slot)], dim=1)
    if same_set:
        k_new = torch.cat([st.k[:, port_slot[f] * l:(port_slot[f] + 1) * l] for f in sorted(port_slot)], dim=1)
        v_new = torch.cat([st.v[:, port_slot[f] * l:(port_slot[f] + 1) * l] for f in sorted(port_slot)], dim=1)
        result["cold_start_cache"] = {"frames": sorted(port_slot), "same_frame_set": True,
                                      "k_cos": cosine(k_new, k_ref), "k_max_abs": max_abs(k_new, k_ref),
                                      "v_cos": cosine(v_new, v_ref), "v_max_abs": max_abs(v_new, v_ref)}
    else:
        result["cold_start_cache"] = {"same_frame_set": False, "orig": sorted(orig_slot), "port": sorted(port_slot)}
    print("A final cache:", result["cold_start_cache"], flush=True)

    # ---- Test B: fully random cache, one call ----
    g = torch.Generator().manual_seed(7)
    shape = (1, cfg.l_cache, cfg.num_heads, cfg.head_dim)
    k0, v0 = torch.randn(shape, generator=g), torch.randn(shape, generator=g)
    n = cfg.num_kv_cache
    kv = new_orig_cache(cm, cfg)
    kv["k"], kv["v"] = k0.clone(), v0.clone()
    kv["global_end_index"].fill_(n * l)
    kv["local_end_index"].fill_(n * l)
    kv["pos"] = torch.arange(n).view(1, n)
    orig.self_attn.evict_idx = [[(s + 1) * l for s in range(cfg.num_sink, n)]]  # ring state of a just-filled cache
    x, e, ctx = make_inputs(cfg, data["embed"], seed=999, timestep=500.0)
    y_ref = call_original(orig, cfg, freqs, kv, x, e, ctx, frame=n)
    cos_, sin_ = rope_cos_sin(cfg, start_frame=n)
    bias = torch.zeros(1, 1, 1, cfg.l_cache)
    y, nk, nv = port(x, e, ctx, cos_, sin_, k0, v0, bias)
    s = cfg.num_sink  # the original overwrote slot `s` (oldest non-sink); the port appended at the tail
    new_ref_k, new_ref_v = kv["k"][:, s * l:(s + 1) * l], kv["v"][:, s * l:(s + 1) * l]
    kept_ref_k = torch.cat([kv["k"][:, :s * l], kv["k"][:, (s + 1) * l:]], dim=1)
    kept_ref_v = torch.cat([kv["v"][:, :s * l], kv["v"][:, (s + 1) * l:]], dim=1)
    result["random_cache"] = {
        "y_cos": cosine(y, y_ref), "y_max_abs": max_abs(y, y_ref),
        "new_k_cos": cosine(nk[:, -l:], new_ref_k), "new_k_max_abs": max_abs(nk[:, -l:], new_ref_k),
        "new_v_cos": cosine(nv[:, -l:], new_ref_v), "new_v_max_abs": max_abs(nv[:, -l:], new_ref_v),
        "kept_k_max_abs": max_abs(nk[:, :-l], kept_ref_k), "kept_v_max_abs": max_abs(nv[:, :-l], kept_ref_v),
    }
    print("B random cache:", result["random_cache"], flush=True)

    # the matmul/softmax fallback path must agree with SDPA too
    port_m, _ = load_port(cfg, manual_attn=True)
    ym, _, _ = port_m(x, e, ctx, cos_, sin_, k0, v0, bias)
    result["manual_attn"] = {"y_cos_vs_original": cosine(ym, y_ref), "y_max_abs": max_abs(ym, y_ref)}
    print("manual-attn fallback:", result["manual_attn"], flush=True)

    # Not gated: how far the plain FIFO (sink_tokens = 0, drops slot 0 instead of the oldest non-sink slot) and an
    # all-zero cache land from the original. Shows the test is sensitive to cache contents.
    port_fifo, _ = load_port(BlockCfg(height=args.height, width=args.width, num_sink=0))
    yf, _, _ = port_fifo(x, e, ctx, cos_, sin_, k0, v0, bias)
    yz, _, _ = port(x, e, ctx, cos_, sin_, torch.zeros_like(k0), torch.zeros_like(v0), bias)
    result["sensitivity"] = {"plain_fifo_y_cos_vs_original": cosine(yf, y_ref), "plain_fifo_y_max_abs": max_abs(yf, y_ref),
                             "zero_cache_y_cos_vs_original": cosine(yz, y_ref), "zero_cache_y_max_abs": max_abs(yz, y_ref)}
    print("sensitivity (not gated):", result["sensitivity"], flush=True)

    cosines = [r["y_cos"] for r in result["cold_start"]]
    cosines += [v for k, v in result["cold_start_cache"].items() if k.endswith("_cos")]
    cosines += [v for k, v in result["random_cache"].items() if k.endswith("_cos")]
    cosines.append(result["manual_attn"]["y_cos_vs_original"])
    result["min_cosine"] = min(cosines)
    result["pass"] = bool(result["min_cosine"] >= GATE and result["cold_start_cache"]["same_frame_set"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"CPU PARITY min_cosine={result['min_cosine']:.8f} gate={GATE} -> {'PASS' if result['pass'] else 'FAIL'}")
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
