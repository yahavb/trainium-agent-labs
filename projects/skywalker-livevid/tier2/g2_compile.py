"""Trace one Neuron graph per DiT layer (same module and compiler args as G1).

    python g2_compile.py extract            write block<i>_fp32.pt for all 30 layers (one checkpoint load)
    python g2_compile.py compile 0 10       trace layers 0..9 -> artifacts/tier2/blockNN_<tag>_bf16.pt
"""
import json
import sys
import time

import torch

from block_port import ART_DIR, BlockCfg, CacheState, extract_all_layers, load_port, rope_cos_sin

BF16 = torch.bfloat16
COMPILER_ARGS = ["--auto-cast=none", "--model-type=transformer"]


def artifact(cfg, layer):
    return ART_DIR / f"block{layer:02d}_{cfg.tag}_bf16.pt"


def compile_layer(cfg, layer):
    import torch_neuronx
    out = artifact(cfg, layer)
    if out.exists():
        print(f"layer {layer}: exists, skip", flush=True)
        return
    port, _ = load_port(cfg, layer)
    st = CacheState(cfg)
    cos_, sin_ = rope_cos_sin(cfg, 0)
    ex = (torch.randn(1, cfg.l_chunk, cfg.dim), torch.randn(1, 1, 6, cfg.dim), torch.randn(1, cfg.text_len, cfg.dim),
          cos_, sin_, st.k, st.v, st.bias())
    ex = tuple(t.to(BF16) for t in ex)
    work = ART_DIR / "work" / out.stem
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    traced = torch_neuronx.trace(port.to(BF16), ex, compiler_workdir=str(work), compiler_args=COMPILER_ARGS)
    dt = time.perf_counter() - t0
    tmp = out.with_suffix(".tmp")
    torch.jit.save(traced, str(tmp))
    tmp.rename(out)
    (ART_DIR / f"{out.stem}_compile.json").write_text(json.dumps(
        {"layer": layer, "cfg": cfg.tag, "compile_s": round(dt, 1), "compiler_args": COMPILER_ARGS}) + "\n")
    print(f"layer {layer}: COMPILE OK {dt:.1f}s", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    if sys.argv[1] == "extract":
        extract_all_layers()
        print("EXTRACT OK", flush=True)
    else:
        cfg = BlockCfg()
        for layer in range(int(sys.argv[2]), int(sys.argv[3])):
            compile_layer(cfg, layer)
