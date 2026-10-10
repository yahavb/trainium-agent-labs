"""Compile one pipeline stage: several consecutive blocks in one graph with their KV caches as resident state.

    python g3_stage.py 0 8      layers 0..7 -> artifacts/tier2/stage_00_08_<tag>_resident_bf16.pt

Initial state is zeros; the pipeline overwrites it with the warmed caches before moving it to the device.
"""
import json
import sys
import time

import torch

from block_port import ART_DIR, BlockCfg, CausalBlockPort, rope_cos_sin
from g2_resident import COMPILER_ARGS, ResidentBlocks

BF16 = torch.bfloat16
STAGES = [(0, 8), (8, 16), (16, 23), (23, 30)]


def stage_path(a, b, cfg=BlockCfg()):
    return ART_DIR / f"stage_{a:02d}_{b:02d}_{cfg.tag}_resident_bf16.pt"


def main():
    import torch_neuronx
    a, b = int(sys.argv[1]), int(sys.argv[2])
    cfg = BlockCfg()
    out = stage_path(a, b, cfg)
    if out.exists():
        print("exists, skip")
        return
    ports = []
    for i in range(a, b):
        m = CausalBlockPort(cfg)
        m.load_state_dict(torch.load(ART_DIR / f"block{i}_fp32.pt")["block"], strict=True)
        ports.append(m.eval().requires_grad_(False))
    z = torch.zeros(1, cfg.l_cache, cfg.num_heads, cfg.head_dim)
    module = ResidentBlocks(ports, [(z, z)] * (b - a)).to(BF16)
    aliases = {}
    for i in range(b - a):
        aliases[module.k[i]] = 1 + 2 * i
        aliases[module.v[i]] = 2 + 2 * i
    cos_, sin_ = rope_cos_sin(cfg, 0)
    ex = tuple(t.to(BF16) for t in (torch.randn(1, cfg.l_chunk, cfg.dim), torch.randn(1, 1, 6, cfg.dim),
                                    torch.randn(1, cfg.text_len, cfg.dim), cos_, sin_,
                                    torch.zeros(1, 1, 1, cfg.l_cache)))
    work = ART_DIR / "work" / out.stem
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    traced = torch_neuronx.trace(module, ex, input_output_aliases=aliases, compiler_workdir=str(work),
                                 compiler_args=COMPILER_ARGS)
    dt = time.perf_counter() - t0
    torch.jit.save(traced, str(out))
    (ART_DIR / f"{out.stem}_compile.json").write_text(json.dumps({"layers": [a, b], "compile_s": round(dt, 1)}) + "\n")
    print(f"STAGE {a}-{b} COMPILE OK {dt:.1f}s", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
