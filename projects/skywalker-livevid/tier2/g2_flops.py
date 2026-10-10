"""FLOPs of one block call at the G1/G2 shapes (torch.utils.flop_counter on the fp32 CPU port).

Uses the matmul+softmax attention path: FlopCounterMode has no formula for aten::scaled_dot_product_attention, so
with SDPA the attention matmuls are silently left out (only the Linear layers, 90.2 GFLOP, get counted).
FlopCounterMode counts 2 FLOPs per multiply-accumulate.
"""
import json

import torch
from torch.utils.flop_counter import FlopCounterMode

from block_port import ART_DIR, BlockCfg, CacheState, load_port, rope_cos_sin

torch.set_grad_enabled(False)
cfg = BlockCfg()
port, _ = load_port(cfg, manual_attn=True)
st = CacheState(cfg)
cos_, sin_ = rope_cos_sin(cfg, 0)
ex = (torch.randn(1, cfg.l_chunk, cfg.dim), torch.randn(1, 1, 6, cfg.dim), torch.randn(1, cfg.text_len, cfg.dim),
      cos_, sin_, st.k, st.v, torch.zeros(1, 1, 1, cfg.l_cache))
with FlopCounterMode(display=False) as fc:
    port(*ex)
res = {"cfg": cfg.tag, "flops_per_block_call": fc.get_total_flops(),
       "by_op": {str(k): v for k, v in fc.get_flop_counts()["Global"].items()}}
(ART_DIR / "g2_flops.json").write_text(json.dumps(res, indent=2) + "\n")
print("FLOPS " + json.dumps(res))
