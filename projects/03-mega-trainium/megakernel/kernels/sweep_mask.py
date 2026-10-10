"""Does the attention block respect the mask / S_ctx? Vary write position and cache capacity."""
import os, sys, json
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_attn import run
configs = [
    dict(H=512, q=4, kv=2, t=100, S_ctx=128, S_max=256),   # baseline: 27 masked zero slots inside S_ctx, 128 beyond
    dict(H=512, q=4, kv=2, t=127, S_ctx=128, S_max=256),   # no masked slots inside S_ctx; 128 zero slots beyond S_ctx
    dict(H=512, q=4, kv=2, t=100, S_ctx=128, S_max=128),   # masked slots inside S_ctx only; nothing beyond
    dict(H=512, q=4, kv=2, t=127, S_ctx=128, S_max=128),   # neither
    dict(H=512, q=4, kv=2, t=255, S_ctx=256, S_max=256),   # neither, longer context
    dict(H=512, q=4, kv=2, t=1,   S_ctx=128, S_max=128),   # almost everything masked
]
for c in configs:
    e, dt = run(**c)
    print(f"{json.dumps(c):70s} attention rel-L2 = {e:.3e}   ({dt:.0f}s)", flush=True)
