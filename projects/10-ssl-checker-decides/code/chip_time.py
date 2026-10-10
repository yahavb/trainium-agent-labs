"""Measured on-chip latency (nki benchmark: warmup 5, timed 10) vs the compiler's prediction.

Kernels: today's level-8 attention kernel, and the three level-4 matmuls from the speed-up experiment
(slow TILE_N=128 start, tutorial reference, Qwen's 109 us kernel). Correctness of each device output
is checked against NumPy.
    LNC=1 NEURON_LOGICAL_NC_CONFIG=1 python chip_time.py
"""
import json
import os
import sys
import time
import traceback
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "/workspace/projects/02-kernel-agent")
import numpy as np
import nkibench
from nki.compiler.ncc_driver import CompileOptions
from nki.compiler.driver import _compile_bir_to_neff
from nki.framework.compiled import compile_kernel_to_nir

LNC = int(os.environ.get("LNC", "1"))


def run(label, kernel, inputs, want, repeat_note=""):
    d = f"/tmp/_chip/{label}"
    os.makedirs(d, exist_ok=True)
    opts = CompileOptions(target="trn2", lnc=LNC, emit_predicted_latency=True, enable_statistics=True,
                          artifacts_dir=d)
    row = dict(kernel=label)
    try:
        nir = compile_kernel_to_nir(kernel, inputs=inputs, compile_opts=opts, enable_cache=False)
        row["pred_us"] = round(nir.total_time_ns / 1000, 2)
        compiled = _compile_bir_to_neff(nir, opts, inputs)
        res = compiled.run(profile=True, **compiled.prepare_inputs(inputs))
        out = next(iter(res.outputs.values())).reshape(want.shape)
        row["device_correct"] = nkibench.describe_mismatch(out, want) is None
        row["chip_us_mean"] = round(res.latency * 1e6, 2)
        row["chip_us_min_max_std"] = [round(x * 1e6, 2) for x in (res.latency_min, res.latency_max, res.latency_std)]
        row["chip_over_pred"] = round(row["chip_us_mean"] / row["pred_us"], 2)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        traceback.print_exc(limit=2)
    print(json.dumps(row), flush=True)
    return row


def load(path, entry):
    return nkibench.load_kernel(path, entry)


rows = []
# level 8, every level-8 shape
k8 = load("/tmp/l8k.py", "nki_attention_")
for spec in nkibench.LEVELS[8]["shapes"]:
    args, _ = nkibench.make_inputs(spec, 8)
    rows.append(run(f"attention_seq{spec['seq']}_dim{spec['dim']}", k8,
                    {"q": args[0], "k": args[1], "v": args[2]}, nkibench.ref_attention(*args)))
# level 4 matmuls on the largest level-4 shape (where the speed-up mattered most)
src = open("/workspace/projects/02-kernel-agent/reference_level4.py").read()
open("/tmp/_slow4.py", "w").write(src.replace("TILE_N = nl.tile_size.gemm_moving_fmax  # 512", "TILE_N = 128"))
spec = dict(K=256, M=512, N=1024)
args, _ = nkibench.make_inputs(spec, 4)
want = nkibench.ref_matmul(*args)
for label, path in (("matmul_slow_TILE_N128", "/tmp/_slow4.py"),
                    ("matmul_tutorial_reference", "/workspace/projects/02-kernel-agent/reference_level4.py"),
                    ("matmul_qwen_109us", "/tmp/k109.py")):
    rows.append(run(label, load(path, "nki_matmul_tiled_"), {"lhsT": args[0], "rhs": args[1]}, want))
json.dump(rows, open("/tmp/chip_time.json", "w"), indent=1)
