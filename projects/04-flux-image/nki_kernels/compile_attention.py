"""Compile (neuronx-cc, CPU only, no NeuronCore needed) the fused FLUX attention kernel and, for an
apples-to-apples baseline, nkilib attention_cte (what NxDI runs today) at the FLUX-1024 per-rank shape.

    python compile_attention.py --out /tmp/nki_flux/neff [--lnc 2]

Prints the compiler's estimated engine utilisation and the NEFF paths; profile the NEFFs with
`neuron-explorer capture -n <neff> -s <ntff>` on a free core (see profile_attention.sh).
"""
import argparse
import os
import sys

import ml_dtypes
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["PYTHONPATH"] = HERE + os.pathsep + os.environ.get("PYTHONPATH", "")

from nki.compiler import KernelSpec, parallel_compile  # noqa: E402

bf16 = ml_dtypes.bfloat16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/nki_flux/neff")
    ap.add_argument("--lnc", type=int, default=2)
    ap.add_argument("--S", type=int, default=4608)
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--n-txt", type=int, default=512)
    ap.add_argument("--only", choices=["fused", "cte", "both"], default="both")
    a = ap.parse_args()
    S, H, D = a.S, a.H, 128
    rng = np.random.default_rng(0)

    specs = []
    if a.only in ("fused", "both"):
        specs.append(KernelSpec(
            kernel_module="flux_attention_nki", kernel_func_name="flux_qknorm_rope_attention",
            inputs={"q": rng.standard_normal((1, S, H * D)).astype(bf16),
                    "k": rng.standard_normal((1, S, H * D)).astype(bf16),
                    "v": rng.standard_normal((1, S, H * D)).astype(bf16),
                    "cos": rng.standard_normal((S, D // 2)).astype(np.float32),
                    "sin": rng.standard_normal((S, D // 2)).astype(np.float32),
                    "q_w_img": np.ones((1, D), np.float32), "k_w_img": np.ones((1, D), np.float32),
                    "q_w_txt": np.ones((1, D), np.float32), "k_w_txt": np.ones((1, D), np.float32)},
            outputs={"out": np.zeros((1, S, H * D), bf16)},
            hyperparams={"n_txt": a.n_txt, "eps": 1e-6},
            config_id=0, config_label=f"fused_qknorm_rope_attn_S{S}_H{H}"))
    if a.only in ("cte", "both"):
        specs.append(KernelSpec(
            kernel_module="nkilib.core.attention.attention_cte", kernel_func_name="attention_cte",
            inputs={"q": rng.standard_normal((H, S, D)).astype(bf16),
                    "k": rng.standard_normal((H, S, D)).astype(bf16),
                    "v": rng.standard_normal((H, S, D)).astype(bf16)},
            outputs={"out": np.zeros((H, S, D), bf16)},
            hyperparams={"scale": D ** -0.5, "causal_mask": False, "tp_q": True, "tp_k": True, "tp_out": False},
            config_id=1, config_label=f"nkilib_attention_cte_S{S}_H{H}"))

    summary = parallel_compile(specs, {"target": "trn2", "lnc": a.lnc}, a.out, max_workers=2)
    for r in summary.results:
        print(f"== {r.config_label}: success={r.success} time={r.compilation_time:.0f}s")
        print(f"   neff={r.neff_path}")
        print(f"   estimated_utilization={r.estimated_utilization}  est_total_ns={r.total_time_ns}")
        if r.error_message:
            print("   ERROR:", r.error_message[-3000:])


if __name__ == "__main__":
    main()
