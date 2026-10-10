"""Smoke test for torch_neuronx.trace: small bf16 MLP, latency, and accuracy vs CPU fp32.

Run with the tnx venv python and NEURON_RT_VISIBLE_CORES=0.
"""
import argparse
import copy
import json
import os
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/livevid/artifacts/smoke.pt")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--soak-seconds", type=float, default=20.0,
                    help="keep running inference afterwards so neuron-monitor can observe activity")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024)).eval()
    x = torch.randn(8, 1024)

    with torch.no_grad():
        ref = model(x)

    model_bf16 = copy.deepcopy(model).to(torch.bfloat16)
    x_bf16 = x.to(torch.bfloat16)

    t0 = time.perf_counter()
    traced = torch_neuronx.trace(model_bf16, x_bf16)
    compile_s = time.perf_counter() - t0
    print(f"compile_s={compile_s:.2f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.jit.save(traced, args.out)
    loaded = torch.jit.load(args.out)

    with torch.no_grad():
        for _ in range(args.warmup):
            out = loaded(x_bf16)
        lat_ms = []
        for _ in range(args.iters):
            t = time.perf_counter()
            out = loaded(x_bf16)
            lat_ms.append((time.perf_counter() - t) * 1e3)

    lat_sorted = sorted(lat_ms)
    p99 = lat_sorted[min(len(lat_sorted) - 1, int(round(0.99 * (len(lat_sorted) - 1))))]
    cos = F.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
    max_abs = (out.float() - ref).abs().max().item()

    result = {
        "torch": torch.__version__,
        "torch_neuronx": torch_neuronx.__version__,
        "visible_cores": os.environ.get("NEURON_RT_VISIBLE_CORES"),
        "compile_s": round(compile_s, 2),
        "latency_ms": {
            "mean": round(statistics.mean(lat_ms), 4),
            "p50": round(statistics.median(lat_ms), 4),
            "p99": round(p99, 4),
            "iters": args.iters,
        },
        "cosine_vs_cpu_fp32": round(cos, 6),
        "max_abs_err": round(max_abs, 5),
        "artifact": args.out,
        "artifact_bytes": os.path.getsize(args.out),
    }
    print("RESULT " + json.dumps(result), flush=True)

    if args.soak_seconds > 0:
        print(f"SOAK START {args.soak_seconds}s", flush=True)
        n = 0
        end = time.time() + args.soak_seconds
        with torch.no_grad():
            while time.time() < end:
                loaded(x_bf16)
                n += 1
        print(f"SOAK END iters={n}", flush=True)


if __name__ == "__main__":
    main()
