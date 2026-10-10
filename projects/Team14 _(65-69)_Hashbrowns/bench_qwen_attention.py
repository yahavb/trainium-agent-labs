#!/usr/bin/env python3
"""
bench_qwen_attention.py — the prefill attention kernel Qwen3-8B runs in the seat pod, against the same
kernel with V4's change (hardware DMA descriptors), at the shapes Qwen3-8B gives it.

What vLLM runs: vllm_neuron/model/qwen3/model.py prefill -> NF.flash_attention ->
nkilib.core.attention.attention_cte, launched at LNC 2. Per tensor-parallel rank (TP 2) it passes
16 query heads of dim 128 in bfloat16, K and V repeated from 4 KV heads up to 16, Q pre-multiplied
by 1/sqrt(128), causal mask on, and these layouts:

    q (16, 128, T)   tp_q=False        k (16, 128, T)   tp_k=False
    v (16, T, 128)                      out (16, 128, T) tp_out=True

    python bench_qwen_attention.py                         # T = 512, 2048, 8192
    python bench_qwen_attention.py --seqlens 512 --no-profile

Both variants are compiled and timed the same way as bench_suites.py (nki standalone compile, the
runtime's device-side benchmark), on a free core: the model server holds 0-1. Each run writes
profiles/qwen3_attention/run-<time>/results.json, plus per variant and length the NEFF, trace and
neuron-explorer summary.

Correctness: the patched output is compared with the original's (the change only alters how DMA
descriptors are built, so they should match exactly), and both with a float32 NumPy reference for
T <= 2048 -- above that the reference's T x T scores per head get slow on the CPU.
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HEADS, HEAD_DIM = 16, 128          # Qwen3-8B: 32 query heads / TP 2; head_dim 128
REFERENCE_MAX_T = 2048


def make_inputs(T, seed=0):
    import ml_dtypes
    r = np.random.default_rng(seed + T)
    bf16 = ml_dtypes.bfloat16
    q = r.standard_normal((HEADS, T, HEAD_DIM), dtype=np.float32) * HEAD_DIM ** -0.5
    k = r.standard_normal((HEADS, T, HEAD_DIM), dtype=np.float32)
    v = r.standard_normal((HEADS, T, HEAD_DIM), dtype=np.float32)
    q, k, v = (a.astype(bf16) for a in (q, k, v))
    # The kernel's layouts, as model.py builds them: q and k transposed to (heads, d, T).
    kernel_args = dict(q=np.ascontiguousarray(q.transpose(0, 2, 1)),
                       k=np.ascontiguousarray(k.transpose(0, 2, 1)), v=v)
    return kernel_args, (q, k, v)


def reference(q, k, v):
    """Causal softmax(q @ k.T) @ v per head in float32, from the bf16 inputs. Returns (heads, d, T)."""
    T = q.shape[1]
    above = np.triu(np.ones((T, T), dtype=bool), 1)
    out = np.empty((HEADS, HEAD_DIM, T), dtype=np.float32)
    for h in range(HEADS):
        s = q[h].astype(np.float32) @ k[h].astype(np.float32).T
        s[above] = -np.inf
        e = np.exp(s - s.max(axis=-1, keepdims=True))
        out[h] = ((e / e.sum(axis=-1, keepdims=True)) @ v[h].astype(np.float32)).T
    return out


def run_on_device(kernel, kwargs, work, warmup, iterations):
    """bench_suites.run_on_device, for keyword arguments and an LNC 2 launch like vLLM's [2]."""
    from nki.framework.compiled import StandaloneKernel

    seen = {}

    def benchmark(compiled, inputs, outputs):
        r = compiled.benchmark(warmup=warmup, iterations=iterations, **inputs)
        for name, arr in r.outputs.items():
            outputs[name][...] = arr
        seen.update(result=r, neff=compiled.neff_path)

    shutil.rmtree(work, ignore_errors=True)
    standalone = kernel._to_subclass(StandaloneKernel, _executor=benchmark, target="trn2", lnc=2,
                                     artifacts_dir=os.path.join(work, "build"))
    got = standalone(**kwargs)
    return np.asarray(got, dtype=np.float32), seen["result"], seen["neff"]


def errors(got, want):
    diff = got - want
    return float(np.abs(diff).max()), float(np.sqrt((diff ** 2).mean() / (want ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("--core", default="2", help="logical core to time on; the model server holds 0-1")
    ap.add_argument("--profile-core", default="3")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--no-profile", action="store_true")
    a = ap.parse_args()

    # Before the runtime starts: it reads this once, at first model load.
    os.environ["NEURON_RT_VISIBLE_CORES"] = a.core
    sys.path.insert(0, HERE)
    import nki
    from nkilib.core.attention.attention_cte import attention_cte as original
    from attention_cte_hwdge import attention_cte as patched
    from bench_suites import change, profile

    variants = [("nkilib original", nki.jit()(original)), ("hwdge patched", nki.jit()(patched))]
    flags = dict(scale=1.0, causal_mask=True, tp_q=False, tp_k=False, tp_out=True)
    root = os.path.join(HERE, "profiles", "qwen3_attention", time.strftime("run-%Y%m%d-%H%M%S"))
    print(f"Qwen3-8B prefill attention, per TP rank: {HEADS} heads x d {HEAD_DIM}, bf16, causal, "
          f"LNC 2 on core {a.core}; {a.iterations} timed iterations after {a.warmup} warmup.\n")

    results = []
    for T in a.seqlens:
        kwargs, (q, k, v) = make_inputs(T)
        want = reference(q, k, v) if T <= REFERENCE_MAX_T else None
        print(f"=== T = {T} tokens")
        base = None
        for label, kernel in variants:
            work = os.path.join(root, f"{label.split()[0]}_T{T}")
            row = dict(variant=label, seqlen=T)
            t0 = time.time()
            try:
                got, r, neff = run_on_device(kernel, dict(kwargs, **flags), work, a.warmup,
                                             a.iterations)
            except Exception as e:
                msg = str(e).strip().splitlines()
                print(f"  {label:<16} FAILED: {type(e).__name__}: {msg[-1] if msg else ''}")
                row["error"] = f"{type(e).__name__}: {e}"
                results.append(row)
                continue
            row.update(mean_us=r.latency * 1e6, min_us=r.latency_min * 1e6, std_us=r.latency_std * 1e6)
            notes = []
            if want is not None:
                row["max_err_vs_ref"], row["rel_rms_vs_ref"] = errors(got, want)
                notes.append(f"vs float32 ref: max {row['max_err_vs_ref']:.3g}, "
                             f"rel RMS {row['rel_rms_vs_ref']:.2g}")
            if base is None:
                base, base_out = row, got
                delta = "baseline"
            else:
                row["identical_to_original"] = bool(np.array_equal(got, base_out))
                notes.append("bit-identical to original" if row["identical_to_original"] else
                             f"DIFFERS from original: max {np.abs(got - base_out).max():.3g}")
                delta = change(row["mean_us"], base["mean_us"])
            print(f"  {label:<16} {row['mean_us']:9.1f} us  +/-{row['std_us']:.1f}  "
                  f"(min {row['min_us']:.1f})  {delta}  {time.time() - t0:.0f}s")
            if notes:
                print(f"  {'':<16} {'; '.join(notes)}")

            if not a.no_profile:
                s = profile(neff, work, a.profile_core)
                if isinstance(s, str):
                    print(f"  {'':<16} profile: {s}")
                else:
                    keys = ("tensor_engine_active_time", "vector_engine_active_time",
                            "scalar_engine_active_time", "gpsimd_engine_active_time",
                            "dma_active_time", "software_dynamic_dma_packet_count",
                            "hardware_dynamic_dma_packet_count", "total_exec_time")
                    row["profile"] = {k: s.get(k) for k in keys}
                    us = lambda k: (s.get(k) or 0) * 1e6
                    print(f"  {'':<16} busy us: Tensor {us(keys[0]):.1f}  Vector {us(keys[1]):.1f}  "
                          f"Scalar {us(keys[2]):.1f}  GpSimd {us(keys[3]):.1f}  DMA {us(keys[4]):.1f}"
                          f"   DMA packets sw {s.get(keys[5])} / hw {s.get(keys[6])}")
            results.append(row)
        print()

    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "results.json"), "w") as f:
        json.dump(dict(heads=HEADS, head_dim=HEAD_DIM, core=a.core, warmup=a.warmup,
                       iterations=a.iterations, results=results), f, indent=1)
    print(f"results: {os.path.relpath(root, HERE)}/")


if __name__ == "__main__":
    main()
