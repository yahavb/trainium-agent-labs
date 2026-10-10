#!/usr/bin/env python3
"""
compare_torch.py -- my_attention.py against torch's scaled_dot_product_attention.

Two questions, kept apart because they have different answers:

  ACCURACY  Both are compared with a float64 NumPy reference, on the level-8 shapes and on hostile
            inputs (large values that overflow a naive exp, identical rows). Runs anywhere NKI
            simulates: the NKI kernel goes through nki.simulate, torch runs on the CPU.

  SPEED     Only meaningful on the chip. The CPU simulator's time says nothing about Trainium, and
            torch-on-CPU says nothing about torch-on-Trainium, so those are printed but labelled.
            --device runs both on a free NeuronCore through torch-xla. That path has never been
            exercised in this repo, so expect to debug it.

    python compare_torch.py                  # accuracy (+ CPU timings, labelled)
    python compare_torch.py --device         # also time both on a free NeuronCore

The vLLM server holds NC 2-3. --device pins to NEURON_RT_VISIBLE_CORES (default "0"), which has
to be set before torch_xla is imported -- this script does that for you.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# (label, seq, dim, input scale). Scale 30 pushes the scores to ~900, where exp() overflows
# float32 unless the row max is subtracted first.
CASES = [
    ("level-8 shape", 128, 64, 1.0),
    ("level-8 shape", 64, 128, 1.0),
    ("level-8 shape", 96, 32, 1.0),
    ("large values", 128, 64, 30.0),
    ("identical rows", 128, 64, 1.0),
]


def make_inputs(seq, dim, scale, label, seed=0):
    r = np.random.default_rng(seed)
    q, k, v = (r.standard_normal((seq, dim)).astype(np.float32) * scale for _ in range(3))
    if label == "identical rows":
        k[:] = k[0]          # every score in a row is equal, so softmax must come out uniform
    return q, k, v


def reference64(q, k, v):
    q, k, v = (a.astype(np.float64) for a in (q, k, v))
    s = q @ k.T / np.sqrt(q.shape[1])
    s -= s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(axis=-1, keepdims=True)) @ v


def rel_error(got, want):
    """Worst absolute error as a fraction of the output's RMS -- the same measure nkibench grades with."""
    got = np.asarray(got, np.float64)
    if not np.all(np.isfinite(got)):
        return float("inf")
    return float(np.abs(got - want).max() / (np.sqrt((want ** 2).mean()) or 1.0))


def run_torch_cpu(q, k, v):
    import torch
    import torch.nn.functional as F
    t = [torch.from_numpy(a)[None, None] for a in (q, k, v)]     # [batch, heads, seq, dim]
    return F.scaled_dot_product_attention(*t)[0, 0].numpy()


def run_nki_sim(kernel, q, k, v):
    import nkibench
    out, counted = nkibench.simulate_and_count(kernel, (q, k, v))
    return np.asarray(out), counted


def median_time(fn, reps):
    fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.percentile(times, 90))


def accuracy(kernel):
    print("ACCURACY -- worst error / RMS of the output, against a float64 reference")
    print("           (nkibench passes a kernel at <= 2e-2)\n")
    print(f"  {'case':<16} {'seq':>4} {'dim':>4}   {'NKI (simulated)':>16}   {'torch SDPA (CPU)':>17}")
    for label, seq, dim, scale in CASES:
        q, k, v = make_inputs(seq, dim, scale, label)
        want = reference64(q, k, v)
        try:
            got_nki, _ = run_nki_sim(kernel, q, k, v)
            e_nki = f"{rel_error(got_nki, want):.2e}"
        except Exception as e:
            e_nki = f"FAILED {type(e).__name__}"
        e_torch = f"{rel_error(run_torch_cpu(q, k, v), want):.2e}"
        print(f"  {label:<16} {seq:>4} {dim:>4}   {e_nki:>16}   {e_torch:>17}")


def cpu_timings(kernel, reps):
    print("\nCPU TIMINGS -- NOT A MEASURE OF TRAINIUM. The simulator interprets every instruction in")
    print("Python, and torch here runs on the host CPU. Use --device for a comparison that means something.\n")
    q, k, v = make_inputs(128, 64, 1.0, "level-8 shape")
    t_sim, _ = median_time(lambda: run_nki_sim(kernel, q, k, v), max(3, reps // 10))
    t_cpu, _ = median_time(lambda: run_torch_cpu(q, k, v), reps)
    print(f"  seq=128 dim=64   NKI simulator {t_sim * 1e3:9.2f} ms    torch CPU {t_cpu * 1e3:9.3f} ms")


def device_timings(kernel, reps):
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
    # Without this torch-xla silently falls back to the CPU and every "device" number is a CPU
    # number. Measured on a seat pod: "Defaulting to PJRT_DEVICE=CPU", 0.1 s first call.
    os.environ.setdefault("PJRT_DEVICE", "NEURON")
    print(f"\nDEVICE TIMINGS -- NEURON_RT_VISIBLE_CORES={os.environ['NEURON_RT_VISIBLE_CORES']}")
    try:
        import torch
        import torch.nn.functional as F
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        dev = xm.xla_device()
        backend = xr.device_type()
    except Exception as e:
        print(f"  cannot reach a Neuron device through torch-xla ({type(e).__name__}: {str(e)[:200]}).")
        print("  No device numbers reported.")
        return
    if str(backend).upper() != "NEURON":
        print(f"  torch-xla is running on {backend}, not a NeuronCore. Refusing to report its timings "
              f"as device numbers.")
        return

    def sync():
        xm.mark_step()
        xm.wait_device_ops()

    for label, seq, dim, scale in CASES[:3]:
        q, k, v = make_inputs(seq, dim, scale, label)
        want = reference64(q, k, v)
        qx, kx, vx = (torch.from_numpy(a).to(dev) for a in (q, k, v))
        print(f"\n  seq={seq} dim={dim}")

        # The first call compiles, so time it separately and never count it as run time.
        for name, call in (("torch SDPA", lambda: F.scaled_dot_product_attention(
                                qx[None, None], kx[None, None], vx[None, None])[0, 0]),
                           ("NKI kernel", lambda: kernel(qx, kx, vx))):
            try:
                t0 = time.perf_counter()
                out = call()
                sync()
                compile_s = time.perf_counter() - t0
                err = rel_error(out.cpu().numpy(), want)

                def step():
                    call()
                    sync()
                med, p90 = median_time(step, reps)
                print(f"    {name:<11} {med * 1e6:9.1f} us median  {p90 * 1e6:9.1f} us p90   "
                      f"first call {compile_s:6.1f} s   error {err:.2e} (on device)")
            except Exception as e:
                print(f"    {name:<11} FAILED on device: {type(e).__name__}: {str(e)[:200]}")
    print("\n  Timings include the host-side launch and sync, which can dominate at these tiny shapes.")
    print("  Compare the two against each other, not against a published per-kernel latency.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", action="store_true", help="also time both on a free NeuronCore")
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()

    try:
        import nkibench
        kernel = nkibench.load_kernel(os.path.join(HERE, "my_attention.py"), "nki_attention_")
    except ImportError as e:
        sys.exit(f"cannot load the NKI kernel here ({e}). Run this inside the seat pod.")

    accuracy(kernel)
    cpu_timings(kernel, a.reps)
    if a.device:
        device_timings(kernel, a.reps)


if __name__ == "__main__":
    main()
