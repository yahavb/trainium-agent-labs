#!/usr/bin/env python3
"""
profile_device.py -- a device profile of each attention kernel: does it run correctly on a
NeuronCore, how long does one execution take, and which engine is busy.

    python profile_device.py                                   # V0 and V1 at seq=128 dim=64
    python profile_device.py --seq 96 --dim 32
    python profile_device.py my_attention_v0.py my_attention.py my_attention_v2_bf16.py

For each kernel file, in its own subprocess (so each gets its own NEFF directory):
  1. RUN      the kernel once on NeuronCore 0 through torch-xla, with profiling inspection on,
              and compare its output with a float64 reference computed on the CPU. The reference
              stays on the CPU on purpose: every on-device torch op compiles its own NEFF, which
              would make the kernel's NEFF hard to find.
  2. FIND     the kernel's NEFF with identify_neffs.py (from the neuron-nki-profiling skill).
  3. CAPTURE  a trace with neuron-explorer (falls back to neuron-profile if that is all there is),
              profiling the 2nd execution so warm-up is excluded.
  4. VIEW     the summary as JSON, saved to profiles/<run>/<kernel>/metrics.json.

Then one table: latency and engine utilisation per kernel. A kernel that is WRONG on the device
is reported and not profiled.

Needs a Python that has torch_xla with the Neuron PJRT plugin. The pod's default python has nki
but not torch_neuronx (verify_sdk.py: "torch_neuronx not installed", torch is a CUDA build), so
this checks first and says so instead of silently running on the CPU. Look for a Neuron venv:
    ls -d /opt/*venv* /opt/aws_neuron* 2>/dev/null
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = "nki_attention_"
TOL = 2e-2                  # nkibench's grading tolerance: worst error as a fraction of output RMS
DEFAULT_KERNELS = ["my_attention_v0.py", "my_attention.py"]


# ---------------------------------------------------------------- child: run one kernel

def child(kernel_path, seq, dim, out_dir):
    # All of these must be set before torch_xla touches the runtime.
    os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
    os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
    os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = out_dir
    # Compile for the LNC the runtime is configured with. Seat pods set NEURON_LOGICAL_NC_CONFIG=2,
    # and a NEFF built for --lnc 1 does not match that runtime.
    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1")
    os.environ.setdefault("NEURON_CC_FLAGS", f"--target trn2 --lnc {lnc}")
    # At LNC=2 the chip has only 2 logical cores, and vLLM at TP=2 holds both: stop it first
    # (/workspace/serve.sh --stop), or the runtime fails with "Logical Neuron Core(s) not available".
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
    os.environ.setdefault("PJRT_DEVICE", "NEURON")             # else torch-xla silently uses CPU

    import importlib.util
    import numpy as np
    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    if str(xr.device_type()).upper() != "NEURON":
        print(json.dumps({"error": f"torch-xla is on {xr.device_type()}, not a NeuronCore"}))
        return 1

    spec = importlib.util.spec_from_file_location("kernel_mod", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, ENTRY)

    r = np.random.default_rng(0)
    q, k, v = (r.standard_normal((seq, dim)).astype(np.float32) for _ in range(3))
    qd, kd, vd = (a.astype(np.float64) for a in (q, k, v))
    s = qd @ kd.T / np.sqrt(dim)
    e = np.exp(s - s.max(1, keepdims=True))
    want = (e / e.sum(1, keepdims=True)) @ vd

    dev = xm.xla_device()
    t0 = time.perf_counter()
    out = kernel(*(torch.from_numpy(a).to(dev) for a in (q, k, v)))
    out = out[0] if isinstance(out, (tuple, list)) else out
    got = out.cpu().numpy().astype(np.float64)          # forces compile + execute
    first_call = time.perf_counter() - t0

    err = float(np.abs(got - want).max() / np.sqrt((want ** 2).mean()))
    print(json.dumps({"error_vs_rms": err, "first_call_s": first_call}))
    return 0


def run_child(kernel_path, seq, dim, out_dir):
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", kernel_path,
                        "--seq", str(seq), "--dim", str(dim), "--out", out_dir],
                       capture_output=True, text=True, timeout=1800)
    for line in reversed(p.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line), p
    return None, p


# ---------------------------------------------------------------- parent: find, capture, view

def find_neff(out_dir):
    """The NKI NEFF for ENTRY. identify_neffs.py records the kernel under its qualified name --
    'kernel_mod.nki_attention_', from the module name child() loads it as -- so its exact-name
    lookup misses; match on the last component of each [NKI:...] line of the full listing instead."""
    listing = subprocess.run([sys.executable, os.path.join(HERE, "identify_neffs.py"), out_dir],
                             capture_output=True, text=True)
    text = (listing.stdout + listing.stderr).strip()
    for line in text.splitlines():
        m = re.match(r"\s*\[NKI:([^\]]+)\]\s+(\S+\.neff)", line)
        if m and m.group(1).split(".")[-1] == ENTRY and os.path.exists(m.group(2)):
            return m.group(2), ""
    return None, text


def profiler():
    for exe in ("neuron-explorer", "neuron-profile"):
        if shutil.which(exe):
            return exe
    return None


def new_ntffs(root, since):
    found = []
    for d, _, fs in os.walk(root):
        found += [os.path.join(d, f) for f in fs
                  if f.endswith(".ntff") and os.path.getmtime(os.path.join(d, f)) >= since]
    return sorted(found, key=os.path.getmtime)


def capture_and_view(exe, neff, prof_dir):
    """Returns (metrics, note). note says which trace was used, or why there are no metrics."""
    cap_dir = os.path.join(prof_dir, "capture")
    os.makedirs(cap_dir, exist_ok=True)
    ntff = os.path.join(cap_dir, "profile.ntff")
    # Two executions, profile the 2nd, so warm-up is excluded. neuron-explorer 2.32 then names the
    # trace profile_exec_2.ntff beside the -s path rather than writing -s itself; new_ntffs finds it.
    cap = [exe, "capture", "-n", neff, "-s", ntff, "--num-exec=2", "--profile-nth-exec=2"]
    if exe == "neuron-explorer":
        cap.append("--enable-dge-notifs")
    print(f"  $ {' '.join(cap)}")
    start = time.time() - 1
    # Run from cap_dir, so a trace written relative to the working directory lands there too.
    p = subprocess.run(cap, capture_output=True, text=True, timeout=900, cwd=cap_dir)
    cap_out = (p.stdout + p.stderr).strip()
    with open(os.path.join(cap_dir, "capture.log"), "w") as f:
        f.write(cap_out + "\n")

    # neuron-explorer has exited 0 without writing the -s path, so take whatever trace it did
    # write. Failing that, the runtime's own trace from the inspected run (beside the NEFF):
    # that one is of the FIRST execution, so it includes warm-up.
    made = [ntff] if os.path.exists(ntff) else new_ntffs(cap_dir, start)
    if made:
        trace, note = made[-1], f"trace {made[-1]} (capture, 2nd execution)"
    else:
        ident = re.search(r"neff_(\d+)_vnc_(\d+)\.neff$", neff)
        runtime = (os.path.join(os.path.dirname(neff), f"{ident.group(1)}_vnc_{ident.group(2)}.ntff")
                   if ident else "")
        if not os.path.exists(runtime):
            return None, (f"capture exited {p.returncode} and wrote no .ntff, and there is no runtime "
                          f"trace beside the NEFF. Capture output: {cap_out[-800:] or '(none)'}")
        trace = runtime
        note = (f"capture wrote no .ntff (exit {p.returncode}; output in capture/capture.log), so "
                f"using the RUNTIME trace {runtime} -- 1st execution, includes warm-up")

    p = subprocess.run([exe, "view", "--output-format", "summary-json", "-n", neff, "-s", trace],
                       capture_output=True, text=True, timeout=900, cwd=cap_dir)
    text = p.stdout.strip()
    try:
        metrics = json.loads(text[text.index("{"):])
    except ValueError:
        return None, (f"{note}\n  view returned no JSON ({p.returncode}): "
                      f"{(p.stdout + p.stderr).strip()[-800:]}")
    with open(os.path.join(prof_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics, note


def pick(metrics, *names):
    """First key present among names, searching one level of nesting -- the summary layout has
    differed between profiler releases."""
    pools = [metrics] + [v for v in metrics.values() if isinstance(v, dict)]
    for name in names:
        for pool in pools:
            if name in pool:
                return pool[name]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernels", nargs="*", default=DEFAULT_KERNELS)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--child")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.child:
        sys.exit(child(a.child, a.seq, a.dim, a.out))

    try:
        import torch_xla  # noqa: F401
    except ImportError:
        sys.exit(f"{sys.executable} has no torch_xla, so nothing can run on the NeuronCore from "
                 f"here.\nActivate a Neuron venv first:  ls -d /opt/*venv* /opt/aws_neuron*")
    exe = profiler()
    print(f"python {sys.executable}\nprofiler {exe or 'NONE on PATH -- runs and checks only'}\n"
          f"shape seq={a.seq} dim={a.dim}   NEURON_RT_VISIBLE_CORES="
          f"{os.environ.get('NEURON_RT_VISIBLE_CORES', '0')}\n")

    run_dir = os.path.join(HERE, "profiles", time.strftime("run_%Y%m%d_%H%M%S"))
    rows = []
    for k in a.kernels:
        path = k if os.path.exists(k) else os.path.join(HERE, k)
        name = os.path.splitext(os.path.basename(path))[0]
        prof_dir = os.path.join(run_dir, name)
        out_dir = os.path.join(prof_dir, "output")
        os.makedirs(out_dir, exist_ok=True)
        print(f"== {name}")

        result, proc = run_child(path, a.seq, a.dim, out_dir)
        if not result or "error" in result:
            text = proc.stdout + proc.stderr
            why = result["error"] if result else text.strip()[-1500:]
            if "not available" in text and "cores busy" in text:
                why = ("the NeuronCores are busy -- vLLM holds them. Stop it with "
                       "/workspace/serve.sh --stop, then rerun.")
            elif "cached failed neff" in text:
                why = ("the neuron compile cache holds an earlier FAILED compile and is replaying it. "
                       "Delete the MODULE_... directory named in the error under "
                       "/var/tmp/neuron-compile-cache, then rerun.\n" + text.strip()[-800:])
            print(f"  RUN FAILED: {why}\n")
            rows.append((name, "run failed", None))
            continue
        err = result["error_vs_rms"]
        ok = err <= TOL
        print(f"  device output error {err:.2e} of RMS -- {'CORRECT' if ok else 'WRONG'} "
              f"(first call {result['first_call_s']:.1f} s, includes compile)")
        if not ok:
            print("  not profiled: a kernel that is wrong on the device has no meaningful latency\n")
            rows.append((name, "WRONG on device", None))
            continue
        if not exe:
            rows.append((name, "correct, not profiled", None))
            continue

        neff, listing = find_neff(out_dir)
        if not neff:
            print(f"  no NEFF for {ENTRY} found. What identify_neffs.py saw:\n    "
                  + listing.replace("\n", "\n    ") + "\n")
            rows.append((name, "no NEFF found", None))
            continue
        print(f"  NEFF {neff}")
        metrics, why = capture_and_view(exe, neff, prof_dir)
        if metrics is None:
            print(f"  {why}\n")
            rows.append((name, "profile failed", None))
            continue
        print(f"  {why}\n  metrics -> {os.path.join(prof_dir, 'metrics.json')}\n")
        rows.append((name, "profiled" if "(capture" in why else "profiled (1st exec)", metrics))

    print(f"SUMMARY  seq={a.seq} dim={a.dim}")
    # neuron-explorer 2.32 summary-json: times are in SECONDS and every *_percent field is a
    # FRACTION (0.28 = 28%), so each column carries the factor that turns it into what it says.
    cols = [("total us", "total_time", 1e6),
            ("TensorE %", "tensor_engine_active_time_percent", 100),
            ("VectorE %", "vector_engine_active_time_percent", 100),
            ("ScalarE %", "scalar_engine_active_time_percent", 100),
            ("GpSimd %", "gpsimd_engine_active_time_percent", 100),
            ("DMA %", "dma_active_time_percent", 100),
            ("HBM read B", "hbm_read_bytes", 1),
            ("HBM write B", "hbm_write_bytes", 1)]
    print(f"  {'kernel':<24} {'status':<22}" + "".join(f"{c:>12}" for c, _, _ in cols))
    for name, status, m in rows:
        vals = [pick(m, key) if m else None for _, key, _ in cols]
        print(f"  {name:<24} {status:<22}" + "".join(
            f"{('-' if v is None else f'{v * k:.4g}'):>12}" for v, (_, _, k) in zip(vals, cols)))
    if any(m for _, _, m in rows) and all(pick(m, "total_time") is None for _, _, m in rows if m):
        print("\n  The summary has no 'total_time' key in this profiler release. The full JSON is in "
              "each metrics.json; send it back and the columns can be fixed.")
    print(f"\nEverything is under {run_dir}. One profile per kernel is one sample: profile again "
          "before calling a small latency difference real.")


if __name__ == "__main__":
    main()
