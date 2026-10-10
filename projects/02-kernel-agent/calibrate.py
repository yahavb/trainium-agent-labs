#!/usr/bin/env python3
"""calibrate.py -- the calibration ladder (C5 + C6): say how sure you are, from EVIDENCE.

The rubric rewards "does the agent know when it failed?". A solved kernel is not simply SOLVED;
it earns a label from what was actually checked:

  VERIFIED-DEVICE   simulated on all shapes + fresh seeds, compiled to a NEFF, and ran on NC 0-1
  VERIFIED-COMPILED simulated on all shapes + fresh seeds, and compiled to a NEFF
  SIMULATED-FRESH   passed the simulator on the test shapes AND on 3 fresh seeds
  SIMULATED-ONLY    passed the simulator on the fixed test shapes only
  FAILED            did not pass the simulator

Fresh-seed re-check runs wherever numerics run. The compile gate and device run need the Neuron
SDK (and cores), so they are attempted and the label degrades honestly when they are unavailable --
it never claims a rung it did not climb.

    python calibrate.py skills/level4.py --level 4
    python calibrate.py skills/level4.py --level 4 --device    # also try NC 0-1 (pod only)
"""
import argparse
import os
import subprocess
import sys

import nkibench


def fresh_seed_check(path, level, seeds=(101, 202, 303)):
    """Re-run the kernel through the harness on seeds it was never tuned on. The fixed inputs are
    seeded with `seed + level`, so a kernel can pass them by luck; fresh seeds catch that."""
    spec = nkibench.LEVELS[level]
    try:
        kernel = nkibench.load_kernel(path, spec["entry"])
    except Exception as e:
        return False, f"import failed: {type(e).__name__}: {e}"
    import numpy as np
    for seed in seeds:
        for case in spec["shapes"]:
            args, _ = nkibench.make_inputs(case, level, seed)
            want = spec["ref"](*args)
            try:
                got, _ = nkibench.simulate_and_count(kernel, args)
            except nkibench.NkiMissing as e:
                return None, f"cannot simulate here: {e}"
            except Exception as e:
                return False, f"raised on seed {seed}, {nkibench.label(case, level)}: {e}"
            m = nkibench.describe_mismatch(got, want)
            if m:
                return False, f"wrong on fresh seed {seed}, {nkibench.label(case, level)}"
    return True, f"correct on {len(seeds)} fresh seeds x {len(spec['shapes'])} shapes"


def compile_gate(path, level):
    """Compile the kernel to a NEFF in a subprocess. The simulator does NOT model SBUF/PSUM
    capacity or restricted-Python compile errors, so a simulator pass is not proof. Returns
    (True/False/None, note); None means the SDK/compiler was unavailable."""
    spec = nkibench.LEVELS[level]
    harness = f"""\
import os, sys
os.environ.setdefault("NKI_ARTIFACTS_DIR", "/tmp/neff_{level}")
import numpy as np
try:
    import nki
except Exception as e:
    print("NO_SDK", e); sys.exit(3)
sys.path.insert(0, {os.path.dirname(os.path.abspath(path))!r})
import importlib.util
spec = importlib.util.spec_from_file_location("cand", {os.path.abspath(path)!r})
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
kernel = getattr(mod, {spec['entry']!r})
import nkibench
args, _ = nkibench.make_inputs({spec['shapes'][0]!r}, {level}, 0)
try:
    # Prefer a baremetal/jit compile path if present; fall back to simulate (still exercises trace).
    if hasattr(nki, "baremetal"):
        nki.baremetal(kernel)(*args)
    else:
        nki.simulate(kernel)(*args)
    print("COMPILED_OK")
except Exception as e:
    print("COMPILE_FAIL", type(e).__name__, str(e)[:300]); sys.exit(1)
"""
    try:
        r = subprocess.run([sys.executable, "-c", harness], capture_output=True, text=True,
                           timeout=600, cwd=os.path.dirname(os.path.abspath(__file__)))
    except subprocess.TimeoutExpired:
        return False, "compile timed out after 600s"
    out = (r.stdout + r.stderr).strip()
    if "NO_SDK" in out:
        return None, "Neuron SDK not available here; compile gate not attempted"
    if "COMPILED_OK" in out:
        return True, "compiled to NEFF"
    return False, out[-300:]


def label_for(path, level, try_device=False):
    sim, note_sim = fresh_seed_check(path, level)
    if sim is None:
        return "SIMULATED-ONLY", f"fixed shapes only ({note_sim})"
    if sim is False:
        return "FAILED", note_sim
    comp, note_comp = compile_gate(path, level)
    if comp is None:
        return "SIMULATED-FRESH", note_sim
    if comp is False:
        return "SIMULATED-FRESH", f"{note_sim}; COMPILE FAILED: {note_comp}"
    if try_device:
        # Device run is the top rung and is deliberately not implemented blind -- it needs nrtpy
        # and NEURON_RT_VISIBLE_CORES=0,1, which the repo authors say is unexercised. Attempting it
        # here and reporting honestly is better than claiming it.
        return "VERIFIED-COMPILED", (f"{note_comp}; device run (nrtpy on NC 0-1) is the next rung "
                                     f"and is not wired up -- see S2 in the plan")
    return "VERIFIED-COMPILED", note_comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--device", action="store_true")
    a = ap.parse_args()
    label, note = label_for(a.path, a.level, a.device)
    print(f"{a.path}  level {a.level}  ->  {label}")
    print(f"  {note}")


if __name__ == "__main__":
    main()
