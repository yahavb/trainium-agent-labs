#!/usr/bin/env python3
"""
device_check.py -- run kernels on the real Trainium2 chip, not only the simulator.

For each kernel and each input set it records the simulator's result and the chip's, each checked
against the NumPy reference with the harness's own describe_mismatch. The kernels:

  reference_level3.py, reference_level4.py   controls: the organisers' kernels
  findings/agent_solved_level3_*.py           level 3 kernels the agent wrote under our checker
  findings/tall_tile_level4.py                finding 1: an SBUF tile with 256/512 rows
  findings/psum_output_level3.py              finding 2: the output returned from PSUM

Inputs: the level's own test shapes, plus hostile values on level 3 (large magnitudes, all zeros,
all negative, one constant row), as the challenge's harness section asks for.

Calling a compiled kernel with a launch grid, kernel[grid](*args), runs it on the device. That is
the method in another team's checks/device_check.py on this repo (Team 20), credited here.

Needs a NeuronCore no model server is using:

    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_RT_NUM_CORES=1 python findings/device_check.py
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
import nkibench as nb  # noqa: E402

LNC = int(os.environ.get("DEVICE_LNC", "2"))

CASES = [("reference_level3.py", 3, "control: organisers' kernel"),
         ("reference_level4.py", 4, "control: organisers' kernel")]
CASES += [(f"findings/{f}", 3, "written by the agent under the checker")
          for f in sorted(os.listdir(HERE)) if f.startswith("agent_solved_level3_")]
CASES += [("findings/tall_tile_level4.py", 4, "finding 1: 256/512-row SBUF tile"),
          ("findings/psum_output_level3.py", 3, "finding 2: output returned from PSUM")]


def hostile_level3():
    """Level 3's one shape with hostile values."""
    r = np.random.default_rng(7)
    lhsT = r.standard_normal((128, 64)).astype(np.float32)
    rhs = r.standard_normal((128, 512)).astype(np.float32)
    const = rhs.copy()
    const[5, :] = 3.0
    return [("large values (x1e3)", (lhsT * 1e3, rhs * 1e3)),
            ("all zeros", (np.zeros_like(lhsT), np.zeros_like(rhs))),
            ("all negative", (-np.abs(lhsT), np.abs(rhs))),
            ("one constant row", (lhsT, const))]


def load(path, entry):
    spec = importlib.util.spec_from_file_location(f"k_{abs(hash(path))}", os.path.join(PROJ, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


def check(got, want):
    try:
        m = nb.describe_mismatch(np.asarray(got), want)
        return "ok" if m is None else "WRONG: " + " ".join(m.split())[:110]
    except Exception as e:
        return f"WRONG: {type(e).__name__}: {e}"[:120]


def main():
    import nki
    print(f"nki {getattr(nki, '__version__', '?')}; launch grid {LNC}; "
          f"NEURON_PLATFORM_TARGET_OVERRIDE={os.environ.get('NEURON_PLATFORM_TARGET_OVERRIDE')}\n")
    rows = []
    for path, level, what in CASES:
        spec = nb.LEVELS[level]
        try:
            fn = load(path, spec["entry"])
        except Exception as e:
            rows.append((path, "-", "load failed", f"{type(e).__name__}: {e}"[:100]))
            continue
        inputs = [(nb.label(c, level), nb.make_inputs(c, level, seed=0)[0]) for c in spec["shapes"]]
        if level == 3:
            inputs += hostile_level3()
        print(f"== {path}  ({what})")
        for label, args in inputs:
            want = spec["ref"](*args)
            try:
                sim, _ = nb.simulate_and_count(fn, list(args))
                s = check(sim, want)
            except Exception as e:
                s = f"RAISED {type(e).__name__}: {str(e)[:90]}"
            try:
                d = check(fn[LNC](*args), want)
            except Exception as e:
                # The compiler's own words are the evidence, so keep enough of them to read.
                msg = " ".join(str(e).split())
                key = next((m for m in ("Allocated memory", "must be a shared_hbm", "NCC_") if m in msg), None)
                start = max(0, msg.find(key) - 40) if key else 0
                d = f"RAISED {type(e).__name__}: {msg[start:start + 400]}"
            print(f"   {label:28s} simulator: {s[:60]:60s} chip: {d}")
            rows.append((path, label, s, d))
        print()
    print("SUMMARY: kernel, cases correct in the simulator / on the chip")
    for path, _, _ in CASES:
        mine = [r for r in rows if r[0] == path]
        print(f"  {path:42s} simulator {sum(r[2] == 'ok' for r in mine)}/{len(mine)}   "
              f"chip {sum(r[3] == 'ok' for r in mine)}/{len(mine)}")


if __name__ == "__main__":
    main()
