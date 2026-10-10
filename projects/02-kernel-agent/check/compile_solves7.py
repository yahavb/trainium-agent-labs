"""Lower every distinct solving kernel in the given attempt logs for trn2 and run it in birsim, the
compiler's instruction-level simulator. The Python simulator accepted all of them; this asks whether
the chip's compiler accepts them and whether the compiled instructions give the same numbers.

Generalizes the desktop's task-06 compile_solves.py (same nkitool.analyze, same 2e-2 of RMS tolerance)
to any level, including task 07's levels 9-11. Run with the neuronx-cc venv ON PATH, from their
projects/02-kernel-agent directory, with this folder and the task folder on PYTHONPATH:

    PATH=~/venvs/cc/bin:$PATH PYTHONPATH=$PWD:$T:$T/check ~/venvs/cc/bin/python compile_solves7.py \\
        --levels 1 8 9 10 11 --out $R/solves  $R/attempts_*.jsonl

BIRSIM=0 lowers only (for a machine where neuronx-cc can't run). Prints one line per kernel and shape, then a summary per level: solves that lower, that fail to
lower (with the compiler's reason), and that lower but give different numbers in birsim.
"""
import collections
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
import nkibench  # noqa: E402
import ops07  # noqa: E402,F401  registers levels 9-11
try:
    import ops08  # noqa: F401  registers levels 12-14 (task 08)
except ImportError:
    pass
from nkitool import analyze  # noqa: E402

argv = sys.argv[1:]
levels = []
out_dir = "solves"
files = []
i = 0
while i < len(argv):
    if argv[i] == "--levels":
        i += 1
        while i < len(argv) and not argv[i].startswith("--") and argv[i].isdigit():
            levels.append(int(argv[i]))
            i += 1
    elif argv[i] == "--out":
        out_dir = argv[i + 1]
        i += 2
    else:
        files.append(argv[i])
        i += 1
os.makedirs(out_dir, exist_ok=True)

# Small shapes where the visible ones would make birsim slow; otherwise the level's own shapes.
SHAPES = {1: [dict(shape=(4, 8, 8), pool_size=2), dict(shape=(8, 12, 12), pool_size=3)],
          4: [dict(K=256, M=256, N=1024)]}

seen = {}
for p in files:
    tag = os.path.basename(p)[len("attempts_"):-len(".jsonl")]
    for line in open(p):
        r = json.loads(line)
        if r["level"] in levels and r["reward"] >= 1 - 1e-9:
            seen.setdefault((r["level"], r["code"]), tag)

summary = collections.defaultdict(collections.Counter)
reasons = collections.defaultdict(collections.Counter)
for (lv, code), src in sorted(seen.items(), key=lambda kv: kv[0][0]):
    h = hashlib.sha1(code.encode()).hexdigest()[:8]
    path = os.path.join(out_dir, f"l{lv}_{src}_{h}.py")
    open(path, "w").write(code)
    print(f"== {os.path.basename(path)}")
    try:
        spec = importlib.util.spec_from_file_location("k_" + h, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        kern = getattr(m, nkibench.LEVELS[lv]["entry"])
    except Exception as e:  # noqa: BLE001
        print(f"   does not load: {type(e).__name__}: {e}")
        summary[lv]["does not load"] += 1
        continue
    verdicts = []
    for s in SHAPES.get(lv, nkibench.LEVELS[lv]["shapes"]):
        args, _ = nkibench.make_inputs(s, lv)
        want = np.asarray(nkibench.LEVELS[lv]["ref"](*args), dtype=np.float64)
        label = nkibench.label(s, lv)
        try:
            if os.environ.get("BIRSIM", "1") == "0":     # lowering only (the Mac cannot run birsim)
                rep = analyze(kern, *args, neff=False)
                verdicts.append("match")
                print(f"   {label:32s} lowers; predicted {rep['predicted_ns'] / 1e3:9.1f} us (birsim off)")
                continue
            rep = analyze(kern, *args, neff=True, birsim=True)
            outs = rep["birsim_outputs"]
            cands = (list(outs.values()) if isinstance(outs, dict) else
                     list(outs) if isinstance(outs, (list, tuple)) else [outs])
            got = next(np.asarray(c).reshape(want.shape).astype(np.float64)
                       for c in cands if np.asarray(c).size == want.size)
            worst = float(np.abs(got - want).max()) / (float(np.sqrt((want ** 2).mean())) or 1.0)
            ok = worst <= 2e-2 and np.all(np.isfinite(got))
            verdicts.append("match" if ok else "wrong numbers")
            print(f"   {label:32s} predicted {rep['predicted_ns'] / 1e3:9.1f} us; birsim "
                  f"{'MATCHES' if ok else 'WRONG NUMBERS'}: worst error {worst:.2e} of RMS")
        except Exception as e:  # noqa: BLE001
            msg = str(e).replace("\n", " ")
            k = msg.find("error:")
            why = msg[k:k + 160] if k >= 0 else msg[:160]
            verdicts.append("does not lower")
            reasons[lv][why[:100]] += 1
            print(f"   {label:32s} FAILS: {type(e).__name__}: {why}")
    worst = ("does not lower" if "does not lower" in verdicts else
             "wrong numbers" if "wrong numbers" in verdicts else "match")
    summary[lv][worst] += 1

print("\n=== summary: distinct simulator-solved kernels, by what the compiler and birsim say")
for lv in sorted(summary):
    print(f"level {lv}: {dict(summary[lv])}")
    for why, n in reasons[lv].most_common():
        print(f"    {n} x {why}")
