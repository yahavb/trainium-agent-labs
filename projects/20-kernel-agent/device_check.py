"""Run saved agent kernels on the REAL Trainium2 chip (a free NeuronCore), not the simulator.

    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_RT_VISIBLE_CORES=2 python device_check.py
"""
import importlib.util, json, sys, time
import numpy as np
import nkibench as nb

CASES = [("solved/level03_matmul_single_tile.py", 3), ("solved/level04_matmul_tiled.py", 4),
         ("solved/level07_matmul_fully_blocked.py", 7), ("solved/level09_transpose_tensor_engine.py", 9),
         ("solved/level10_row_softmax.py", 10)]
LNC = 2
out = []
for path, lv in CASES:
    spec_mod = importlib.util.spec_from_file_location(f"k{lv}", path)
    mod = importlib.util.module_from_spec(spec_mod); spec_mod.loader.exec_module(mod)
    fn = next(getattr(mod, n) for n in dir(mod) if n.startswith("nki_") or n.startswith("tensor_"))
    for case in nb.LEVELS[lv]["shapes"]:
        args, _ = nb.make_inputs(case, lv, seed=0)
        want = nb.LEVELS[lv]["ref"](*args)
        rec = dict(level=lv, case=nb.label(case, lv))
        try:
            t0 = time.perf_counter(); got = np.asarray(fn[LNC](*args)); t1 = time.perf_counter()
            got2 = np.asarray(fn[LNC](*args)); t2 = time.perf_counter()
            m = nb.describe_mismatch(got, want)
            rec.update(ok=m is None, first_call_s=round(t1 - t0, 2), second_call_s=round(t2 - t1, 2),
                       repeatable=bool(np.array_equal(got, got2)),
                       max_rel_err=float(np.abs(got - want).max() / (np.sqrt((want ** 2).mean()) + 1e-30)),
                       mismatch=(m or "")[:200])
        except Exception as e:
            rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:200]}")
        out.append(rec)
        print(json.dumps(rec), flush=True)
json.dump(out, open("device_results.json", "w"), indent=1)
print(f"\nON DEVICE: {sum(r['ok'] for r in out)}/{len(out)} cases correct")
