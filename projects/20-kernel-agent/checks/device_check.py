"""Run the solved kernels on the REAL Trainium2 chip (not the simulator): correctness, and timing.

    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_RT_NUM_CORES=1 python checks/device_check.py

Needs a NeuronCore no model server is using. Timing is OFF by default (DEVICE_REPS=0): measured, an NKI
0.6 call re-compiles the kernel every time (~1.7 s per call for level 1), so host wall-clock measures the
compiler, not the kernel. With DEVICE_REPS=N it times N calls after 3 warm-ups (median, p10-p90) and a
trivial copy kernel as the launch floor. Real kernel latency needs neuron-profile on a compiled NEFF.
"""
import importlib.util, json, os, time
import numpy as np
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the project folder
sys.path.insert(0, ROOT); os.chdir(ROOT)
import nki
import nki.isa as nisa
import nki.language as nl
import nkibench as nb

CASES = [("solved/level01_avgpool.py", 1), ("solved/level02_transpose.py", 2),
         ("solved/level03_matmul_single_tile.py", 3), ("solved/level04_matmul_tiled.py", 4),
         ("solved/level05_matmul_hoisted.py", 5), ("solved/level06_matmul_blocked.py", 6),
         ("solved/level07_matmul_fully_blocked.py", 7), ("solved/level09_transpose_tensor_engine.py", 9),
         ("solved/level10_row_softmax.py", 10), ("solved/level11_attention_scores.py", 11),
         ("solved/level08_attention.py", 8)]   # generated with --attention-plan (seat 97)
LNC, REPS = 2, int(os.environ.get("DEVICE_REPS", "0"))


@nki.jit
def copy_kernel(a):
    """The floor: one HBM -> SBUF -> HBM round trip of the first input."""
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=a)
    nisa.dma_copy(dst=out, src=t)
    return out


def timed(fn, args):
    for _ in range(3):
        fn[LNC](*args)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); fn[LNC](*args); ts.append((time.perf_counter() - t0) * 1e6)
    p10, p50, p90 = np.percentile(ts, [10, 50, 90])
    return round(p50, 1), round(p10, 1), round(p90, 1)


out = []
for path, lv in CASES:
    spec = importlib.util.spec_from_file_location(f"k{lv}", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    fn = getattr(mod, nb.LEVELS[lv]["entry"])
    for case in nb.LEVELS[lv]["shapes"]:
        args, _ = nb.make_inputs(case, lv, seed=0)
        want = nb.LEVELS[lv]["ref"](*args)
        rec = dict(level=lv, case=nb.label(case, lv), kernel=path)
        try:
            got = np.asarray(fn[LNC](*args))
            m = nb.describe_mismatch(got, want)
            rec.update(ok=m is None, repeatable=bool(np.array_equal(got, np.asarray(fn[LNC](*args)))),
                       max_rel_err=float(np.abs(got - want).max() / (np.sqrt((want ** 2).mean()) + 1e-30)),
                       mismatch=(m or "")[:200])
            if not REPS:
                out.append(rec); print(json.dumps(rec), flush=True); continue
            rec["us_median"], rec["us_p10"], rec["us_p90"] = timed(fn, args)
            first = next(a for a in args if hasattr(a, "shape") and a.ndim == 2 and a.shape[0] <= 128) \
                if any(hasattr(a, "shape") and a.ndim == 2 and a.shape[0] <= 128 for a in args) else None
            if first is not None:
                rec["copy_us_median"] = timed(copy_kernel, (first,))[0]
                rec["kernel_minus_copy_us"] = round(rec["us_median"] - rec["copy_us_median"], 1)
        except Exception as e:
            rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:200]}")
        out.append(rec)
        print(json.dumps(rec), flush=True)
json.dump(out, open(os.environ.get("DEVICE_OUT", "device_results.json"), "w"), indent=1)
print(f"\nON DEVICE: {sum(r['ok'] for r in out)}/{len(out)} cases correct")
