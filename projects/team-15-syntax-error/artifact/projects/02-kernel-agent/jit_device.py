"""Run @nki.jit kernels on the Trainium device by calling them with numpy arrays (nki.jit docstring:
'np.ndarray: compiles and executes standalone kernel'). Checks the device output against NumPy and
times repeated calls (wall clock incl. host<->device copies; the first call includes compilation)."""
import importlib.util, sys, time, json
import numpy as np
K, M, N = 512, 512, 1024
r = np.random.default_rng(0)
lhsT = r.standard_normal((K, M)).astype(np.float32)
rhs = r.standard_normal((K, N)).astype(np.float32)
want = (lhsT.astype(np.float64).T @ rhs.astype(np.float64)).astype(np.float32)
rows = []
for spec in sys.argv[1:]:
    path, entry = spec.split(":")
    s = importlib.util.spec_from_file_location("k", path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    kern = getattr(m, entry)
    try:
        t0 = time.perf_counter(); got = np.asarray(kern(lhsT, rhs)); t_first = time.perf_counter() - t0
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); kern(lhsT, rhs); ts.append(time.perf_counter() - t0)
        err = float(np.abs(got - want).max() / np.sqrt((want ** 2).mean()))
        row = dict(kernel=path, ok=err < 2e-2, err=err, first_call_s=t_first, warm_call_s_min=min(ts), warm_call_s_mean=sum(ts)/len(ts))
    except Exception as e:
        row = dict(kernel=path, error=f"{type(e).__name__}: {str(e)[:300]}")
    rows.append(row); print(row, flush=True)
json.dump(rows, open("runs/device_jit.json", "w"), indent=1)
