"""Infer the peaks the compiler's latency model ASSUMES, by differencing predicted times.

For each engine: compile a kernel doing the op N1 times and N2 times on data already in SBUF;
(t2 - t1) / ((N2 - N1) * work) is the modelled per-unit cost. DMA: vary the size of a load+store.
"""
import json, os, sys
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import predict, kernel_from_source

def dma_kernel_factory():
    @nki.jit
    def k(x):
        P, F = x.shape
        out = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm)
        t = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x)
        nisa.dma_copy(dst=out, src=t)
        return out
    return k

HDR = "import nki\nimport nki.isa as nisa\nimport nki.language as nl\n"

def engine_kernel_factory(kind, nrep, dname):
    body = {
        "vector_tt": "nisa.tensor_tensor(dst=dst, data1=src, data2=src, op=nl.add)",
        "vector_ts": "nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=0.5)",
        "scalar_exp": "nisa.activation(dst=dst, op=nl.exp, data=src, scale=0.001)",
        "vector_max": "nisa.tensor_reduce(dst=dst[:, 0:1], op=nl.maximum, data=src, axis=1)",
    }[kind]
    ops = []
    for i in range(nrep):
        s, d = ("a", "b") if i % 2 == 0 else ("b", "a")
        ops.append("    src = %s\n    dst = %s\n    %s\n" % (s, d, body))
    fin = "a" if nrep % 2 == 0 else "b"
    src = HDR + f"""
@nki.jit
def k(x):
    P, F = x.shape
    out = nl.ndarray((P, 1), dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    a = nl.ndarray((P, F), dtype=nl.{dname}, buffer=nl.sbuf)
    b = nl.ndarray((P, F), dtype=nl.{dname}, buffer=nl.sbuf)
    nisa.tensor_copy(dst=a, src=t)
    nisa.tensor_copy(dst=b, src=t)
""" + "".join(ops) + f"""
    o = nl.ndarray((P, 1), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o, src={fin}[:, 0:1])
    nisa.dma_copy(dst=out, src=o)
    return out
"""
    return kernel_from_source(src, "k", f"{kind}_{nrep}_{dname}")

def mm_kernel_factory(nrep, dname):
    src = HDR + f"""
@nki.jit
def k(lhsT, rhs):
    K, M = lhsT.shape
    K2, N = rhs.shape
    out = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    l32 = nl.ndarray((K, M), dtype=lhsT.dtype, buffer=nl.sbuf)
    r32 = nl.ndarray((K, N), dtype=rhs.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=l32, src=lhsT)
    nisa.dma_copy(dst=r32, src=rhs)
    l = nl.ndarray((K, M), dtype=nl.{dname}, buffer=nl.sbuf)
    r = nl.ndarray((K, N), dtype=nl.{dname}, buffer=nl.sbuf)
    nisa.tensor_copy(dst=l, src=l32)
    nisa.tensor_copy(dst=r, src=r32)
    ps = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
""" + "".join("    nisa.nc_matmul(dst=ps, stationary=l, moving=r)\n" for _ in range(nrep)) + """
    o = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o, src=ps)
    nisa.dma_copy(dst=out, src=o)
    return out
"""
    return kernel_from_source(src, "k", f"mm_{nrep}_{dname}")

def main():
    rows = {}
    rng = np.random.default_rng(0)
    # DMA: load+store of P x F fp32, bytes moved = 2 * P * F * 4
    pts = []
    for F in (512, 2048, 8192, 32768):
        x = rng.standard_normal((128, F)).astype(np.float32)
        r = predict(dma_kernel_factory(), {"x": x})
        pts.append((2 * x.nbytes, r["ns"]))
        print("dma", F, 2 * x.nbytes, r, flush=True)
    (b1, t1), (b2, t2) = pts[-2], pts[-1]
    rows["dma_GBps_model"] = (b2 - b1) / (t2 - t1)
    rows["dma_points"] = pts
    # engines: F=8192 so each op is 1M elements
    F = 8192
    x = rng.standard_normal((128, F)).astype(np.float32)
    for kind in ("vector_tt", "vector_ts", "scalar_exp", "vector_max"):
        for dname, dt in (("fp32", "float32"), ("bf16", "bfloat16")):
            t = {}
            for n in (2, 10):
                r = predict(engine_kernel_factory(kind, n, dt), {"x": x})
                t[n] = r["ns"]
                print(kind, dname, n, r, flush=True)
            per_op_ns = (t[10] - t[2]) / 8
            rows[f"{kind}_{dname}_Gelem_per_s"] = 128 * F / per_op_ns
            rows[f"{kind}_{dname}_ns_per_op"] = per_op_ns
    for dname, dt in (("fp32", "float32"), ("bf16", "bfloat16")):
        l = rng.standard_normal((128, 128)).astype(np.float32)
        rr = rng.standard_normal((128, 512)).astype(np.float32)
        t = {}
        for n in (2, 34):
            r = predict(mm_kernel_factory(n, dt), {"lhsT": l, "rhs": rr})
            t[n] = r["ns"]
            print("matmul", dname, n, r, flush=True)
        per = (t[34] - t[2]) / 32
        rows[f"matmul_{dname}_TFLOPs"] = 2 * 128 * 128 * 512 / per / 1e3
        rows[f"matmul_{dname}_ns_per_op"] = per
    print(json.dumps(rows, indent=1))
    json.dump(rows, open("/tmp/roof_peaks_probe.json", "w"), indent=1)

if __name__ == "__main__":
    main()
