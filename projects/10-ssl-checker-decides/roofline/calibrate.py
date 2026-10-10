"""Calibrate the roofline against the compiler's prediction (no device).

For each shipped reference kernel (levels 1-4, plus the TILE_N=128 level-4 variant) and the three
softmax kernels: simulate (correctness + DMA bytes, via nkibench.simulate_and_count), compile with
predicted latency, and compare with roofline.py's HW and MODEL floors.

    PYTHONDONTWRITEBYTECODE=1 python calibrate.py      # in a seat pod; reads /workspace, writes /tmp only
"""
import importlib.util
import json
import re
import os
import sys
import warnings

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, "/workspace/projects/02-kernel-agent")
import numpy as np
import nkibench
import roofline as rl
import softmax_kernels as sk
from predict import predict

HERE = "/workspace/projects/02-kernel-agent"
warnings.simplefilter("ignore")


def load_baked(path, entry, consts, tag):
    """Copy a reference kernel to /tmp, turning its non-array arguments into constants (the
    compiler's `inputs` only takes arrays)."""
    src = open(path).read()
    if consts:
        m = re.search(rf"def {entry}\(([^)]*)\):\n", src)
        args = [a.strip() for a in m.group(1).split(",")]
        keep = [a for a in args if a not in consts]
        body = "".join(f"  {k} = {v!r}\n" for k, v in consts.items())
        src = src[:m.start()] + f"def {entry}({', '.join(keep)}):\n" + body + src[m.end():]
    if tag == "ref4_narrow":
        src = src.replace("TILE_N = nl.tile_size.gemm_moving_fmax  # 512", "TILE_N = 128")
    out = f"/tmp/_calib_{tag}.py"
    open(out, "w").write(src)
    spec = importlib.util.spec_from_file_location(f"_calib_{tag}", out)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


def run_case(label, kernel_sim, kernel_cmp, sim_args, cmp_inputs, want, op):
    row = dict(kernel=label, op=op.name)
    try:
        got, counted = nkibench.simulate_and_count(kernel_sim, list(sim_args))
        row["sim_ok"] = nkibench.describe_mismatch(got, want) is None
        row["dma_bytes"] = counted["bytes"]
    except Exception as e:
        row["sim_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    try:
        p = predict(kernel_cmp, cmp_inputs)
        row["pred_ns"], row["util"] = p["ns"], p["util"]
    except Exception as e:
        row["compile_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    hw, md = rl.analyze(op, rl.HW), rl.analyze(op, rl.MODEL)
    row.update(hw_floor_ns=round(hw["floor_ns"]), hw_bound=hw["bound"],
               model_floor_ns=round(md["floor_ns"]), model_bound=md["bound"], min_bytes=op.hbm_bytes)
    if "pred_ns" in row:
        row["pred_over_hw_floor"] = round(row["pred_ns"] / hw["floor_ns"], 2)
        row["pred_over_model_floor"] = round(row["pred_ns"] / md["floor_ns"], 2)
        row["feedback"] = rl.feedback(op, row["pred_ns"], row.get("dma_bytes"), row.get("util"))
    print(json.dumps(row), flush=True)
    return row


def main():
    rows = []
    # level 4 (+ narrow variant) and level 3: matmul, fp32 inputs
    k4 = load_baked(f"{HERE}/reference_level4.py", "nki_matmul_tiled_", {}, "ref4")
    k4n = load_baked(f"{HERE}/reference_level4.py", "nki_matmul_tiled_", {}, "ref4_narrow")
    for spec in nkibench.LEVELS[4]["shapes"]:
        args, _ = nkibench.make_inputs(spec, 4)
        want = nkibench.ref_matmul(*args)
        op = rl.matmul(spec["M"], spec["K"], spec["N"], "float32")
        for lab, k in (("ref4", k4), ("ref4_TILE_N128", k4n)):
            rows.append(run_case(lab, k, k, args, {"lhsT": args[0], "rhs": args[1]}, want, op))
    k3 = load_baked(f"{HERE}/reference_level3.py", "nki_matmul_basic_", {}, "ref3")
    for spec in nkibench.LEVELS[3]["shapes"][:1]:
        args, _ = nkibench.make_inputs(spec, 3)
        want = nkibench.ref_matmul(*args)
        op = rl.matmul(spec["M"], spec["K"], spec["N"], "float32")
        rows.append(run_case("ref3", k3, k3, args, {"lhsT": args[0], "rhs": args[1]}, want, op))
    # level 1 avgpool and level 2 transpose: non-array args baked in for the compile
    for n, path, entry, kw in ((1, "reference_level1.py", "tensor_avgpool_kernel", "pool_size"),
                               (2, "reference_level2.py", "tensor_transpose2D_kernel_", "shape2D")):
        ksim = load_baked(f"{HERE}/{path}", entry, {}, f"ref{n}")
        for i, spec in enumerate(nkibench.LEVELS[n]["shapes"]):
            args, _ = nkibench.make_inputs(spec, n)
            want = nkibench.LEVELS[n]["ref"](*args)
            kc = load_baked(f"{HERE}/{path}", entry, {kw: args[1]}, f"ref{n}_{i}")
            if n == 1:
                C, H, W = spec["shape"]
                op = rl.avgpool(C, H, W, spec["pool_size"])
                first = "in_tensor"
            else:
                op = rl.transpose2d(*spec["shape"])
                first = "in_tensor"
            rows.append(run_case(f"ref{n}", ksim, kc, args, {first: args[0]}, want, op))
    # softmax kernels
    def ref_softmax(x):
        m = x.max(axis=1, keepdims=True)
        e = np.exp(x - m)
        return e / e.sum(axis=1, keepdims=True)
    for (R, C) in ((128, 4096), (512, 1024)):
        x = (np.random.default_rng(1).standard_normal((R, C)) * 4).astype(np.float32)
        want = ref_softmax(x)
        op = rl.softmax(R, C, "float32")
        for name in ("softmax_fused", "softmax_reload", "softmax_extrapass"):
            k = getattr(sk, name)
            rows.append(run_case(name, k, k, (x,), {"x": x}, want, op))
    json.dump(rows, open("/tmp/roof_calibrate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
