"""Compile and execute Level 8 on a Neuron device, never nki.simulate_kernel.

Run only on a seat with a confirmed free core:
NEURON_RT_VISIBLE_CORES=2 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python tests/check_attention_device.py solved/level08_attention.py --lnc 2

Times include host/framework/transfer overhead; they are not kernel latency measurements.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nkibench as nb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--lnc", type=int, default=2)
    parser.add_argument("--suite", choices=("official", "independent"), default="official")
    parser.add_argument("--json", default="results/attention-device.json")
    parser.add_argument("--stop-on-error", action="store_true")
    options = parser.parse_args()
    if not os.environ.get("NEURON_RT_VISIBLE_CORES") and os.environ.get("NEURON_RT_NUM_CORES") != "1":
        parser.error("Set NEURON_RT_VISIBLE_CORES, or NEURON_RT_NUM_CORES=1 to request any free core.")
    if int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1")) != options.lnc:
        parser.error("--lnc must match NEURON_LOGICAL_NC_CONFIG.")
    candidate = Path(options.candidate).resolve()
    output = Path(options.json)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(host=os.uname().nodename, execution="nki.jit standalone device execution",
                  candidate=str(candidate), sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                  lnc=options.lnc, cores=os.environ.get("NEURON_RT_VISIBLE_CORES", "automatic"),
                  requested_core_count=os.environ.get("NEURON_RT_NUM_CORES"),
                  platform=os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE"),
                  suite=options.suite, cases=[])
    print(json.dumps({k:v for k,v in report.items() if k != "cases"}), flush=True)
    kernel = nb.load_kernel(str(candidate), "nki_attention_")
    if options.suite == "official":
        inputs = [(nb.label(shape, 8), nb.make_inputs(shape, 8, seed=0)[0], None)
                  for shape in nb.LEVELS[8]["shapes"]]
    else:
        from check_attention import cases
        inputs = list(cases())
    for name, args, invariant in inputs:
        print("DEVICE_CASE_START " + name, flush=True)
        before = [x.copy() for x in args]
        want = nb.ref_attention(*args)
        record = dict(case=name, passed=False)
        try:
            t0 = time.perf_counter()
            got = np.asarray(kernel[options.lnc](*args))
            t1 = time.perf_counter()
            repeated = np.asarray(kernel[options.lnc](*args))
            t2 = time.perf_counter()
            error = nb.describe_mismatch(got, want) or nb.describe_mismatch(repeated, want)
            if invariant is not None:
                error = error or nb.describe_mismatch(got, invariant)
            unchanged = all(np.array_equal(x, y) for x, y in zip(args, before))
            if not unchanged:
                error = "input mutated"
            scale = float(np.sqrt(np.mean(want.astype(np.float64) ** 2))) or 1.0
            record.update(passed=error is None, error=error, shape=list(got.shape),
                          normalized_max_error=float(np.max(np.abs(got-want))) / scale,
                          inputs_unchanged=unchanged, repeatable=bool(np.array_equal(got, repeated)),
                          first_call_wall_s=t1-t0, second_call_wall_s=t2-t1)
        except Exception as exc:
            record.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            print(record["traceback"], flush=True)
        report["cases"].append(record)
        report["passed"] = sum(r["passed"] for r in report["cases"])
        report["total"] = len(report["cases"])
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if not record["passed"] and options.stop_on_error:
            break
    print(f"ON DEVICE: {report['passed']}/{report['total']} cases correct", flush=True)
    return 0 if report["passed"] == len(inputs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
