"""Disposable evaluation process. Candidate Python executes here, not in the search driver."""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(actual, expected, task):
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            raise AssertionError("Output structure mismatch")
        for a, e in zip(actual, expected):
            check(a, e, task)
        return
    if not isinstance(expected, torch.Tensor):
        raise TypeError("Reference must return tensors or a tuple/list of tensors")
    e = expected.detach().cpu().numpy()
    a = np.asarray(actual)
    if a.shape != e.shape or a.dtype != e.dtype:
        raise AssertionError(f"Output shape/dtype mismatch: {a.shape}/{a.dtype} vs {e.shape}/{e.dtype}")
    if not np.isfinite(a).all() or not np.isfinite(e).all():
        raise AssertionError("Nonfinite output")
    np.testing.assert_allclose(a, e, rtol=getattr(task, "RTOL", 1e-4),
                               atol=getattr(task, "ATOL", 1e-5))


def evaluate(candidate, task_path, mode, seeds, baseline=None):
    import nki
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from nki_kernel_search.hardware import HardwareKernel

    task = load(task_path, "search_task")
    fn = load(candidate, "search_candidate").kernel
    same_baseline = bool(baseline) and Path(candidate).read_bytes() == Path(baseline).read_bytes()
    sim = nki.simulate(nki.jit(fn))
    device = HardwareKernel(fn) if mode == "hardware" else None
    baseline_fn = fn if same_baseline else load(baseline, "search_baseline").kernel if baseline else None
    baseline_sim = nki.simulate(nki.jit(baseline_fn)) if baseline_fn else None
    baseline_device = device if same_baseline else HardwareKernel(baseline_fn) if baseline_fn and device else None
    latencies, ratios = [], []
    count = 0
    timing_cases = {}
    for seed in seeds:
        for inputs in task.cases(seed):
            if not isinstance(inputs, (tuple, list)):
                raise TypeError("Each case must be a tuple/list of positional inputs")
            with torch.no_grad():
                expected = task.reference(*[x.clone() if isinstance(x, torch.Tensor) else x for x in inputs])
            args = tuple(x.detach().cpu().numpy().copy() if isinstance(x, torch.Tensor) else x
                         for x in inputs)
            check(sim(*args), expected, task)
            if baseline_sim and not same_baseline:
                check(baseline_sim(*args), expected, task)
            if device:
                check(device(*args), expected, task)
                if baseline_device and not same_baseline:
                    check(baseline_device(*args), expected, task)
                timing_cases.setdefault(device.key(args), args)
            count += 1
    if not count:
        raise ValueError("Task produced no cases")
    # Accuracy gates the entire suite before any timing score is assigned.
    for args in timing_cases.values():
        latency = device.benchmark(*args)
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError("Invalid hardware latency")
        latencies.append(latency)
        if baseline_device:
            base_us = latency if same_baseline else baseline_device.benchmark(*args)
            if not math.isfinite(base_us) or base_us <= 0:
                raise ValueError("Invalid baseline latency")
            ratios.append(base_us / latency)
    metrics = {"correctness": 1.0, "cases_passed": float(count), "combined_score": 1.0,
               "hardware_measured": float(mode == "hardware")}
    if latencies:
        metrics["latency_us"] = float(np.mean(latencies))
        metrics["combined_score"] = math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else 1 / metrics["latency_us"]
    return {"metrics": metrics, "artifacts": {"mode": mode, "case_latencies_us": json.dumps(latencies)}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--mode", choices=["simulate", "hardware"], required=True)
    p.add_argument("--result", required=True)
    p.add_argument("--baseline")
    p.add_argument("--seeds", default="0,17,42")
    a = p.parse_args()
    try:
        result = evaluate(a.candidate, a.task, a.mode, [int(s) for s in a.seeds.split(",")], a.baseline)
    except Exception as exc:
        result = {"metrics": {"correctness": 0.0, "combined_score": 0.0},
                  "artifacts": {"error": f"{type(exc).__name__}: {exc}"[:8000]}}
    Path(a.result).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
