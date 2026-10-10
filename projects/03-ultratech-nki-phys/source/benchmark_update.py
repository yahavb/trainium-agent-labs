"""Validate a preserved NKI kernel and measure it with the nrtpy runtime."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
import traceback
import uuid

import numpy as np
from contact import build_fixture, diagnostics, oracle
from update_reference import fixed_steps, prepare


def timing_stats(result):
    values = {name: float(getattr(result, name))
              for name in ("mean_ms", "min_ms", "max_ms", "std_dev_ms")}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Invalid benchmark latency")
    if not 0 < values["min_ms"] <= values["mean_ms"] <= values["max_ms"] or values["std_dev_ms"] < 0:
        raise ValueError("Invalid benchmark latency range")
    values["iterations"] = int(result.iterations)
    if values["iterations"] < 1:
        raise ValueError("No benchmark iterations")
    samples = [float(value) for value in result.durations_ms]
    if len(samples) != values["iterations"] or not all(np.isfinite(v) and v > 0 for v in samples):
        raise ValueError("Invalid or missing timing samples")
    values["durations_ms"] = samples
    values["warmup_iterations"] = int(result.warmup_iterations)
    return values


def runtime_inputs(model, inputs, tensor_type):
    arrays = dict(zip(("a_transposed", "bias", "initial"), inputs[:3]))
    names = set(model.input_tensors_info)
    if names != set(arrays):
        raise ValueError(f"Compiled input interface mismatch: {sorted(names)}; expected {sorted(arrays)}")
    return {name: tensor_type.from_numpy(array, name=name) for name, array in arrays.items()}


def runtime_outputs(model):
    tensors = model.allocate_output_tensors()
    if len(tensors) != 1 or not tensors[0].name:
        raise ValueError("Expected exactly one named output tensor")
    return {tensors[0].name: tensors[0]}


def execute_runtime(model, inputs, outputs):
    model(inputs=inputs, outputs=outputs)
    actual = np.asarray(next(iter(outputs.values())).numpy())
    if actual.dtype.kind == "V":
        actual = actual.view(np.float32)
    if actual.dtype != np.float32:
        raise ValueError(f"Expected FP32 output, got {actual.dtype}")
    return actual.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contacts", type=int, default=33)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.contacts <= 128 or min(args.steps, args.runs, args.warmup, args.iters) < 1:
        parser.error("Positive counts required; contacts must be at most 128")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        parser.error("Set NEURON_RT_VISIBLE_CORES to confirmed available cores")
    run_id = uuid.uuid4().hex
    out = Path("data") / f"benchmark-update-{run_id}"
    out.mkdir(parents=True)
    context = dict(run_id=run_id, hostname=platform.node(), python=platform.python_version(),
                   visible_cores=os.environ["NEURON_RT_VISIBLE_CORES"], **vars(args))
    context["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("benchmark_update.py", "nki_contact.py", "update_reference.py", "contact.py")}
    context["neuron_bench_path"] = shutil.which("neuron-bench")
    results = []
    try:
        os.environ["NKI_ARTIFACTS_DIR"] = str((out / "compiled").resolve())
        import nki
        from nki_contact import contact_steps
        from nrtpy import SpikeModel, SpikeTensor
        context["nki_version"] = getattr(nki, "__version__", "unknown")
        context["timing_backend"] = "nrtpy"
        fixture = build_fixture("plane", args.contacts, 0)
        inputs = prepare(fixture)
        expected = fixed_steps(*inputs, args.steps)
        reference = oracle(fixture["A"], fixture["b"])
        print("Compiling/executing baseline and preserving NEFF", flush=True)
        start = time.perf_counter()
        compiled_output = np.asarray(contact_steps(*inputs, args.steps)).copy()
        context["first_jit_call_seconds"] = time.perf_counter() - start
        if compiled_output.shape != expected.shape or not np.isfinite(compiled_output).all() or not np.allclose(compiled_output, expected, rtol=0, atol=1e-4):
            raise ValueError("Initial JIT correctness check failed")
        neffs = sorted((out / "compiled").rglob("*.neff"))
        context["neff_paths"] = [str(p) for p in neffs]
        if len(neffs) != 1:
            raise RuntimeError(f"Expected exactly one preserved NEFF, found {len(neffs)}")
        context["neff_sha256"] = hashlib.sha256(neffs[0].read_bytes()).hexdigest()
        model = SpikeModel.load_from_neff(str(neffs[0]))
        context["compiled_inputs"] = list(model.input_tensors_info)
        device_outputs = runtime_outputs(model)
        context["compiled_outputs"] = list(device_outputs)
        device_inputs = runtime_inputs(model, inputs, SpikeTensor)
        for repeat in range(args.runs):
            entry = dict(run_id=run_id, repeat=repeat, score=0, timing_ok=False)
            print(f"Run {repeat + 1}/{args.runs}: numerical validation", flush=True)
            try:
                originals = [x.copy() for x in inputs[:3]]
                start = time.perf_counter()
                actual = execute_runtime(model, device_inputs, device_outputs)
                entry["runtime_host_call_seconds"] = time.perf_counter() - start
                limit = 1e-4 * max(1.0, float(np.max(np.abs(expected))))
                if actual.shape != expected.shape or not np.isfinite(actual).all():
                    raise ValueError("Invalid output shape or nonfinite values")
                entry["update_max_absolute_error"] = float(np.max(np.abs(actual - expected)))
                entry["update_error_limit"] = limit
                entry["padding_ok"] = bool(np.all(np.abs(actual[args.contacts:]) <= 1e-6))
                entry["inputs_unchanged"] = all(np.array_equal(x, y) for x, y in zip(inputs[:3], originals))
                entry["device_inputs_unchanged"] = all(
                    np.array_equal(np.asarray(device_inputs[name].numpy()).view(np.float32), original)
                    for name, original in zip(("a_transposed", "bias", "initial"), originals))
                entry["score"] = int(entry["update_max_absolute_error"] <= limit and
                                     entry["padding_ok"] and entry["inputs_unchanged"] and entry["device_inputs_unchanged"])
                entry["physics_checker"] = diagnostics(fixture["A"], fixture["b"],
                                                       actual[:args.contacts, 0], reference, fixture)
                if not entry["score"]:
                    raise ValueError("Correctness gate failed; refusing timing")
                for mode in ("device", "host"):
                    print(f"  Benchmarking {mode} timing", flush=True)
                    entry[f"{mode}_stats"] = timing_stats(model.benchmark(
                        inputs=device_inputs, outputs=device_outputs, warmup_iter=args.warmup,
                        benchmark_iter=args.iters, mode=mode))
                after = execute_runtime(model, device_inputs, device_outputs)
                if after.shape != expected.shape or not np.isfinite(after).all() or not np.allclose(after, expected, rtol=0, atol=limit):
                    entry["score"] = 0
                    raise ValueError("Post-benchmark correctness failed")
                if not all(np.array_equal(np.asarray(device_inputs[name].numpy()).view(np.float32), original)
                           for name, original in zip(("a_transposed", "bias", "initial"), originals)):
                    entry["score"] = 0
                    raise ValueError("Benchmark mutated device inputs")
                entry["timing_ok"] = True
                print(f"  device mean={entry['device_stats']['mean_ms']:.6f} ms; host mean={entry['host_stats']['mean_ms']:.6f} ms", flush=True)
            except Exception:
                entry["error"] = traceback.format_exc()
                print(entry["error"], flush=True)
            results.append(entry)
            with Path("data/benchmark-attempts.jsonl").open("a") as log:
                log.write(json.dumps(dict(context=context, **entry), allow_nan=False) + "\n")
            if "error" in entry:
                break
    except Exception:
        context["error"] = traceback.format_exc()
        print(context["error"], flush=True)
        with Path("data/benchmark-attempts.jsonl").open("a") as log:
            log.write(json.dumps(dict(context=context, score=0, timing_ok=False)) + "\n")
    report = dict(context=context, results=results)
    medians = [r["device_stats"]["mean_ms"] for r in results if r["timing_ok"]]
    if medians:
        report["device_mean_spread_ms"] = dict(min=min(medians), median=float(np.median(medians)), max=max(medians))
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    note = f"""# Fixed-Update Timing Diagnostic

Run {run_id}; host {context['hostname']}; visible cores {context['visible_cores']}.
Plane scene, seed 0; {args.contacts} contacts; {args.steps} fixed updates.
Requested {args.runs} runs; recorded {len(results)}; timed successfully {len(medians)}.
Each timing requests {args.warmup} warmups and {args.iters} measured iterations.
Spread of per-run device means in milliseconds: {report.get('device_mean_spread_ms', 'unavailable')}.

Score is numerical update equivalence, zero padding and input preservation.
Full physics checks are separate. This is not a converged-solver benchmark.
nrtpy device mode measures NeuronCore execution including on-device DMA.
Host mode includes host-device communication; inputs are allocated before timing.
Compilation/loading and ordinary output materialization are excluded from device timing.
The first JIT call is reported separately, including possible compilation.
No speedup, time-to-accuracy, batching or multi-core scaling claim is made.
Source hashes, failures, timing summaries and individual samples: results.json.
Append-only attempt log: data/benchmark-attempts.jsonl.
"""
    (out / "run-note.md").write_text(note)
    print(f"Artifacts: {out}", flush=True)
    return int(len(medians) != args.runs)


if __name__ == "__main__":
    raise SystemExit(main())
