"""Separate correctness and paired-throughput evaluation for distinct math tasks."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import traceback
import uuid

import numpy as np

from benchmark_update import execute_runtime, runtime_outputs, timing_stats
from math_tasks import TASKS, check, cpu, fixture, sources
from math_program import evaluate, lower


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load(task, form, source, directory, backend, program=None):
    path = directory / "candidate.py"
    path.write_text(source)
    if backend == "cpu":
        if program is not None:
            return lambda *arrays: evaluate(task, program, arrays), None, None
        return lambda *arrays: cpu(task, arrays, form != "baseline"), None, None
    import nki
    compiled = directory / "compiled"
    os.environ["NKI_ARTIFACTS_DIR"] = str(compiled.resolve())
    spec = importlib.util.spec_from_file_location("math_candidate_" + uuid.uuid4().hex, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if backend == "simulate":
        return nki.simulate(module.calculate), None, None
    from nrtpy import SpikeModel, SpikeTensor
    module.calculate(*fixture(task, 3))
    neffs = list(compiled.rglob("*.neff"))
    if len(neffs) != 1:
        raise ValueError("Expected one preserved compiled kernel")
    model = SpikeModel.load_from_neff(str(neffs[0]))
    if set(model.input_tensors_info) != set(TASKS[task]["input_names"]):
        raise ValueError("Unexpected runtime input names")
    return None, model, SpikeTensor


def validate(task, executor, model, tensor_type, directory):
    rows = []
    for seed in range(16):
        arrays = fixture(task, seed)
        originals = tuple(a.copy() for a in arrays)
        if model is None:
            actual = np.asarray(executor(*arrays)).copy()
            readback = arrays
        else:
            names = TASKS[task]["input_names"]
            inputs = {name: tensor_type.from_numpy(a, name=name) for name, a in zip(names, arrays)}
            actual = execute_runtime(model, inputs, runtime_outputs(model))
            readback = [np.asarray(inputs[name].numpy()).view(np.float32).reshape(a.shape)
                        for name, a in zip(names, arrays)]
        row = dict(seed=seed, **check(task, actual, readback, originals))
        artifact = directory / f"case-{seed:02d}.npz"
        np.savez_compressed(artifact, actual=actual, **dict(zip(TASKS[task]["input_names"], originals)))
        row.update(output_file=artifact.name, output_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        rows.append(row)
    dump(directory / "checks.json", rows)
    if not all(r["score"] == 1 for r in rows):
        raise ValueError(f"Equation checks: {sum(r['score'] for r in rows)}/16; refusing timing")
    return rows


def benchmark(task, model, tensor_type, args):
    arrays = fixture(task, 3)
    names = TASKS[task]["input_names"]
    inputs = {name: tensor_type.from_numpy(a, name=name) for name, a in zip(names, arrays)}
    outputs = runtime_outputs(model)
    rows = []
    for repeat in range(args.runs):
        checks = []
        for phase in ("before", "after"):
            actual = execute_runtime(model, inputs, outputs)
            readback = [np.asarray(inputs[name].numpy()).view(np.float32).reshape(a.shape)
                        for name, a in zip(names, arrays)]
            checked = check(task, actual, readback, arrays)
            checks.append(dict(phase=phase, **checked))
            if not checked["score"]:
                raise ValueError("Timing pre/post equation check failed")
            if phase == "before":
                timing = timing_stats(model.benchmark(inputs=inputs, outputs=outputs,
                    warmup_iter=20, benchmark_iter=args.iters, mode="device"))
                if timing["iterations"] != args.iters:
                    raise ValueError("Unexpected timing sample count")
        rows.append(dict(repeat=repeat, checks=checks, timing=timing))
    return rows


def average(rows):
    return statistics.mean(v for r in rows for v in r["timing"]["durations_ms"])


def run_task(task, args, out):
    out.mkdir()
    prepared = {}
    attempts = []
    programs = sources(task)
    proposal = None
    if args.program:
        proposal = json.loads(args.program.read_text())["program"]
        programs = {"baseline": programs["baseline"], "agent-proposal": lower(task, proposal)}
        (out / "proposal.json").write_bytes(args.program.read_bytes())
    for form, source in programs.items():
        directory = out / form
        directory.mkdir()
        row = dict(task=task, form=form, status="pending", correctness_score=None,
                   throughput_ratio=None, source_sha256=hashlib.sha256(source.encode()).hexdigest())
        try:
            executor, model, tensor_type = load(task, form, source, directory, args.backend,
                                                proposal if form == "agent-proposal" else None)
            row["checks"] = validate(task, executor, model, tensor_type, directory)
            row.update(status="correct_not_benchmarked", correctness_score=1.0)
            prepared[form] = (model, tensor_type)
        except Exception:
            row.update(status="error_or_rejected", error=traceback.format_exc())
            checks_path = directory / "checks.json"
            if checks_path.exists():
                checks = json.loads(checks_path.read_text())
                row.update(checks=checks, correctness_score=sum(r["score"] for r in checks) / len(checks),
                           status="correctness_failed")
        attempts.append(row)
        print(f"{task}/{form}: {row['status']}", flush=True)
    winner = "baseline"
    if args.backend == "device" and len(prepared) == 2:
        candidate = next(name for name in prepared if name != "baseline")
        try:
            before = benchmark(task, *prepared["baseline"], args)
            timed = benchmark(task, *prepared[candidate], args)
            after = benchmark(task, *prepared["baseline"], args)
            for form in prepared:
                post = out / form / "post-timing-suite"
                post.mkdir()
                validate(task, None, *prepared[form], post)
            drift = abs(average(after) - average(before)) / average(before)
            ratio = average(before + after) / average(timed)
            evidence = dict(baseline_before=before, candidate=timed, baseline_after=after,
                            baseline_drift_fraction=drift, stable=drift <= .1, observed_ratio=ratio)
            dump(out / "timing.json", evidence)
            attempts[1].update(status="benchmarked" if drift <= .1 else "unstable_baseline",
                               throughput_ratio=ratio if drift <= .1 else None)
            if drift <= .1 and ratio > 1:
                winner = candidate
        except Exception:
            attempts[1].update(status="benchmark_error", error=traceback.format_exc())
    for row in attempts:
        with (out / "attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    summary = dict(task=task, equation=TASKS[task]["equation"], structure=TASKS[task]["structure"],
                   backend=args.backend, winner=winner if args.backend == "device" else None,
                   attempts=attempts, note="No global optimality or statistical significance claim")
    dump(out / "results.json", summary)
    (out / "next-feedback.txt").write_text(json.dumps(summary, indent=2) +
        "\nPreserve this task's equation and inputs. Retain baseline unless a correct, stable measured candidate wins.\n")
    return all(r["correctness_score"] == 1.0 and r["status"] not in ("benchmark_error", "unstable_baseline")
               for r in attempts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", *TASKS], default="all")
    parser.add_argument("--backend", choices=["cpu", "simulate", "device"], default="cpu")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--program", type=Path, help="Model-proposed JSON graph; lowered by trusted code")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / f"data/math-harness-{uuid.uuid4().hex}")
    args = parser.parse_args()
    if args.program and args.task == "all":
        parser.error("A proposal must target one explicit task")
    if args.runs < 5 or args.iters < 20:
        parser.error("At least 5 repeats and 20 timing samples required")
    if args.backend == "device":
        if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
            parser.error("Set confirmed exclusive cores and stop Qwen before timing")
        import subprocess
        devices = json.loads(subprocess.check_output(["neuron-ls", "--json-output"]))
        if any(d.get("neuron_processes") for d in devices):
            parser.error("Neuron device has existing owners; stop your Qwen server or wait for an idle seat")
    args.out.mkdir(parents=True, exist_ok=False)
    snapshots = args.out / "sources"
    snapshots.mkdir()
    for name in ("math_harness.py", "math_tasks.py", "math_program.py", "benchmark_update.py"):
        (snapshots / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    dump(args.out / "context.json", dict(hostname=platform.node(), python=platform.python_version(),
         cores=os.environ.get("NEURON_RT_VISIBLE_CORES"), backend=args.backend,
         runs=args.runs, iters=args.iters, tasks=TASKS,
         source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("math_harness.py", "math_tasks.py", "math_program.py", "benchmark_update.py")}))
    tasks = list(TASKS) if args.task == "all" else [args.task]
    passed = [run_task(task, args, args.out / task) for task in tasks]
    (args.out / "run-note.md").write_text(
        f"# Distinct Math Tasks\n\nBackend: {args.backend}; tasks: {tasks}. "
        "Each has 16 deterministic cases, a baseline and one reviewed candidate. "
        "FP64 references use exact saved FP32 inputs; fixed error budget is 1e-6 + "
        "2e-6 times the sum of absolute contributing terms. Shape, finite FP32 output "
        "and unchanged inputs are required. This is equation validation, not MuJoCo rollout validation.\n\n"
        f"Device timing: seed 3 only, {args.runs} repeats of {args.iters} samples, 20 warmups. "
        "Baseline-before/candidate/baseline-after; >10% baseline drift invalidates reward. "
        "Device-only timing excludes compilation, transfers and readback; no end-to-end claim. "
        "CPU results test reference/checker plumbing, not NKI execution or speed. "
        "Without --program templates are human supplied. With --program the model's bounded "
        "operation graph is lowered by trusted code. This runner does not itself call Qwen. "
        "Sources, input/output hashes, every attempt, checks and timing samples are retained.\n")
    print(f"Artifacts: {args.out}")
    return int(not all(passed))


if __name__ == "__main__":
    raise SystemExit(main())
