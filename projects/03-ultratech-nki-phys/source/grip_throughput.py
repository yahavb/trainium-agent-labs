"""Public-only execution and separate trusted grading for gripping throughput."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time
import traceback
import uuid

import numpy as np

from batch_reference import fixed_batch, prepare_batch
from benchmark_update import execute_runtime, runtime_outputs, timing_stats


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def public_cases(public):
    manifest = json.loads((public / "manifest.json").read_text())
    if digest(public / "contract.json") != manifest["contract_sha256"]:
        raise ValueError("Public contract hash mismatch")
    records = manifest["cases"]
    ids = [row["case_id"] for row in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate public cases")
    fixtures = []
    for record in records:
        path = public / Path(record["input_file"]).name
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"A", "b"}:
                raise ValueError("Candidate inputs must only contain A and b")
            a, b = arrays["A"].copy(), arrays["b"].copy()
        if (b.ndim != 1 or len(b) != 8 or a.shape != (len(b), len(b)) or
                a.dtype != np.float32 or b.dtype != np.float32 or
                not np.isfinite(a).all() or not np.isfinite(b).all()):
            raise ValueError("Invalid public FP32 QP")
        fixtures.append(dict(A=a, b=b))
    return records, fixtures


def check_output(value, expected_shape):
    if value.shape != expected_shape or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("Invalid output shape/dtype or nonfinite forces")


def execute(public, out, backend, batch, steps, runs, warmup, iters, candidate=None, candidate_sha256=None):
    if batch not in (1, 2, 4, 8, 16) or min(steps, runs, warmup, iters) < 1:
        raise ValueError("Batch must be 1/2/4/8/16 and counts positive")
    if backend == "device" and not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Set NEURON_RT_VISIBLE_CORES to confirmed available cores")
    if candidate is not None:
        if backend == "cpu":
            raise ValueError("CPU backend cannot execute an NKI candidate; use simulate")
        if not candidate_sha256 or digest(candidate) != candidate_sha256:
            raise ValueError("Explicit reviewed candidate SHA256 must match before loading code")
    elif candidate_sha256:
        raise ValueError("Candidate hash supplied without candidate")
    records, fixtures = public_cases(public)
    if len(records) % batch:
        raise ValueError("Batch must divide the full public suite; no dropped cases")
    out.mkdir(parents=True, exist_ok=False)
    context = dict(run_id=uuid.uuid4().hex, backend=backend, batch=batch, steps=steps,
                   runs=runs, warmup=warmup, iters=iters, host=platform.node(),
                   python=platform.python_version(), numpy=np.__version__,
                   visible_cores=os.environ.get("NEURON_RT_VISIBLE_CORES"),
                   public_manifest_sha256=digest(public / "manifest.json"),
                   contract_sha256=digest(public / "contract.json"),
                   public_input_sha256={row["case_id"]: digest(public / Path(row["input_file"]).name)
                                        for row in records},
                   source_sha256={name: digest(Path(__file__).with_name(name)) for name in
                                  ("grip_throughput.py", "batch_reference.py", "update_reference.py",
                                   "nki_contact_batch.py", "benchmark_update.py")})
    source = out / "sources"
    source.mkdir()
    for name in context["source_sha256"]:
        (source / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    if candidate is not None:
        context["candidate_sha256"] = candidate_sha256
        context["source_sha256"]["generated_candidate.py"] = candidate_sha256
        (source / "generated_candidate.py").write_bytes(candidate.read_bytes())
        if digest(source / "generated_candidate.py") != candidate_sha256:
            raise ValueError("Candidate changed during snapshot creation")
    entries = []
    model = None
    names = ("a_transposed", "bias", "initial", "alpha")
    try:
        executor = fixed_batch
        if backend != "cpu":
            os.environ["NKI_ARTIFACTS_DIR"] = str((out / "compiled").resolve())
            import nki
            if candidate is None:
                from nki_contact_batch import contact_batch
            else:
                # Operator-reviewed source only; this is not a generated-code sandbox.
                spec = importlib.util.spec_from_file_location("reviewed_grip_candidate", source / "generated_candidate.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                contact_batch = module.contact_batch
            context["nki_version"] = getattr(nki, "__version__", "unknown")
            executor = nki.simulate(contact_batch) if backend == "simulate" else contact_batch
        for group in range(len(records) // batch):
            start = group * batch
            began = time.perf_counter()
            inputs = prepare_batch(fixtures[start:start + batch])
            packing_seconds = time.perf_counter() - began
            originals = [array.copy() for array in inputs]
            if backend == "device":
                from nrtpy import SpikeModel, SpikeTensor
                if model is None:
                    began = time.perf_counter()
                    first = np.asarray(executor(*inputs, steps)).copy()
                    context["first_jit_call_seconds"] = time.perf_counter() - began
                    check_output(first, inputs[1].shape)
                    if not all(np.array_equal(x, y) for x, y in zip(inputs, originals)):
                        raise ValueError("JIT mutated public inputs")
                    neffs = list((out / "compiled").rglob("*.neff"))
                    if len(neffs) != 1:
                        raise ValueError("Expected exactly one compiled NEFF")
                    context["neff_sha256"] = digest(neffs[0])
                    model = SpikeModel.load_from_neff(str(neffs[0]))
                    if set(model.input_tensors_info) != set(names):
                        raise ValueError("Unexpected compiled input interface")
                device_inputs = {name: SpikeTensor.from_numpy(array, name=name) for name, array in zip(names, inputs)}
                device_outputs = runtime_outputs(model)
            for run in range(runs):
                entry = dict(group=group, run=run, case_ids=[r["case_id"] for r in records[start:start + batch]],
                             packing_seconds=packing_seconds, status="ungraded", execution_ok=False)
                try:
                    before = (execute_runtime(model, device_inputs, device_outputs) if backend == "device"
                              else np.asarray(executor(*inputs, steps)).copy())
                    check_output(before, inputs[1].shape)
                    if backend == "device":
                        for mode in ("device", "host"):
                            entry[f"{mode}_stats"] = timing_stats(model.benchmark(
                                inputs=device_inputs, outputs=device_outputs, warmup_iter=warmup,
                                benchmark_iter=iters, mode=mode))
                        after = execute_runtime(model, device_inputs, device_outputs)
                        readback = [np.asarray(device_inputs[name].numpy()).view(np.float32) for name in names]
                    else:
                        after = np.asarray(executor(*inputs, steps)).copy()
                        readback = inputs
                    check_output(after, inputs[1].shape)
                    entry["inputs_unchanged"] = all(np.array_equal(x, y) for x, y in zip(readback, originals))
                    entry["padding_ok"] = bool(np.all(np.abs(before[8:]) <= 1e-6) and
                                               np.all(np.abs(after[8:]) <= 1e-6))
                    file = f"outputs-g{group}-r{run}.npz"
                    np.savez_compressed(out / file, before=before, after=after)
                    entry.update(output_file=file, output_sha256=digest(out / file),
                                 execution_ok=bool(entry["inputs_unchanged"] and entry["padding_ok"]))
                except Exception:
                    entry["error"] = traceback.format_exc()
                entries.append(entry)
                append(out / "execution-attempts.jsonl", entry)
            print(f"[{backend}] batch={batch} group={group + 1}/{len(records) // batch}: outputs saved, physics pending", flush=True)
    except Exception:
        context["error"] = traceback.format_exc()
        append(out / "execution-attempts.jsonl", dict(status="execution_error", score=0, error=context["error"]))
    dump(out / "execution.json", dict(context=context, attempts=entries))
    print(f"Ungraded artifacts: {out}; no throughput claim until trusted grading", flush=True)
    return int("error" in context or not entries or any(not row["execution_ok"] for row in entries))


def aggregate_timing(entries, batch, iters, warmup, mode):
    samples = []
    for entry in entries:
        stats = entry[f"{mode}_stats"]
        durations = stats["durations_ms"]
        if (stats["iterations"] != iters or stats["warmup_iterations"] != warmup or
                len(durations) != iters or not np.isfinite(durations).all() or min(durations) <= 0):
            raise ValueError("Invalid benchmark samples or iteration counts")
        np.testing.assert_allclose(stats["mean_ms"], np.mean(durations), rtol=1e-5, atol=1e-8)
        samples.extend(durations)
    return dict(accepted_worlds_per_second=batch * len(samples) * 1000 / sum(samples),
                batch_latency_ms=dict(median=float(np.median(samples)), min=min(samples),
                                      max=max(samples), p99=float(np.percentile(samples, 99))),
                measured_launches=len(samples))


def grade(suite, result_dir):
    # Trusted evaluator only; execution never imports or sees these references.
    from grip_physics import CONTRACT, check_grip
    from grip_feedback import explain_check, next_prompt
    report = json.loads((result_dir / "execution.json").read_text())
    context, entries = report["context"], report["attempts"]
    manifest = json.loads((suite / "manifest.json").read_text())
    commitments = json.loads((suite / "commitment.json").read_text())
    for name, expected in commitments.items():
        if digest(suite / name) != expected:
            raise ValueError("Trusted suite commitment mismatch")
    if (context["public_manifest_sha256"] != digest(suite / "public/manifest.json") or
            context["contract_sha256"] != digest(suite / "public/contract.json") or
            json.loads((suite / "public/contract.json").read_text()) != CONTRACT):
        raise ValueError("Dataset or checker contract mismatch")
    for name, expected in manifest["source_sha256"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("Trusted checker source mismatch")
    for name, expected in context["source_sha256"].items():
        if Path(name).name != name or digest(result_dir / "sources" / name) != expected:
            raise ValueError("Execution source hash mismatch")
    batch, runs = context["batch"], context["runs"]
    records = manifest["cases"]
    if batch not in (1, 2, 4, 8, 16) or runs < 1 or len(records) % batch:
        raise ValueError("Invalid execution workload")
    expected_keys = {(group, run) for group in range(len(records) // batch) for run in range(runs)}
    keys = [(row["group"], row["run"]) for row in entries]
    if len(keys) != len(expected_keys) or set(keys) != expected_keys:
        raise ValueError("Missing/duplicate groups or repeats; refusing cherry-picked scoring")
    lookup = {row["case_id"]: row for row in records}
    if set(context["public_input_sha256"]) != set(lookup):
        raise ValueError("Wrong input coverage")
    for case_id, row in lookup.items():
        if digest(suite / row["input_file"]) != context["public_input_sha256"][case_id]:
            raise ValueError("Public input hash mismatch")
        for name, expected in row["sha256"].items():
            if digest(suite / name) != expected:
                raise ValueError("Trusted reference hash mismatch")
    scored = []
    for entry in entries:
        start = entry["group"] * batch
        expected_ids = [r["case_id"] for r in records[start:start + batch]]
        if entry["case_ids"] != expected_ids:
            raise ValueError("Changed case ordering or omitted workload")
        checked = dict(group=entry["group"], run=entry["run"], score=0, cases=[])
        if entry["execution_ok"] and "output_file" in entry:
            file = entry["output_file"]
            if Path(file).name != file or digest(result_dir / file) != entry["output_sha256"]:
                raise ValueError("Output hash mismatch")
            with np.load(result_dir / file, allow_pickle=False) as arrays:
                if set(arrays.files) != {"before", "after"}:
                    raise ValueError("Missing pre/post benchmark outputs")
                for value in arrays.values():
                    check_output(value, (16, batch))
                    if not np.all(np.abs(value[8:]) <= 1e-6):
                        raise ValueError("Nonzero padding")
                for world, case_id in enumerate(expected_ids):
                    with np.load(suite / lookup[case_id]["reference_file"], allow_pickle=False) as trusted:
                        fixture = dict(trusted)
                    checks = [check_grip(fixture, arrays[phase][:8, world], fixture["reference_forces"])
                              for phase in ("before", "after")]
                    checked["cases"].append(dict(case_id=case_id, passed=all(c["passed"] for c in checks),
                                                  before=checks[0], after=checks[1],
                                                  guidance={phase: explain_check(check) for phase, check in
                                                            zip(("before", "after"), checks)}))
            checked["score"] = int(bool(entry["inputs_unchanged"] and entry["padding_ok"] and
                                         all(c["passed"] for c in checked["cases"])))
        else:
            checked["error"] = entry.get("error", "Execution failed")
        if not entry.get("inputs_unchanged", False) or not entry.get("padding_ok", False):
            checked["harness_failure"] = "Input mutation or nonzero padding failed the harness gate. Preserve all inputs and zero padded output rows."
        scored.append(checked)
    all_pass = bool("error" not in context and all(row["score"] == 1 for row in scored))
    summary = dict(context=context, all_cases_pass=all_pass, scored_attempts=scored,
                   evaluator_source_sha256={name: digest(Path(__file__).with_name(name)) for name in
                                            ("grip_throughput.py", "grip_feedback.py", "grip_physics.py", "force_checker.py", "contact.py")},
                   passing_case_evaluations=sum(c["passed"] for row in scored for c in row["cases"]),
                   expected_case_evaluations=len(records) * runs,
                   throughput_eligible=bool(all_pass and context["backend"] == "device"))
    if summary["throughput_eligible"]:
        for mode in ("device", "host"):
            summary[f"{mode}_throughput"] = aggregate_timing(entries, batch, context["iters"], context["warmup"], mode)
    output = result_dir / f"grading-{uuid.uuid4().hex}"
    output.mkdir()
    dump(output / "results.json", summary)
    (output / "next-prompt.txt").write_text(next_prompt(scored) + "\n")
    for row in scored:
        append(output / "attempts.jsonl", dict(run_id=context["run_id"], **row))
    (output / "run-note.md").write_text(f"""# Gripping Throughput Evaluation

Host {context['host']}; backend {context['backend']}; visible cores {context['visible_cores']}.
Run {context['run_id']}; all {len(records)} public cases; batch {batch}; fixed updates {context['steps']}.
Repeats {runs}; accepted case evaluations {summary['passing_case_evaluations']}/{len(records) * runs}.
Warmups {context['warmup']}; measured iterations {context['iters']} per group/repeat/mode.
All-case accuracy gate: {all_pass}; throughput eligible: {summary['throughput_eligible']}.
Device throughput and batch latency spread: {summary.get('device_throughput', 'withheld')}.
Host throughput and batch latency spread: {summary.get('host_throughput', 'withheld')}.
Every workload must pass before AND after timing; failed cases are not dropped.
CPU/simulator runs are correctness tests and produce no hardware throughput metric.
Worlds execute serially inside one launch; no parallel-world or multi-core claim.
Device timing excludes compilation/loading. Host timing uses preallocated tensors,
excludes fixture packing, transfers into initial tensors and final readback; not end-to-end simulation.
Packing time is separately logged. Physics/reference extraction is offline.
Raw timing samples, saved outputs, source hashes and every score are preserved.
References were available only to the trusted evaluator. This harness does not
isolate generated code or provide cryptographic attestation of remote execution.
No private gripping generalization, Qwen run or MuJoCo integration is claimed.
""")
    print(f"Physics: {summary['passing_case_evaluations']}/{len(records) * runs}; throughput eligible={summary['throughput_eligible']}; {output}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--public", type=Path, required=True)
    run.add_argument("--out", type=Path)
    run.add_argument("--backend", choices=("cpu", "simulate", "device"), default="simulate")
    run.add_argument("--batch", type=int, default=1)
    run.add_argument("--steps", type=int, default=32)
    run.add_argument("--runs", type=int, default=5)
    run.add_argument("--warmup", type=int, default=20)
    run.add_argument("--iters", type=int, default=200)
    run.add_argument("--candidate", type=Path, help="Only operator-reviewed source; this runner is not a sandbox")
    run.add_argument("--candidate-sha256", help="Required reviewed code hash when selecting a candidate")
    check = sub.add_parser("grade")
    check.add_argument("--suite", type=Path, required=True)
    check.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "grade":
        return int(not grade(args.suite, args.results)["all_cases_pass"])
    return execute(args.public, args.out or Path("data") / f"grip-throughput-{uuid.uuid4().hex}",
                   args.backend, args.batch, args.steps, args.runs, args.warmup, args.iters,
                   args.candidate, args.candidate_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
