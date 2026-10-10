"""Validate independent-world batching before measuring launch amortization."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import traceback
import uuid

import numpy as np
from batch_reference import fixed_batch, prepare_batch
from benchmark_update import execute_runtime, runtime_outputs, timing_stats
from contact import build_fixture, diagnostics, oracle


def check_result(actual, expected, fixtures, originals, inputs):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("Invalid output shape or nonfinite values")
    errors = np.max(np.abs(actual - expected), axis=0)
    limits = 1e-4 * np.maximum(1, np.max(np.abs(expected), axis=0))
    count = len(fixtures[0]["b"])
    padding_ok = bool(np.all(np.abs(actual[count:]) <= 1e-6))
    unchanged = all(np.array_equal(x, y) for x, y in zip(inputs, originals))
    physics = [diagnostics(f["A"], f["b"], actual[:count, w], oracle(f["A"], f["b"]), f)
               for w, f in enumerate(fixtures)]
    return dict(score=int(np.all(errors <= limits) and padding_ok and unchanged),
                per_world_update_error=errors.tolist(), update_error_limits=limits.tolist(),
                padding_ok=padding_ok, inputs_unchanged=unchanged,
                physics_checks=physics, physics_passes=sum(p["passed"] for p in physics))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "simulate", "device"), default="simulate")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--contacts", type=int, default=33)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--scene", choices=("plane", "pairs", "stack"), default="plane")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--candidate", type=Path, help="Manually reviewed candidate only; not a sandbox")
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--require-physics", action="store_true", help="Reject benchmark if any physics case fails")
    args = parser.parse_args()
    if not 1 <= args.batch <= 16 or not 1 <= args.contacts <= 128 or min(args.steps, args.runs, args.warmup, args.iters) < 1:
        parser.error("Positive counts required; batch<=16 and contacts<=128")
    if args.backend == "device" and not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        parser.error("Set NEURON_RT_VISIBLE_CORES to confirmed free cores")
    if args.candidate:
        if args.backend == "cpu":
            parser.error("NKI candidates require simulate or device backend")
        if hashlib.sha256(args.candidate.read_bytes()).hexdigest() != args.candidate_sha256:
            parser.error("Candidate must match its manually reviewed SHA256")
        args.require_physics = True
    elif args.candidate_sha256:
        parser.error("Candidate hash supplied without candidate")
    run_id = uuid.uuid4().hex
    out = Path("data") / f"batch-{args.backend}-{run_id}"
    out.mkdir(parents=True)
    context = dict(run_id=run_id, hostname=platform.node(),
                   visible_cores=os.environ.get("NEURON_RT_VISIBLE_CORES"),
                   **{name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()})
    context["source_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ("check_batch.py", "batch_reference.py", "nki_contact_batch.py", "benchmark_update.py", "contact.py", "update_reference.py")}
    sources = out / "sources"
    sources.mkdir()
    for name in context["source_sha256"]:
        (sources / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    if args.candidate:
        snapshot = sources / "candidate.py"
        snapshot.write_bytes(args.candidate.read_bytes())
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != args.candidate_sha256:
            raise ValueError("Candidate changed during snapshot creation")
        context["source_sha256"]["candidate.py"] = args.candidate_sha256
    results = []
    try:
        fixtures = [build_fixture(args.scene, args.contacts, w) for w in range(args.batch)]
        inputs = prepare_batch(fixtures)
        originals = [x.copy() for x in inputs]
        expected = fixed_batch(*inputs, args.steps)
        executor = fixed_batch
        if args.backend != "cpu":
            os.environ["NKI_ARTIFACTS_DIR"] = str((out / "compiled").resolve())
            import nki
            if args.candidate:
                spec = importlib.util.spec_from_file_location("reviewed_plane_candidate", snapshot)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                contact_batch = module.contact_batch
            else:
                from nki_contact_batch import contact_batch
            context["nki_version"] = getattr(nki, "__version__", "unknown")
            executor = nki.simulate(contact_batch) if args.backend == "simulate" else contact_batch
        print(f"[{args.backend}] {args.batch} independent {args.scene} worlds, C={args.contacts}, steps={args.steps}", flush=True)
        actual = np.asarray(executor(*inputs, args.steps)).copy()
        initial_check = check_result(actual, expected, fixtures, originals, inputs)
        context["initial_check"] = initial_check
        if not initial_check["score"]:
            raise ValueError("Initial update correctness failed; refusing benchmark")
        if args.require_physics and initial_check["physics_passes"] != args.batch:
            raise ValueError("Initial physics correctness failed; refusing benchmark")
        if args.backend == "device":
            from nrtpy import SpikeModel, SpikeTensor
            neffs = sorted((out / "compiled").rglob("*.neff"))
            if len(neffs) != 1:
                raise ValueError(f"Expected one NEFF, found {len(neffs)}")
            context["neff_sha256"] = hashlib.sha256(neffs[0].read_bytes()).hexdigest()
            model = SpikeModel.load_from_neff(str(neffs[0]))
            names = ("a_transposed", "bias", "initial", "alpha")
            context["compiled_inputs"] = list(model.input_tensors_info)
            if set(model.input_tensors_info) != set(names):
                raise ValueError(f"Unexpected compiled input names: {context['compiled_inputs']}")
            device_inputs = {name: SpikeTensor.from_numpy(array, name=name) for name, array in zip(names, inputs)}
            device_outputs = runtime_outputs(model)
        for repeat in range(args.runs):
            entry = dict(repeat=repeat, score=0, timing_ok=False)
            try:
                actual = execute_runtime(model, device_inputs, device_outputs) if args.backend == "device" else np.asarray(executor(*inputs, args.steps))
                entry.update(check_result(actual, expected, fixtures, originals, inputs))
                if args.require_physics and entry["physics_passes"] != args.batch:
                    entry["score"] = 0
                before = actual.copy()
                if not entry["score"]:
                    raise ValueError("Numerical gate failed")
                if args.backend == "device":
                    for mode in ("device", "host"):
                        entry[f"{mode}_stats"] = timing_stats(model.benchmark(inputs=device_inputs, outputs=device_outputs,
                            warmup_iter=args.warmup, benchmark_iter=args.iters, mode=mode))
                    after = execute_runtime(model, device_inputs, device_outputs)
                    readback = [np.asarray(device_inputs[name].numpy()).view(np.float32) for name in names]
                    post = check_result(after, expected, fixtures, originals, readback)
                    entry["post_benchmark_check"] = post
                    entry["score"] = min(entry["score"], post["score"])
                    if args.require_physics and post["physics_passes"] != args.batch:
                        entry["score"] = 0
                    if not entry["score"]:
                        raise ValueError("Post-benchmark correctness failed")
                    entry["timing_ok"] = True
                    for mode in ("device", "host"):
                        milliseconds = entry[f"{mode}_stats"]["mean_ms"]
                        entry[f"{mode}_worlds_per_second"] = args.batch * 1000 / milliseconds
                    print(f"Run {repeat + 1}: device={entry['device_stats']['mean_ms']:.6f} ms/batch; host={entry['host_stats']['mean_ms']:.6f} ms/batch", flush=True)
                file = out / f"outputs-r{repeat}.npz"
                np.savez_compressed(file, before=before, after=after if args.backend == "device" else actual,
                                    expected=expected)
                entry["output_file"] = file.name
                entry["output_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
                print(f"  update score={entry['score']}; physics={entry['physics_passes']}/{args.batch}", flush=True)
            except Exception:
                entry["error"] = traceback.format_exc()
                print(entry["error"], flush=True)
            results.append(entry)
            with Path("data/batch-attempts.jsonl").open("a") as log:
                log.write(json.dumps(dict(context=context, **entry), allow_nan=False) + "\n")
            if "error" in entry:
                break
    except Exception:
        context["error"] = traceback.format_exc()
        print(context["error"], flush=True)
        with Path("data/batch-attempts.jsonl").open("a") as log:
            log.write(json.dumps(dict(context=context, score=0, timing_ok=False), allow_nan=False) + "\n")
    report = dict(context=context, results=results)
    for mode in ("device", "host"):
        times = [r[f"{mode}_stats"]["mean_ms"] for r in results if r["timing_ok"]]
        if times:
            report[f"{mode}_mean_spread_ms"] = dict(min=min(times), median=float(np.median(times)), max=max(times))
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "run-note.md").write_text(f"""# Independent-World Launch Baseline

Host: {context['hostname']}; cores: {context['visible_cores']}; backend: {args.backend}.
Run: {run_id}; scene: {args.scene}; seeds: 0..{args.batch - 1}.
Batch: {args.batch}; contacts/world: {args.contacts}; fixed steps: {args.steps}.
Runs completed: {len(results)}/{args.runs}; update passes: {sum(r['score'] for r in results)}.
Device mean spread (ms/batch): {report.get('device_mean_spread_ms', 'not measured')}.
Host mean spread (ms/batch): {report.get('host_mean_spread_ms', 'not measured')}.
Warmups/mode/run: {args.warmup}; measured iterations/mode/run: {args.iters}.

Each world has a distinct matrix, bias and step size. Worlds execute serially in
one kernel launch. This tests launch amortization, not multi-core scaling or an
optimized parallel algorithm. Device timing excludes compilation/loading; host
timing uses preallocated inputs/outputs and excludes fixture packing/readback.
Score tests fixed updates, padding and input preservation. Full physics checks
are recorded per world and may fail, especially for stacks. No time-to-accuracy
or converged-solver speedup is claimed. CPU/simulator timing is not hardware data.
All samples, checks and source hashes: results.json.
Every attempt is appended to data/batch-attempts.jsonl.
Full-physics gate required for this run: {args.require_physics}.
When enabled, every world must pass physics before and after timing; a candidate
always enables this gate. Reviewed source snapshots and outputs are saved.
""")
    print(f"Artifacts: {out}", flush=True)
    return int(len(results) != args.runs or any(not r["score"] or "error" in r for r in results) or "error" in context)


if __name__ == "__main__":
    raise SystemExit(main())
