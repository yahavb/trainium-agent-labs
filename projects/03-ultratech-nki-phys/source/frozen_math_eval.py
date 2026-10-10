"""Freeze winners, execute held-out inputs once, and grade locally without Qwen."""

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

from math_program import lower
from math_tasks import TASKS, check, fixture


ROOT = Path(__file__).resolve().parent
CASE_COUNT = 32


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def input_signature(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def heldout_inputs(task):
    seen = {input_signature(fixture(task, seed)) for seed in range(16)}
    cases = []
    for seed in range(10000, 11000):
        arrays = fixture(task, seed)
        signature = input_signature(arrays)
        if signature in seen:
            continue
        seen.add(signature)
        cases.append((seed, arrays, signature))
        if len(cases) == CASE_COUNT:
            return cases
    raise ValueError("Could not generate sufficient unique held-out inputs")


def verified(bundle, relative, expected):
    path = (bundle / relative).resolve()
    if not path.is_relative_to(bundle.resolve()) or sha(path) != expected:
        raise ValueError(f"Frozen artifact changed or path escaped bundle: {relative}")
    return path


def manifest(bundle, expected=None):
    path = bundle / "manifest.json"
    pinned = expected or (bundle / "manifest.sha256").read_text().strip()
    if sha(path) != pinned:
        raise ValueError("Frozen manifest hash mismatch")
    value = json.loads(path.read_text())
    for relative, digest in value["support_sources"].items():
        verified(bundle, relative, digest)
        # The grader/lowerer must use the exact checker version frozen in advance.
        if sha(ROOT / Path(relative).name) != digest:
            raise ValueError("Installed evaluation support differs from frozen source")
    return value


def freeze(run, bundle):
    winners = json.loads((run / "best.json").read_text())
    attempts = [json.loads(line) for line in (run / "attempts.jsonl").read_text().splitlines()]
    selected = {}
    # Validate winners before creating any held-out data.
    for task in TASKS:
        winner = winners[task]
        if winner["winner"] != "agent-proposal":
            raise ValueError("This final evaluator requires a measured selected proposal for each task")
        index = winner["attempt"]
        row = next(r for r in attempts if r["attempt"] == index and r["task"] == task)
        if row["status"] != "benchmarked" or row["correctness_score"] != 1.0:
            raise ValueError("Winner lacks correctness-gated device evidence")
        proposal_path = run / f"attempt-{index:03d}/generation/proposal.json"
        proposal = json.loads(proposal_path.read_text())
        source = lower(task, proposal["program"])
        digest = hashlib.sha256(source.encode()).hexdigest()
        if digest != row["candidate_sha256"] or sha(run / f"best-{task}.py") != digest:
            raise ValueError("Selected source no longer matches the measured winner")
        selected[task] = dict(proposal=proposal, source=source, row=row)
    bundle.mkdir(parents=True, exist_ok=False)
    support = {}
    (bundle / "support").mkdir()
    for name in ("frozen_math_eval.py", "math_tasks.py", "math_program.py"):
        path = bundle / "support" / name
        path.write_bytes((ROOT / name).read_bytes())
        support[str(path.relative_to(bundle))] = sha(path)
    tasks = {}
    for task, selection in selected.items():
        directory = bundle / task
        directory.mkdir()
        kernel = directory / "kernel.py"
        kernel.write_text(selection["source"])
        save(directory / "proposal.json", selection["proposal"])
        cases = []
        for seed, arrays, signature in heldout_inputs(task):
            path = directory / f"input-{seed}.npz"
            np.savez_compressed(path, **dict(zip(TASKS[task]["input_names"], arrays)))
            cases.append(dict(seed=seed, file=str(path.relative_to(bundle)), sha256=sha(path),
                              input_signature=signature))
        tasks[task] = dict(kernel=str(kernel.relative_to(bundle)), kernel_sha256=sha(kernel),
                           proposal=str((directory / "proposal.json").relative_to(bundle)),
                           proposal_sha256=sha(directory / "proposal.json"),
                           input_names=TASKS[task]["input_names"], cases=cases,
                           development_attempt=selection["row"])
    save(bundle / "manifest.json", dict(version=1, development_run=str(run.resolve()),
        development_seeds=list(range(16)),
        heldout_seeds={task: [c["seed"] for c in info["cases"]] for task, info in tasks.items()}, tasks=tasks,
        support_sources=support, policy="One fixed-kernel assessment; no generation, feedback or revision",
        scope="New values in the same distributions, equations and fixed shapes; not unseen-task generalization"))
    (bundle / "manifest.sha256").write_text(sha(bundle / "manifest.json") + "\n")
    (bundle / "README.md").write_text(
        "# Frozen Final Evaluation\n\nTwo measured winners frozen by source SHA256. "
        "32 unique new cases per task (64 total), seeds starting at 10000; development used 0..15. "
        "Exact duplicates of development inputs and earlier held-out inputs are excluded. "
        "Input NPZ bytes, checker/lowerer sources and manifest are pinned before execution. "
        "The original FP32 error gates are unchanged. No model is called or receives these results. "
        "This tests held-out values, not new equations, shapes or out-of-distribution conditions. "
        "No final results or speedups are claimed until device outputs are graded locally.\n")
    return bundle


def execute(bundle, out, pinned):
    value = manifest(bundle, pinned)
    out.mkdir(parents=True, exist_ok=False)
    save(out / "context.json", dict(hostname=platform.node(), python=platform.python_version(),
         cores=os.environ.get("NEURON_RT_VISIBLE_CORES"), manifest_sha256=pinned, backend="device"))
    import nki
    rows = []
    for task, info in value["tasks"].items():
        directory = out / task
        directory.mkdir()
        try:
            source = verified(bundle, info["kernel"], info["kernel_sha256"])
            proposal = json.loads(verified(bundle, info["proposal"], info["proposal_sha256"]).read_text())
            if lower(task, proposal["program"]) != source.read_text():
                raise ValueError("Frozen graph and kernel do not match")
            os.environ["NKI_ARTIFACTS_DIR"] = str((directory / "compiled").resolve())
            spec = importlib.util.spec_from_file_location("frozen_" + task.replace("-", "_"), source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception:
            rows.append(dict(task=task, status="setup_error", error=traceback.format_exc()))
            continue
        for case in info["cases"]:
            row = dict(task=task, seed=case["seed"], input_sha256=case["sha256"], status="pending")
            started = time.monotonic()
            try:
                path = verified(bundle, case["file"], case["sha256"])
                with np.load(path, allow_pickle=False) as stored:
                    arrays = [stored[name].copy() for name in info["input_names"]]
                # One kernel invocation per held-out case, including the first JIT call.
                actual = np.asarray(module.calculate(*arrays)).copy()
                artifact = directory / f"output-{case['seed']}.npz"
                np.savez_compressed(artifact, actual=actual,
                    **dict(zip(info["input_names"], arrays)))
                row.update(status="executed", output_file=str(artifact.relative_to(out)), output_sha256=sha(artifact))
            except Exception:
                row.update(status="execution_error", error=traceback.format_exc())
            row["seconds_including_jit_and_transfers"] = time.monotonic() - started
            rows.append(row)
            with (out / "execution.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(f"{task} seed={case['seed']}: {row['status']}", flush=True)
    save(out / "execution.json", rows)


def grade(bundle, outputs, out):
    value = manifest(bundle)
    context = json.loads((outputs / "context.json").read_text())
    if context["backend"] != "device" or context["manifest_sha256"] != sha(bundle / "manifest.json"):
        raise ValueError("Outputs do not match the frozen device evaluation")
    records = json.loads((outputs / "execution.json").read_text())
    seen = set()
    for record in records:
        key = (record["task"], record.get("seed"))
        if key in seen:
            raise ValueError("Duplicate final case execution")
        seen.add(key)
    rows = []
    for task, info in value["tasks"].items():
        verified(bundle, info["kernel"], info["kernel_sha256"])
        for case in info["cases"]:
            row = dict(task=task, seed=case["seed"], score=None, status="unmeasured")
            matching = [r for r in records if r["task"] == task and r.get("seed") == case["seed"]]
            if matching and matching[0]["status"] == "executed":
                record = matching[0]
                if record["input_sha256"] != case["sha256"]:
                    raise ValueError("Executed input hash mismatch")
                with np.load(verified(bundle, case["file"], case["sha256"]), allow_pickle=False) as stored:
                    originals = [stored[name].copy() for name in info["input_names"]]
                with np.load(verified(outputs, record["output_file"], record["output_sha256"]), allow_pickle=False) as stored:
                    actual = stored["actual"].copy()
                    readback = [stored[name].copy() for name in info["input_names"]]
                row.update(check(task, actual, readback, originals), status="graded")
            else:
                row["error"] = matching[0].get("error") if matching else "Missing case output"
            rows.append(row)
    summary = {task: dict(passed=sum(r["score"] == 1 for r in rows if r["task"] == task),
                          measured=sum(r["score"] is not None for r in rows if r["task"] == task),
                          expected=len(info["cases"])) for task, info in value["tasks"].items()}
    out.mkdir(parents=True, exist_ok=False)
    save(out / "results.json", dict(context=context, tasks=summary, cases=rows,
        note="Held-out values only; no model revision or final performance benchmark"))
    (out / "run-note.md").write_text(
        "# Final Frozen-Kernel Assessment\n\n" + json.dumps(context) + "\n\n" +
        json.dumps(summary) + "\n\n64 fixed held-out cases total, one invocation per case. "
        "Seeds start at 10000; exact input duplicates excluded; development used 0..15. Frozen input bytes "
        "and source/checker hashes verified. Original error budget unchanged. "
        "Results were graded locally and are not sent to Qwen. Scores with missing "
        "outputs are unmeasured, not physics failures. Compilation/transfers are "
        "included in execution durations, which are NOT kernel latency benchmarks. "
        "No unseen-equation, new-shape, private-task or final speedup claim.\n")
    print(json.dumps(summary))
    return all(s["passed"] == s["expected"] and s["measured"] == s["expected"] for s in summary.values())


def remote_evaluation(args):
    from full_plane_loop import FullLoop
    value = manifest(args.bundle)
    if (args.bundle / "assessment-started.json").exists():
        raise ValueError("This bundle already has a final assessment; do not retry or tune against these cases")
    args.out = args.bundle / "assessment"
    controller = FullLoop(args)
    controller.remote = f"/workspace/projects/03-contact-physics/data/{args.bundle.name}-assessment"
    args.out.mkdir(exist_ok=True)
    log = args.out / "commands.log"
    controller.stop_model(log)
    controller.command(["kubectl", "exec", args.seat, "--", "mkdir", "-p", controller.remote], log)
    frozen_remote = controller.remote + "/bundle"
    controller.command(["kubectl", "exec", args.seat, "--", "mkdir", "-p", frozen_remote], log)
    for name in ("manifest.json", "manifest.sha256", "README.md"):
        controller.upload(args.bundle / name, frozen_remote + "/" + name, log)
    for name in ["support", *value["tasks"]]:
        controller.upload(args.bundle / name, frozen_remote + "/" + name, log)
    for relative in value["support_sources"]:
        controller.upload(args.bundle / relative, "/workspace/projects/03-contact-physics/" + Path(relative).name, log)
    save(args.bundle / "assessment-started.json", dict(seat=args.seat, cores=args.cores,
        manifest_sha256=sha(args.bundle / "manifest.json"), policy="No re-generation or feedback"))
    remote_outputs = controller.remote + "/device-outputs"
    controller.remote_command(["env", "NEURON_RT_VISIBLE_CORES=" + args.cores, "python", "-u",
        "frozen_math_eval.py", "execute", "--bundle", frozen_remote, "--out", remote_outputs,
        "--manifest-sha256", sha(args.bundle / "manifest.json")], log)
    controller.download(remote_outputs, args.out / "device-outputs", log)
    return grade(args.bundle, args.out / "device-outputs", args.out / "grading")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    freezing = sub.add_parser("freeze")
    freezing.add_argument("--run", type=Path, required=True)
    freezing.add_argument("--out", type=Path, default=ROOT / f"data/final-math-{uuid.uuid4().hex}")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--seat", default="seat-260")
    evaluate.add_argument("--cores", required=True)
    evaluate.add_argument("--minutes", type=float, default=15)
    execution = sub.add_parser("execute")
    execution.add_argument("--bundle", type=Path, required=True)
    execution.add_argument("--out", type=Path, required=True)
    execution.add_argument("--manifest-sha256", required=True)
    grading = sub.add_parser("grade")
    grading.add_argument("--bundle", type=Path, required=True)
    grading.add_argument("--outputs", type=Path, required=True)
    grading.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "freeze":
        print(f"Frozen bundle: {freeze(args.run, args.out)}")
        return 0
    if args.mode == "evaluate":
        if not 0 < args.minutes <= 120 or not args.cores or any(not v.isdigit() for v in args.cores.split(",")):
            parser.error("Set positive budget and confirmed exclusive core IDs")
        args.bundle = args.bundle.resolve()
        return int(not remote_evaluation(args))
    if args.mode == "execute":
        if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
            parser.error("Device execution requires confirmed visible cores")
        execute(args.bundle, args.out, args.manifest_sha256)
        return 0
    return int(not grade(args.bundle, args.outputs, args.out))


if __name__ == "__main__":
    raise SystemExit(main())
