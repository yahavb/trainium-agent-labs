"""Repeat frozen winners on the development benchmark, then package evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import uuid

from frozen_math_eval import manifest, sha, verified
from grip_loop import ROOT
from math_program import lower


def independent_repeat(args):
    from full_plane_loop import FullLoop
    pinned = manifest(args.bundle)
    args.out.mkdir(parents=True, exist_ok=False)
    controller = FullLoop(args)
    setup = args.out / "commands.log"
    controller.stop_model(setup)
    controller.command(["kubectl", "exec", args.seat, "--", "mkdir", "-p", controller.remote], setup)
    for name in ("math_harness.py", "math_tasks.py", "math_program.py", "benchmark_update.py",
                 "contact.py", "update_reference.py"):
        controller.upload(ROOT / name, "/workspace/projects/03-contact-physics/" + name, setup)
    summaries = {}
    for task, info in pinned["tasks"].items():
        proposal = verified(args.bundle, info["proposal"], info["proposal_sha256"])
        generated = lower(task, json.loads(proposal.read_text())["program"])
        if hashlib.sha256(generated.encode()).hexdigest() != info["kernel_sha256"]:
            raise ValueError("Frozen proposal no longer lowers to the selected source")
        remote_proposal = controller.remote + f"/{task}-proposal.json"
        controller.upload(proposal, remote_proposal, setup)
        remote_output = controller.remote + "/" + task
        try:
            controller.remote_command(["env", "NEURON_RT_VISIBLE_CORES=" + args.cores,
                "python", "-u", "math_harness.py", "--task", task, "--backend", "device",
                "--program", remote_proposal, "--runs", "5", "--iters", "200", "--out", remote_output], setup)
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 1:
                raise
        controller.download(remote_output, args.out / task, setup)
        report = json.loads((args.out / task / task / "results.json").read_text())
        row = report["attempts"][1]
        if row["source_sha256"] != info["kernel_sha256"]:
            raise ValueError("Repeat timed a different candidate")
        context = json.loads((args.out / task / "context.json").read_text())
        if context["backend"] != "device" or context["cores"] != args.cores:
            raise ValueError("Unexpected repeat backend or core allocation")
        if row["status"] != "benchmarked" or row["correctness_score"] != 1.0:
            raise ValueError(f"{task} repeat failed; preserve report and do not claim replication")
        timing = json.loads((args.out / task / task / "timing.json").read_text())
        means = [r["timing"]["mean_ms"] for r in timing["candidate"]]
        summaries[task] = dict(throughput_ratio=row["throughput_ratio"],
            baseline_drift_fraction=timing["baseline_drift_fraction"],
            candidate_repeat_mean_ms=dict(min=min(means), median=statistics.median(means), max=max(means)),
            source_sha256=row["source_sha256"])
        print(f"{task}: independent observed ratio={row['throughput_ratio']:.6f}x", flush=True)
    (args.out / "summary.json").write_text(json.dumps(dict(
        frozen_manifest_sha256=sha(args.bundle / "manifest.json"), tasks=summaries), indent=2) + "\n")
    return args.out


def package(bundle, destination, repeat=None):
    pinned = manifest(bundle)
    run = Path(pinned["development_run"])
    final = json.loads((bundle / "assessment/grading/results.json").read_text())
    if final["context"]["manifest_sha256"] != sha(bundle / "manifest.json"):
        raise ValueError("Final report does not match frozen manifest")
    if any(s["passed"] != s["expected"] or s["measured"] != s["expected"] for s in final["tasks"].values()):
        raise ValueError("Final assessment is not completely passing")
    repeat_summary = None
    if repeat:
        repeat_summary = json.loads((repeat / "summary.json").read_text())
        if repeat_summary["frozen_manifest_sha256"] != sha(bundle / "manifest.json"):
            raise ValueError("Repeat belongs to another frozen selection")
        for task, info in pinned["tasks"].items():
            if repeat_summary["tasks"][task]["source_sha256"] != info["kernel_sha256"]:
                raise ValueError("Repeat candidate differs from frozen winner")
    destination.mkdir(parents=True, exist_ok=False)
    source = destination / "source"
    source.mkdir()
    for path in [*ROOT.glob("*.py"), *ROOT.glob("*.md")]:
        if path.name == "submission.md":
            continue
        shutil.copy2(path, source / path.name)
    if (ROOT / "submission.md").exists():
        shutil.copy2(ROOT / "submission.md", destination / "submission.md")
    if (ROOT / "submission-assets").exists():
        shutil.copytree(ROOT / "submission-assets", destination / "submission-assets")
    ignored = shutil.ignore_patterns("compiled", "__pycache__", "*.pyc")
    shutil.copytree(run, destination / "development", ignore=ignored)
    shutil.copytree(bundle, destination / "final-evaluation", ignore=ignored)
    if repeat:
        shutil.copytree(repeat, destination / "independent-repeat", ignore=ignored)
    lines = ["# One-Page Run Note", "",
        "Objective: a bounded, checker-guided Qwen agent optimizes distinct Trainium physics calculations. "
        "Qwen chooses operation graphs from equations, source and measured feedback; trusted lowering emits NKI. "
        "No weight training or unrestricted compiler discovery.", "",
        "Hardware: seat-260 on Trainium2 (trn2.48xlarge), confirmed visible cores 0,1, "
        "Python 3.13.7, installed NKI 0.6.0. Qwen/Qwen3-8B is stopped during device timing.", "",
        "Development: four proposals across spring F=-(k*x+c*v) and net-force sum(F_i). "
        "Attempt 0 spring had a double-negation error (0/16); attempt 1 net-force passed; "
        "attempt 2 spring corrected its sign after feedback and passed; attempt 3 repeated "
        "the net-force graph and was rejected as a duplicate without another benchmark.", "",
        "| Task | Development Ratio | Independent Repeat | Final Holdout |", "| --- | ---: | ---: | ---: |"]
    for task, info in pinned["tasks"].items():
        row = info["development_attempt"]
        replicated = f"{repeat_summary['tasks'][task]['throughput_ratio']:.6f}x" if repeat_summary else "Pending"
        final_count = final["tasks"][task]
        lines.append(f"| {task} | {row['throughput_ratio']:.6f}x | {replicated} | {final_count['passed']}/{final_count['expected']} |")
    lines += ["", "Timing: five repeats, 20 warmups and 200 device samples per repeat, on development seed 3. "
        "Original baseline is measured before and after the candidate; >10% drift invalidates the reward. "
        "Both kernels are checked on 16 development cases and before/after timing. "
        "Device timing excludes compilation, loading, transfers and readback.", ""]
    for task, info in pinned["tasks"].items():
        initial = json.loads((run / f"attempt-{info['development_attempt']['attempt']:03d}/device" /
                              task / "timing.json").read_text())
        means = [r["timing"]["mean_ms"] for r in initial["candidate"]]
        lines.append(f"{task} development candidate repeat means (min/median/max ms): "
                     f"{min(means):.8f}/{statistics.median(means):.8f}/{max(means):.8f}.")
        if repeat_summary:
            spread = repeat_summary["tasks"][task]["candidate_repeat_mean_ms"]
            lines.append(f"{task} independent repeat means (min/median/max ms): "
                         f"{spread['min']:.8f}/{spread['median']:.8f}/{spread['max']:.8f}.")
    lines += ["", "Final correctness: source/input/checker hashes were frozen before one invocation per "
        "held-out case; 32 unique unseen inputs per task passed (64/64), unchanged gates, no Qwen feedback. "
        "This tests new values in the same equations, distributions and shapes, not new algorithms.", "",
        "Checker accepts finite FP32 outputs with exact shape, unchanged inputs and error <= "
        "1e-6 + 2e-6 times the sum of absolute contributing terms, against FP64 equations on saved inputs. "
        "It rejects wrong signs/values, altered inputs, invalid shapes/dtypes and nonfinite outputs. "
        "Compilation errors and duplicates are unmeasured, not correctness scores.", "",
        "Evidence: complete four-attempt generation history, requests/replies/hypotheses, lowered sources, "
        "input/output snapshots, pre/post checks and raw timing samples are included. Compiled NEFFs are "
        "omitted; reproduce with the recorded sources and installed SDK. No statistical-significance, "
        "global-optimality or full-simulator claim. Independent benchmark replication is " +
        ("recorded above." if repeat_summary else "still pending; this package is provisional.")]
    (destination / "RUN_NOTE.md").write_text("\n".join(lines) + "\n")
    (destination / "README.md").write_text(
        "# Submission Evidence\n\nStart with submission.md (when included), then RUN_NOTE.md. Checker/reasoning: source/math_tasks.py "
        "and source/DISTINCT_MATH.md. All four attempts: development/attempts.jsonl. "
        "Final holdout report: final-evaluation/assessment/grading/results.json. "
        "Agent: source/math_agent_loop.py; model request: source/qwen_math.py; "
        "trusted lowering: source/math_program.py. API and runtime dependencies require the "
        "provided Neuron seat environment. Older contact/gripping work is documented in "
        "source/AWS_PROGRESS.md; this package's complete attempt-history claim refers to the "
        "four-proposal distinct-math experiment, not every earlier exploratory run.\n")
    files = {str(path.relative_to(destination)): sha(path) for path in sorted(destination.rglob("*")) if path.is_file()}
    (destination / "SHA256SUMS.json").write_text(json.dumps(files, indent=2) + "\n")
    archive = shutil.make_archive(str(destination), "zip", root_dir=destination.parent, base_dir=destination.name)
    print(f"Submission directory: {destination}\nArchive: {archive}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repeat", action="store_true", help="Repeat frozen winners on original benchmark before packaging")
    parser.add_argument("--repeat-report", type=Path)
    parser.add_argument("--seat", default="seat-260")
    parser.add_argument("--cores", default="0,1")
    parser.add_argument("--minutes", type=float, default=15)
    parser.add_argument("--out", type=Path, default=ROOT / f"data/math-repeat-{uuid.uuid4().hex}")
    parser.add_argument("--package-out", type=Path, default=ROOT / f"data/submission-math-{uuid.uuid4().hex}")
    args = parser.parse_args()
    if args.repeat and args.repeat_report:
        parser.error("Choose a new repeat or an existing repeat report")
    if not 0 < args.minutes <= 120 or not args.cores or any(not v.isdigit() for v in args.cores.split(",")):
        parser.error("Set positive budget and confirmed exclusive core IDs")
    args.bundle = args.bundle.resolve()
    repeated = independent_repeat(args) if args.repeat else args.repeat_report
    package(args.bundle, args.package_out, repeated)


if __name__ == "__main__":
    main()
