"""Correctness-gated, paired device throughput feedback loop on one exclusive seat."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
import uuid

from grip_loop import Controller, ROOT
from plane_loop import canonical, review, reviewed_forms
from optimization_harness import plan


def measured(report, runs, iters):
    context = report["context"]
    rows = report["results"]
    if context.get("error") or len(rows) != runs or context["backend"] != "device":
        raise ValueError("Incomplete device benchmark")
    initial = context["initial_check"]
    if not initial["score"] or initial["physics_passes"] != 8:
        raise ValueError("Initial correctness failed")
    samples = []
    means = []
    for row in rows:
        post = row.get("post_benchmark_check", {})
        if (row.get("error") or not row.get("timing_ok") or not row["score"] or
                row["physics_passes"] != 8 or not post.get("score") or post.get("physics_passes") != 8):
            raise ValueError("Benchmark failed pre/post physics checks")
        values = row["device_stats"]["durations_ms"]
        if len(values) != iters or not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Invalid device samples")
        samples.extend(values)
        means.append(statistics.mean(values))
    return dict(worlds_per_second=8000 * len(samples) / sum(samples),
                mean_ms=statistics.mean(samples), repeat_mean_spread_ms=dict(
                    min=min(means), median=statistics.median(means), max=max(means)),
                samples_ms=samples)


def comparison(before, candidate, after):
    results = {}
    all_base = []
    all_candidate = []
    for scene in ("plane", "pairs"):
        first, value, last = before[scene], candidate[scene], after[scene]
        baseline_samples = first["samples_ms"] + last["samples_ms"]
        baseline_rate = 8000 * len(baseline_samples) / sum(baseline_samples)
        drift = abs(last["mean_ms"] - first["mean_ms"]) / first["mean_ms"]
        results[scene] = dict(baseline_worlds_per_second=baseline_rate,
                              candidate_worlds_per_second=value["worlds_per_second"],
                              speedup=value["worlds_per_second"] / baseline_rate,
                              baseline_drift_fraction=drift,
                              candidate_repeat_spread_ms=value["repeat_mean_spread_ms"])
        all_base.extend(baseline_samples)
        all_candidate.extend(value["samples_ms"])
    baseline_rate = 8000 * len(all_base) / sum(all_base)
    candidate_rate = 8000 * len(all_candidate) / sum(all_candidate)
    return dict(scenes=results, baseline_worlds_per_second=baseline_rate,
                candidate_worlds_per_second=candidate_rate, speedup=candidate_rate / baseline_rate,
                stable_baseline=all(r["baseline_drift_fraction"] <= .1 for r in results.values()),
                note="Observed ratio, not statistical significance or global optimality")


class FullLoop(Controller):
    def start_model(self, log):
        self.remote_command(["/workspace/serve.sh"], log)

    def stop_model(self, log):
        self.remote_command(["/workspace/serve.sh", "--stop"], log)
        # Do not kill unknown runtime owners. Refuse timing until the seat is idle.
        probe = ("import json,subprocess,time; "
                 "\nfor _ in range(30):\n"
                 " devices=json.loads(subprocess.check_output(['neuron-ls','--json-output']))\n"
                 " owners=[p for d in devices for p in d.get('neuron_processes',[])]\n"
                 " if not owners: break\n"
                 " time.sleep(1)\n"
                 "else: raise RuntimeError('Neuron processes still own the device; refusing concurrent timing: '+str(owners))\n")
        self.remote_command(["python", "-c", probe], log)

    def workload(self, candidate, candidate_hash, scene, backend, directory):
        directory.mkdir()
        log = directory / "commands.log"
        command = ["python", "-u", "check_batch.py", "--backend", backend, "--scene", scene,
                   "--batch", "8", "--contacts", "33", "--steps", "32", "--require-physics"]
        if candidate:
            command += ["--candidate", candidate, "--candidate-sha256", candidate_hash]
        if backend == "device":
            command = ["env", "NEURON_RT_VISIBLE_CORES=" + self.args.cores, *command,
                       "--runs", str(self.args.runs), "--warmup", "20", "--iters", str(self.args.iters)]
        try:
            self.remote_command(command, log)
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 1 or "Artifacts: " not in log.read_text():
                raise
        lines = log.read_text().splitlines()
        remote = next(line.removeprefix("Artifacts: ") for line in reversed(lines) if line.startswith("Artifacts: "))
        local = directory / "artifacts"
        self.download("/workspace/projects/03-contact-physics/" + remote, local, log)
        report = json.loads((local / "results.json").read_text())
        context = report["context"]
        check = context.get("initial_check", {})
        rows = report["results"]
        if (context.get("error") or not check.get("score") or check.get("physics_passes") != 8 or
                not rows or any(not r["score"] or r.get("error") or r["physics_passes"] != 8 for r in rows)):
            raise ValueError("Correctness/execution failed: " + json.dumps(context.get("error") or check)[:1800])
        if backend == "device":
            if context["visible_cores"] != self.args.cores:
                raise ValueError("Unexpected core allocation")
            return measured(report, self.args.runs, self.args.iters)
        return dict(physics_passes=8, report=str(local / "results.json"))

    def suite(self, candidate, digest, backend, directory):
        directory.mkdir()
        return {scene: self.workload(candidate, digest, scene, backend, directory / scene)
                for scene in ("plane", "pairs")}

    def run(self):
        args = self.args
        self.out.mkdir(parents=True, exist_ok=False)
        setup = self.out / "setup.log"
        self.command(["kubectl", "exec", args.seat, "--", "mkdir", "-p", self.remote], setup)
        for name in ("qwen_grip.py", "nki_guidance.py", "optimization_knowledge.py", "check_batch.py",
                     "nki_contact_batch.py", "batch_reference.py", "update_reference.py", "benchmark_update.py", "contact.py"):
            self.upload(ROOT / name, f"/workspace/projects/03-contact-physics/{name}", setup)
        previous = self.remote + "/baseline.py"
        self.upload(ROOT / "nki_contact_batch.py", previous, setup)
        feedback = (ROOT / "plane-feedback.txt").read_text()
        search_forms = getattr(args, "search_forms", "copyfree,scale-fused").split(",")
        applicability = plan(reviewed_forms()["baseline"])
        (self.out / "applicability.json").write_text(json.dumps(applicability, indent=2) + "\n")
        # Explicit legacy 'fused' experiments remain available, but the default
        # search is routed through the applicability catalog.
        search_forms = [name for name in search_forms
                        if name in applicability["candidate_forms"] or name == "fused"]
        reference = self.out / "reference-copyfree.py"
        reference.write_text(reviewed_forms()["copyfree"])
        reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
        reference_remote = self.remote + "/reference-copyfree.py"
        self.upload(reference, reference_remote, setup)
        seen = {canonical(reviewed_forms()["baseline"])}
        best_ratio = 1.0
        (self.out / "best-candidate.py").write_text(reviewed_forms()["baseline"])
        for index in range(args.attempts):
            directory = self.out / f"attempt-{index:03d}"
            directory.mkdir()
            log = directory / "commands.log"
            row = dict(attempt=index, status="pending", physics_score=None, throughput_score=None)
            stage = "generation"
            began = time.monotonic()
            try:
                if index == 0 and args.initial_candidate:
                    source = args.initial_candidate.read_text()
                    candidate = args.initial_candidate
                    remote_candidate = self.remote + "/initial-candidate.py"
                    review(source)
                    self.upload(candidate, remote_candidate, log)
                    row["origin"] = str(candidate.resolve())
                else:
                    prompt = directory / "feedback.txt"
                    forms = reviewed_forms()
                    remaining = [name for name in search_forms if canonical(forms[name]) not in seen]
                    if not remaining:
                        break
                    self.start_model(log)
                    target = remaining[0]
                    prompt.write_text(feedback[:1700] + f"\nThe next untested reviewed form is {target}. "
                        "Applicable techniques for the input: " +
                        "; ".join(t["name"] + ": " + t["reason"] for t in applicability["techniques"]
                                  if t["applicable"]) + "\n" +
                        "Return this exact human-reviewed template. Do not substitute the other form "
                        "or return the previous candidate. Preserve rate multiplication. "
                        "The template is supplied by the human, not an autonomous discovery.\n" + forms[target])
                    self.upload(prompt, self.remote + "/feedback.txt", log)
                    generated = self.remote + f"/generation-{index:03d}"
                    try:
                        self.remote_command(["python", "-u", "qwen_grip.py", "--task", "plane-throughput",
                            "--feedback", self.remote + "/feedback.txt", "--previous", previous,
                            "--baseline", self.remote + "/baseline.py", "--base-url", args.base_url,
                            "--max-tokens", "1200", "--temperature", "0.4", "--out", generated], log)
                    except subprocess.CalledProcessError:
                        self.download(generated, directory / "generation", log)
                        record = json.loads((directory / "generation/generation-attempt.json").read_text())
                        raise ValueError("Qwen generation status: " + record["status"] + "; " +
                                         record.get("error", record.get("feedback", "No valid new candidate")))
                    self.download(generated, directory / "generation", log)
                    candidate = directory / "generation/candidate.py"
                    source = candidate.read_text()
                    remote_candidate = generated + "/candidate.py"
                form = review(source)
                row.update(form=form, candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
                if canonical(source) in seen:
                    raise ValueError("Duplicate form; no new timing. Select the other untested reviewed form.")
                seen.add(canonical(source))
                stage = "correctness"
                row["simulation"] = self.suite(remote_candidate, row["candidate_sha256"], "simulate", directory / "simulation")
                row["physics_score"] = 1.0
                if not (self.out / "first-correct-candidate.py").exists():
                    (self.out / "first-correct-candidate.py").write_bytes(candidate.read_bytes())
                stage = "benchmark"
                self.stop_model(log)
                before = self.suite(None, None, "device", directory / "baseline-before")
                reference_before = (self.suite(reference_remote, reference_hash, "device", directory / "copyfree-before")
                                    if form == "scale-fused" else None)
                timed = self.suite(remote_candidate, row["candidate_sha256"], "device", directory / "candidate-device")
                reference_after = (self.suite(reference_remote, reference_hash, "device", directory / "copyfree-after")
                                   if form == "scale-fused" else None)
                after = self.suite(None, None, "device", directory / "baseline-after")
                result = comparison(before, timed, after)
                if reference_before is not None:
                    result["versus_copyfree"] = comparison(reference_before, timed, reference_after)
                    result["stable_baseline"] = result["stable_baseline"] and result["versus_copyfree"]["stable_baseline"]
                row["comparison"] = result
                row["status"] = "benchmarked" if result["stable_baseline"] else "unstable_baseline"
                if result["stable_baseline"]:
                    row["throughput_score"] = result["speedup"]
                    if result["speedup"] > best_ratio:
                        best_ratio = result["speedup"]
                        (self.out / "best-candidate.py").write_bytes(candidate.read_bytes())
                        (self.out / "best.json").write_text(json.dumps(row, indent=2) + "\n")
                previous = remote_candidate
                feedback = ("Measured Trainium device throughput, all plane/pair physics gates passed.\n" +
                            json.dumps(result) + "\nImprove throughput without changing the workload or solver. "
                            "If baseline drift exceeds 10%, this ratio is not a valid performance reward.")
            except Exception as exc:
                row.update(status="error_or_rejected", error=f"{type(exc).__name__}: {exc}", failed_stage=stage)
                feedback = feedback[:900] + "\nLast attempt failed: " + row["error"]
            row["seconds"] = time.monotonic() - began
            (directory / "next-feedback.txt").write_text(feedback + "\n")
            with (self.out / "attempts.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            self.rows.append(row)
            (self.out / "run-note.md").write_text(
                f"# Full Plane Throughput Loop\n\nSeat {args.seat}; cores {args.cores}; "
                f"attempts {len(self.rows)}/{args.attempts}; budget {args.minutes} minutes.\n"
                f"Workload: plane and pairs, 8 worlds/scene, 33 contacts, 32 updates. "
                f"Timing: {args.runs} repeats, 20 warmups, {args.iters} measurements per repeat/mode.\n"
                f"Best valid observed device throughput ratio versus paired baseline: {best_ratio:.6g}x. "
                "Baseline remains best if no valid candidate exceeds 1x.\n"
                "Every proposal/rejection and pre/post correctness result is retained. Device timing "
                "excludes compile/load; host timing excludes packing/transfers/readback. Baseline is "
                "measured before and after each candidate; >10% mean-latency drift invalidates the reward. "
                "Per-repeat min/median/max and raw samples are in saved reports.\n"
                "Qwen server is stopped for timing; restarts consume the budget. Only human-reviewed "
                "AST forms execute; curated keyword documentation retrieval is saved per generation. "
                "No unrestricted discovery, weight training, statistical significance or global optimality claim.\n")
            print(f"Attempt {index}: {row['status']}; physics={row['physics_score']}; "
                  f"throughput_ratio={row['throughput_score']}", flush=True)
            if (all(canonical(reviewed_forms()[name]) in seen for name in search_forms) or time.monotonic() >= self.deadline or
                    row.get("failed_stage") == "benchmark"):
                break
        print(f"Artifacts: {self.out}; Qwen may be stopped after timing. Best measured ratio: {best_ratio:.6g}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seat", default="seat-260")
    parser.add_argument("--cores", required=True, help="Confirmed exclusive cores; this loop stops Qwen on this seat")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--initial-candidate", type=Path, help="Reuse a saved proposal after AST review")
    parser.add_argument("--search-forms", default="copyfree,scale-fused", help="Comma-separated reviewed targets")
    parser.add_argument("--out", type=Path, default=ROOT / f"data/full-plane-loop-{uuid.uuid4().hex}")
    args = parser.parse_args()
    if not 0 < args.minutes <= 120 or args.attempts < 1 or args.runs < 5 or args.iters < 20:
        parser.error("Positive budget/attempts, at least 5 repeats and 20 iterations required")
    if not args.cores or any(not value.isdigit() for value in args.cores.split(",")):
        parser.error("Use comma-separated confirmed core IDs")
    if args.base_url != "http://localhost:8000/v1":
        parser.error("This single-seat lifecycle controller requires the seat-local default Qwen endpoint")
    if any(name not in reviewed_forms() or name == "baseline" for name in args.search_forms.split(",")):
        parser.error("Search forms must be reviewed non-baseline targets")
    FullLoop(args).run()


if __name__ == "__main__":
    main()
