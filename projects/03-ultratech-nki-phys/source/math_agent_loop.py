"""Model-driven hypotheses, independent equation checks and paired device feedback."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

from full_plane_loop import FullLoop
from grip_loop import ROOT
from math_program import lower
from math_tasks import TASKS


class MathAgent(FullLoop):
    def evaluate_math(self, task, proposal, backend, remote, local, log):
        command = ["python", "-u", "math_harness.py", "--task", task, "--backend", backend,
                   "--program", proposal, "--runs", str(self.args.runs), "--iters", str(self.args.iters),
                   "--out", remote]
        if backend == "device":
            command = ["env", "NEURON_RT_VISIBLE_CORES=" + self.args.cores, *command]
        try:
            self.remote_command(command, log)
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 1:
                raise
        self.download(remote, local, log)
        report = json.loads((local / task / "results.json").read_text())
        candidate = report["attempts"][1]
        return report, candidate

    def run(self):
        args = self.args
        self.out.mkdir(parents=True, exist_ok=False)
        setup = self.out / "setup.log"
        self.command(["kubectl", "exec", args.seat, "--", "mkdir", "-p", self.remote], setup)
        for name in ("qwen_math.py", "math_harness.py", "math_program.py", "math_tasks.py",
                     "benchmark_update.py", "contact.py", "update_reference.py"):
            self.upload(ROOT / name, "/workspace/projects/03-contact-physics/" + name, setup)
            (self.out / name).write_bytes((ROOT / name).read_bytes())
        tasks = args.tasks.split(",")
        feedback = {task: "No proposal evaluated yet. Choose a technique from the computation, not the task name."
                    for task in tasks}
        previous = {}
        seen = {task: set() for task in tasks}
        best = {task: dict(winner="baseline", throughput_ratio=1.0) for task in tasks}
        for index in range(args.attempts):
            if time.monotonic() >= self.deadline:
                break
            task = tasks[index % len(tasks)]
            directory = self.out / f"attempt-{index:03d}"
            directory.mkdir()
            log = directory / "commands.log"
            row = dict(attempt=index, task=task, status="pending", correctness_score=None, throughput_ratio=None)
            started = time.monotonic()
            try:
                self.start_model(log)
                prompt = directory / "feedback.txt"
                prompt.write_text(feedback[task])
                self.upload(prompt, self.remote + "/feedback.txt", log)
                generated = self.remote + f"/generation-{index:03d}"
                command = ["python", "-u", "qwen_math.py", "--task", task, "--feedback",
                           self.remote + "/feedback.txt", "--base-url", args.base_url, "--out", generated]
                if task in previous:
                    command += ["--previous", previous[task]]
                try:
                    self.remote_command(command, log)
                except subprocess.CalledProcessError as exc:
                    if exc.returncode != 1:
                        raise
                self.download(generated, directory / "generation", log)
                record = json.loads((directory / "generation/generation-attempt.json").read_text())
                if record["status"] != "generated_not_evaluated":
                    raise ValueError(record.get("error", record["status"]))
                proposal = json.loads((directory / "generation/proposal.json").read_text())
                source = lower(task, proposal["program"])
                if hashlib.sha256(source.encode()).hexdigest() != record["candidate_sha256"]:
                    raise ValueError("Proposal source hash mismatch")
                # Renaming IDs alone is not a new executable program.
                from plane_loop import canonical
                signature = canonical(source)
                if signature in seen[task]:
                    raise ValueError("Duplicate executable graph; no new evaluation")
                seen[task].add(signature)
                row.update(hypothesis=proposal["hypothesis"], candidate_sha256=record["candidate_sha256"])
                previous[task] = generated + "/proposal.json"
                report, candidate = self.evaluate_math(task, previous[task], "simulate",
                    self.remote + f"/simulation-{index:03d}", directory / "simulation", log)
                if candidate.get("source_sha256") != record["candidate_sha256"]:
                    raise ValueError("Simulated source does not match the proposed source")
                row["correctness_score"] = candidate["correctness_score"]
                if candidate["correctness_score"] != 1.0:
                    feedback[task] = json.dumps(candidate)
                    row["status"] = "correctness_failed"
                else:
                    self.stop_model(log)
                    report, candidate = self.evaluate_math(task, previous[task], "device",
                        self.remote + f"/device-{index:03d}", directory / "device", log)
                    if candidate.get("source_sha256") != record["candidate_sha256"]:
                        raise ValueError("Benchmarked source does not match the proposed source")
                    row.update(status=candidate["status"], correctness_score=candidate["correctness_score"],
                               throughput_ratio=candidate["throughput_ratio"])
                    feedback[task] = json.dumps(candidate)[:2200] + "\nCurrent best: " + json.dumps(best[task])
                    ratio = row["throughput_ratio"]
                    if row["status"] == "benchmarked" and ratio is not None and ratio > best[task]["throughput_ratio"]:
                        best[task] = dict(winner="agent-proposal", throughput_ratio=ratio, attempt=index,
                                          hypothesis=proposal["hypothesis"])
                        (self.out / f"best-{task}.py").write_text(source)
            except Exception as exc:
                row.update(status="error_or_rejected", error=f"{type(exc).__name__}: {exc}")
                feedback[task] = feedback[task][:900] + "\nLast attempt rejected: " + row["error"]
            row["seconds"] = time.monotonic() - started
            (directory / "next-feedback.txt").write_text(feedback[task])
            with (self.out / "attempts.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            self.rows.append(row)
            (self.out / "best.json").write_text(json.dumps(best, indent=2) + "\n")
            (self.out / "run-note.md").write_text(
                f"# Model-Driven Math Agent\n\nSeat {args.seat}; confirmed cores {args.cores}; "
                f"attempts {len(self.rows)}/{args.attempts}; budget {args.minutes} minutes. Tasks {tasks}.\n"
                "Qwen selects and revises bounded operation graphs from equations, source and feedback; "
                "the controller does not prescribe target transformations or accept generated Python. "
                "Trusted lowering emits NKI. Supported operations are a finite DSL, not unrestricted discovery.\n"
                "Each task: 16 public correctness cases. Device timing: seed 3, paired original baseline, "
                f"{args.runs} repeats, {args.iters} samples, 20 warmups. Pre/post gates required; "
                ">10% baseline drift invalidates reward. Raw samples and spread derivable in timing.json. "
                "Qwen stops for timing and restarts for generation. Every rejection remains in attempts.jsonl. "
                "No model weight training, private-task generalization or statistically significant gain claimed.\n"
                f"Best observed per-task results: {json.dumps(best)}\n")
            print(f"Attempt {index} ({task}): {row['status']}; correctness={row['correctness_score']}; "
                  f"throughput_ratio={row['throughput_ratio']}", flush=True)
        print(f"Artifacts: {self.out}; Qwen may be stopped after timing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seat", default="seat-260")
    parser.add_argument("--cores", required=True)
    parser.add_argument("--tasks", default="spring,net-force")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--out", type=Path, default=ROOT / f"data/math-agent-{uuid.uuid4().hex}")
    args = parser.parse_args()
    tasks = args.tasks.split(",")
    if any(t not in TASKS for t in tasks) or len(set(tasks)) != len(tasks):
        parser.error("Use unique task names spring,net-force")
    if args.attempts < 1 or not 0 < args.minutes <= 120 or args.runs < 5 or args.iters < 20:
        parser.error("Positive budget/attempts, >=5 repeats, >=20 samples required")
    if not args.cores or any(not v.isdigit() for v in args.cores.split(",")):
        parser.error("Use confirmed exclusive comma-separated core IDs")
    if args.base_url != "http://localhost:8000/v1":
        parser.error("The model must use the seat-local endpoint for exclusive timing")
    MathAgent(args).run()


if __name__ == "__main__":
    main()
