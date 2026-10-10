"""Local controller: restricted Qwen revisions, remote simulation, trusted grading."""

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid


SEED_SHA = "23fd2f7c0580a17561f9a320d255d08979077e31c372df6af35b14703b0f06b9"
ROOT = Path(__file__).resolve().parent


def restricted_revision(source, approved):
    """Accept only a finite beta literal change in the pinned reviewed program."""
    tree = ast.parse(source)
    reference = ast.parse(approved)

    def normalize(value):
        assignments = [n for n in ast.walk(value) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "beta" for t in n.targets)]
        if len(assignments) != 1:
            raise ValueError("Expected exactly one beta assignment")
        literal = assignments[0].value
        if not isinstance(literal, ast.Constant) or type(literal.value) not in (int, float):
            raise ValueError("beta must be a numeric literal")
        beta = float(literal.value)
        if not math.isfinite(beta) or not 0 <= beta < 1:
            raise ValueError("beta must be finite and in [0,1)")
        assignments[0].value = ast.Constant(value=0.9)
        return beta, ast.dump(value, include_attributes=False)

    beta, normalized = normalize(copy.deepcopy(tree))
    _, expected = normalize(reference)
    if normalized != expected:
        raise ValueError("Only beta may change; algorithm, imports and all other syntax are frozen")
    return beta


class Controller:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        self.remote = f"/workspace/projects/03-contact-physics/data/{self.out.name}"
        self.deadline = time.monotonic() + args.minutes * 60
        self.rows = []

    def command(self, argv, log):
        remaining = int(self.deadline - time.monotonic())
        if remaining <= 2:
            raise TimeoutError("Loop time budget exhausted")
        with log.open("a") as stream:
            stream.write(json.dumps(argv) + "\n")
            stream.flush()
            subprocess.run(argv, stdout=stream, stderr=stream, check=True, timeout=remaining)

    def remote_command(self, argv, log):
        seconds = max(1, int(self.deadline - time.monotonic()) - 1)
        self.command(["kubectl", "exec", self.args.seat, "--", "bash", "-lc",
                      "cd /workspace/projects/03-contact-physics && " +
                      shlex.join(["timeout", "--kill-after=1s", f"{seconds}s", *argv])], log)

    def upload(self, local, remote, log):
        self.command(["kubectl", "cp", str(local), f"{self.args.seat}:{remote}"], log)

    def download(self, remote, local, log):
        self.command(["kubectl", "cp", f"{self.args.seat}:{remote}", str(local)], log)

    def evaluate(self, results, log):
        existing = set(results.glob("grading-*/results.json"))
        try:
            self.command([sys.executable, str(ROOT / "grip_throughput.py"), "grade", "--suite",
                          str(self.args.suite.resolve()), "--results", str(results)], log)
        except subprocess.CalledProcessError as exc:
            # The grader exits 1 for a valid report with any failing physics case.
            if exc.returncode != 1:
                raise
        grading = set(results.glob("grading-*/results.json")) - existing
        if len(grading) != 1:
            raise ValueError("Grading failed: expected exactly one fresh trusted report; inspect commands.log")
        report = grading.pop()
        return report, json.loads(report.read_text())

    def record(self, row):
        self.rows.append(row)
        with (self.out / "attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        scored = [r for r in self.rows if r.get("passed") is not None]
        scores = [r["passed"] for r in scored]
        (self.out / "run-note.md").write_text(
            "# Restricted Qwen Feedback Loop\n\n"
            f"Seat: {self.args.seat}; CPU NKI simulator, not hardware timing.\n"
            f"Public suite: {self.args.suite.resolve()}; steps: {self.args.steps}; batch: 2; repeats: 1.\n"
            f"Attempt limit: {self.args.attempts}; wall-clock budget: {self.args.minutes} minutes.\n"
            f"Recorded attempts: {len(self.rows)}; graded: {len(scored)}; "
            f"passing-case count range: {min(scores) if scores else 'unmeasured'}"
            f"..{max(scores) if scores else 'unmeasured'}.\n"
            "Every attempt, rejection and error is retained. Checker gates are unchanged.\n"
            "Qwen weights are unchanged. Only beta in the pinned, human-reviewed accelerated "
            "kernel may change; this is parameter tuning, not unrestricted kernel discovery.\n"
            "AST equality to the reviewed template is a narrow execution gate, not a general sandbox.\n"
            "References remain local to the trusted grader. No private-test or throughput claim.\n"
            "Each grade contains detailed measured failures and natural-language next-prompt feedback.\n")

    def run(self):
        args = self.args
        seed = args.seed.resolve()
        approved = seed.read_text()
        if hashlib.sha256(seed.read_bytes()).hexdigest() != SEED_SHA:
            raise ValueError("Seed must match the pinned, manually reviewed beta=0.9 kernel")
        self.out.mkdir(parents=True, exist_ok=False)
        log = self.out / "setup.log"
        self.command(["kubectl", "exec", args.seat, "--", "mkdir", "-p", self.remote], log)
        for name in ("qwen_grip.py", "nki_guidance.py", "grip_throughput.py"):
            self.upload(ROOT / name, f"/workspace/projects/03-contact-physics/{name}", log)
        self.upload(seed, self.remote + "/seed.py", log)
        feedback = self.out / "feedback.txt"
        feedback.write_text("Evaluate the seed first.\n")
        previous_remote = self.remote + "/seed.py"
        best = -1
        seen = set()
        for index in range(args.attempts):
            attempt = self.out / f"attempt-{index:03d}"
            attempt.mkdir()
            log = attempt / "commands.log"
            row = dict(attempt=index, steps=args.steps, physics_score=None, passed=None,
                       status="pending", backend="simulate")
            started = time.monotonic()
            stage = "generation"
            try:
                if index == 0:
                    candidate = seed
                    candidate_remote = previous_remote
                else:
                    prompt = attempt / "feedback.txt"
                    prompt.write_text(feedback.read_text() +
                                      "\nAUTOMATIC EXECUTION CONTRACT: Change ONLY the numeric literal beta "
                                      "in the previous candidate, keeping every other AST node identical. "
                                      "Use 0 <= beta < 1. Return the full module. Do not change steps. "
                                      f"Already evaluated beta values: {sorted(seen)}. Propose a different value.\n")
                    self.upload(prompt, self.remote + "/feedback.txt", log)
                    generated = self.remote + f"/generation-{index:03d}"
                    generation_error = None
                    try:
                        self.remote_command(["python", "-u", "qwen_grip.py", "--feedback",
                                             self.remote + "/feedback.txt", "--previous", previous_remote,
                                             "--base-url", args.base_url, "--temperature", "0.5",
                                             "--max-tokens", "1200", "--out", generated], log)
                    except subprocess.CalledProcessError as exc:
                        generation_error = exc
                    self.download(generated, attempt / "generation", log)
                    if generation_error:
                        raise generation_error
                    candidate = attempt / "generation/candidate.py"
                    candidate_remote = generated + "/candidate.py"
                beta = restricted_revision(candidate.read_text(), approved)
                row.update(beta=beta, candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
                if beta in seen:
                    raise ValueError("Repeated beta; no new evaluation")
                seen.add(beta)
                remote_result = self.remote + f"/simulation-{index:03d}"
                self.remote_command(["python", "-u", "grip_throughput.py", "run", "--public",
                                     "gripping-public", "--backend", "simulate", "--batch", "2",
                                     "--steps", str(args.steps), "--runs", "1", "--candidate",
                                     candidate_remote, "--candidate-sha256", row["candidate_sha256"],
                                     "--out", remote_result], log)
                results = attempt / "simulation"
                self.download(remote_result, results, log)
                stage = "grading"
                report, summary = self.evaluate(results, log)
                passed = summary["passing_case_evaluations"]
                total = summary["expected_case_evaluations"]
                row.update(status="graded", passed=passed, total=total, physics_score=passed / total,
                           all_cases_pass=summary["all_cases_pass"], grading=str(report))
                feedback.write_text((report.parent / "next-prompt.txt").read_text())
                if passed > best:
                    best = passed
                    previous_remote = candidate_remote
                    (self.out / "best-candidate.py").write_bytes(candidate.read_bytes())
                    (self.out / "best.json").write_text(json.dumps(row, indent=2) + "\n")
            except Exception as exc:
                row.update(status="error_or_rejected", error=f"{type(exc).__name__}: {exc}")
                row["checker_failure"] = stage == "grading"
                feedback.write_text(feedback.read_text()[:5000] + "\nLast proposal failed: " + row["error"] + "\n")
            row["seconds"] = time.monotonic() - started
            self.record(row)
            print(f"Attempt {index}: {row['status']}; beta={row.get('beta')}; "
                  f"physics={row.get('passed')}/{row.get('total')}", flush=True)
            if row.get("checker_failure") or row.get("all_cases_pass") or time.monotonic() >= self.deadline:
                break
        print(f"Loop artifacts: {self.out}")


def recover(args):
    """Regrade saved outputs offline without changing the historical attempt log."""
    from grip_throughput import grade

    directory = args.recover.resolve()
    output = directory / f"recovery-{uuid.uuid4().hex}"
    output.mkdir()
    rows = []
    for attempt in sorted(directory.glob("attempt-*/simulation")):
        if not (attempt / "execution.json").exists():
            continue
        summary = grade(args.suite.resolve(), attempt)
        passed = summary["passing_case_evaluations"]
        total = summary["expected_case_evaluations"]
        row = dict(attempt=attempt.parent.name, passed=passed, total=total,
                   physics_score=passed / total, all_cases_pass=summary["all_cases_pass"],
                   context=summary["context"])
        rows.append(row)
        with (output / "attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"Recovered {attempt.parent.name}: {passed}/{total}")
    (output / "run-note.md").write_text(
        "# Recovered Physics Scores\n\n"
        f"Regraded {len(rows)} saved public simulator attempts against the unchanged trusted checker.\n"
        "The original controller mistook the grader's exit code 1 (partial physics pass) "
        "for a checker failure. Original logs are preserved; these are corrected scores, "
        "not new generation or simulation attempts. No device throughput claim.\n")
    print(f"Recovery artifacts: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seat", default="seat-260")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--suite", type=Path, default=ROOT / "data/gripping-development-v1")
    parser.add_argument("--seed", type=Path, default=ROOT / "data/qwen-grip-6074b673bb6348ce932b631a7a079f4c/candidate.py")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--attempts", type=int, default=6, help="Includes seed evaluation")
    parser.add_argument("--minutes", type=float, default=20)
    parser.add_argument("--out", type=Path, default=ROOT / f"data/grip-loop-{uuid.uuid4().hex}")
    parser.add_argument("--recover", type=Path, help="Regrade an existing loop offline; no Qwen or remote work")
    args = parser.parse_args()
    if args.steps < 1 or args.attempts < 1 or not math.isfinite(args.minutes) or args.minutes <= 0:
        parser.error("Positive steps, attempts and finite minutes required")
    if args.recover:
        recover(args)
    else:
        Controller(args).run()


if __name__ == "__main__":
    main()
