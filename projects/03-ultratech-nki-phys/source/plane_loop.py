"""Bounded plane loop with an AST whitelist, remote simulation and physics gates."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from grip_loop import Controller, ROOT
from experimental_forms import experimental_forms


def canonical(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body.pop(0)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    arguments = {a.arg for a in function.args.args}
    names = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id not in arguments:
            names.setdefault(node.id, f"local_{len(names)}")
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and node.id in names:
            node.id = names[node.id]
    return ast.dump(tree, include_attributes=False)


def reviewed_forms():
    baseline = (ROOT / "nki_contact_batch.py").read_text()
    copyfree = baseline.replace(
        "            gradient = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)\n"
        "            nisa.tensor_copy(dst=gradient, src=product)\n", "").replace(
        "data1=gradient, data2=b", "data1=product, data2=b")
    fused = copyfree.replace(
        "nisa.tensor_tensor(dst=updated, data1=impulses, data2=scaled, op=nl.subtract)",
        "nisa.tensor_scalar(dst=updated, data=scaled, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=impulses)")
    scale_fused = (ROOT / "nki_contact_scale_fused.py").read_text()
    return dict(baseline=baseline, copyfree=copyfree, fused=fused,
                **{"scale-fused": scale_fused}, **experimental_forms(baseline))


def review(source):
    value = canonical(source)
    for name, approved in reviewed_forms().items():
        if value == canonical(approved):
            return name
    raise ValueError("Unreviewed code form. Preserve the solver and per-world step size. "
                     "Use the exact reviewed operation sequence supplied in feedback.")


class PlaneLoop(Controller):
    def run(self):
        args = self.args
        self.out.mkdir(parents=True, exist_ok=False)
        log = self.out / "setup.log"
        self.command(["kubectl", "exec", args.seat, "--", "mkdir", "-p", self.remote], log)
        for name in ("qwen_grip.py", "nki_guidance.py", "optimization_knowledge.py", "check_batch.py"):
            self.upload(ROOT / name, f"/workspace/projects/03-contact-physics/{name}", log)
        baseline = ROOT / "nki_contact_batch.py"
        self.upload(baseline, self.remote + "/baseline.py", log)
        previous = self.remote + "/baseline.py"
        feedback = (ROOT / "plane-feedback.txt").read_text()
        seen = {canonical(baseline.read_text())}
        self.rows = []
        for index in range(args.attempts):
            attempt = self.out / f"attempt-{index:03d}"
            attempt.mkdir()
            log = attempt / "commands.log"
            row = dict(attempt=index, physics_score=None, status="pending", task="plane-throughput")
            started = time.monotonic()
            try:
                # Human-reviewed templates constrain unattended execution, not a sandbox.
                prompt = attempt / "feedback.txt"
                prompt.write_text(feedback[:3000] + "\nFor automatic execution use precisely this human-reviewed "
                                  "copy-removal module, or change ONLY its updated subtraction to "
                                  "nisa.tensor_scalar(dst=updated, data=scaled, op0=nl.multiply, operand0=-1.0, "
                                  "op1=nl.add, operand1=impulses). Other structural changes require human review.\n" +
                                  reviewed_forms()["copyfree"])
                self.upload(prompt, self.remote + "/feedback.txt", log)
                generated = self.remote + f"/generation-{index:03d}"
                try:
                    self.remote_command(["python", "-u", "qwen_grip.py", "--task", "plane-throughput",
                                         "--feedback", self.remote + "/feedback.txt", "--baseline",
                                         self.remote + "/baseline.py", "--previous", previous,
                                         "--base-url", args.base_url, "--max-tokens", "1600",
                                         "--temperature", "0.4", "--out", generated], log)
                except subprocess.CalledProcessError:
                    self.download(generated, attempt / "generation", log)
                    raise
                self.download(generated, attempt / "generation", log)
                candidate = attempt / "generation/candidate.py"
                source = candidate.read_text()
                row["candidate_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
                form = review(source)
                row["reviewed_form"] = form
                if canonical(source) in seen:
                    raise ValueError("Duplicate reviewed program: propose the other reviewed form; no new execution")
                seen.add(canonical(source))
                previous = generated + "/candidate.py"
                row["status"] = "reviewed"
                for scene in ("plane", "pairs"):
                    # The existing runner chooses a UUID output path; capture it from its log.
                    scene_log = attempt / f"{scene}.log"
                    try:
                        self.remote_command(["python", "-u", "check_batch.py", "--backend", "simulate",
                                             "--scene", scene, "--contacts", "33", "--batch", "8",
                                             "--steps", "32", "--candidate", previous,
                                             "--candidate-sha256", row["candidate_sha256"]], scene_log)
                    except subprocess.CalledProcessError as exc:
                        if exc.returncode != 1 or "Artifacts: " not in scene_log.read_text():
                            raise
                    lines = scene_log.read_text().splitlines()
                    path = next(line.removeprefix("Artifacts: ") for line in reversed(lines)
                                if line.startswith("Artifacts: "))
                    local = attempt / scene
                    self.download("/workspace/projects/03-contact-physics/" + path, local, log)
                    report = json.loads((local / "results.json").read_text())
                    check = report["context"].get("initial_check")
                    if check is None:
                        raise ValueError(f"{scene} execution error: {report['context'].get('error', 'No checker report')}")
                    row[scene] = dict(physics_passes=check["physics_passes"], update_score=check["score"],
                                      report=str(local / "results.json"))
                    if "error" in report["context"] or not check["score"] or check["physics_passes"] != 8:
                        raise ValueError(f"{scene} checker failed: " + json.dumps(check)[:2200])
                row.update(status="physics_passed_not_benchmarked", physics_score=1.0)
                (self.out / "accepted-candidate.py").write_bytes(candidate.read_bytes())
                (self.out / "accepted.json").write_text(json.dumps(row, indent=2) + "\n")
            except Exception as exc:
                row.update(status="error_or_rejected", error=f"{type(exc).__name__}: {exc}")
                feedback = "Last proposal rejected: " + row["error"]
            row["seconds"] = time.monotonic() - started
            self.rows.append(row)
            with (self.out / "attempts.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            (self.out / "run-note.md").write_text(
                f"# Plane Qwen Loop\n\nSeat {args.seat}; attempts {len(self.rows)}/{args.attempts}; "
                f"budget {args.minutes} minutes.\n"
                "Human-supplied, AST-whitelisted copy-removal/fusion templates only; not unrestricted "
                "Qwen optimization, model training or a general sandbox. CPU NKI simulation only.\n"
                "Candidate workload: plane and pairs, 8 worlds, 33 contacts, 32 updates. "
                "Fixed-update and every physics gate required. No timing or speedup claim. "
                "Every proposal, rejection, request, source and available report is retained.\n")
            print(f"Attempt {index}: {row['status']}; physics={row['physics_score']}", flush=True)
            if row["physics_score"] == 1.0 or time.monotonic() >= self.deadline:
                break
        print(f"Loop artifacts: {self.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seat", default="seat-260")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--out", type=Path, default=ROOT / f"data/plane-loop-{uuid.uuid4().hex}")
    args = parser.parse_args()
    if not 0 < args.minutes <= 120 or args.attempts < 1:
        parser.error("Positive minutes <=120 and attempts required")
    PlaneLoop(args).run()


if __name__ == "__main__":
    main()
