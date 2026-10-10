"""Auditable gripping development fixtures and scored iterative CPU attempts."""

import argparse
import itertools
import json
from pathlib import Path
import platform
import time
import uuid

import numpy as np

from contact import projected_gradient
from engine_validation import ENGINE_VERSION, digest, problem_digest
from grip_physics import CONTRACT, build_fixture, certify, check_grip, extract


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def replay(out):
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["engine_version"] != ENGINE_VERSION:
        raise ValueError("Engine version mismatch")
    for name, expected in manifest["source_sha256"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Source changed: {name}; preserve this version for replay")
    if digest(out / "public/contract.json") != manifest["contract_sha256"]:
        raise ValueError("Contract hash mismatch")
    seen = set()
    for record in manifest["cases"]:
        for name, expected in record["sha256"].items():
            if digest(out / name) != expected:
                raise ValueError(f"Artifact hash mismatch: {name}")
        with np.load(out / record["reference_file"], allow_pickle=False) as saved:
            fixture = extract((out / record["scene_file"]).read_text(), saved["qpos"], saved["qvel"])
            for name, value in fixture.items():
                np.testing.assert_allclose(value, saved[name], rtol=1e-10, atol=1e-10)
            fingerprint = problem_digest(fixture)
            if fingerprint != record["problem_sha256"] or fingerprint in seen:
                raise ValueError("Mismatched or duplicate physics")
            seen.add(fingerprint)
            reference, check = certify(fixture)
            if not check["passed"]:
                raise ValueError("Failed engine/reference agreement")
            np.testing.assert_allclose(reference, saved["reference_forces"], rtol=1e-10, atol=1e-10)
            with np.load(out / record["input_file"], allow_pickle=False) as inputs:
                if set(inputs.files) != {"A", "b"}:
                    raise ValueError("Public inputs must only contain A and b")
                for name in inputs.files:
                    np.testing.assert_array_equal(inputs[name], fixture[name].astype(np.float32))
    attempts = [json.loads(line) for line in (out / "attempts.jsonl").read_text().splitlines()]
    expected = {(row["case_id"], run) for row in manifest["cases"] for run in range(manifest["runs"])}
    actual = [(row["case_id"], row["run"]) for row in attempts]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Missing or duplicate baseline attempts")
    for attempt in attempts:
        path = out / attempt["candidate_file"]
        if digest(path) != attempt["candidate_sha256"]:
            raise ValueError("Candidate output hash mismatch")
        row = next(row for row in manifest["cases"] if row["case_id"] == attempt["case_id"])
        with np.load(out / row["reference_file"], allow_pickle=False) as saved:
            with np.load(path, allow_pickle=False) as candidate:
                check = check_grip(dict(saved), candidate["forces"], saved["reference_forces"])
        if check != attempt["check"] or int(check["passed"]) != attempt["score"]:
            raise ValueError("Attempt score cannot be reproduced")
    print(f"Verified {len(seen)} gripping cases and reproduced {len(attempts)} scores")


def generate(out, runs):
    out.mkdir(parents=True, exist_ok=False)
    public, evaluator = out / "public", out / "evaluator-only"
    public.mkdir()
    evaluator.mkdir(mode=0o700)
    outputs = out / "candidate-outputs"
    outputs.mkdir()
    write_json(public / "contract.json", CONTRACT)
    manifest = dict(schema=1, run_id=uuid.uuid4().hex, engine_version=ENGINE_VERSION,
                    host=platform.node(), python=platform.python_version(), numpy=np.__version__,
                    runs=runs, scope="public development snapshots; not private evaluation or rollouts",
                    candidate="FP32 projected gradient; residual stopping 1e-7; cap 8192",
                    contract_sha256=digest(public / "contract.json"),
                    source_sha256={name: digest(Path(__file__).with_name(name)) for name in
                                   ("generate_grip.py", "grip_physics.py", "contact.py", "force_checker.py", "engine_validation.py")},
                    cases=[])
    seen, attempts = set(), []
    parameters = itertools.product((0.5, 2.0), (0.2, 0.8), (0.0002, 0.001), (0.0, 0.2))
    for index, (mass, friction, penetration, slip) in enumerate(parameters):
        case_id = f"grip-{index:03d}"
        xml, fixture = build_fixture(mass, friction, penetration, slip)
        reference, certification = certify(fixture)
        with (out / "certification-attempts.jsonl").open("a") as log:
            log.write(json.dumps(dict(case_id=case_id, score=int(certification["passed"]),
                                      check=certification), allow_nan=False) + "\n")
        if not certification["passed"]:
            raise ValueError(f"Uncertified case: {case_id}; failure preserved in log")
        fingerprint = problem_digest(fixture)
        if fingerprint in seen:
            raise ValueError(f"Duplicate physics: {case_id}")
        seen.add(fingerprint)
        scene_file = f"public/{case_id}.xml"
        input_file = f"public/{case_id}.npz"
        reference_file = f"evaluator-only/{case_id}.npz"
        (out / scene_file).write_text(xml + "\n")
        np.savez_compressed(out / input_file, A=fixture["A"].astype(np.float32), b=fixture["b"].astype(np.float32))
        np.savez_compressed(out / reference_file, **fixture, reference_forces=reference)
        (out / reference_file).chmod(0o600)
        record = dict(case_id=case_id, parameters=dict(mass=mass, friction=friction,
                      penetration=penetration, downward_speed=slip), constraints=len(fixture["b"]),
                      condition_number=float(np.linalg.cond(fixture["A"])), certification=certification,
                      problem_sha256=fingerprint, scene_file=scene_file, input_file=input_file,
                      reference_file=reference_file,
                      sha256={name: digest(out / name) for name in (scene_file, input_file, reference_file)})
        manifest["cases"].append(record)
        for run in range(runs):
            started = time.perf_counter()
            forces, iterations = projected_gradient(fixture["A"].astype(np.float32),
                                                   fixture["b"].astype(np.float32),
                                                   tolerance=1e-7, max_iterations=8192)
            seconds = time.perf_counter() - started
            check = check_grip(fixture, forces, reference)
            candidate_file = f"candidate-outputs/{case_id}-r{run}.npz"
            np.savez_compressed(out / candidate_file, forces=forces)
            attempt = dict(run_id=manifest["run_id"], case_id=case_id, run=run,
                           score=int(check["passed"]), iterations=iterations, seconds=seconds,
                           check=check, candidate_file=candidate_file,
                           candidate_sha256=digest(out / candidate_file))
            attempts.append(attempt)
            with (out / "attempts.jsonl").open("a") as log:
                log.write(json.dumps(attempt, allow_nan=False) + "\n")
        print(f"{case_id}: certified; baseline {sum(a['score'] for a in attempts[-runs:])}/{runs}", flush=True)
    write_json(out / "manifest.json", manifest)
    write_json(public / "manifest.json", dict(contract_sha256=manifest["contract_sha256"],
               cases=[{key: row[key] for key in ("case_id", "parameters", "constraints", "input_file", "scene_file")}
                      for row in manifest["cases"]]))
    write_json(out / "commitment.json", {name: digest(out / name) for name in
               ("manifest.json", "public/manifest.json", "attempts.jsonl", "certification-attempts.jsonl")})
    times = np.array([a["seconds"] * 1000 for a in attempts])
    passed = sum(a["score"] for a in attempts)
    (out / "run-note.md").write_text(f"""# Gripping Development Run

Run {manifest['run_id']}; host {manifest['host']}; Python {manifest['python']};
MuJoCo {ENGINE_VERSION}; CPU only, no Trainium or Qwen execution.
16 distinct two-pad sphere snapshots: two masses, two sliding-friction coefficients,
two pad penetrations, and two downward speeds. Two contacts, eight pyramid edges.
All 16 references certified by independent FP64 NNLS and MuJoCo Newton agreement.
Force, contact-wrench and acceleration certification errors must each be <=1e-7;
oracle residual <=1e-9. Candidate gates: public/contract.json and GRIPPING.md.
Baseline FP32 projected gradient starts at zero; stopping residual 1e-7; cap 8192.
Scored attempts: {passed}/{len(attempts)} pass; {runs} repeats per case.
CPU solve milliseconds over all workloads: median {np.median(times):.6f},
min {times.min():.6f}, max {times.max():.6f}. This mixes workloads; per-case samples
and iterations are in attempts.jsonl. Solve timing includes eigenvalue preparation
and stopping checks, excludes fixture generation, oracle, final scoring and file IO.
This is a development baseline, not a controlled accelerator performance comparison.
All returned force arrays are preserved and their scores can be replayed.
No sustained hold, real-world fidelity, private generalization or model-training claim.
Only public/ may be exposed to candidates; same-user permissions are not isolation.
Replay: python generate_grip.py --replay {out}
""")
    print(f"Artifacts: {out}; baseline passes {passed}/{len(attempts)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.replay:
        commitments = json.loads((args.replay / "commitment.json").read_text())
        for name, expected in commitments.items():
            if digest(args.replay / name) != expected:
                raise ValueError(f"Commitment mismatch: {name}")
        replay(args.replay)
    else:
        if args.runs < 1:
            parser.error("Positive runs required")
        generate(args.out or Path("data") / f"gripping-{uuid.uuid4().hex}", args.runs)


if __name__ == "__main__":
    main()
