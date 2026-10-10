"""Engine-backed frictionless force QPs; distinct from our impulse snapshots."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import uuid
import xml.etree.ElementTree as ET

import numpy as np
from contact import diagnostics, oracle

ENGINE_VERSION = "3.15.0"


def load_engine():
    import mujoco
    if mujoco.__version__ != ENGINE_VERSION:
        raise RuntimeError(f"Expected MuJoCo {ENGINE_VERSION}, got {mujoco.__version__}; use requirements-engine.txt")
    return mujoco


def scene_xml(scene, count, seed):
    if scene not in ("plane", "pairs", "stack") or not 1 <= count <= 64:
        raise ValueError("Choose plane/pairs/stack and 1..64 contacts")
    rng = np.random.default_rng(seed)
    root = ET.Element("mujoco", model=f"frictionless_{scene}")
    option = ET.SubElement(root, "option", timestep="0.002", gravity="0 0 0" if scene == "pairs" else "0 0 -9.81",
                           solver="Newton", jacobian="dense", iterations="200", tolerance="1e-12")
    ET.SubElement(option, "flag", warmstart="disable")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", condim="1", friction="0 0 0", solref="0.02 1", solimp="0.9 0.95 0.001")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="20 20 0.1")
    for i in range(count * (2 if scene == "pairs" else 1)):
        if scene == "plane":
            position = (i * 1.0, 0, 0.1995)
        elif scene == "stack":
            position = (0, 0, 0.1995 + i * 0.399)
        else:
            position = ((i % 2) * 0.399, (i // 2) * 1.0, 2)
        body = ET.SubElement(world, "body", name=f"body{i}", pos=" ".join(map(str, position)))
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "geom", name=f"sphere{i}", type="sphere", size="0.2", mass=f"{rng.uniform(0.5, 2):.17g}")
    return ET.tostring(root, encoding="unicode")


def initial_state(model, scene, seed):
    rng = np.random.default_rng(seed + 10000)
    velocity = np.zeros(model.nv)
    for i in range(model.nv // 6):
        velocity[6 * i:6 * i + 3] = rng.uniform(-0.5, 0.5, 3)
        if scene == "pairs":
            velocity[6 * i] = 0.5 if i % 2 == 0 else -0.5
        else:
            velocity[6 * i + 2] -= 0.5
    return model.qpos0.copy(), velocity


def extract(xml, qpos, qvel):
    mj = load_engine()
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mj.mj_forward(model, data)
    if data.nefc < 1 or not np.all(data.efc_type == int(mj.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)):
        raise ValueError("Only nonempty, purely frictionless contact constraints are supported")
    mass = np.zeros((model.nv, model.nv))
    mj.mj_fullM(model, data, mass)
    jacobian = np.asarray(data.efc_J).reshape(data.nefc, model.nv).copy()
    response = np.linalg.solve(mass, jacobian.T)
    regularizer = np.asarray(data.efc_R).copy()
    a = jacobian @ response + np.diag(regularizer)
    b = np.asarray(data.efc_b).copy()
    np.testing.assert_allclose(b, jacobian @ data.qacc_smooth - data.efc_aref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(data.qfrc_constraint, jacobian.T @ data.efc_force, rtol=1e-10, atol=1e-10)
    if np.linalg.eigvalsh(a)[0] <= 0 or not np.all(regularizer > 0):
        raise ValueError("Non-positive-definite force QP")
    return dict(A=a, b=b, M=mass, J=jacobian, R=regularizer, response=response,
                qpos=np.asarray(qpos).copy(), qvel=np.asarray(qvel).copy(),
                qacc_smooth=np.asarray(data.qacc_smooth).copy(),
                engine_qacc=np.asarray(data.qacc).copy(), engine_forces=np.asarray(data.efc_force).copy(),
                contact_geom=np.array([c.geom for c in data.contact], dtype=np.int32),
                contact_position=np.array([c.pos for c in data.contact]))


def verify(fixture):
    reference = oracle(fixture["A"], fixture["b"])
    engine = fixture["engine_forces"]
    oracle_check = diagnostics(fixture["A"], fixture["b"], reference, reference)
    engine_check = diagnostics(fixture["A"], fixture["b"], engine, reference)
    predicted_acceleration = fixture["qacc_smooth"] + fixture["response"] @ reference
    force_error = float(np.max(np.abs(engine - reference)) / max(1, np.max(np.abs(reference))))
    acceleration_error = float(np.max(np.abs(predicted_acceleration - fixture["engine_qacc"])) /
                               max(1, np.max(np.abs(fixture["engine_qacc"]))))
    passed = bool(oracle_check["passed"] and oracle_check["residual"] <= 1e-9 and engine_check["passed"]
                  and force_error <= 1e-7 and acceleration_error <= 1e-7)
    return reference, dict(passed=passed, oracle=oracle_check, engine=engine_check,
                           force_error=force_error, acceleration_error=acceleration_error)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def problem_digest(fixture):
    fingerprint = hashlib.sha256()
    for name in ("A", "b", "M", "J", "R", "qacc_smooth", "qpos", "qvel"):
        value = np.ascontiguousarray(fixture[name], dtype="<f8")
        fingerprint.update(json.dumps([name, value.shape]).encode("ascii"))
        fingerprint.update(value.tobytes())
    return fingerprint.hexdigest()


def replay(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["engine_version"] != ENGINE_VERSION:
        raise ValueError("Unsupported engine version")
    for name, expected in manifest["source_sha256"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Source hash mismatch: {name}")
    failures = []
    for record in manifest["fixtures"]:
        try:
            for name, expected in record["sha256"].items():
                if digest(directory / name) != expected:
                    raise ValueError(f"Artifact hash mismatch: {name}")
            with np.load(directory / record["data_file"], allow_pickle=False) as saved:
                fresh = extract((directory / record["scene_file"]).read_text(), saved["qpos"], saved["qvel"])
                for name, value in fresh.items():
                    np.testing.assert_allclose(value, saved[name], rtol=1e-10, atol=1e-10, err_msg=name)
                if "problem_sha256" in record and problem_digest(dict(saved)) != record["problem_sha256"]:
                    raise ValueError("Problem fingerprint mismatch")
                reference, check = verify(fresh)
                np.testing.assert_allclose(reference, saved["reference_forces"], rtol=1e-10, atol=1e-10)
                if not check["passed"]:
                    raise ValueError(f"Engine agreement failed: {check}")
        except Exception as exc:
            failures.append(dict(case=record["case"], error=f"{type(exc).__name__}: {exc}"))
    print(json.dumps(dict(replayed=len(manifest["fixtures"]), failures=failures), indent=2))
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--contacts", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    if args.replay:
        return replay(args.replay)
    if args.seeds < 1 or len(set(args.contacts)) != len(args.contacts) or not all(1 <= c <= 64 for c in args.contacts):
        parser.error("Positive seeds and unique contact counts in 1..64 required")
    load_engine()
    run_id = uuid.uuid4().hex
    out = args.out or Path("data") / f"engine-validation-{run_id}"
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema=1, run_id=run_id, engine_version=ENGINE_VERSION,
                    python=platform.python_version(), numpy=np.__version__, host=platform.node(),
                    scope="MuJoCo frictionless soft-contact force QP snapshots; not impulse data or rollouts",
                    source_sha256={name: digest(Path(__file__).with_name(name)) for name in ("engine_validation.py", "contact.py")}, fixtures=[])
    for scene in ("plane", "pairs", "stack"):
        for count in args.contacts:
            for seed in range(args.seeds):
                case = f"{scene}-c{count}-s{seed}"
                record = dict(case=case, scene=scene, requested_contacts=count, seed=seed, score=0)
                try:
                    xml = scene_xml(scene, count, seed)
                    model = load_engine().MjModel.from_xml_string(xml)
                    fixture = extract(xml, *initial_state(model, scene, seed))
                    reference, check = verify(fixture)
                    record.update(check=check, score=int(check["passed"]), actual_contacts=len(fixture["b"]),
                                  problem_sha256=problem_digest(fixture))
                    scene_file, data_file = f"{case}.xml", f"{case}.npz"
                    (out / scene_file).write_text(xml + "\n")
                    np.savez_compressed(out / data_file, **fixture, reference_forces=reference)
                    record.update(scene_file=scene_file, data_file=data_file,
                                  sha256={name: digest(out / name) for name in (scene_file, data_file)})
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                manifest["fixtures"].append(record)
                with Path("data/engine-validation-attempts.jsonl").open("a") as log:
                    log.write(json.dumps(dict(run_id=run_id, **record), allow_nan=False) + "\n")
                print(f"{case}: score={record['score']} {record.get('error', '')}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    passed = sum(r["score"] for r in manifest["fixtures"])
    unique = len({r["problem_sha256"] for r in manifest["fixtures"] if "problem_sha256" in r})
    (out / "run-note.md").write_text(f"""# Engine-Backed Reference Validation

Run {run_id}; host {platform.node()}; MuJoCo {ENGINE_VERSION}; CPU only.
Scenes: plane, independent pairs, vertical stacks. Contact counts: {args.contacts}.
Seeds per configuration: {args.seeds}; certified snapshots: {passed}/{len(manifest['fixtures'])}.
Distinct exact problem fingerprints: {unique}; duplicates must not cross dataset splits.
This validates extraction and independent FP64 NNLS agreement with MuJoCo's
Newton solver on normal contact forces and generalized accelerations.
Force and acceleration normalized discrepancies must each be <=1e-7.
Oracle projected residual must be <=1e-9; existing objective/feasibility gates apply.
These are scene/state snapshots, not trajectory or real-world validation.
No Trainium timing, Qwen generation or training claim is made.
Physics follows the engine's regularizer and reference acceleration, not our
earlier constant-epsilon impulse model. Reference outputs belong to the evaluator.
Scene XML, initial states, matrices and outputs are preserved with SHA256 hashes.
Replay: python engine_validation.py --replay {out}
Detailed checks and errors: manifest.json; attempts: data/engine-validation-attempts.jsonl.
""")
    print(f"Certified {passed}/{len(manifest['fixtures'])}; artifacts: {out}", flush=True)
    return int(passed != len(manifest["fixtures"]))


if __name__ == "__main__":
    raise SystemExit(main())
