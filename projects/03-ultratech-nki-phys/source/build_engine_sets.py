"""Certify disjoint engine datasets; retain references and private cases locally."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import uuid
import xml.etree.ElementTree as ET

import numpy as np
from engine_validation import ENGINE_VERSION, digest, extract, initial_state, load_engine, problem_digest, scene_xml, verify
from force_checker import CONTRACT


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_scene(scene, count, seed):
    if scene != "multi_stack":
        return scene_xml(scene, count, seed)
    root = ET.fromstring(scene_xml("stack", count, seed))
    root.set("model", "frictionless_multi_stack")
    for i, body in enumerate(root.findall("worldbody/body")):
        body.set("pos", f"{i % 3} 0 {0.1995 + (i // 3) * 0.399:.17g}")
    return ET.tostring(root, encoding="unicode")


def build(out):
    load_engine()
    out.mkdir(parents=True, exist_ok=False)
    public = out / "public"
    trusted = out / "evaluator-only"
    public.mkdir()
    trusted.mkdir(mode=0o700)
    suite_id = uuid.uuid4().hex
    write_json(public / "contract.json", CONTRACT)
    contract_hash = digest(public / "contract.json")
    source_hashes = {name: digest(Path(__file__).with_name(name)) for name in
                     ("build_engine_sets.py", "engine_validation.py", "force_checker.py", "contact.py")}
    manifests = {split: dict(schema=1, suite_id=suite_id, split=split, engine_version=ENGINE_VERSION,
                             contract_sha256=contract_hash, source_sha256=source_hashes, fixtures=[])
                 for split in ("public", "private")}
    seen = set()
    private_seeds = [1000000 + secrets.randbelow(1000000000) for _ in range(3)]
    while len(set(private_seeds)) != 3:
        private_seeds = [1000000 + secrets.randbelow(1000000000) for _ in range(3)]
    specifications = [("public", ("plane", "pairs", "stack"), (1, 4, 8), (2, 3, 4)),
                      ("private", ("plane", "pairs", "stack", "multi_stack"), (3, 7, 17), private_seeds)]
    duplicates = dict(public=0, private=0)
    for split, scenes, counts, seeds in specifications:
        for scene in scenes:
            for count in counts:
                for seed in seeds:
                    xml = make_scene(scene, count, seed)
                    model = load_engine().MjModel.from_xml_string(xml)
                    fixture = extract(xml, *initial_state(model, scene, seed))
                    reference, certification = verify(fixture)
                    if not certification["passed"]:
                        raise ValueError(f"Uncertified {split} fixture; no usable manifest will be published")
                    fingerprint = problem_digest(fixture)
                    if fingerprint in seen:
                        duplicates[split] += 1
                        continue
                    seen.add(fingerprint)
                    case_id = uuid.uuid4().hex
                    destination = public if split == "public" else trusted
                    input_name, scene_name, reference_name = f"{case_id}-input.npz", f"{case_id}.xml", f"{case_id}-reference.npz"
                    np.savez_compressed(destination / input_name, A=fixture["A"], b=fixture["b"])
                    (destination / scene_name).write_text(xml + "\n")
                    np.savez_compressed(trusted / reference_name, **fixture, reference_forces=reference)
                    record = dict(case_id=case_id, scene=scene, seed=seed, contacts=len(fixture["b"]),
                                  problem_sha256=fingerprint, input_file=input_name, scene_file=scene_name,
                                  input_sha256=digest(destination / input_name), scene_sha256=digest(destination / scene_name))
                    manifests[split]["fixtures"].append(record)
                    with (trusted / "certification-attempts.jsonl").open("a") as log:
                        log.write(json.dumps(dict(suite_id=suite_id, split=split, case_id=case_id, score=1,
                                                 certification=certification), allow_nan=False) + "\n")
                    with (trusted / "references.jsonl").open("a") as refs:
                        refs.write(json.dumps(dict(case_id=case_id, split=split, file=reference_name,
                                                   sha256=digest(trusted / reference_name))) + "\n")
    write_json(public / "manifest.json", manifests["public"])
    write_json(trusted / "private-manifest.json", manifests["private"])
    commitment = dict(suite_id=suite_id, contract_sha256=contract_hash,
                      public_manifest_sha256=digest(public / "manifest.json"),
                      private_manifest_sha256=digest(trusted / "private-manifest.json"),
                      references_index_sha256=digest(trusted / "references.jsonl"),
                      public_cases=len(manifests["public"]["fixtures"]),
                      private_cases=len(manifests["private"]["fixtures"]))
    write_json(out / "commitment.json", commitment)
    for path in trusted.iterdir():
        path.chmod(0o600)
    (out / "README.md").write_text(f"""# Frozen Engine Dataset

Suite {suite_id}; MuJoCo {ENGINE_VERSION}.
Public cases: {commitment['public_cases']}; private cases: {commitment['private_cases']}.
Exact duplicates removed: {duplicates}.
Public cases use plane/pairs/stack, counts 1/4/8 and seeds 2/3/4.
Private cases add held-out counts and disconnected multiple stacks.
All cases were certified against MuJoCo and independent FP64 NNLS.
Certification is not candidate testing. Private candidate evaluation has not run.
Only public/ may be deployed to a model/candidate environment. evaluator-only/
contains every reference answer and all private scenes, states and seeds.
File permissions help prevent accidental access but do not isolate the same OS user.
Use a separate evaluator host/account before running model-generated code.
Verify hashes and replay: python build_engine_sets.py --verify {out}
Do not edit frozen files or regenerate this suite after seeing candidate results.
Public reports must bind the candidate hash, suite ID and contract hash and pass
every case before final private evaluation. Private detailed feedback stays hidden.
""")
    print(f"Created {commitment['public_cases']} public and {commitment['private_cases']} private certified cases")
    print(f"Exact duplicates removed: {duplicates}; suite: {out}")
    return commitment


def verify_set(out):
    commitment = json.loads((out / "commitment.json").read_text())
    paths = {"contract_sha256": out / "public/contract.json",
             "public_manifest_sha256": out / "public/manifest.json",
             "private_manifest_sha256": out / "evaluator-only/private-manifest.json",
             "references_index_sha256": out / "evaluator-only/references.jsonl"}
    for field, path in paths.items():
        if digest(path) != commitment[field]:
            raise ValueError(f"Commitment mismatch: {field}")
    refs = [json.loads(line) for line in (out / "evaluator-only/references.jsonl").read_text().splitlines()]
    refs_by_id = {r["case_id"]: r for r in refs}
    if len(refs_by_id) != len(refs):
        raise ValueError("Duplicate reference IDs")
    fingerprints = set()
    checked = 0
    for split, directory, manifest_path in (("public", out / "public", paths["public_manifest_sha256"]),
                                             ("private", out / "evaluator-only", paths["private_manifest_sha256"])):
        manifest = json.loads(manifest_path.read_text())
        if manifest["suite_id"] != commitment["suite_id"] or manifest["contract_sha256"] != commitment["contract_sha256"]:
            raise ValueError("Manifest identity mismatch")
        if len(manifest["fixtures"]) != commitment[f"{split}_cases"]:
            raise ValueError("Manifest case count mismatch")
        for name, expected in manifest["source_sha256"].items():
            if digest(Path(__file__).with_name(name)) != expected:
                raise ValueError(f"Source mismatch: {name}")
        for row in manifest["fixtures"]:
            for file_field, hash_field in (("input_file", "input_sha256"), ("scene_file", "scene_sha256")):
                if digest(directory / row[file_field]) != row[hash_field]:
                    raise ValueError("Case artifact hash mismatch")
            reference = refs_by_id[row["case_id"]]
            if reference["split"] != split or digest(out / "evaluator-only" / reference["file"]) != reference["sha256"]:
                raise ValueError("Reference artifact mismatch")
            with np.load(out / "evaluator-only" / reference["file"], allow_pickle=False) as saved:
                fingerprint = problem_digest(dict(saved))
                if fingerprint != row["problem_sha256"] or fingerprint in fingerprints:
                    raise ValueError("Duplicate or changed physical problem")
                fingerprints.add(fingerprint)
                fresh = extract((directory / row["scene_file"]).read_text(), saved["qpos"], saved["qvel"])
                for name, value in fresh.items():
                    np.testing.assert_allclose(value, saved[name], rtol=1e-10, atol=1e-10)
                force, proof = verify(fresh)
                np.testing.assert_allclose(force, saved["reference_forces"], rtol=1e-10, atol=1e-10)
                with np.load(directory / row["input_file"], allow_pickle=False) as inputs:
                    if set(inputs.files) != {"A", "b"}:
                        raise ValueError("Candidate input schema includes unexpected data")
                    for name in ("A", "b"):
                        np.testing.assert_array_equal(inputs[name], saved[name])
                if not proof["passed"]:
                    raise ValueError("Reference certification failed")
                checked += 1
    if checked != len(refs):
        raise ValueError("Orphan reference records")
    print(f"Verified hashes, disjoint fingerprints and engine replay for {checked} cases; no private details printed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify_set(args.verify)
    else:
        build(args.out or Path("data") / f"engine-sets-{uuid.uuid4().hex}")


if __name__ == "__main__":
    main()
