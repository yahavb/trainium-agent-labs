"""Grade force snapshots or replay the packaged evidence without accelerator access."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
from math_tasks import TASKS, check


def grade_snapshot(task, path):
    with np.load(path, allow_pickle=False) as stored:
        inputs = tuple(stored[name].copy() for name in TASKS[task]["input_names"])
        actual = stored["actual"].copy()
    result = check(task, actual, inputs, inputs)
    result["input_preservation_independently_verified"] = False
    return result


def verify_hashes(root):
    hashes = json.loads((root / "SHA256SUMS.json").read_text())
    for relative, expected in hashes.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Hash manifest path escapes the submission")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"File hash mismatch: {relative}")
    return len(hashes)


def replay(root):
    from frozen_math_eval import grade

    count = verify_hashes(root)
    log = root / "ATTEMPTS.jsonl"
    if log.read_bytes() != (root / "development/attempts.jsonl").read_bytes():
        raise ValueError("Top-level attempt log differs from original")
    attempts = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    public = {}
    for directory in (root / "development", root / "independent-repeat"):
        for path in sorted(directory.rglob("case-*.npz")):
            task = next((name for name in TASKS if name in path.relative_to(directory).parts), None)
            if task is None:
                raise ValueError(f"Cannot identify task: {path}")
            key = str(path.parent.relative_to(root))
            group = public.setdefault(key, {"passed": 0, "measured": 0})
            group["passed"] += grade_snapshot(task, path)["score"]
            group["measured"] += 1
    # Reuse the frozen evaluator in a temporary output folder, leaving evidence untouched.
    bundle = root / "final-evaluation"
    with tempfile.TemporaryDirectory(prefix="ultratech-regrade-") as tmp:
        passed = grade(bundle, bundle / "assessment/device-outputs", Path(tmp) / "grading")
        final = json.loads((Path(tmp) / "grading/results.json").read_text())["tasks"]
    summary = dict(verified_files=count, attempts=attempts,
                   public_snapshot_groups=public, frozen_final=final,
                   note="Recorded-output regrading only; no new timing or device execution")
    if not passed:
        raise ValueError("Frozen final evidence no longer passes its original checker")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.replay:
        if args.task or args.snapshot:
            parser.error("Use --replay alone")
        print(json.dumps(replay(ROOT), indent=2))
        return 0
    if not args.task or not args.snapshot:
        parser.error("Specify --replay or both --task and --snapshot")
    result = grade_snapshot(args.task, args.snapshot)
    print(json.dumps(result, indent=2))
    return int(not result["score"])


if __name__ == "__main__":
    raise SystemExit(main())
