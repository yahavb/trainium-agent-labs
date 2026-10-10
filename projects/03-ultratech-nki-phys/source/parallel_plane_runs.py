"""Independent, physically disjoint seats; paired benchmarks remain per seat."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid

from grip_loop import ROOT


def device_keys(devices):
    if not devices:
        raise ValueError("No devices exposed")
    keys = set()
    for device in devices:
        instance = device.get("instance_id")
        bdf = device.get("bdf")
        if not instance or not bdf:
            raise ValueError("Missing physical device identity; refusing potentially shared timing")
        keys.add((instance, bdf))
    return keys


def ensure_disjoint(inventories):
    owners = {}
    for seat, devices in inventories.items():
        for key in device_keys(devices):
            if key in owners:
                raise ValueError(f"{seat} and {owners[key]} expose the same chip {key}; do not benchmark concurrently")
            owners[key] = seat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seats", required=True, help="Authorized seats, comma separated")
    parser.add_argument("--cores", required=True, help="Confirmed core IDs valid on every listed seat")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / f"data/parallel-plane-{uuid.uuid4().hex}")
    args = parser.parse_args()
    seats = args.seats.split(",")
    if len(seats) < 2 or len(set(seats)) != len(seats) or any(not re.fullmatch(r"seat-\d+", s) for s in seats):
        parser.error("At least two distinct authorized seat-NUMBER names required")
    if args.minutes < 1 or args.attempts < 1:
        parser.error("Positive budgets required")
    args.out.mkdir(parents=True, exist_ok=False)
    inventories = {}
    for seat in seats:
        raw = subprocess.check_output(["kubectl", "exec", seat, "--", "neuron-ls", "--json-output"], timeout=60)
        inventories[seat] = json.loads(raw)
    (args.out / "device-inventories.json").write_text(json.dumps(inventories, indent=2) + "\n")
    ensure_disjoint(inventories)

    def worker(index, seat):
        strategy = "copyfree" if index % 2 == 0 else "scale-fused"
        directory = args.out.resolve() / f"{seat}-{strategy}"
        command = [sys.executable, str(ROOT / "full_plane_loop.py"), "--seat", seat,
                   "--cores", args.cores, "--minutes", str(args.minutes), "--attempts", str(args.attempts),
                   "--search-forms", strategy, "--out", str(directory)]
        with (args.out / f"{seat}.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=log, timeout=args.minutes * 60 + 90)
        rows = []
        if (directory / "attempts.jsonl").exists():
            rows = [json.loads(line) for line in (directory / "attempts.jsonl").read_text().splitlines()]
        return dict(seat=seat, strategy=strategy, returncode=completed.returncode, attempts=rows)

    with ThreadPoolExecutor(max_workers=len(seats)) as executor:
        futures = [executor.submit(worker, index, seat) for index, seat in enumerate(seats)]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(dict(error=f"{type(exc).__name__}: {exc}"))
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "run-note.md").write_text(
        f"# Parallel Seat Experiments\n\n{len(seats)} physically disjoint seat inventories checked. "
        "Alternating independent copy-removal replication and scaling/update-fusion searches. "
        "Baselines, correctness and timing remain paired within each seat. Do not pool raw "
        "throughput across chips as one speedup. Each worker retains every attempt and source.\n"
        "This requires exclusive access to every seat; each worker manages its Qwen server.\n")
    print(f"Parallel artifacts: {args.out}")


if __name__ == "__main__":
    main()
