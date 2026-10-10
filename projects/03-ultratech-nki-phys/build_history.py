"""Collect available local attempt records and reports, without inventing scores."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def collect(source, destination):
    files = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part.startswith("submission-math-") for part in relative.parts):
            continue
        if ("attempt" in path.name and path.suffix in (".jsonl", ".json")) or path.name == "results.json":
            files.append(path)
    history = destination / "history"
    history.mkdir(exist_ok=False)
    manifest = []
    records = []
    for path in files:
        relative = path.relative_to(source)
        target = history / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest.append(dict(file=str(relative), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             bytes=path.stat().st_size))
        if path.suffix == ".jsonl":
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                records.append(dict(source_file="history/" + str(relative), source_line=number,
                                    record_type="log_row", record=json.loads(line)))
        elif "attempt" in path.name:
            records.append(dict(source_file="history/" + str(relative),
                                record_type="standalone_record", record=json.loads(path.read_text())))
    (history / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "ALL_ATTEMPTS.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records))
    print(json.dumps(dict(original_files=len(files), indexed_records=len(records),
                          note="Records/stages are not unique model proposals")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    collect(args.source.resolve(), args.destination.resolve())
