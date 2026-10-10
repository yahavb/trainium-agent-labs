#!/usr/bin/env python3
"""The final tables, one command: report.py per group and one compare.py table across groups.

    python scripts/final_tables.py OUT_DIR GROUP=ATTEMPTS[,ATTEMPTS...] [GROUP=...] [--ref GROUP]
           [--extra LABEL=DIR[,DIR] ...]

A GROUP is a list of attempts files, which may come from different folders and versions (e.g. "final" =
96a9fc9's L1/L3/L4 files + 5c3aba2's L2 files). For each attempts file X.jsonl, the files next to it named
X_usage.jsonl, X_verdicts.jsonl, X_nki_verdicts.jsonl and run_X.log (feedback_v8's naming) are taken
along. Each group is linked into a scratch folder, report.py runs on it into OUT_DIR/<group>/, and
compare.py puts every group (plus --extra LABEL=DIR, e.g. the baseline) in OUT_DIR/compare.md.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SIBLINGS = ("{}_usage.jsonl", "{}_verdicts.jsonl", "{}_nki_verdicts.jsonl")


def stage(name, files, root):
    d = os.path.join(root, name)
    os.makedirs(d)
    missing = []
    for f in files:
        f = os.path.abspath(f)
        stem = os.path.basename(f)[:-len(".jsonl")]
        folder = os.path.dirname(f)
        prefix = os.path.basename(os.path.dirname(folder)) + "_" + os.path.basename(folder) + "__"
        for src in [f] + [os.path.join(folder, s.format(stem)) for s in SIBLINGS] \
                + [os.path.join(folder, f"run_{stem}.log")]:
            if os.path.exists(src):
                os.symlink(src, os.path.join(d, prefix + os.path.basename(src)))
            elif src != f:
                missing.append(os.path.basename(src))
            else:
                sys.exit(f"{name}: no such attempts file {f}")
    return d, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("groups", nargs="+", help="GROUP=ATTEMPTS[,ATTEMPTS...]")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--extra", nargs="*", default=[], help="LABEL=DIR[,DIR...] passed to compare.py as is")
    ap.add_argument("--baseline", default="analysis/logs/baseline/attempts.jsonl")
    a = ap.parse_args()
    root = tempfile.mkdtemp(prefix="final_tables_")
    os.makedirs(a.out, exist_ok=True)
    staged = []
    for g in a.groups:
        name, _, files = g.partition("=")
        d, missing = stage(name, files.split(","), root)
        staged.append((name, d))
        print(f"== {name}: {len(files.split(','))} attempts file(s)"
              + (f"; missing companions: {', '.join(missing)}" if missing else ""))
        subprocess.run([sys.executable, os.path.join(HERE, "report.py"), os.path.join(a.out, name), d,
                        "--baseline", a.baseline], check=False)
    cmd = [sys.executable, os.path.join(HERE, "compare.py")] + [f"{n}={d}" for n, d in staged] + a.extra \
        + ["--ref", a.ref or staged[0][0], "-o", os.path.join(a.out, "compare")]
    subprocess.run(cmd, check=False)
    shutil.rmtree(root)


if __name__ == "__main__":
    main()
