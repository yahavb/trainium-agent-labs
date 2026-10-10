#!/usr/bin/env python3
"""Group the final logs by file-name prefix and run final_tables.py on the groups.

    python scripts/final_auto.py analysis/logs/final analysis/final [--l57 PREFIX] [--dry-run]

Prefixes (the part of the name before `_L<level>`, e.g. v82_L1_s117 -> v82): v82 and v82x are 96a9fc9's
runs, v83 is 5c3aba2's (L2CAT=1), final_all is 116's --all run, anything else (e.g. warm) is a
level 5-7 configuration. Groups:
  final      L1, L3, L4 from v82 + v82x;  L2 from v83;  L5-L7 from --l57 (default: warm if present,
             else v83)
  v8.2_L2    L2 from v82 + v82x (the comparison)
  final_all  116's --all run, by itself
  L9-L14     every file whose levels are all in 9-14 (e.g. final_L9_s116), whatever its prefix
  <prefix>   every other prefix that ran levels 5-7, by itself
Levels come from the files' contents, not their names. The baseline + replica go into the compare table.
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report  # noqa: E402


def prefix_of(path):
    name = os.path.basename(path)[:-len(".jsonl")]
    m = re.match(r"^(.*?)_L\d", name)
    if m:
        return m.group(1)
    return re.sub(r"_s\d+$", "", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("out")
    ap.add_argument("--l57", default=None, help="prefix whose L5-L7 runs go into the final group")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files, _ = report.find([a.logs], "")
    by_prefix = collections.defaultdict(list)
    levels = {}
    for f in files["attempts"]:
        by_prefix[prefix_of(f)].append(f)
        levels[f] = sorted({json.loads(l)["level"] for l in open(f) if l.strip()})
    l57 = a.l57 or ("warm" if "warm" in by_prefix else "v83")
    pick = lambda prefixes, lv: [f for p in prefixes for f in by_prefix.get(p, []) if set(levels[f]) <= set(lv)]
    groups = collections.OrderedDict()
    groups["final"] = (pick(["v82", "v82x"], (1, 3, 4)) + pick(["v83"], (2,))
                       + pick([l57], (5, 6, 7)))
    groups["v8.2_L2"] = pick(["v82", "v82x"], (2,))
    groups["final_all"] = by_prefix.get("final_all", [])
    groups["L9-L14"] = [f for fs in by_prefix.values() for f in fs if levels[f] and set(levels[f]) <= set(range(9, 15))]
    for p, fs in by_prefix.items():
        if p not in ("v82", "v82x", "final_all", l57) and any(set(levels[f]) & {5, 6, 7} for f in fs):
            groups[p] = [f for f in fs if set(levels[f]) & {5, 6, 7}]
    print(f"prefixes found: {dict((p, len(fs)) for p, fs in by_prefix.items())}; L5-L7 for final from '{l57}'")
    unused = [f for f in files["attempts"] if not any(f in g for g in groups.values())]
    for name, fs in groups.items():
        lv = collections.Counter(l for f in fs for l in levels[f])
        print(f"  {name}: {len(fs)} files, levels {dict(sorted(lv.items()))}")
    if unused:
        print("  not in any group: " + ", ".join(os.path.relpath(f, a.logs) for f in unused))
    groups = [(n, fs) for n, fs in groups.items() if fs]
    cmd = [sys.executable, os.path.join(HERE, "final_tables.py"), a.out] \
        + [f"{n}={','.join(fs)}" for n, fs in groups] \
        + ["--extra", "baseline=analysis/logs/baseline,analysis/logs/replica_seat119"]
    if a.dry_run:
        print("would run:", " ".join(cmd[:3]), "...")
        return
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
