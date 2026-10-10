#!/usr/bin/env python3
"""
dashboard/collect.py -- the live page in one command: every seat's logs out of its pod, every branch's
results files out of git, then dashboard/index.html built from exactly those files.

    python dashboard/collect.py                               # the team's seats, in the terminal with the AWS credentials
    python dashboard/collect.py --seats 102 --pod-dir 102=/workspace/chipboost/projects/03-chipboost
    python dashboard/collect.py --no-pods                     # git only: results files, no pod logs

Pods: `kubectl cp` copies each seat's whole log folder (SOURCES below: P1's comparison output in /tmp on
seat-100, P3's repo at /workspace on seat-101, <POD_DIR>/logs/seat-N otherwise), so every attempts*.jsonl,
sweep*.jsonl and <arm>-r<n>.jsonl comes along (P2 writes one file per concurrent run). Git: the newest version of every
*results*.json, attempts*.jsonl and sweep*.jsonl on any origin branch. A seat's pod copy wins over its git
copy, so no line is counted twice. Everything lands in dashboard/collected/ (gitignored).

Read-only: nothing in a pod or on a branch changes. A log caught mid-append loses only its last half
line, which build.py skips and lists; the next collect picks it up.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT = HERE / "collected"
POD_DIR = "/workspace/chipboost/projects/03-chipboost"   # where the fork is cloned inside the seat pods
SEATS = [100, 101, 102]
# Folders to copy, per seat, where they differ from <POD_DIR>/logs/seat-N:
SOURCES = {
    100: "/tmp/p1-comparison-20261010-2",                   # P1: the pinned three-arm comparison (run_comparison.py)
    101: "/workspace/projects/03-chipboost/logs/seat-101",  # P3: its own referee and model_alone loops
}
# run_comparison.py names its per-arm logs <arm>-r<repeat>.jsonl: attempt logs, though not named attempts*.
COMPARISON_LOG = re.compile(r"^(referee|model_alone|random_search)-r\d+\.jsonl$")
# Single log files, by glob, copied one by one: P1's v2 runs each write pilot.jsonl into their own /tmp
# folder, beside full repo snapshots whose results files must not be collected twice. P1's continuation
# and recovery runs (/tmp/p1-qwen-continuation-*, /tmp/redteam-recovery-*) are left out on purpose: they
# start from the 1.517x winner, not the start kernel, and neither went beyond it.
POD_FILES = {
    100: ("/tmp/p1-qwen-v2-*/pilot.jsonl",),
}


def run(cmd, cwd, text=True):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=text)


def wanted(name):
    if name.startswith("fake_"):
        return False
    return ((name.endswith(".json") and "results" in name) or
            (name.endswith(".jsonl") and (name.startswith(("attempts", "sweep")) or bool(COMPARISON_LOG.match(name)))))


def from_git():
    """The newest committed version of each result or log file, across every origin branch."""
    top = Path(run(["git", "rev-parse", "--show-toplevel"], PROJECT).stdout.strip())
    prefix = PROJECT.relative_to(top).as_posix() + "/"
    if run(["git", "fetch", "--all", "--prune", "-q"], top).returncode:
        print("  git fetch failed: using the branches as last fetched")
    refs = [r for r in run(["git", "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin"], top).stdout.split()
            if r != "origin" and not r.endswith("/HEAD")]
    newest = {}
    for ref in refs:
        for f in run(["git", "ls-tree", "-r", "--name-only", ref, "--", prefix], top).stdout.splitlines():
            if not wanted(f.rsplit("/", 1)[-1]):
                continue
            ts = int(run(["git", "log", "-1", "--format=%ct", ref, "--", f], top).stdout.strip() or 0)
            if f not in newest or ts > newest[f][0]:
                newest[f] = (ts, ref)
    got = {}
    for f, (_, ref) in sorted(newest.items()):
        rel = f[len(prefix):]
        dest = OUT / "git" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(run(["git", "show", f"{ref}:{f}"], top, text=False).stdout)
        got[rel] = dest
        print(f"  git  {ref.split('/', 1)[-1]:<16} {rel}")
    return got


def from_pods(seats, pod_dirs):
    got, copied = [], set()
    for seat in seats:
        src = f"{pod_dirs[seat]}/logs/seat-{seat}" if seat in pod_dirs else SOURCES.get(seat, f"{POD_DIR}/logs/seat-{seat}")
        rel = Path("pods") / f"seat-{seat}"
        shutil.rmtree(OUT / rel, ignore_errors=True)
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        # kubectl cp reads "C:" in a Windows path as a pod name, so the destination stays relative.
        try:
            p = run(["kubectl", "cp", f"seat-{seat}:{src}", rel.as_posix()], OUT)
        except FileNotFoundError:
            print("  kubectl is not on PATH in this terminal (README step 1); skipping the pods")
            break
        files = sorted(f for f in (OUT / rel).rglob("*.jsonl") if wanted(f.name)) if (OUT / rel).exists() else []
        if p.returncode or not files:
            why = (p.stderr.strip().splitlines() or ["no attempts*/sweep*.jsonl there yet"])[-1]
            print(f"  pod  seat-{seat:<11} nothing copied: {why}")
            continue
        copied.add(seat)
        got += files
        print(f"  pod  seat-{seat:<11} {', '.join(f.name for f in files)}")
    for seat, patterns in POD_FILES.items():
        if seat not in seats:
            continue
        ls = run(["kubectl", "exec", f"seat-{seat}", "-c", "app", "--", "sh", "-c", f"ls -1 {' '.join(patterns)} 2>/dev/null"], OUT)
        for src in ls.stdout.split():
            # /tmp/p1-qwen-v2-20261010-1/pilot.jsonl -> pods/seat-100/extra/attempts-p1-qwen-v2-20261010-1.jsonl
            rel = Path("pods") / f"seat-{seat}" / "extra" / f"attempts-{src.rstrip('/').split('/')[-2]}.jsonl"
            (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
            p = run(["kubectl", "cp", f"seat-{seat}:{src}", rel.as_posix()], OUT)
            if p.returncode or not (OUT / rel).exists():
                print(f"  pod  seat-{seat:<11} {src}: not copied")
                continue
            got.append(OUT / rel)
            print(f"  pod  seat-{seat:<11} {src} -> {rel.name}")
    return got, copied


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seats", nargs="*", type=int, default=SEATS, help=f"seats to copy logs from (default {SEATS})")
    ap.add_argument("--pod-dir", action="append", default=[], metavar="SEAT=DIR",
                    help=f"where the repo's 03-chipboost folder is in that seat's pod (default {POD_DIR})")
    ap.add_argument("--no-pods", action="store_true", help="git only")
    a = ap.parse_args()
    pod_dirs = {int(s.split("=", 1)[0]): s.split("=", 1)[1] for s in a.pod_dir}
    sys.stdout.reconfigure(line_buffering=True)   # keep this file's lines in order with build.py's

    OUT.mkdir(exist_ok=True)
    print("from git:")
    git = from_git()
    pods, copied = [], set()
    if a.seats and not a.no_pods:
        print("from the pods:")
        pods, copied = from_pods(a.seats, pod_dirs)
    # A seat copied from its pod is newer than anything it pushed: drop its git copies.
    files = pods + [p for rel, p in git.items()
                    if not any(rel.startswith(f"logs/seat-{s}/") for s in copied)]
    logs = [str(p) for p in files if p.name.startswith("attempts") or COMPARISON_LOG.match(p.name)]
    sweep = [str(p) for p in files if p.name.startswith("sweep")]
    results = [str(p) for p in files if p.name.endswith(".json")]
    cmd = [sys.executable, str(HERE / "build.py"), *logs, "--results", *results, "--sweep", *sweep]
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
