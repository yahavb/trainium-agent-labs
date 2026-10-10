#!/bin/bash
# M1: bring every seat's results into the repo's results/ folder (run on your laptop, from the repo root,
# in a terminal where the workshop credentials are pasted -- see SERVER.md).
#
#     bash veriloop/collect.sh 30 31 33 230 231 232 233 234
#
# Copies each seat's /workspace/veriloop/results/*_summary.csv and *_attempts.jsonl. File names already
# carry the seat's tag (e.g. 2026-10-10_s30_summary.csv), so nothing is overwritten across seats.
set -u
[ $# -gt 0 ] || { echo "usage: bash veriloop/collect.sh SEAT [SEAT ...]"; exit 1; }
mkdir -p results
for n in "$@"; do
  files=$(kubectl exec "seat-$n" -c app -- bash -c 'ls /workspace/veriloop/results/*_summary.csv /workspace/veriloop/results/*_attempts.jsonl 2>/dev/null' 2>/dev/null)
  if [ -z "$files" ]; then echo "seat-$n: no results yet"; continue; fi
  for f in $files; do
    kubectl cp "seat-$n:$f" "results/$(basename "$f")" -c app >/dev/null 2>&1 && echo "seat-$n: $(basename "$f")"
  done
done
