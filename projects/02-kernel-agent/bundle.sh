#!/usr/bin/env bash
# Pack everything the write-up needs into ONE file that can be copied off the pod.
#
#   bash bundle.sh
#
# Safe to run while an experiment is still going: it only reads. It writes
#   RESULTS.md                      the comparison table (report.py)
#   <log>.taxonomy.md               mistakes grouped and counted, next to every attempt log
#   /workspace/results-bundle.tgz   all of the above plus the attempt logs and run logs
set -uo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
command -v "$PY" >/dev/null 2>&1 || PY=python3

"$PY" report.py --cut 8 > RESULTS.md 2> report.err || { echo "report.py failed:"; cat report.err; }
for f in results*/*.jsonl results*/previous-*/*.jsonl; do
  [ -f "$f" ] && "$PY" taxonomy.py --log "$f" --examples > "${f%.jsonl}.taxonomy.md" 2>&1
done

OUT=${BUNDLE:-/workspace/results-bundle.tgz}
"$PY" - "$OUT" <<'EOF'
import glob, os, sys, tarfile
out = sys.argv[1]
keep = (".jsonl", ".log", ".md")
files = [f for f in ["RESULTS.md", "handover.log"] if os.path.isfile(f)]
for d in sorted(glob.glob("results*")):
    for root, _, names in os.walk(d):
        files += [os.path.join(root, n) for n in sorted(names) if n.endswith(keep)]
with tarfile.open(out, "w:gz") as tar:
    for f in files:
        tar.add(f)
size = os.path.getsize(out)
print(f"Packed {len(files)} files into {out} ({size / 1024:.0f} KB).")
EOF

echo
echo "To hand it over:"
echo "  1. Type   exit   to leave the pod (the experiment keeps running)."
echo "  2. On your Mac, run:"
echo "       kubectl exec $(hostname) -- cat $OUT > ~/Desktop/results-bundle.tgz"
echo "  3. Attach results-bundle.tgz from your Desktop to the chat."
echo "  4. To come back:  kubectl exec -it $(hostname) -- bash"
