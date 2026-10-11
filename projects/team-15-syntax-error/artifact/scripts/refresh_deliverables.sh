#!/bin/bash
# Refresh every generated deliverable from the latest runs, snapshot, commit, push.
#   scripts/refresh_deliverables.sh <label>
# 1. mirror all five seats into runs/live/ and snapshot it into runs/<HHMM>_<label>/ (tracked in git)
# 2. regenerate FAILURES.md, TOKENS.md, ATTEMPTS.md + attempts_all.csv, RESULTS_AUTO.md
# 3. commit + push
# CALIBRATION.md needs the SDK (held-out simulation), so it is regenerated in the pod by
# scripts/refresh_calibration.sh; NOTE.md / RESULTS.md / CHECKER.md / SUBMISSION.md are written by hand.
set -euo pipefail
cd "$(dirname "$0")/.."
label=${1:-refresh}
scripts/pullall5.sh
T=$(scripts/now | tr -d :)
snap="runs/${T}_${label}"
mkdir -p "$snap"
cp runs/live/*.jsonl runs/live/*.log "$snap"/ 2>/dev/null || true
ALL="runs/live/*.jsonl"   # the complete mirror of all five seats; runs/<HHMM>_<label>/ is the committed archive
python scripts/taxonomy.py "$ALL" > FAILURES.md
python scripts/token_report.py "$ALL" > TOKENS.md
python scripts/attempt_log.py "$ALL" >/dev/null
python scripts/results.py "$ALL" > RESULTS_AUTO.md
{ echo "# RESULTS_AUTO — every experiment's per-level rate, regenerated from all logs ($(scripts/now) ET)"; echo;
  echo '```'; cat RESULTS_AUTO.md; echo '```'; } > RESULTS_AUTO.tmp && mv RESULTS_AUTO.tmp RESULTS_AUTO.md
echo "- **$(scripts/now)** refresh_deliverables ($label): snapshot \`$snap/\`; regenerated FAILURES.md, TOKENS.md, ATTEMPTS.md, attempts_all.csv, RESULTS_AUTO.md." >> LOG.md
git add -A
git commit -qm "refresh deliverables: $label

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
git push -q 2>&1 | tail -1
echo "refreshed ($snap)"
