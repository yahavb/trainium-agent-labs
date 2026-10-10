#!/usr/bin/env bash
# Runs inside the seat-233 pod. Usage: tier1/status.sh [lines] [log-name...]
# Prints the tail of the tier1 job logs with progress bars and known noise removed.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
lines="${1:-12}"; shift || true
names=("$@")
[[ ${#names[@]} -eq 0 ]] && names=($(ls "$ROOT/logs/tier1" | grep -v '^dl_' | sed 's/\.log$//'))
for n in "${names[@]}"; do
  echo "=== $n"
  tr '\r' '\n' < "$ROOT/logs/tier1/$n.log" \
    | grep -vE 'Loading weights|unauthenticated|local_dir_use_symlinks|warnings.warn|torch_dtype|deprecate\(|Inserted NKI|^  }|^\.*$|Loading pipeline' \
    | tail -n "$lines" | cut -c1-1500
done
ls -la /workspace/livevid/artifacts/tier1/ | grep -v work
