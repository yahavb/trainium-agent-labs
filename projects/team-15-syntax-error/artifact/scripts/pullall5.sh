#!/bin/bash
# Mirror run logs from all five seats into runs/live/. seat-73 keeps its logs in the experiment dir;
# the teammate seats (used with their consent, 14:3x) run the identical code from /workspace/s73/.
cd "$(dirname "$0")/.."
export MSYS_NO_PATHCONV=1
scripts/pod.sh pullall >/dev/null 2>&1
for s in 70 71 72 74; do
  POD=seat-$s PROJ=/workspace/s73/exp scripts/pod.sh pullall >/dev/null 2>&1 || echo "seat-$s: pull failed"
done
echo "mirrored 5 seats into runs/live"
