#!/bin/bash
# Emit a ladder schedule, copy it to a Trainium node, and check it three ways:
#   1. nkibench (rule scan + nki.simulate numerics + HBM traffic bar)
#   2. nki.simulate with float32 + bfloat16 inputs
#   3. compiled with neuronx-cc and executed on the NeuronCore (numerics only; wall time is launch-dominated)
#
#   ENV_FILE=/path/to/.env scripts/device_check.sh l5
#
# Environment:
#   ENV_FILE  file exporting AWS credentials     (default: $ROOT/../.env; pass it explicitly from a git worktree)
#   POD       pod to use                         (default: seat-270)
#   NKIBENCH_DIR  directory holding nkibench.py on the node (default: /workspace/projects/02-kernel-agent)
#   PYTHON    interpreter with torch + numpy     (default: python3)
#   SKIP_HW=1 skip step 3
set -euo pipefail
LEVEL_NAME=${1:?usage: device_check.sh l4|l5|l6|l7}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
POD=${POD:-seat-270}
NKIBENCH_DIR=${NKIBENCH_DIR:-/workspace/projects/02-kernel-agent}
PYTHON=${PYTHON:-python3}
ENV_FILE=${ENV_FILE:-$ROOT/../.env}
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
N=${LEVEL_NAME#l}
TMP=$(mktemp -d)
OUT=$TMP/kernel_$LEVEL_NAME.py
NAME=$(basename "$OUT")

"$PYTHON" "$ROOT/examples/matmul_ladder.py" "$LEVEL_NAME" "$OUT" >/dev/null
ENTRY=$(grep -m1 '^def ' "$OUT" | sed 's/def \([a-zA-Z0-9_]*\).*/\1/')
kubectl cp -c app "$OUT" "$POD:/tmp/$NAME"
kubectl cp -c app "$ROOT/scripts/sim_dtypes.py" "$POD:/tmp/sim_dtypes.py"
kubectl cp -c app "$ROOT/scripts/device_run.py" "$POD:/tmp/device_run.py"
echo "== 1. nkibench level $N =="
kubectl exec "$POD" -c app -- bash -c "cd $NKIBENCH_DIR && python nkibench.py --level $N --check /tmp/$NAME" | grep -E "rules|numerics|HBM traffic|VIOLATION|FAILED" | cut -c1-160
echo "== 2. simulator, float32 + bfloat16 =="
kubectl exec "$POD" -c app -- python /tmp/sim_dtypes.py "/tmp/$NAME" "$ENTRY"
if [ -z "${SKIP_HW:-}" ]; then
  echo "== 3. NeuronCore =="
  kubectl exec "$POD" -c app -- bash -c "cd /tmp && NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python /tmp/device_run.py /tmp/$NAME $ENTRY 2"
fi
