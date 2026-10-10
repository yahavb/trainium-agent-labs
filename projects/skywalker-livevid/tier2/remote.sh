#!/usr/bin/env bash
# Local helpers for gate G2, seat-230 by default (G2_SEAT=seat-231 for extra compiles; no other seat is allowed).
#   tier2/remote.sh push                 sync local tier2/ -> pod /workspace/livevid/tier2
#   tier2/remote.sh run "<cmd>"          run <cmd> in the pod from tier2/ with the tnx venv, foreground
#   tier2/remote.sh bg <name> "<cmd>"    start <cmd> detached (nohup setsid); log -> logs/tier2/<name>.log, ends "EXIT <rc>"
#   tier2/remote.sh pull <remote> <dir>  copy a pod file/dir into local <dir>
set -euo pipefail
SEAT="${G2_SEAT:-seat-230}"
[[ "$SEAT" == "seat-230" || "$SEAT" == "seat-231" ]] || { echo "error: tier2-g2 only uses seat-230 / seat-231, got '$SEAT'" >&2; exit 1; }
CONTAINER="app"
REMOTE_ROOT="/workspace/livevid"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The pod sees 192 CPUs but its cgroup quota is 11; cap torch threads so the CPU reference is not throttled.
ENVSETUP="source /workspace/venvs/tnx/neuron_env.sh && export OMP_NUM_THREADS=\${THREADS:-8} MKL_NUM_THREADS=\${THREADS:-8} && mkdir -p $REMOTE_ROOT/logs/tier2 $REMOTE_ROOT/artifacts/tier2 && cd $REMOTE_ROOT/tier2"

cmd="${1:-}"; shift || true
case "$cmd" in
  push)
    cd "$REPO_ROOT"
    kubectl exec -i "$SEAT" -c "$CONTAINER" -- mkdir -p "$REMOTE_ROOT/tier2" "$REMOTE_ROOT/logs/tier2"
    COPYFILE_DISABLE=1 tar -cz --no-mac-metadata --no-xattrs --exclude=.DS_Store --exclude=__pycache__ tier2 2>/dev/null \
      | kubectl exec -i "$SEAT" -c "$CONTAINER" -- tar -xz --no-same-owner -C "$REMOTE_ROOT" 2>/dev/null
    echo "pushed tier2 -> $SEAT:$REMOTE_ROOT/tier2"
    ;;
  run)
    kubectl exec -i "$SEAT" -c "$CONTAINER" -- bash -lc "$ENVSETUP && $*"
    ;;
  bg)
    name="$1"; shift
    kubectl exec -i "$SEAT" -c "$CONTAINER" -- bash -lc \
      "$ENVSETUP && (nohup setsid bash -c '$*; echo \"EXIT \$?\"' > $REMOTE_ROOT/logs/tier2/$name.log 2>&1 < /dev/null &) ; echo started $name"
    ;;
  pull)
    remote="$1"; local_dir="${2%/}"
    [[ "$remote" = /* ]] || remote="$REMOTE_ROOT/$remote"
    mkdir -p "$local_dir"
    kubectl exec -i "$SEAT" -c "$CONTAINER" -- tar -cz -C "$(dirname "$remote")" "$(basename "$remote")" | tar -xz -C "$local_dir"
    echo "pulled $SEAT:$remote -> $local_dir/"
    ;;
  *) echo "usage: $0 push | run \"<cmd>\" | bg <name> \"<cmd>\" | pull <remote> <local_dir>" >&2; exit 2 ;;
esac
