#!/usr/bin/env bash
# sync.sh — move the project and its traces between this machine and the seat pod.
#
# The workshop RBAC allows `kubectl exec` into seat-* pods but NOT `pods/portforward`,
# so the harness runs INSIDE the pod (against its local model server) and the dashboard
# runs on the laptop over files this script copies. tar-over-exec is the whole trick --
# the same mechanism kubectl cp uses.
#
#   ./sync.sh up            push this project dir to $SEAT:/workspace/projects/
#   ./sync.sh down          pull runs/ (traces) back to this machine
#   ./sync.sh sh            a shell in the pod, at the project dir
#   ./sync.sh run ARGS...   run agent.py in the pod (nohup, survives a dropped exec)
#   ./sync.sh log           tail the pod's run.log
#   ./sync.sh ps            is the agent still running in the pod?
#
# SEAT defaults to seat-217; override with SEAT=seat-42 ./sync.sh ...
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="03-kernel-agent-v2"
SEAT="${SEAT:-seat-217}"
POD_BASE="/workspace/projects"

# Credentials live in .context/aws-env.sh (gitignored). If kubectl suddenly reports
# ExpiredToken, ask for a fresh EKSExec block and re-source.
if [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
  if [ -f "$PROJECT_DIR/../../.context/aws-env.sh" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/../../.context/aws-env.sh"
  fi
fi

case "${1:-}" in
  up)
    echo "pushing $PROJ to $SEAT:$POD_BASE/ ..."
    # COPYFILE_DISABLE=1 keeps macOS bsdtar from smuggling AppleDouble (._*) members,
    # whose bytes broke every utf-8 read of docs/ in the pod. --no-xattrs is NOT used:
    # macOS bsdtar rejects it, which made every push a silent no-op for an hour (the
    # local tar died, the pipeline's `|| true` swallowed it, and the import check does
    # not exercise the changed code). The md5 comparison below is the real check.
    COPYFILE_DISABLE=1 tar cf - -C "$PROJECT_DIR/.." --exclude runs --exclude '__pycache__' \
      --exclude '._*' "$PROJ" \
      | kubectl exec -i "$SEAT" -- tar xf - -C "$POD_BASE/" 2>&1 \
      | grep -v "Ignoring unknown extended header" || true
    kubectl exec "$SEAT" -- bash -c "find $POD_BASE/$PROJ -name '._*' -delete; true"
    LOCAL_MD5=$(md5 -q "$PROJECT_DIR/errors.py")
    POD_MD5=$(kubectl exec "$SEAT" -- md5sum "$POD_BASE/$PROJ/errors.py" | cut -d' ' -f1)
    if [ "$LOCAL_MD5" != "$POD_MD5" ]; then
      echo "PUSH FAILED: pod errors.py md5 $POD_MD5 != local $LOCAL_MD5" >&2
      exit 1
    fi
    echo "push verified (errors.py md5 match)"
    ;;
  down)
    mkdir -p "$PROJECT_DIR/runs"
    echo "pulling runs/ from $SEAT ..."
    kubectl exec "$SEAT" -- tar cf - -C "$POD_BASE/$PROJ" runs 2>/dev/null | tar xf - -C "$PROJECT_DIR"
    echo "done; open the dashboard:  python3 dashboard/server.py"
    ;;
  sh)
    kubectl exec -it "$SEAT" -- bash -c "cd $POD_BASE/$PROJ && bash"
    ;;
  run)
    shift
    ARGS="$*"
    [ -z "$ARGS" ] && ARGS="--all --rounds 8 --samples 4"
    kubectl exec "$SEAT" -- bash -c "cd $POD_BASE/$PROJ && nohup python3 agent.py $ARGS > run.log 2>&1 < /dev/null & echo started: \$!"
    echo "watch it:  ./sync.sh log"
    ;;
  log)
    kubectl exec "$SEAT" -- bash -c "tail -n 40 -f $POD_BASE/$PROJ/run.log"
    ;;
  ps)
    kubectl exec "$SEAT" -- bash -c "pgrep -af 'agent.py' || echo 'nothing running'"
    ;;
  watch)
    # pull runs/ every 60s so the local dashboard shows the pod run live
    echo "watching $SEAT runs/ every 60s (Ctrl+C to stop)"
    while true; do
      "$0" down >/dev/null 2>&1 || true
      sleep 60
    done
    ;;
  *)
    sed -n '2,14p' "$0"
    exit 1
    ;;
esac
