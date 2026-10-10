#!/usr/bin/env bash
# Usage: tier1/run_live.sh [--restart-server | --stop-server] [client.py args...]
# Starts tier1/server.py on the seat-233 pod if it is not already running, waits until it is
# ready, then starts the webcam client on this Mac. Client args: --record, --duration N, ...
set -euo pipefail
source "$(dirname "$0")/_seat.sh"

VENV="$REPO_ROOT/venvs/client"
PIDFILE="$REMOTE_ROOT/logs/tier1/server.pid"
LOG="$REMOTE_ROOT/logs/tier1/server.log"
pod() { kubectl exec -i "$SEAT" -c "$CONTAINER" -- bash -lc "$1"; }
server_running() { pod "test -f $PIDFILE && kill -0 \$(cat $PIDFILE) 2>/dev/null" 2>/dev/null; }
stop_server() {
  pod "test -f $PIDFILE && kill \$(cat $PIDFILE) 2>/dev/null; rm -f $PIDFILE; sleep 2" || true
  echo "server stopped"
}

case "${1:-}" in
  --stop-server) stop_server; exit 0 ;;
  --restart-server) stop_server; shift ;;
esac

if [[ ! -x "$VENV/bin/python" ]]; then
  echo ">> creating client venv at $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q opencv-python numpy
fi

if server_running; then
  echo ">> server already running on $SEAT"
else
  echo ">> starting server on $SEAT"
  "$TIER1_DIR/push.sh" > /dev/null
  # The daemon is backgrounded on its own line: backgrounding a whole "a && b &" list would leave a
  # subshell holding the exec stream open. setsid execs python directly here, so $! is its pid.
  pod "source /workspace/venvs/tnx/neuron_env.sh; mkdir -p $REMOTE_ROOT/logs/tier1; cd $REMOTE_ROOT/tier1 || exit 1
       nohup setsid python -u server.py > $LOG 2>&1 < /dev/null &
       echo \$! > $PIDFILE"
fi

echo -n ">> waiting for server"
for _ in $(seq 1 120); do
  if pod "grep -q '^READY' $LOG 2>/dev/null" 2>/dev/null; then ready=1; break; fi
  if ! server_running; then echo; echo "error: server exited; last log lines:" >&2; pod "tail -n 20 $LOG" >&2; exit 1; fi
  echo -n "."; sleep 3
done
echo
[[ "${ready:-}" == 1 ]] || { echo "error: server not ready after 6 minutes; see $LOG on the pod" >&2; exit 1; }
pod "grep '^READY' $LOG | tail -1"

exec "$VENV/bin/python" "$TIER1_DIR/client.py" "$@"
