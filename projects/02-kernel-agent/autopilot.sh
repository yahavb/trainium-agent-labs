#!/usr/bin/env bash
# Keeps the seat and the fork in step, so nobody has to carry files by hand.
#
#   bash autopilot.sh           start it (background; survives a dropped connection)
#   bash autopilot.sh status    is it running, and what did it last do
#   bash autopilot.sh stop      stop it (running experiments are left alone)
#
# Once a minute it does three things:
#
#   1. RESULTS UP.   Copies every attempt log, run log and report under results*/ to the branch
#                    `seat17-results` on the fork, with a STATUS.txt snapshot. Read-only on the
#                    experiment: it never touches a running job.
#   2. CODE DOWN.    Fast-forwards this checkout to the branch on the fork. A job that is already
#                    running keeps the code it started with; the next job gets the new code.
#   3. NEXT JOB.     When no experiment is running or queued, starts the next file in queue/ that
#                    has not been started on this seat. One job at a time, always: two agents on one
#                    model were measured at 4 minutes a round instead of 1.
#
# A queue file is four settings and nothing else -- it is parsed, not executed:
#
#       OUT=results-example
#       MODES="directed3 enriched"
#       REPEAT=1
#       AGENT_ARGS="--levels 3,4 --rounds 12 --samples 4 --context 8192"
#
#   or, to drop jobs that are waiting or running:   CANCEL="results-repeat2 results-baselines"
#
# Needs the `mine` remote to be able to push (the token set once with git remote set-url --push).
set -uo pipefail
cd "$(dirname "$0")"
PROJ=$(pwd)
ROOT=$(git rev-parse --show-toplevel)
EVERY=${AUTOPILOT_EVERY:-60}
REMOTE=${AUTOPILOT_REMOTE:-mine}
RESULTS_BRANCH=${AUTOPILOT_RESULTS_BRANCH:-seat17-results}
TREE=$PROJ/.seat-results
DONE=$PROJ/.queue-done
PIDFILE=$PROJ/.autopilot.pid
LOG=$PROJ/autopilot.log
PY=${PYTHON:-python}
command -v "$PY" >/dev/null 2>&1 || PY=python3

say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
alive() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null \
          && ! grep -q '^State:[[:space:]]*Z' "/proc/$(cat "$PIDFILE")/status" 2>/dev/null; }
# Busy means an experiment process exists: a go.sh job (running or waiting its turn), an agent, or
# the hand-over waiter. Matched on the interpreter plus the script, so a shell or editor that merely
# has one of these file names on its command line does not count.
busy() { pgrep -f "bash [^ ]*go[.]sh _run" >/dev/null \
         || pgrep -f "python[^ ]* ([^ ]*/)?agent[.]py" >/dev/null \
         || pgrep -f "bash [^ ]*after_directed[.]sh _wait" >/dev/null; }

# One entry per agent process on this seat, ours or not: how long it has run, where, and how far its
# attempt log has got. Counts and positions only -- the contents of a log that is not ours are never
# copied anywhere. This is what tells us when a chip someone else is using will be free.
agents_progress() {
  "$PY" - <<'EOF'
import json, os, subprocess
found = False
for pid in sorted((p for p in os.listdir("/proc") if p.isdigit()), key=int):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = [c.decode("utf-8", "replace") for c in f.read().split(b"\0") if c]
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        continue
    if len(cmd) < 2 or "python" not in os.path.basename(cmd[0]) or os.path.basename(cmd[1]) != "agent.py":
        continue
    found = True
    try:
        up = subprocess.run(["ps", "-o", "etime=", "-p", pid], capture_output=True, text=True).stdout.strip()
    except OSError:
        up = "?"

    def arg(name, default=None):
        return cmd[cmd.index(name) + 1] if name in cmd and cmd.index(name) + 1 < len(cmd) else default

    print(f"pid {pid}   running for {up or '?'}   in {cwd}")
    print(f"   options: {' '.join(cmd[2:])[:120]}")
    log = arg("--log")
    if not log:
        print("   no --log option, so its progress cannot be read")
        continue
    path = log if os.path.isabs(log) else os.path.join(cwd, log)
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError as e:
        print(f"   log {path}: cannot read ({e.strerror})")
        continue
    runs, prev, last, rounds, n = 1, None, None, set(), 0
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        n += 1
        lv = r.get("level")
        if isinstance(lv, int) and isinstance(prev, int) and lv < prev:
            runs += 1                      # the level number went back down: the next repeat began
        prev, last = lv, r
        rounds.add((runs, lv, r.get("round")))
    if last is None:
        print(f"   log {path}: empty so far")
        continue
    print(f"   log {path}: {n} attempts, {len(rounds)} rounds done; now on run {runs} of "
          f"{arg('--repeat', '1')}, level {last.get('level')}, round {last.get('round')} "
          f"(up to {arg('--rounds', '?')} rounds a level)")
if not found:
    print("none")
EOF
}

results_up() {
  if [ ! -d "$TREE/.git" ]; then
    git init -q "$TREE" && git -C "$TREE" checkout -q -b "$RESULTS_BRANCH" \
      && git -C "$TREE" remote add origin "$(git -C "$ROOT" remote get-url --push "$REMOTE")" \
      || { say "results: could not set up $TREE"; return; }
  fi
  "$PY" - "$PROJ" "$TREE" <<'EOF' || { say "results: copy failed"; return; }
import glob, os, shutil, sys
proj, tree = sys.argv[1:3]
keep = (".jsonl", ".log", ".md")
for d in sorted(glob.glob(os.path.join(proj, "results*"))):
    for root, _, names in os.walk(d):
        for n in names:
            if not n.endswith(keep):
                continue
            src = os.path.join(root, n)
            dst = os.path.join(tree, os.path.relpath(src, proj))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst) \
                    or os.path.getsize(src) != os.path.getsize(dst):
                shutil.copy2(src, dst)
for n in ("autopilot.log", "handover.log"):
    if os.path.isfile(os.path.join(proj, n)):
        shutil.copy2(os.path.join(proj, n), os.path.join(tree, n))
EOF
  {
    echo "seat $(hostname)   $(date -u +%Y-%m-%dT%H:%M:%SZ)   code $(git -C "$ROOT" log -1 --format='%h %s' | cut -c1-70)"
    echo "busy: $(busy && echo yes || echo no)   started on this seat: $(ls "$DONE" 2>/dev/null | tr '\n' ' ')"
    echo
    for d in results*/; do
      [ -d "$d" ] || continue
      echo "==================== ${d%/} ===================="
      for log in "$d"*.log; do
        [ -f "$log" ] || continue
        name=$(basename "$log" .log); case "$name" in ablation|selftest) continue ;; esac
        echo "--- feedback = $name ---"
        grep -hE "^=========== level|^round |SOLVED on round|STOPPING|not solved in|^  level [0-9]|^  solved " "$log" | cut -c1-110 | tail -30
      done
      grep -h "^#####" "$d/ablation.log" 2>/dev/null | tail -4; echo
    done
    echo "==================== processes ===================="
    pgrep -af "agent[.]py|go[.]sh _run" | cut -c1-150
    echo
    echo "==================== agents on this chip: how far along ===================="
    agents_progress
  } > "$TREE/STATUS.txt" 2>&1
  git -C "$TREE" add -A >/dev/null 2>&1
  if [ -n "$(git -C "$TREE" status --porcelain 2>/dev/null)" ]; then
    git -C "$TREE" -c user.name="seat17 autopilot" -c user.email="seat17@users.noreply.github.com" \
      commit -q -m "results $(date -u +%H:%M:%S)" >/dev/null 2>&1
    if git -C "$TREE" push -q -f origin "$RESULTS_BRANCH" >/dev/null 2>&1; then
      say "results: pushed"
    else
      say "results: PUSH FAILED (token expired or no network?)"
    fi
  fi
}

code_down() {
  local branch before after
  branch=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)
  before=$(git -C "$ROOT" rev-parse --short HEAD)
  if git -C "$ROOT" pull -q --ff-only "$REMOTE" "$branch" >/dev/null 2>&1; then
    after=$(git -C "$ROOT" rev-parse --short HEAD)
    [ "$before" != "$after" ] && say "code: $before -> $after"
  else
    say "code: pull failed on $branch (left as it is)"
  fi
}

setting() {   # setting FILE KEY -> the value, quotes removed; only the first matching line counts
  sed -n "s/^$2=//p" "$1" | head -1 | sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/"
}

next_job() {
  mkdir -p "$DONE"
  local job name out modes repeat args cancel d
  for job in queue/*.job; do
    [ -f "$job" ] || continue
    name=$(basename "$job")
    [ -e "$DONE/$name" ] && continue
    cancel=$(setting "$job" CANCEL)
    if [ -n "$cancel" ]; then                       # a cancel runs at once, busy or not
      touch "$DONE/$name"
      for d in $cancel; do
        [[ "$d" =~ ^results(-[a-z0-9-]+)?$ ]] || { say "queue: $name: refused to cancel '$d'"; continue; }
        [ -d "$d" ] && OUT=$d bash go.sh stop >/dev/null 2>&1 && say "queue: $name: stopped $d"
      done
      continue
    fi
    busy && return                                  # everything else waits its turn
    out=$(setting "$job" OUT); modes=$(setting "$job" MODES)
    repeat=$(setting "$job" REPEAT); args=$(setting "$job" AGENT_ARGS)
    touch "$DONE/$name"
    # Whole-value checks: a job may only name a results folder, feedback modes, a count and
    # agent.py options. Anything else is skipped, not run.
    [[ "$out" =~ ^results-[a-z0-9-]+$ ]] || { say "queue: $name: bad OUT, skipped"; continue; }
    [[ "$modes" =~ ^[a-z0-9]+(\ [a-z0-9]+)*$ ]] || { say "queue: $name: bad MODES, skipped"; continue; }
    [[ "${repeat:-1}" =~ ^[0-9]+$ ]] || { say "queue: $name: bad REPEAT, skipped"; continue; }
    [[ "$args" =~ ^[A-Za-z0-9\ ,._=-]+$ ]] || { say "queue: $name: bad AGENT_ARGS, skipped"; continue; }
    say "queue: starting $name  ->  $out  modes '$modes'  repeat ${repeat:-1}  args '$args'"
    OUT=$out MODES=$modes REPEAT=${repeat:-1} AGENT_ARGS=$args bash go.sh >> "$LOG" 2>&1
    return                                          # one job per pass
  done
}

case "${1:-}" in
  _loop)
    say "autopilot started on $(hostname), every ${EVERY}s"
    # If a pull brings a newer copy of this script, switch to it. Otherwise the loop keeps running
    # the code it started with, and a fix pushed to the fork would need a restart by hand.
    MINE=$(cksum < "$PROJ/autopilot.sh")
    while :; do
      results_up
      code_down
      if [ "$(cksum < "$PROJ/autopilot.sh")" != "$MINE" ]; then
        say "autopilot: this script changed, switching to the new copy"
        exec bash "$PROJ/autopilot.sh" _loop
      fi
      next_job
      sleep "$EVERY"
    done ;;
  status)
    if alive; then echo "RUNNING."; else echo "NOT RUNNING. Start it with:  bash $PROJ/autopilot.sh"; fi
    echo "experiment in progress: $(busy && echo yes || echo no)"
    echo "--- last lines ---"; tail -8 "$LOG" 2>/dev/null
    exit 0 ;;
  stop)
    if alive; then kill -- "-$(cat "$PIDFILE")" 2>/dev/null || kill "$(cat "$PIDFILE")" 2>/dev/null
      rm -f "$PIDFILE"; echo "Stopped. Running experiments are left alone."
    else echo "It is not running."; fi
    exit 0 ;;
  "") ;;
  *) echo "usage: bash autopilot.sh [status|stop]"; exit 2 ;;
esac

if alive; then echo "Already running. See:  bash $PROJ/autopilot.sh status"; exit 0; fi
if [ -z "${AUTOPILOT_SKIP_TOKEN_CHECK:-}" ] && \
   ! git -C "$ROOT" remote get-url --push "$REMOTE" 2>/dev/null | grep -q "@"; then
  echo "The '$REMOTE' remote has no push token, so results cannot be sent. Set it first."; exit 1
fi
SETSID=$(command -v setsid || true)
nohup $SETSID bash "$PROJ/$(basename "$0")" _loop > /dev/null 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "Autopilot is on. Once a minute it sends results to the fork, takes new code, and starts the"
echo "next queued run when the seat is free. It keeps going if your connection drops."
echo "  Check it:  bash $PROJ/autopilot.sh status"
echo "  Stop it:   bash $PROJ/autopilot.sh stop"
