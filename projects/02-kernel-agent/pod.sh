#!/usr/bin/env bash
# pod.sh -- run from your LAPTOP, in a terminal that has the workshop credentials pasted.
# Moves this folder to your seat pod, starts measured runs with the same settings every time, and
# brings the logs back. The pod is only a runner: edit here, never in the pod.
#
#   export SEAT=154                  your own seat number, once per terminal
#
#   ./pod.sh serve                   start the model on this seat and wait for READY (about 4 min)
#   ./pod.sh push                    copy this folder's code into the pod
#   ./pod.sh check                   prove the checker in the pod: nkibench + diagnose selftests, probe
#   ./pod.sh launch NAME [FEATURES] [LEVEL]   serve, push, check and run, stopping at the first
#                                    that fails. For bringing up a free seat in one go.
#   ./pod.sh run NAME [FEATURES] [LEVEL]   start a measured run in the background, e.g.
#                                      ./pod.sh run baseline
#                                      ./pod.sh run locate locate
#                                      ./pod.sh run state-l3 state 3     level 3 only
#   ./pod.sh replay NAME             what the features would have told the model, for every failing
#                                    kernel in NAME.jsonl. Calls no model; safe beside a run.
#   ./pod.sh findings                run findings/run_findings.py on the real simulator; saves
#                                    results/seat-N/findings.out. Needs no model; safe beside a run.
#   ./pod.sh device                  stop the model server and run findings/device_check.py on the
#                                    real chip; saves results/seat-N/device.out
#   ./pod.sh stop                    stop the run going on this seat. Its log so far is kept.
#   ./pod.sh status [NAME]           is a run going, and the last lines of its output
#   ./pod.sh pull NAME               copy NAME.jsonl and NAME.out into results/seat-$SEAT/
#   ./pod.sh get REMOTE [LOCAL]      copy any one file from the pod's project folder
#
# Every measured run uses the same command, so only the code and the features differ between two
# runs. Change RUN_ARGS and you can no longer compare against anything run before. Running one
# LEVEL instead of all four is fine: the loop treats each level on its own, so a level's result
# does not depend on which other levels were in the run.
set -euo pipefail

RUN_ARGS=${RUN_ARGS:---all --rounds 8 --samples 4 --context 8192 --repeat 5}
REMOTE=${REMOTE:-/workspace/projects/02-kernel-agent}
HERE=$(cd "$(dirname "$0")" && pwd)
RESULTS=$(cd "$HERE/../.." && pwd)/results

[ -n "${SEAT:-}" ] || { echo "Set your seat first:  export SEAT=<your number>" >&2; exit 1; }
POD=seat-$SEAT
# No -i: nothing here reads from the keyboard, and a kubectl that asks for the terminal is
# suspended the moment the command is put in the background.
in_pod() { kubectl exec "$POD" -- bash -c "cd $REMOTE && $1" < /dev/null; }

case "${1:-}" in
  serve)
    # serve.sh starts the server detached, so it keeps running after this returns or drops.
    kubectl exec "$POD" -- bash -c "cd /workspace && ./serve.sh" < /dev/null
    ;;
  push)
    # A run that is going keeps the code it started with, so pushing does not disturb it. But
    # the files in the pod are replaced for whatever runs NEXT, which matters on a seat that is
    # someone else's: say so, loudly, before doing it.
    if kubectl exec "$POD" -- bash -c "pgrep -fa 'python agent.py' | grep -v 'bash -c'" 2>/dev/null | grep -q .; then
      echo "NOTE: a run is going on $POD. It is not affected, but the code files in this pod are" >&2
      echo "about to be replaced with yours. If this is a teammate's seat, tell them to push" >&2
      echo "their own code again before their next run. Ctrl+C in the next 5 seconds to stop." >&2
      sleep 5
    fi
    # The harness itself must be the one the organisers shipped, or scores stop being comparable
    # with theirs and with a baseline run on the untouched pod. Say so before overwriting it.
    mine=$(shasum -a 256 "$HERE/nkibench.py" | cut -d' ' -f1)
    theirs=$(kubectl exec "$POD" -- sha256sum "$REMOTE/nkibench.py" | cut -d' ' -f1)
    if [ "$mine" != "$theirs" ]; then
      echo "STOP: nkibench.py in $POD differs from yours. Nothing was copied." >&2
      echo "Bring the pod's copy here and look at the difference first:" >&2
      echo "  ./pod.sh get nkibench.py nkibench.pod.py" >&2
      [ "${FORCE:-}" = 1 ] || exit 1
    fi
    # devtools/ stays on the laptop: its stand-in simulator must never be importable in the pod.
    if PREFIX=$(git -C "$HERE" rev-parse --show-prefix 2>/dev/null) && [ -n "$PREFIX" ]; then
      # Only COMMITTED code goes to a seat. A run is then always some commit, never an edit that
      # happened to be half made when the push ran -- which is how a measured run stops being
      # reproducible without anyone noticing.
      REV=$(git -C "$HERE" rev-parse --short HEAD)
      if ! git -C "$HERE" diff --quiet HEAD -- . ; then
        echo "NOTE: this folder has uncommitted changes. They are NOT pushed: the seat gets commit $REV." >&2
      fi
      git -C "$(git -C "$HERE" rev-parse --show-toplevel)" archive "HEAD:${PREFIX%/}" -- . ':!devtools' \
        | kubectl exec -i "$POD" -- tar -xf - -C "$REMOTE" --no-same-owner
    else
      # Not a git checkout (a copy of the folder): send the files as they are.
      REV=copy
      COPYFILE_DISABLE=1 tar --no-xattrs -cf - -C "$HERE" \
          --exclude '__pycache__' --exclude './devtools' --exclude '*.jsonl' --exclude '*.out' \
          --exclude '*.log' . \
        | kubectl exec -i "$POD" -- tar -xf - -C "$REMOTE" --no-same-owner
    fi
    in_pod "echo $REV > .pushed_rev"
    in_pod "python -c 'import agent, diagnose; print(\"pushed to $POD; agent.py knows features:\", sorted(agent.KNOWN_FEATURES))'"
    ;;
  check)
    in_pod "python nkibench.py --selftest > check_nkibench.out 2>&1; echo \"nkibench selftest exit \$?\"; \
            python diagnose.py --selftest > check_diagnose.out 2>&1; echo \"diagnose selftest exit \$?\"; \
            python probe_sim.py > check_probe.out 2>&1; echo \"probe exit \$?\""
    mkdir -p "$RESULTS/$POD"
    for f in check_nkibench.out check_diagnose.out check_probe.out; do
      kubectl exec "$POD" -- cat "$REMOTE/$f" > "$RESULTS/$POD/$f"
    done
    echo "saved to $RESULTS/$POD/check_*.out"
    # A check that cannot fail is not a check. No scored run on a seat where this did not pass.
    if ! grep -q "Located lines: all are the planted ones." "$RESULTS/$POD/check_diagnose.out" \
       || ! grep -q "State on the line: every planted fact was reported." "$RESULTS/$POD/check_diagnose.out" \
       || ! grep -q "Origin of the tile: every planted line was traced." "$RESULTS/$POD/check_diagnose.out" \
       || ! grep -q "In pieces: every planted case was told to cut." "$RESULTS/$POD/check_diagnose.out" \
       || ! grep -q "Looking ahead: every planted case was explained by its later use." "$RESULTS/$POD/check_diagnose.out" \
       || ! grep -q "Too tall for the chip: the planted tile was caught and the reference was not." "$RESULTS/$POD/check_diagnose.out" \
       || ! grep -q "Missing names: the planted name was explained." "$RESULTS/$POD/check_diagnose.out" \
       || ! grep -q "Loaded askew: the planted operand was traced to its load." "$RESULTS/$POD/check_diagnose.out" \
       || ! grep -q "Worked example: it runs, it is correct, and no tile in it is too tall." "$RESULTS/$POD/check_diagnose.out" \
       || ! grep -q "SELFTEST PASSED" "$RESULTS/$POD/check_nkibench.out"; then
      echo "CHECK FAILED on $POD. Read $RESULTS/$POD/check_*.out before running anything here." >&2
      exit 1
    fi
    echo "check passed on $POD"
    ;;
  launch)
    "$0" serve && "$0" push && "$0" check && "$0" run "${2:?give the run a name}" "${3:-}" "${4:-}"
    ;;
  run)
    NAME=${2:?give the run a name, e.g. ./pod.sh run locate locate}
    FEATURES=${3:-}
    [ -z "${4:-}" ] || RUN_ARGS=${RUN_ARGS/--all/--level $4}
    in_pod "python3 -c 'import urllib.request; urllib.request.urlopen(\"http://localhost:8000/health\", timeout=5)' 2>/dev/null \
              || { echo 'The model is not answering on this seat yet. In the pod: cd /workspace && ./serve.sh, and wait for READY.' >&2; exit 1; }; \
            if [ -e $NAME.jsonl ]; then echo '$NAME.jsonl already exists in the pod. Pick another name: a log holding two runs cannot be told apart.' >&2; exit 1; fi; \
            if pgrep -fa 'python agent.py' | grep -qv 'bash -c'; then echo 'A run is already going on this seat. Two at once share one model and change each other. Wait, or use another seat.' >&2; exit 1; fi; \
            nohup python agent.py $RUN_ARGS --features '$FEATURES' --log $NAME.jsonl > $NAME.out 2>&1 < /dev/null & \
            sleep 2; echo \"started $NAME on $POD: agent.py $RUN_ARGS --features '$FEATURES'\"; head -3 $NAME.out"
    # The run sheet: which code, on which seat, with which settings. Written by the tool so it
    # cannot be forgotten. The commit is the one that was pushed to this seat.
    REV=$(kubectl exec "$POD" -- cat "$REMOTE/.pushed_rev" 2>/dev/null || echo "?")
    mkdir -p "$RESULTS"
    [ -e "$RESULTS/RUNS.tsv" ] || printf 'started\tseat\tname\tfeatures\tcommit\targs\n' > "$RESULTS/RUNS.tsv"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M')" "$SEAT" "$NAME" "${FEATURES:-none}" "$REV" "$RUN_ARGS" >> "$RESULTS/RUNS.tsv"
    ;;
  replay)
    NAME=${2:?which log? e.g. ./pod.sh replay baseline}
    in_pod "python diagnose.py --replay $NAME.jsonl > replay_$NAME.out 2>&1; echo \"replay exit \$?\"; tail -n 4 replay_$NAME.out"
    mkdir -p "$RESULTS/$POD"
    kubectl exec "$POD" -- cat "$REMOTE/replay_$NAME.out" > "$RESULTS/$POD/replay_$NAME.out"
    echo "saved to $RESULTS/$POD/replay_$NAME.out"
    ;;
  findings)
    in_pod "python findings/run_findings.py > findings.out 2>&1; echo \"findings exit \$?\"; sed -n '/SUMMARY/,\$p' findings.out"
    mkdir -p "$RESULTS/$POD"
    kubectl exec "$POD" -- cat "$REMOTE/findings.out" > "$RESULTS/$POD/findings.out"
    echo "saved to $RESULTS/$POD/findings.out"
    ;;
  device)
    # The model server holds NeuronCores; a kernel cannot run on the chip while it does. The runs
    # are finished, so stop it, then run findings/device_check.py on the chip.
    in_pod "pkill -f '[p]ython agent.py' || true; cd /workspace && ./serve.sh --stop; sleep 5; cd $REMOTE && \
            NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_RT_NUM_CORES=1 timeout 1500 python findings/device_check.py > device.out 2>&1; \
            echo \"device check exit \$?\"; sed -n '/SUMMARY/,\$p' device.out"
    mkdir -p "$RESULTS/$POD"
    kubectl exec "$POD" -- cat "$REMOTE/device.out" > "$RESULTS/$POD/device.out"
    echo "saved to $RESULTS/$POD/device.out"
    ;;
  stop)
    # [p]ython, so the pattern cannot match the shell that is carrying this very command.
    in_pod "pkill -f '[p]ython agent.py' && echo 'stopped the run on $POD' || echo 'no run was going on $POD'"
    ;;
  status)
    in_pod "pgrep -fa 'python agent.py' | grep -v 'bash -c' || echo 'no run going'; \
            ls -la *.jsonl *.out 2>/dev/null | awk '{print \$5, \$9}'; \
            f=${2:+$2.out}; f=\${f:-\$(ls -t *.out 2>/dev/null | grep -v '^check_' | head -1)}; \
            if [ -n \"\$f\" ]; then echo \"---- \$f\"; tail -n 12 \"\$f\"; fi"
    ;;
  pull)
    NAME=${2:?which run? e.g. ./pod.sh pull baseline}
    mkdir -p "$RESULTS/$POD"
    for f in "$NAME.jsonl" "$NAME.out"; do
      if kubectl exec "$POD" -- test -e "$REMOTE/$f"; then
        kubectl exec "$POD" -- cat "$REMOTE/$f" > "$RESULTS/$POD/$f"
        echo "$RESULTS/$POD/$f  ($(wc -l < "$RESULTS/$POD/$f" | tr -d ' ') lines)"
      else
        echo "no $f in the pod"
      fi
    done
    ;;
  get)
    SRC=${2:?which file?}
    mkdir -p "$RESULTS/$POD"
    kubectl exec "$POD" -- cat "$REMOTE/$SRC" > "$RESULTS/$POD/${3:-$(basename "$SRC")}"
    echo "$RESULTS/$POD/${3:-$(basename "$SRC")}"
    ;;
  *)
    sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
