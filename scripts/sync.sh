#!/usr/bin/env bash
# Move files between this checkout and your seat pod's /workspace.
#
#   scripts/sync.sh <seat> push      local projects/ files you changed  ->  pod /workspace
#   scripts/sync.sh <seat> pull      pod run*.log and *.jsonl           ->  local runs/seat-<n>/<time>/
#   scripts/sync.sh <seat> both      push, then pull (default)
#   DRY_RUN=1 scripts/sync.sh ...    list what would be copied, copy nothing
#
# "Changed" = differs from upstream's default branch (committed or not) plus new untracked files, under projects/.
# Push OVERWRITES those paths in the pod — if you also edited them inside the pod, pull first.
set -uo pipefail

CONTAINER=app
REMOTE_ROOT=/workspace

die()  { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m==>\033[0m %s\n' "$*"; }

SEAT=${1:-}
MODE=${2:-both}
[[ "$SEAT" =~ ^[0-9]+$ ]] || die "usage: $0 <seat> [push|pull|both]"
[[ "$MODE" =~ ^(push|pull|both)$ ]] || die "mode must be push, pull or both (got '$MODE')"
POD="seat-$SEAT"
DRY_RUN=${DRY_RUN:-0}

cd "$(git rev-parse --show-toplevel 2>/dev/null)" || die "run this from inside the repo"

kc() {
  local out
  if ! out=$(kubectl "$@" 2>&1); then
    case "$out" in
      *ExpiredToken*|*expired*|*"provide credentials"*|*Unauthorized*)
        die "credentials missing or expired — paste the newest block from the channel, then: scripts/connect.sh $SEAT" ;;
      *localhost:8080*)
        die "kubectl is not pointed at the cluster yet — run: scripts/connect.sh $SEAT" ;;
      *) die "kubectl $1 failed: $out" ;;
    esac
  fi
  [[ -n "$out" ]] && printf '%s\n' "$out"
  return 0
}

push() {
  local base=upstream/HEAD files=()
  git rev-parse --verify -q "$base" >/dev/null || base=HEAD
  while IFS= read -r f; do
    [[ -f "$f" ]] && files+=("$f")         # skip deletions
  done < <( { git diff --name-only "$base" -- projects/; git ls-files --others --exclude-standard -- projects/; } | sort -u )

  if (( ${#files[@]} == 0 )); then info "push: nothing under projects/ differs from $base"; return; fi
  info "push: ${#files[@]} file(s) -> $POD:$REMOTE_ROOT"
  for f in "${files[@]}"; do
    echo "    $f"
    (( DRY_RUN )) && continue
    kc exec "$POD" -c "$CONTAINER" -- mkdir -p "$REMOTE_ROOT/$(dirname "$f")" >/dev/null
    kc cp "$f" "$POD:$REMOTE_ROOT/$f" -c "$CONTAINER" >/dev/null
  done
}

pull() {
  local dest listing remote=()
  dest="runs/$POD/$(date +%Y%m%d-%H%M%S)"
  listing=$(kc exec "$POD" -c "$CONTAINER" -- sh -c \
      "cd $REMOTE_ROOT && find projects -maxdepth 3 -type f \( -name 'run*.log' -o -name '*.jsonl' \)") || exit 1
  while IFS= read -r f; do
    [[ -n "$f" ]] && remote+=("$f")
  done <<< "$listing"

  if (( ${#remote[@]} == 0 )); then info "pull: no run*.log or *.jsonl under $REMOTE_ROOT/projects"; return; fi
  info "pull: ${#remote[@]} file(s) -> $dest/"
  for f in "${remote[@]}"; do
    echo "    $f"
    (( DRY_RUN )) && continue
    mkdir -p "$dest/$(dirname "$f")"
    kc cp "$POD:$REMOTE_ROOT/$f" "$dest/$f" -c "$CONTAINER" >/dev/null
  done
  (( DRY_RUN )) || ln -sfn "$(basename "$dest")" "runs/$POD/latest"
}

case "$MODE" in
  push) push ;;
  pull) pull ;;
  both) push; pull ;;
esac
