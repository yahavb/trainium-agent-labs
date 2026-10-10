# Sourced by the local helper scripts. SEAT comes from the environment
# (e.g. `SEAT=seat-232 scripts/pod.sh ...`), falling back to the repo's .env.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${SEAT:-}" ]]; then
  if [[ ! -f "$REPO_ROOT/.env" ]]; then
    echo "error: SEAT not set and $REPO_ROOT/.env not found (expected SEAT=seat-<N>)" >&2
    exit 1
  fi
  SEAT="$(grep -E '^SEAT=' "$REPO_ROOT/.env" | tail -1 | cut -d= -f2- | tr -d '"'"'"' ')"
fi
if [[ ! "$SEAT" =~ ^seat-[0-9]+$ ]]; then
  echo "error: SEAT must look like seat-<N>, got '$SEAT'" >&2
  exit 1
fi
# These seats run someone else's vLLM server and experiments; touching them needs explicit opt-in.
PROTECTED_SEATS=" "
is_protected() { [[ "$PROTECTED_SEATS" == *" $SEAT "* ]]; }
if is_protected && [[ "${ALLOW_PROTECTED:-}" != "readonly" ]]; then
  echo "error: $SEAT is protected; set ALLOW_PROTECTED=readonly for read-only pod.sh commands" >&2
  exit 1
fi
CONTAINER="app"
REMOTE_ROOT="/workspace/livevid"
