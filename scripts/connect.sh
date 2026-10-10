#!/usr/bin/env bash
# Connect this terminal to your seat pod.
#
#   scripts/connect.sh <seat>
#
# Credentials come ONLY from the environment of this terminal (paste the export block from the
# workshop channel first). This script never writes them anywhere.
set -uo pipefail

CLUSTER=hack-hyd
REGION=ap-south-2

die()  { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m==>\033[0m %s\n' "$*"; }

creds_hint() {
  cat >&2 <<'EOF'

  Paste the newest macOS/Linux credential block from the workshop channel into THIS terminal:
      export AWS_ACCESS_KEY_ID="ASIA..."
      export AWS_SECRET_ACCESS_KEY="..."
      export AWS_SESSION_TOKEN="..."
  then run this script again. (A new terminal or tab needs them pasted again.)
EOF
}

# Turn an AWS/kubectl error into a clear next step.
explain() {
  local out=$1
  case "$out" in
    *ExpiredToken*|*"token has expired"*|*"security token included in the request is expired"*)
      printf '\033[31mERROR:\033[0m the workshop credentials have EXPIRED.\n' >&2; creds_hint; exit 1 ;;
    *InvalidClientTokenId*|*SignatureDoesNotMatch*|*"Unable to locate credentials"*|*"provide credentials"*|*Unauthorized*)
      printf '\033[31mERROR:\033[0m the credentials are missing, mistyped or incomplete.\n' >&2; creds_hint; exit 1 ;;
    *Forbidden*|*forbidden*)
      die "Kubernetes refused the request (RBAC). Your credentials only allow seat-1..seat-150 — check the seat number. Detail: $out" ;;
    *NotFound*|*"not found"*)
      die "pod not found. Check your seat number, or ask an organiser. Detail: $out" ;;
    *)
      die "$out" ;;
  esac
}

# --- 0. arguments --------------------------------------------------------------------------
SEAT=${1:-}
[[ "$SEAT" =~ ^[0-9]+$ ]] || die "usage: $0 <seat number>   e.g. $0 42"
(( SEAT >= 1 && SEAT <= 150 )) || die "seat must be between 1 and 150 (got $SEAT)"
POD="seat-$SEAT"

command -v aws >/dev/null     || die "aws CLI not found — brew install awscli"
command -v kubectl >/dev/null || die "kubectl not found — brew install kubectl"

# --- 1. credentials present? ---------------------------------------------------------------
missing=()
for v in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN; do
  [[ -n "${!v:-}" ]] || missing+=("$v")
done
if (( ${#missing[@]} )); then
  printf '\033[31mERROR:\033[0m not set in this terminal: %s\n' "${missing[*]}" >&2
  creds_hint; exit 1
fi
info "AWS credentials found in environment (values not shown)"

# --- 2. credentials valid? (catches ExpiredToken before kubectl gives a vaguer error) -------
if ! out=$(aws sts get-caller-identity --region "$REGION" --query Arn --output text 2>&1); then
  explain "$out"
fi
info "authenticated as: $out"

# --- 3. kubeconfig -------------------------------------------------------------------------
info "aws eks update-kubeconfig --name $CLUSTER --region $REGION"
if ! out=$(aws eks update-kubeconfig --name "$CLUSTER" --region "$REGION" 2>&1); then
  explain "$out"
fi
echo "    $out"

# --- 4. the pod ----------------------------------------------------------------------------
info "kubectl get pod $POD"
if ! out=$(kubectl get pod "$POD" 2>&1); then
  explain "$out"
fi
echo "$out"

status=$(kubectl get pod "$POD" -o jsonpath='{.status.phase} {.status.containerStatuses[0].ready}' 2>/dev/null)
case "$status" in
  "Running true") info "pod is Running and READY" ;;
  Pending*|*false|"Running ") echo; echo "Pod is still starting (Init / 0/1). Wait a minute and run this again." ;;
  *) echo; echo "Unexpected state '$status' — find an organiser; they can move you to a spare seat." ;;
esac

cat <<EOF

Next — get a shell in your pod (wait for the 'root@$POD:/workspace#' prompt before typing):

    kubectl exec -it $POD -- bash

First time inside:  cd /workspace && ./serve.sh   (leave it running; open a 2nd terminal for the agent)
EOF
