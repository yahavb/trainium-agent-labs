#!/bin/bash
# Talk to the Trainium pod (credentials from ENV_FILE, default aws-hack/.env).
#   scripts/pod.sh 'neuron-ls'              run a command in the pod
#   scripts/pod.sh put local remote         copy a file to the pod
#   scripts/pod.sh get remote local         copy a file from the pod
ENV_FILE=${ENV_FILE:-/home/luka/aws-hack/.env}
. "$ENV_FILE"
P=${POD:-seat-270}
case "$1" in
  put) exec kubectl cp -c app "$2" "$P:$3" ;;
  get) exec kubectl cp -c app "$P:$2" "$3" ;;
  *) exec kubectl exec "$P" -c app -- bash -c "$1" ;;
esac
