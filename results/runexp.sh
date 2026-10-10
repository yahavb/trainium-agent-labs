#!/bin/sh
# usage: runexp.sh NAME [extra agent.py args]   -> NAME.log, NAME_L{1,3,4}.jsonl
N=$1; shift
for L in 1 3 4; do
  echo "##### LEVEL $L start $(date +%T)"
  python agent.py --level $L --rounds 8 --context 8192 --repeat 2 "$@" --log ${N}_L$L.jsonl
done
echo "##### DONE $(date +%T)"
