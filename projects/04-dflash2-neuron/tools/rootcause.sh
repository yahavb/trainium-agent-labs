#!/usr/bin/env bash
# rootcause.sh — print the first worker traceback (deduped) from a vLLM log.
LOG=${1:-/tmp/dflash2_test.log}
grep -aE '^\(Worker_TP0' "$LOG" \
  | sed -E 's/^\(Worker_TP0 pid=[0-9]+\) (ERROR [0-9: -]+\[[^]]+\] )?//' \
  | awk '/Traceback/{f=1} f' \
  | grep -vE '^\s*[~^]+\s*$' \
  | awk '!seen[$0]++' \
  | grep -E 'File "|Error|error|Exception|meta|Expected|shape|size|dtype|device' \
  | sed -E 's#/opt/conda/lib/python3.13/site-packages/##' \
  | tail -${2:-25}
