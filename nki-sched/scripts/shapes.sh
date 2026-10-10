#!/bin/bash
# best config over several shapes: scripts/shapes.sh "K M N" ...
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export CORES=2
for sh in "$@"; do
  echo -n "K M N = $sh: "
  $ROOT/scripts/sweep.sh $sh "4 2 1 sw nm rhs n" | sed 's/^\[[^]]*\] cores=2: //'
done
