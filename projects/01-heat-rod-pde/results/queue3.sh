#!/bin/bash
cd /workspace/projects/01-heat-rod-pde
# (no wait: nothing else is running)
run() { name=$1; shift; echo "$(date -u +%T) start $name"; python agent.py "$@" > results/$name.txt 2>&1 < /dev/null; echo "$(date -u +%T) done  $name"; }
T="--temps 0.6,0.8,1.0,1.0"
run T3d-s0 --level 1 --all --seed 0 --mode full $T --arm T3d
run T3d-s1 --level 1 --all --seed 1 --mode full $T --arm T3d
run T3d-s2 --level 1 --all --seed 2 --mode full $T --arm T3d
run T2c-s0 --level 1 --all --seed 0 --mode ansatz $T --arm T2c
run T2c-s1 --level 1 --all --seed 1 --mode ansatz $T --arm T2c
run T2c-s2 --level 1 --all --seed 2 --mode ansatz $T --arm T2c
echo "$(date -u +%T) queue3 finished"
