#!/bin/bash
# Runs the experiment arms one after another (one agent at a time on the server).
cd /workspace/projects/01-heat-rod-pde
until curl -s -m 5 localhost:8000/v1/models > /dev/null; do sleep 10; done   # wait for the server
run() { name=$1; shift; echo "$(date -u +%T) start $name"; python agent.py "$@" > results/$name.txt 2>&1 < /dev/null; echo "$(date -u +%T) done  $name"; }
T="--temps 0.6,0.8,1.0,1.0"
run T2-s1  --level 1 --all --seed 1 --mode ansatz $T --arm T2
run T2-s2  --level 1 --all --seed 2 --mode ansatz $T --arm T2
run T2A-s0 --level 1 --all --seed 0 --mode ansatz --aggregate all --arm T2A
run T2A-s1 --level 1 --all --seed 1 --mode ansatz --aggregate all --arm T2A
run T2A-s2 --level 1 --all --seed 2 --mode ansatz --aggregate all --arm T2A
run T3-s0  --level 1 --all --seed 0 --mode full $T --arm T3
run T3-s1  --level 1 --all --seed 1 --mode full $T --arm T3
run T3-s2  --level 1 --all --seed 2 --mode full $T --arm T3
run B-13s0  --level 1 --sub 3 --seed 0 --arm B
run F2-13s0 --level 1 --sub 3 --seed 0 --aggregate all --splice --arm F2
run B-13s2  --level 1 --sub 3 --seed 2 --arm B
run F2-13s2 --level 1 --sub 3 --seed 2 --aggregate all --splice --arm F2
run B-13s1  --level 1 --sub 3 --seed 1 --arm B
run F2-13s1 --level 1 --sub 3 --seed 1 --aggregate all --splice --arm F2
echo "$(date -u +%T) queue finished"
