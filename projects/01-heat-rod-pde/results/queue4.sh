#!/bin/bash
# Change A / B ablation on the seed Idea 2 failed (level 1.3, seed 1). One run at a time.
cd /workspace/projects/01-heat-rod-pde
run() { name=$1; shift; echo "$(date -u +%T) start $name"; python agent.py "$@" > results/$name.txt 2>&1 < /dev/null; echo "$(date -u +%T) done  $name"; }
F="--level 1 --sub 3 --aggregate all --splice"
run F2A-13s1  $F --seed 1 --verify-rates --arm F2A
run F2B-13s1  $F --seed 1 --v4-fix --arm F2B
run F2AB-13s1 $F --seed 1 --verify-rates --v4-fix --arm F2AB
echo "$(date -u +%T) queue4 finished"
