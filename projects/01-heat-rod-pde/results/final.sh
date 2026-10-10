#!/bin/bash
# Final benchmark: free-form + all-candidate feedback + decay-rate verification (A) + fixed V4 (B). Seed 0.
cd /workspace/projects/01-heat-rod-pde
F="--aggregate all --verify-rates --v4-fix --seed 0 --arm FINAL"
echo "$(date -u +%T) start level 1"; python agent.py --level 1 --all $F > results/FINAL-L1-s0.txt 2>&1 < /dev/null
echo "$(date -u +%T) start level 0"; python agent.py --level 0 --all $F > results/FINAL-L0-s0.txt 2>&1 < /dev/null
echo "$(date -u +%T) final finished"
