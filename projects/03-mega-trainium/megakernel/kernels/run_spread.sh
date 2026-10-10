#!/usr/bin/env bash
# Repeated runs for the one-page note's "how many runs, and the spread". Strictly sequential, one core,
# nothing else on the device. Appends to results/spread_*.jsonl.
#   NEURON_RT_VISIBLE_CORES=2 bash kernels/run_spread.sh
set -u
cd "$(dirname "$0")/.."
export NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-2}
R=results
for i in 1 2 3; do   # full step: one launch vs per-op, S_ctx 1024, random weights, median of 30 synced calls
  MASTER_PORT=$((29680 + i)) python kernels/bench_step.py --layers 36 --resident --out $R/spread_bench_step.jsonl > $R/spread_bench_step_$i.log 2>&1
done
for i in 1 2 3; do   # end-to-end greedy generation, real weights, host loop incl. read-back, 32 tokens
  MASTER_PORT=$((29690 + i)) python kernels/generate.py --layers 36 --new 32 --resident --feed host --no-logits \
    --timing-iters 30 --out $R/spread_generate_timing.jsonl > $R/spread_generate_timing_$i.log 2>&1
done
P1="The Trainium2 chip has eight physical NeuronCores, and the Neuron Kernel Interface lets you"
P2="Write a Python function that returns the n-th Fibonacci number."
P3="The three most important ideas in thermodynamics are"
i=0
for p in "$P1" "$P2" "$P3"; do   # teacher-forced correctness vs HF float32 on three prompts
  i=$((i + 1))
  MASTER_PORT=$((29700 + i)) python kernels/generate.py --layers 36 --new 32 --resident --teacher-force --timing-iters 0 \
    --prompt "$p" --out $R/spread_teacher_forced.jsonl > $R/spread_teacher_forced_$i.log 2>&1
done
echo SPREAD_DONE
