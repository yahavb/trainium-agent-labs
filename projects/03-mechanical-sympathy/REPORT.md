# Samudra on AWS Trainium

## What ran

Pending.

## Correctness

The CPU implementation is the reference. A Trainium result must pass the frozen
checker before it receives a performance result.

## Hardware and software

Pending. Record the pod type, Neuron SDK, PyTorch, runner commit, Samudra commit,
checkpoint SHA-256, numeric precision, and fixed input shape.

## Performance

Pending. Compare the best correct Trainium result with the first correct
Trainium result. Report compile time and warm-up separately from steady-state
runtime. Use rollout throughput as the primary metric.

## Attempts and spread

Pending. Summarize the accepted and failed attempts in `results/attempts.csv`.
