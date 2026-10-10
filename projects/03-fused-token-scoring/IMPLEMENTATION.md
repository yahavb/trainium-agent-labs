# Implementation and benchmark checkpoints

Implementation branch: `feature/fused-token-scoring`. Submission lives entirely in
`projects/03-fused-token-scoring/`. The earlier proposal remains in `proposals/`.

1. **Prove the execution path.** Inspect the installed SDK; compile and execute a
   minimal NKI kernel on seat 256 using an available logical core. Preserve other
   projects and the model server. Record the actual environment.
2. **Freeze the contract and reference.** BF16 contiguous logits `[T,V]`, int32
   selected indices `[T]`, two FP32 outputs. Compare against the same BF16 inputs
   promoted to FP32. Fix tolerance at `1e-3 + 1e-4 * abs(reference)`.
3. **Build comparable baselines.** Separate NKI log-probability and entropy
   computations, plus AWS's installed cross-entropy implementation where
   compatible. Both baseline paths must return both required outputs.
4. **Implement fusion.** Tile rows and vocabulary; share loads, maxima and
   exponentials; accumulate stable online sums for log-probability and entropy.
   Handle partial tiles and initialize from a valid first tile.
5. **Validate on hardware.** Uniform, singleton, random, peaked, extreme offsets,
   boundary indices and ragged shapes. Check every output, numeric invariants,
   and rejection of invalid host inputs. Keep the original accuracy gate.
6. **Tune and benchmark.** Predefine A=`128x8192`, B=`512x32768`,
   C=`129x8193`; optional larger vocabularies. Tune separately from the final
   measurement. Use the same cores, inputs and precision for each path. Alternate
   order across three rounds with 10 warmups and 100 measured executions. Collect
   device trace durations and report p50, p95, variability, speedup and latency
   reduction. Keep compilation and transfers out of device timing; report host
   timing separately. Preserve raw samples and source/environment identifiers.
7. **Document the contribution.** Write the final README around observed benefits,
   reproducible commands, accuracy, baseline fairness and remaining integration
   work. Never use simulator timing or external GPU numbers as our result.

Performance objective: at least 20% lower median device latency (1.25x speedup)
on A and B versus the strongest applicable measured separate-operation baseline.
This is a project objective, not an organizer scoring rule. Report misses and
regressions explicitly. No claim of full training-step acceleration without a
training integration measurement.

## Completed checks and outcomes

- The smoke kernel compiled, ran correctly on seat-256 and produced ten complete
  device execution traces using the supplied NKI 0.6 snapshot.
- Both complete baselines and the fused kernel passed all 37 numerical cases:
  111 hardware comparisons and ten invalid-input rejection checks, with the
  original tolerance. The final selected tile configurations were validated.
- All 36 tuning candidates passed correctness. Final measurement used a different
  input seed, three rotated rounds and 300 samples per implementation, mode and
  workload. The evidence contains 5400 final device/host timing samples.
- The offline audit passed on both the pod and the local Mac: source hashes,
  every physical-core trace, iteration counts, percentiles and baseline choice
  agree with the published statistics.
- Median device latency improved by 9.9% on A, 20.7% on B and 13.6% on C against
  the fastest measured separate baseline. The 20% objective is partially met:
  B passes and A misses. Full training performance remains unmeasured.
- The optional 512 × 151936 experiment did not finish; no result is claimed.
  It is outside the completed A/B/C acceptance and benchmark suite.

See [README.md](README.md) for impact, commands, measured limitations and links
to the retained evidence.
