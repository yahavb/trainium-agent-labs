# Measured results

Device medians from real Trainium2 execution; see summary.json and raw_samples.json.

| Workload | Shape | Strongest separate baseline | Before p50 (ms) | Fused p50 (ms) | Speedup | Reduction |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| A | 128 × 8192 | separate_score | 0.073176 | 0.065931 | 1.11x | 9.9% |
| B | 512 × 32768 | separate_score | 0.406309 | 0.322317 | 1.26x | 20.7% |
| C | 129 × 8193 | separate_score | 0.077624 | 0.067073 | 1.16x | 13.6% |
