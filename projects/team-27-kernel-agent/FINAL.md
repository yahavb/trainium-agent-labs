# Final kernels

One kernel per level, chosen by `scripts/final_kernels.py`: written by the agent, verified by our
harness, and passing the organizers' `kernelbench.py`; among those, the lowest roofline time, then
the fewest instructions. Kernels seeded from our hand-written references (e.g. the optimised
softmax of notes E19) are excluded: only kernels the agent wrote count.

| level | kernel file | from | organizers' checker | roofline time | instructions | candidates passing / checked |
|---|---|---|---|---|---|---|
| L1 relu_affine | `kernels/final/l1_relu_affine.py` | naive ve22cca run3 | 32/32 cases | 1.00x | 18 | 24/24 |
| L2 row_sum | `kernels/final/l2_row_sum.py` | agent run2, then optimised | 32/32 cases | 1.00x | 2068 | 19/19 |
| L3 row_max | `kernels/final/l3_row_max.py` | agent run3, then optimised | 32/32 cases | 1.00x | 2068 | 17/17 |
| L4 rmsnorm | `kernels/final/l4_rmsnorm.py` | ours vc5e98d run5 (final agent) | 32/32 cases | 1.50x | 272193 | — |
| L5 softmax | `kernels/final/l5_softmax.py` | ours vc5e98d run1 (teammate, in progress) | 32/32 cases | 2.00x | 540244 | — |
| L6 transpose | `kernels/final/l6_transpose.py` | ours v29fd8d run2 | 32/32 cases | 1.00x | 0 | 3/3 |
| L7 matmul | — | none passes yet | — | — | — | 0/7 |
| L8 layernorm | `kernels/final/l8_layernorm.py` | ours vf0c39f run3 | 32/32 cases | 2.00x | 675323 | 1/6 |
| L9 band_attention | `kernels/final/l9_band_attention.py` | ours vc5e98d run1 (teammate, in progress) | 5/5 cases | 61.19x | 284120 | — |
| L10 conv1d | `kernels/final/l10_conv1d.py` | ours ve22cca run2 | 4/4 cases | 12.24x | 30420 | 1/17 |
