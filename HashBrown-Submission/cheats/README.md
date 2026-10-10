# Cheats

Deliberately wrong kernels, one per file, used to measure whether the checker catches them.
Each file's header comment states what it does, why it's wrong, and which test is expected to
catch it. Run all of them through both scoring paths with `run_cheats.py` from the project
root; results land in `results/cheats_before.csv`.

| id | cheat | level | base | wrong because |
|---|---|---|---|---|
| C1 | do_nothing | 1 (avgpool) | hand-written | output tensor is never written |
| C2 | partial_rows | 2 (transpose) | variant of `reference_level2.py` | hardcoded 64-row cap on the partition axis |
| C3 | hardcoded_shape | 3 (matmul) | near-copy of `reference_level3.py` | asserts a single hardcoded (K,M,N), same as the shipped reference, but level 3 has only one declared test shape so this is indistinguishable from "correct" today |
| C4 | constant_output | 1 (avgpool) | variant of `reference_level1.py` | real input loaded, then multiplied by 0.0 before storing |
| C5 | input_tamper | 2 (transpose) | variant of `reference_level2.py` | writes the (numerically correct) result back into the input tensor instead of a fresh output |
| C6 | almost_right | 2 (transpose) | variant of `reference_level2.py` | adds a uniform +1e-3 offset after an otherwise-correct transpose |
| C7 | edge_skip | 2 (transpose) | variant of `reference_level2.py` | outer loop runs `sz_f1 - 1` instead of `sz_f1`, dropping the last column group |
| C8 | path_mismatch | 2 (transpose) | variant of `reference_level2.py` | destination stride uses `sz_f2` instead of `sz_f1` -- identical to correct only when the tile is square |

C8 was not in the original cheat list. It fell out directly while reading
`reference_level2.py`'s two index formulas for Phase 2 (per HARDENING.md's instruction to add it
only if it falls out naturally), so it's included.

## Second generation (Phase 4): attacking the hardened A0

After Phase 3's fixes (level 3's second shape, the measured tolerance, tamper-check parity),
all of C1-C8 are caught at A0 with `--augment` off. These are the smarter cheats built to pass
that hardened A0 and fail only at a specific augmentation tier:

| id | cheat | level | base | passes A0 because | caught at |
|---|---|---|---|---|---|
| C9 | memorise_all_shapes | 3 (matmul) | variant of `reference_level3.py` | memorizes BOTH of level 3's current declared `(K,M,N)` triples, not just one | A2 (either A2a or A2b -- neither is memorized) |
| C11 | assume_divisible | 2 (transpose) | variant of `reference_level2.py` | tiles the partition axis in chunks of 8; all 4 original shapes AND A2a's extra shape have `sz_p` divisible by 8 | A2b specifically (ragged `sz_p=37` and `sz_p=1` are not multiples of 8) |

C10 (`fixed_input_lookup`) and C12 (`precision_shortcut`) were attempted and dropped -- see
HARDENING.md's Phase 4 Findings log for the concrete, empirically-verified technical reasons
(an NKI API limitation for C10; a dynamic-range argument for C12).
