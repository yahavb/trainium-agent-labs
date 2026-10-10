# Semantic planning and shape-consistency findings

The full adaptive original-model Level 1 run, controlled-20261010T192716-jp07xp0l, ended with .30, zero of four numerical shapes, 32 candidates / 8 rounds, 37,976 actual endpoint tokens and 505.37 seconds. Thirty candidates contain the same operand in both nc_matmul inputs; two contain reductions. A self-product is generally quadratic whereas average pooling is linear; AST presence alone cannot establish whether a result reaches the returned output. The checker is authoritative.

Level 2 from the same run ended with .30, zero shapes, 20 candidates / 5 rounds, 26,763 tokens and 327.96 seconds. The unchanged repetition stopping rule fired before its eight-round ceiling. Dominant category was DMA_SHAPE_MISMATCH (14 candidates); no solve.

The first planner-enabled full-agent Level 1 run, controlled-20261010T194114-ln8yl9ak, also ended at .30, zero shapes, 20 candidates / 5 rounds, 22,069 tokens and 224.80 seconds. All 20 candidates contain reductions and none contain nc_matmul. This is a shift toward the desired operation, not proof of correct average pooling. Seventeen failed DMA, three failed tensor dimensions. Example source allocates (C,H,p,W) while loading [C,H,W], then proposes an incompatible view. For C=32,H=W=32,p=2, that allocation contains 65,536 elements and the input contains 32,768.

The diagnosed gap: failure_input_shapes supplied tensor shapes but omitted actual pool_size, so targeted/SymPy allocation inference returned unresolved dimensions. New failure_input_values binds scalar/tuple parameters only from the exact checker-reported case and actual entry argument order. Rebound/unsupported inputs remain UNKNOWN. Both analyzers now consume these safe bindings. Offline replay of the 20 first-planner candidates detects proven DMA mismatches in 17 cases versus zero without scalar bindings. This measures static diagnosis on recorded source, not a live correctness improvement.

The planner now records operation, output dimensions, window reduction axes, SymPy normalization 1/p², required memory spaces and appropriate verified APIs. Its compact guidance states that splitting a spatial axis must preserve element count, rather than adding p dimensions alongside full H/W. Source-side semantic checks conservatively flag contributing self-products as POSSIBLE_VIOLATION, leave constant-weight matmul unflagged, and do not declare reductions mathematically correct just because nl.sum appears. A NumPy linearity/self-product test checks the mathematical distinction independently, without executing generated source.

Full eight-round-ceiling shape-consistency refinement is running at controlled-20261010T195135-qyw82zco. Early selected candidates moved past DMA into reduction-rank and Python NkiTensor division failures. Their mathematical output remains unverified. A further tested root-cause rule identifies source-specific tensor division and recommends installed tensor_scalar with nl.multiply and a host-computed reciprocal. It explicitly preserves reduction semantics rather than introducing matmul to normalize a tensor. That rule is excluded from the already frozen refinement run.

New independent grouped-mean-plus-one-eighth synthetic task, not spatial pooling or an official reference kernel, was simulator/NumPy-verified on three varied inputs. Four actual injected failures/restorations cover extra shape axes, wrong divisor, max instead of sum, and unsupported Python tensor division. All four were accepted with real feedback. data_v6 contains 17 unique clean kernels and 32 repair pairs (30 train / 2 held-out), zero device-verified. Active training remains frozen to data_v4; the new examples are not secretly introduced mid-run.

Latest validation: 158 tests plus 72 parametrized subtests passed; both unchanged benchmark self-tests passed. Original checker/references remain byte-identical. No reward, tolerance, test-case, stopping or grading changes.

```bash
python -B -m pytest -q tests
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python -B nkibench.py --selftest
python -B kernelbench.py --selftest
python -B -m synthetic_nki.semantic_curriculum --output synthetic_nki/NEW-CORPUS
python -B -u run_controlled.py --run --full-agent-only --planner-policy hardware --rounds 8 --samples 4 --repeat 1 --levels 1
```

Only launch the final command after process guards are clear. Preserve unique snapshots and logs. The experiment differs by a coordinated scalar-binding/shape-guidance refinement, so its components cannot be attributed separately without another ablation.


## Completed refinement and primitive sprint

The scalar/shape refinement completed at 0.30 with 28 candidates, zero numerical shapes, 37,536 endpoint tokens and 545.78 seconds. Follow-up Qwen warm repairs also remained at 0.30: wrong instruction namespace then wrong arithmetic-opcode namespace. A generic optional instruction legalizer replayed one exact generated source at 1.00 on all four cases, with no expert edits; this is warm-start evidence only. Full cold-start validation is in runs/controlled-20261010T201403-vxbwe5dd. Read PRIMITIVE_CORRECTNESS_SPRINT.md for SDK checks, attribution and latest artifacts. LoRA completed 82 steps; correctness evaluation is queued and remains unmeasured.
