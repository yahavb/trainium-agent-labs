# Project 2 evidence


## Operation-aware primitive and LoRA update (2026-10-10)

Latest evidence is summarized in LEVEL_SCORECARD.md, PRIMITIVE_CORRECTNESS_SPRINT.md, LORA_TRAINING_REPORT.md and LORA_COMPARISON.md. Full untuned L1/L2 remained .30, zero numerical cases. Current cold L1 planner/legalizer has a provisional .50 execution-only candidate, zero shapes; no cold solve claimed. A generic deterministic instruction-legalizer warm replay reached L1 1.00, 4/4 cases, separately labeled and excluded from cold-start rates/training. L3 remains 1.00; L4 .625 (1/4). No hardware performance measured.

Corpus data_v8: 21 independently simulator/NumPy-verified clean kernels, 38 verified error/restoration pairs, 36 training/retrieval candidates and two held-out records. Existing retrieval and actual LoRA training remain on frozen data_v4; later data did not enter the running training. Diagnosis rubric on 38 actual injected failures: legacy 21 correct, targeted 30, targeted+SymPy 30; ten supported symbolic causes, four UNKNOWN, zero clean static false-positive kernels among 21. These are task-specific offline checks, not general diagnostic accuracy or proof of live improvement.

LoRA has completed 82 optimizer steps on 41 examples (15 generation, 26 repair), with three held-out examples from one completely excluded family. Rank 8/alpha 16, q_proj/v_proj, lr1e-4, completion-only loss, batch1; CPU training avoided Neuron device resources. Held-out mean example loss .9951169491 -> .1274786662 is not correctness evidence. Adapter saved in runs/qwen3-nki-lora-mdzxuq1k/training/adapter. The preserved full-adapter evaluation waits for active cold-run completion; a matched original/LoRA x legacy/planner study follows sequentially on the same isolated CPU backend. New held-out repairs are queued afterwards. Shared Qwen/Neuron vLLM was not restarted or modified.

SDK-verified miniature compositions exercise access-pattern reduction/scalar scaling, free-axis permutation, partition/free transpose and explicit K accumulation. Optional --primitive-policy legalize corrects narrowly verified instruction/opcode namespaces and known HBM scalar-result staging; it never supplies an algorithm or benchmark-specific code. Original generated source and transformed source are both recorded. Original grading/references/shapes/tolerances/hazards remain unchanged.
