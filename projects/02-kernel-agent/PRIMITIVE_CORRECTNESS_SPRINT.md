# Operation-aware correctness sprint

The original five-repeat study remains 0/5 solves on each level. The full untuned agent completed Level 1 and 2 at 0.30. Operation-aware planning corrected the observed Level 1 generation direction toward window reductions, but mathematical direction alone did not produce legal NKI execution.

## Verified programming constraints

Reviewed AWS [AveragePool2D](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/tutorials/average_pool2d.html), [tensor_scalar](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_scalar.html), [Transpose2D](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/guides/tutorials/transpose2d.html), [nc_transpose](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_transpose.html), [matrix multiplication](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.0/nki/guides/tutorials/matrix_multiplication.html), [nc_matmul](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul.html), and the existing isolated AWS samples/reference reviews. Conceptual attribution to AWS; new miniature implementations are project-authored. No completed benchmark solution is included in prompts or training.

Installed NKI is 0.6.0+31049202112.g85070674; SymPy is 1.14.0. Inspect.signature confirms explicit dst-based dma_copy, tensor_scalar, tensor_copy, nc_transpose and nc_matmul. Earlier return-style examples are not transplanted. nc_matmul exposes accumulate=None/False/True. First matmul must overwrite its PSUM location; later disjoint K contributions accumulate. Non-matmul value initialization before accumulation is an unsupported hardware pattern. Simulation is not device verification.

A pool window sum is scaled by a host-computed reciprocal using **nki.isa.tensor_scalar**, with the arithmetic opcode **nki.language.multiply**. The result is staged through SBUF before dma_copy to HBM. A strided access-pattern view can group elements without allocating additional data or changing element count. Reduction rank depends on actual retained dimensions, rather than blindly requiring keepdims.

Level 2 preserves P and permutes two flattened free dimensions. This is distinct from nc_transpose, which exchanges partition/free dimensions. Independent NumPy-checked miniature tests exercise both cases. Installed nc_transpose selects vector when dst is SBUF and tensor otherwise; tensor requires SBUF input/PSUM output and <=128x128, vector <=32x32. Operand memory constraints and dtype still matter.

NKI integer indexing differs from NumPy: SBUF/PSUM preserve dimension zero as a singleton and pad views to rank two; HBM integer selections drop axes but preserve at least rank one. Both host analyzers now reflect this verified SDK rule. Unknown buffer provenance remains UNKNOWN.

## What changed

- Compact mathematical/hardware planning before initial generation; optional semantic evidence before grading. Self-product pooling is a hypothesis, not automatically rejected: constant-weight matmul can implement a linear reduction. Only a side-effect-free identity function receives the narrow PROVEN_INVALID task proof. The official checker still runs and determines rewards.
- Source-specific scalar-instruction/opcode namespace guidance, legal on-chip result placement, and preservation of reduction/normalization.
- Safe binding of actual reported scalar task arguments for literal/SymPy shape analysis. First planner-run DMA diagnostics improved from zero to 17/20 resolved candidates after pool size was bound; this is offline inference coverage, not a live correctness gain.
- Optional `--primitive-policy legalize`: deterministic instruction-role correction for tensor_scalar plus staging of a known on-chip scalar result through SBUF to its original known shared_hbm destination. It contains no task-specific dimensions, benchmark names, or algorithm templates. It never creates a missing pooling/transpose/matmul computation. Unknown operands remain untouched. Every change and both raw generated and transformed sources are logged; the unchanged checker verifies the transformed candidate. Default is off.
- Four independent miniature compositions: grouped mean plus bias using .ap; free-axis permutation plus bias; partition/free transpose plus scaling; and three-block contraction plus bias. They cover DMA, reductions, scalar scaling, PSUM transfers, transpose, explicit accumulation and output writes.

## Actual results and attribution

- `runs/controlled-20261010T192716-jp07xp0l`: full agent L1 0.30, 32 candidates; L2 0.30, 20 candidates. No numerical cases passed.
- `runs/controlled-20261010T194114-ln8yl9ak`: first planner L1 0.30, 20 candidates; all 20 use reduction direction, 17 DMA/3 rank errors.
- `runs/controlled-20261010T195135-qyw82zco`: scalar/shape refinement L1 0.30, 28 candidates; 37,536 actual endpoint tokens, 545.78 seconds. Reduction rank was corrected along one trajectory, followed by tensor division and later PSUM/SBUF errors.
- `runs/level1-expert-diagnostic-o4u29a4q`: expert-edited saved candidate 1.00, 4/4 shapes. Explicitly excluded from autonomous cold-start solve rates and training/retrieval.
- `runs/level1-warm-targeted-_ptryalc`: four Qwen repairs 0.30; wrong nl.tensor_scalar namespace and HBM dst.
- `runs/level1-warm-namespace-u3_qnpj4`: four further Qwen repairs 0.30; the instruction namespace changed but op0 became nonexistent nisa.multiply. 71.78 seconds. Source and real feedback preserved.
- `runs/level1-automatic-primitive-replay-72i8u7g7`: generic instruction legalizer applied to that exact generated candidate, without expert per-candidate edits: **1.00, 4/4 official shapes**. This is a warm-start deterministic system repair, not independent cold-start success or a fine-tuning result. No complete solution was supplied to the model.
- `runs/controlled-20261010T201403-vxbwe5dd`: full eight-round-ceiling, four-candidate cold-start L1 test, planner + optional legalizer. Consult its raw JSONL and final summary for current results; do not infer success from the warm replay.

## Synthetic and diagnostic evidence

Immutable data_v8 contains **21 clean simulator/NumPy-verified kernels and 38 verified repair pairs (36 train, 2 held-out)**. Six new pairs passed the intended actual simulator/numerical failure and restoration checks; zero rejected in this final expansion. data_v7 preserves an earlier rejected access-pattern mutation: the real out-of-storage error was UNKNOWN until its exact pattern was added to the taxonomy. Never relabel the original artifact retroactively.

On all 38 bug-specific records, deterministic diagnosis checks score legacy 21/38, targeted 30/38, targeted+SymPy 30/38. SymPy independently detects 10 supported shape causes; 4 records remain UNKNOWN. Zero reported static violations on 21 clean examples. Analysis time was 0.06005 seconds for the broken records. These are corpus-specific rubric checks, not general diagnostic accuracy; syntactically valid line references are not runtime causality proof. Numerical wrong-normalization/operation failures require task semantics beyond shape consistency. The first LoRA training run and current default retrieval remain frozen on data_v4, excluding later examples and held-out records.

## Reproduction

```bash
cd /tmp/trainium-kernel-dev/projects/02-kernel-agent
python -B -m pytest tests -q
python -B nkibench.py --selftest
python -B kernelbench.py --selftest
python -B -m synthetic_nki.primitives --base synthetic_nki/data_v6 --output synthetic_nki/NEW-CORPUS
python -B -m synthetic_nki.symbolic_eval --data synthetic_nki/NEW-CORPUS --output synthetic_nki/NEW-CORPUS/diagnosis.json
# Only when no evaluation is active:
python -B run_controlled.py --full-agent-only --planner-policy hardware --primitive-policy legalize --levels 1 --rounds 8 --samples 4 --repeat 1 --run
```

Keep evaluations sequential and preserve frozen source/private grading directories. Pending LoRA studies use an isolated CPU endpoint because the installed Neuron vLLM worker cannot hot-load PEFT adapters. Their timing cannot be compared as a speedup over Neuron vLLM. The matched original/LoRA × legacy/planner study uses the same CPU model/backend with adapters disabled/enabled, unchanged budgets and checker. Held-out repairs are separately queued after the preserved studies; only completed artifacts are results.
