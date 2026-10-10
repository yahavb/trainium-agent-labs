# Project: CHIPBOOST (Hack the Chip, Oct 10 2026)
Plan: projects/03-chipboost/README.md. Qwen3-8B on the Trainium chip optimizes its own kernels
(matmul, RMSNorm). Our referee (speedcheck.py) checks correctness in the simulator, on the chip and on
held-out shapes, times on the chip, and returns ONE named change. We also red-team the referee with
planted cheating kernels.

## Environment facts
- Code runs ONLY in the seat pod (Neuron SDK 2.32, NKI 0.6.0). This laptop has no `nki`: never try to
  run or "test" kernels here, and never add workarounds that fake nki. Edit here, push, pull and run in
  the pod.
- Fork: github.com/likhith2366/trainium-agent-labs, main branch `master`. One branch per person.
- Pod: vLLM serves Qwen3-8B on NeuronCores 2-3; cores 0-1 are free (NEURON_RT_VISIBLE_CORES=0,1).
- NKI 0.6.0 imports: `import nki`, `import nki.language as nl`, `import nki.isa as nisa`. Simulate with
  `nki.simulate(kernel)(*args)`. There is no public `nki.benchmark`; a plain kernel(*args) call
  recompiles (~1.5 s), so never time that call.
- Harness: projects/02-kernel-agent/ (agent.py = loop, nkibench.py = checker). Reuse check_rules,
  describe_mismatch, check_inputs_untouched, simulate_and_count, reuse_report, check_traffic_bar.
  Add new ops with level(...) like level 8. The byte counter only hooks nisa.dma_copy.

## Rules
- Constraints go in the checker, not the prompt. Feedback = one instruction, never the answer.
- Never report a number from --offline or from a single run; use --repeat and give the spread.
- Label every number: simulator / chip / projection.
- Never commit AWS credentials or tokens.

## Results to include in the final report
- **P3 headline (chip, seat 101): Qwen3-8B + referee feedback made the matmul kernel 1.517x faster**
  (960.9 -> 633.2 us), verified `faster` by speedcheck (chip correctness with hostile inputs, held-out shapes)
  and replicated by a standalone re-check. Read `projects/03-chipboost/P3_HEADLINE.md` and put it in the
  report and slides as P3's main result, with its one-line context (one verified run, reproduced; the gain
  is scheduling, not bytes; separate from P2's expert-tuning claim). Full history: `P3_STATUS.md`.
