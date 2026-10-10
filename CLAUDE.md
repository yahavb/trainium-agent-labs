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
- Pod: vLLM serves Qwen3-8B on NeuronCores 0-1 (measured, STATUS.md; older docs say 2-3). The referee
  times kernels on core 2, fallback 3, so at most two core holders per seat (check_isolated,
  RefereeWorker, heldout_grid.py, timing --selftest); give each its own CHIPBOOST_CORE.
- Referee API: REFEREE.md. Loops call speedcheck.check_isolated(path, op=...); None means the referee
  failed (retry), never a verdict on the kernel.
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
