# Hackathon development instructions

- Work on Project 2, the Autonomous NKI Kernel Agent, only. Do not modify Project 1.
- Read repository README.md, STATE.md, and all Project 2 source before proposing changes.
- Make edits only in a separate Git worktree; preserve /workspace and its running baseline.
- Never stop, signal, restart, or modify baseline/model processes. Verify current process identity before reporting status.
- Never overwrite experiment artifacts. Use unique run directories and record revision, settings, repeat, round, and candidate identifiers.
- Never expose or commit credentials, tokens, environment dumps, or secrets.
- Application changes require review of the implementation plan first. Documentation is authorized now.
- Keep changes focused and testable. Validate the checker before trusting scores.
- Compare repeated baseline/improved runs using success, attempts, numerics, reward, tokens, and elapsed time. Distinguish CPU simulation metrics from device latency.
- Label reported, historical, preliminary, and independently verified results explicitly. Never claim unverified improvements.
- Maintain HACKATHON_CONTEXT.md with evidence, decisions, progress, and next steps.

## Approved development scope
- The user approved Phases 1-3 plus minimal evaluation instrumentation and guarded controlled experiments. This supersedes the earlier documentation-only gate.
- Develop in /tmp/trainium-kernel-dev on feat/nki-diagnostic-selection.
- Default standard generation/standard repair/reward selection and existing checker, scoring and stopping stay unchanged. New policies are opt-in.
- During development use pure/mocked tests or tiny CPU API exercises; never call grade() against shared /tmp/_agent_levelN.py while the baseline runs. Use private grade directories.
- Do not implement a full adaptive repair planner. After tests, a small sequential pilot is authorized only once the original baseline is finished; otherwise prepare it and report pending completion without waiting indefinitely.

## Official AWS references
- Consult /tmp/aws-neuron-agentic-reference as read-only source material; preserve AWS attribution. Do not install its plugins or execute its hardware workflows during protected experiments.
- Verify guidance against installed NKI 0.6.0 signatures/source and Trainium2 constraints before adding reference cards. SDK presence, simulation success, and device correctness are distinct evidence levels.
- The current user request authorizes Phase 2 diversity, Phase 3 local grounded repair, instrumentation and a conditional pilot. It forbids interference with the active original baseline.

## Approved Phase 4 scope
- Current user instructions authorize optional constrained generation, targeted root-cause/shape-aware coordinated repairs, sequential cold-start comparisons, documentation and local commits. This supersedes the earlier prohibition on repair planning within this bounded scope. Keep baseline defaults, scoring and stopping unchanged.
- Reference Wan2 read-only at /tmp/wan2-trainium-reference. Do not install its dependencies, deploy it, copy incompatible API calls, or displace Qwen. Device profiling remains a future verification-gated plan.
- Live evaluations must use frozen source snapshots, exclusive logs/private grading directories and endpoint/process guards. Do not start a duplicate evaluation.

## Approved synthetic/diagnostic evaluation scope
- Current user authorization includes a 12-kernel corpus, 20-40 verified injected-error repairs, train-only retrieval, bounded adaptive history, bug-specific diagnosis checks and sequential live ablations. Preserve existing work and original benchmark files.
- Prioritize the earliest actual simulator failure. Installed nl.load and nl.store exist; do not classify them as invented functions. Diagnostic coverage does not establish a causal correctness benefit.
