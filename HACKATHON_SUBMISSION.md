# Autonomous NKI kernel generation and repair agent

NYU x Amazon Annapurna Labs Hack the Chip 2026, Project 2. Qwen3-8B on Trainium2 generates candidate NKI kernels, then an unchanged independent checker verifies static legality, CPU simulation, numerical correctness, input integrity and relevant traffic/hardware constraints. The agent feeds failures back, stops on verification, and records unsuccessful attempts honestly.

The latest original five-repeat experiment solved 0/5 on each of Levels 1-4. Diversity prompting increased AST-distinct candidates in the initial pilot but did not improve correctness. A later existing-full-agent cold start solved Level 3 (reward 1.0), independently replayed; new constrained generation failed. No causal treatment improvement or device speedup is claimed.

We implemented opt-in deterministic diagnosis, reward-primary diagnostic selection, prompt diversity, SDK-verified local guidance, conservative AST shapes/root causes, private grading, endpoint token accounting and sequential frozen-source ablations. A synthetic pipeline independently validates twelve tiny non-benchmark kernels and twenty-four actual error/restoration pairs. Twenty-two training pairs and two held-out pairs from a separate family are saved. Retrieval exposes short transferable changes, never full benchmark solutions; a bounded adaptive controller escalates repeated failures without extra model calls.

Added-feedback bug-specific coverage is 18/24 legacy versus 22/24 targeted, with unproven numeric causes counted as incorrect. This skewed small corpus does not establish general accuracy or Qwen improvement. All 95 deterministic tests and both original checker self-tests passed before live ablation. No device-verified examples. Original source, checker, tolerances, reference kernels, SDK and logs remain unchanged.

Reproducibility: development branch feat/nki-diagnostic-selection; complete source/data/reports and frozen experiment manifests/raw attempts in /tmp/trainium-kernel-dev. SYNTHETIC_NKI_EXPERIMENTS.md records measured comparisons; SYNTHETIC_NKI_ITERATION.md records the evidence-based refinement and its limits.
