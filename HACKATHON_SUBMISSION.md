# Autonomous NKI kernel generation and repair agent

NYU x Amazon Annapurna Labs Hack the Chip 2026, Project 2. Qwen3-8B on Trainium2 generates candidate NKI kernels, then an unchanged independent checker verifies static legality, CPU simulation, numerical correctness, input integrity and relevant traffic/hardware constraints. The agent feeds failures back, stops on verification, and records unsuccessful attempts honestly.

The latest original five-repeat experiment solved 0/5 on each of Levels 1-4. Diversity prompting increased AST-distinct candidates in the initial pilot but did not improve correctness. A later existing-full-agent cold start solved Level 3 (reward 1.0), independently replayed; new constrained generation failed. No causal treatment improvement or device speedup is claimed.

We implemented opt-in deterministic diagnosis, reward-primary diagnostic selection, prompt diversity, SDK-verified local guidance, conservative AST shapes/root causes, private grading, endpoint token accounting and sequential frozen-source ablations. A synthetic pipeline independently validates twelve tiny non-benchmark kernels and twenty-four actual error/restoration pairs. Twenty-two training pairs and two held-out pairs from a separate family are saved. Retrieval exposes short transferable changes, never full benchmark solutions; a bounded adaptive controller escalates repeated failures without extra model calls.

Added-feedback bug-specific coverage is 18/24 legacy versus 22/24 targeted, with unproven numeric causes counted as incorrect. This skewed small corpus does not establish general accuracy or Qwen improvement. All 95 deterministic tests and both original checker self-tests passed before live ablation. No device-verified examples. Original source, checker, tolerances, reference kernels, SDK and logs remain unchanged.

Reproducibility: development branch feat/nki-diagnostic-selection; complete source/data/reports and frozen experiment manifests/raw attempts in /tmp/trainium-kernel-dev. SYNTHETIC_NKI_EXPERIMENTS.md records measured comparisons; SYNTHETIC_NKI_ITERATION.md records the evidence-based refinement and its limits.


## Latest verified evidence, 2026-10-10

Level 3 remains independently verified at 1.00 from autonomous initial generation. A saved Qwen Level 1 candidate now reaches 1.00 on all four official shapes after the generic optional instruction legalizer corrects API-role namespaces and stages the on-chip scalar result through SBUF to HBM. This is a verified warm-start system repair, excluded from cold-start solve rates; no reference solution or expert per-candidate edit was used by this legalizer. The locked source and independent full replay are in projects/02-kernel-agent/runs/verified-level1-locked-gowa0nvv. Active cold L1 has .50 execution credit and zero passing shapes; L2 .30/zero shapes; L4 .625/one of four.

Corpus: 21 clean NumPy/simulator-verified synthetic kernels and 38 actual mutation/restoration pairs. Bug-specific checks: legacy21/38 vs targeted30/38; SymPy independently catches ten supported causes but does not improve the combined30/38 score. Four UNKNOWN, zero clean shape false-positive kernels among21. Test suite184 tests plus76 subtests, both unchanged checker self-tests pass.

Actual Qwen3-8B CPU LoRA training completed82 steps on41 examples (15 generation/26 repair), with3 held-out examples in an excluded family. Rank8/alpha16 q_proj/v_proj adapters are saved. Held-out loss .9951->.1275; benchmark correctness effects remain pending. Original and adapter controls use the same isolated CPU endpoint and matched settings; original Neuron vLLM is preserved. Full results, exclusions and reproduction commands are in PRIMITIVE_CORRECTNESS_SPRINT.md, LEVEL_SCORECARD.md and LORA_COMPARISON.md. No device execution/speedup claimed.
