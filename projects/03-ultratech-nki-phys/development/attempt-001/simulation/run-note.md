# Distinct Math Tasks

Backend: simulate; tasks: ['net-force']. Each has 16 deterministic cases, a baseline and one reviewed candidate. FP64 references use exact saved FP32 inputs; fixed error budget is 1e-6 + 2e-6 times the sum of absolute contributing terms. Shape, finite FP32 output and unchanged inputs are required. This is equation validation, not MuJoCo rollout validation.

Device timing: seed 3 only, 5 repeats of 200 samples, 20 warmups. Baseline-before/candidate/baseline-after; >10% baseline drift invalidates reward. Device-only timing excludes compilation, transfers and readback; no end-to-end claim. CPU results test reference/checker plumbing, not NKI execution or speed. Without --program templates are human supplied. With --program the model's bounded operation graph is lowered by trusted code. This runner does not itself call Qwen. Sources, input/output hashes, every attempt, checks and timing samples are retained.
