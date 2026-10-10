# Challenge 1 — Heat-Rod PDE Agent results

> Historical report: the current agent uses only the unchanged upstream checker.
> The additional physics validator described below has been removed; see README.md.

Hardware: seat-85, Trainium2 (`trn2.48xlarge` host, one allocated chip); Qwen3-8B, TP=2, context 8192.
Real model inference only. No offline generator and no analytic-answer fallback.

Comparison: upstream baseline versus enhanced agent WITH the experimental concise output contract. Each variant: 2 samples/round, at most 3 rounds, 512 output tokens, 1 calculator exchange/attempt; 180-second case budget. Level 0 and Level 1 subproblems 1/2: one run per subproblem. Level 1 subproblem 3: three problem seeds. Sampling is stochastic; problem seeds do not fix generated answers.

| Variant | Level | Validated solved | Service failures | Budget timeouts | Reward range on completed runs | Mean rounds on completed runs |
|---|---|---|---|---|---|---|
| baseline | 0 | 3/3 | 0 | 0 | 1.0–1.0 | 1.33 |
| baseline | 1 | 0/5 | 0 | 0 | 0.0–0.8 | 3.00 |
| improved | 0 | 1/3 | 0 | 0 | 0.4–1.0 | 2.33 |
| improved | 1 | 0/5 | 0 | 1 | 0.6–0.8 | 3.00 |

The improved checker preserves the original physics reward and qualitative coefficient feedback, then requires a separate 384-point space/time check and a refined 1601-point initial-shape check. It does not consult known exact solutions. A tested high-frequency error that earns full marks from the original checker is rejected by the added near-zero-time checks.

Both original checker selftests and all nine new regression tests pass. The client records calculator requests/results, retries transient errors, preserves its best candidate, and keeps a short failure ledger.

The concise contract regressed on Level 0 (baseline 3/3 versus concise 1/3), so it is now optional (`--concise`) and disabled by default. This table measures that experimental mode, not the final default agent. No claim of improved default solve rate is made; a separate real-model smoke run checks the default path.

Final default-mode smoke run: level0.1, solved, reward 1.0, 1 round(s), 35.2s; independent validation accepted: True. This is one runnable-path check, not an estimated solve rate.

Observed duplicate final answers in 26/40 recorded multi-sample rounds. The two samples must not be treated as independent statistical trials.

Limits: only the hardest parabola case uses three problem seeds; the easier cases use one. These are exploratory results, not statistical proof of improvement. Numerical validation is not a universal symbolic proof. Wall-clock costs may differ because the improved client retries transport errors. The first batch crashed on a Neuron-incompatible per-request seed; its logs are retained separately and are not included in this comparison.

A second preliminary batch exposed long derivations exhausting output budgets; it was stopped to add a concise output contract and truncation-aware feedback. Its partial logs are retained separately. Only the final batch is summarized here.

Budget timeouts are inconclusive: they do not imply a zero mathematical reward or a server failure. Any full-score answer in a completed baseline run must pass the independent verifier before being counted as solved.

Evidence: `comparison.json` plus case-level `attempts.jsonl`, `console.log` and `summary.json` in the same batch directory.
