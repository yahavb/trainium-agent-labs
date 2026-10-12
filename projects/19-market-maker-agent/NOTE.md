# One-page note: the market-maker agent

**What ran, on what.** Qwen3-8B on one Trainium2 chip (seat 94; vLLM-Neuron, TP 2, context 8192,
thinking off) wrote market-maker quoting functions. 798 generations ran on the device
(14.5 attempt-hours, median 5.6 tokens/s per request). Every grade ran on the CPU,
in a deterministic seeded order-book simulator; no number here is a device-performance claim. Each
arm: 3 levels × 3 repeats × (6 rounds × 2 samples). The level-3-only follow-ups had 5 repeats.

**The comparison** (frozen at commit `b73b918`, equal budgets): score only (A) **0 of 9** levels
verified; raw metrics (B) **0 of 9**; the checker naming the cause (C) **4 of 9** (one-sided
Fisher p = 0.04 against B). That's suggestive, not conclusive: repeats often reproduced identical
code.

**Every verified claim held on 32 held-out seeds frozen before any run.** Confidence calibration
over 79 claims: Brier 0.029. Every claim at ≥ 97.5% confidence held, and of the
claims below that, 0 held.

**What the runs taught us about the checker** (each fix one change, measured as its own arm):
1. *A budget failure looked like a model failure.* A cut-off answer was graded as a syntax error.
   It is now tagged `TRUNCATED`.
2. *Our message blamed the wrong line* for the most common crash (`a, b = state`, 39–44% of failures
   in A–C). Naming the cause cut it to 20–25% (C3: 6 of 9 verified).
3. *"Never trade" was a stable trap* (29–42%) until it got its own message (C2).
4. *The held-out set was too easy.* The real-data replay showed real stocks at a 1-tick spread; every
   verified strategy broke a rule on 99% of real steps. Adding the 1-tick case to the checker (C4)
   made every new strategy legal on all four real stocks, including two unseen ones.
5. *A pass is not a mechanism.* A level-3 "solve" (C3, margin +4) never read the order book.
   252-state hostile probes now catch that kind of latent bug in a second.
6. *A safety hole:* numpy file I/O was not blocked. No strategy used it; it is now blocked.
7. *Evidence did not help on level 3 (C5, pre-registered, 0 of 5).* The model saw its own losing
   fills and repeated itself.
8. *Concepts did (C6, pre-registered).* Three cited microstructure ideas in the prompt produced the
   real mechanism in 3 of 5 repeats, profitable at 2 ticks (dev lcb +267). It was not verified: it
   broke even at 1 tick. Correcting that diagnosis (C7, pre-registered ≥ 2 of 5) produced **one robust, verified level-3 strategy** (1 of 5; held-out lcb +347, survives every hostile test). The hypothesis is rejected as written, but it is the first level-3 solve that holds up. On real days it is legal but not profitable.

**Real data (sanity check).** LOBSTER samples, 2012-06-21. No strategy, ours or the agent's, is
profitable on all four stocks. Our hand-written reference wins only on INTC and loses on the two
unseen high-priced stocks, because its rules are in absolute ticks.

**Limits.** Synthetic market with a simple queue model; the replay cannot react to quotes; profit
in ticks, no fees; level 4 excluded (not fair); small samples; one model.
