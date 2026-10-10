# H5: does "runs" credit for NaN output distort the results? (offline re-score)

- **Who / seat:** Rishabh (+ Claude), offline, no seat
- **Commit measured:** the logs already in `logs/`; script `projects/02-kernel-agent/rescore.py`
- **Hypothesis:** a kernel that runs but never writes its result (NaN or zero output) earns the 0.2
  "runs" credit and outranks a nearly correct kernel that crashes, which steers the loop wrong.
- **Change:** none to the agent. r2 = today's reward minus 0.2 when the feedback says the output is
  non-finite or mostly zeros. Every log is re-scored with it.
- **Command:** `cd logs && python3 ../projects/02-kernel-agent/rescore.py`

## Scores

```
log (attempts)                         degenerate   best per run: today -> r2
baseline (432, agent.py, 4 samples)         0       unchanged; repair pick changed in 0/108 rounds
api-messages (240, agent.py)                0       unchanged; 0/60 rounds
plan-given (28, agent.py)                   0       unchanged; 0/7 rounds
loop-tool given (39)                        8       [0.5, 0.5, 0.5, 0.5, 0.3] -> [0.3, 0.3, 0.3, 0.3, 0.3]
loop-tool given, v2.1 (40)                  2       [0.5, 0.3, 0.3, 0.3, 0.5] -> all 0.3
loop-tool given, v2.1 + structure (40)      2       [0.5, 0.3, 0.3, 0.5, 0.3] -> all 0.3
loop-tool staged (20 + 10)                  0       unchanged: 1.0 x5
```

The staged-loose and unstaged-steps ablation logs keep per-stage rewards, so their per-run best isn't a
level score. They're left out of the reading.

## Verdict

**The flaw inflated reported scores; it didn't steer the loop.** Every 0.50 reported for the loop tool
today was a kernel whose output was never written. Under r2 all of them are 0.30, the same as the
crashing attempts, so "0.50 beats 0.30" in those write-ups is an artifact. But the loop repairs the
LATEST attempt, not the best (agent.py, `solve()`), and those runs used 1 sample per round. So the
reward never chose which attempt got repaired, and no trajectory would have changed. In the agent.py
logs, no attempt was degenerate at all.

**Fix for reporting honesty:** give "runs" credit only when the output is finite and was actually
written. That's a one-line change in the grader, to apply before quoting any more loop-tool means.
There's no need to rerun anything. The 5/5 staged result and every agent.py result stand as reported.
