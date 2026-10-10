# Level 8 baseline: the original agent

- **Who / seat:** Rishabh (+ Claude), seat 48
- **Commit measured:** the ORIGINAL agent.py and nkibench.py from git HEAD `8f1ca41` (md5 7d0dcc0c… and
  9f268b38…), unmodified. The only addition: M/K/N keys on the level-8 shape dicts, so the stock
  grader's roofline line can't crash on a correct kernel.
- **Command:** `l8_baseline.py` = `agent.py --level 8 --rounds 8 --samples 1 --context 8192 --repeat 2`
- **Log:** `logs/attempts-rishabh-l8-baseline.jsonl` (14 attempts)

## Scores

```
level 8 (original agent): solved 0/2   all = [0.30, 0.30]
level 8 (staged, results/2026-10-10-rishabh-attention.md): solved 2/2   all = [1.00, 1.00]
```

Top failures: reshape instead of slicing (8/14), Python `*` on two tiles (2), `nc_matmul(src=...)` (1).

## Verdict

The baseline for the level-8 claim: **0.30 → solved**. There is no earlier correct kernel, so there's
no speed baseline. The solved kernel measures about 20 µs per call on the chip
(results/2026-10-10-rishabh-chip-timing.md).
