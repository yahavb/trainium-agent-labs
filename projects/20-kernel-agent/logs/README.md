# Attempt log

`attempts.tar.gz` holds the agent's attempts from our seats (95, 96, 98, 99): **1,418 attempts** in 43 files,
one folder per seat, one JSON object per line. The level-8 solves are in `../results/seat97-repair/`.

**Removed:** the 1,096 failed level-8 attempts (many were byte-identical repeats across seats, because the
server's sampling is deterministic). What they failed on is kept in `level8_failures_summary.json`; the
largest group (476) is the q·kᵀ layout mistake: q passed to `nc_matmul` without being transposed.

```bash
mkdir -p /tmp/attempts && tar xzf logs/attempts.tar.gz -C /tmp/attempts
python analysis/trace_analysis.py /tmp/attempts/seat99/attempts-l2-tp4-fixed.jsonl
```

Each line: `level`, `run`, `round`, `sample`, `reward` (0.1 parses + 0.2 rules + 0.2 runs + 0.5 correct),
`parts`, `code` (the kernel), `feedback` (exactly what the model was told next), `selected` (the sample the
loop carried forward), and, in later runs, `prompt_tokens`, `completion_tokens`, `finish`, `seconds`,
`temperature`, `session`, `source_hash`, `commit` and `flags`. File names give the level and setup
(`l11-device-safe` = level 11 with the traffic bar). Older files predate some fields.
