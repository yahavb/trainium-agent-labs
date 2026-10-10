# Attempt log

`attempts.tar.gz` holds **every attempt** the agent made on our seats (95, 96, 98, 99): 2,106 attempts in
81 files, one folder per seat, one JSON object per line.

```bash
mkdir -p /tmp/attempts && tar xzf logs/attempts.tar.gz -C /tmp/attempts
python analysis/trace_analysis.py /tmp/attempts/seat99/attempts-l2-tp4-fixed.jsonl
```

Each line: `level`, `run`, `round`, `sample`, `reward` (0.1 parses + 0.2 rules + 0.2 runs + 0.5 correct),
`parts`, `code` (the kernel), `feedback` (exactly what the model was told next), `selected` (the sample the
loop carried forward), and, in later runs, token counts, `finish`, `seconds`, `temperature`, `session`,
`source_hash`, `commit` and `flags`. File names say the level and setup (`l8-tp4-seed95` = level 8, fast
server, seat 95). Older files predate some fields.
