# Hostile held-out: the 32 held-out seeds at a 1-tick spread

Only strategies the agent claimed as verified (plus the baseline and reference). Same pass rule as the checker.

| strategy | level | spread 2 (held-out) | spread 1 (hostile) | lcb at spread 1 | hostile states (252 probes) |
|---|---|---|---|---|---|
| baseline.py (level 1) | 1 | PASS | PASS | +256 | PASS |
| baseline.py (level 2) | 2 | fail | profit lcb -388 | -388 | PASS |
| baseline.py (level 3) | 3 | fail | profit lcb -603 | -603 | PASS |
| reference.py (level 1) | 1 | PASS | PASS | +128 | PASS |
| reference.py (level 2) | 2 | PASS | PASS | +239 | PASS |
| reference.py (level 3) | 3 | PASS | PASS | +149 | PASS |
| C rep 1 | 1 | PASS | rules: CROSSED x48000 | +0 | 63 fail (CROSSED, TAKES) |
| C rep 1 | 2 | PASS | rules: CROSSED x48000 | +0 | 63 fail (CROSSED, TAKES) |
| C rep 3 | 1 | PASS | rules: CROSSED x48000 | +0 | 63 fail (CROSSED, TAKES) |
| C rep 3 | 2 | PASS | rules: CROSSED x48000 | +0 | 63 fail (CROSSED, TAKES) |
| C2 rep 1 | 2 | PASS | rules: CROSSED x48000 | +0 | 63 fail (CROSSED, TAKES) |
| C2 rep 2 | 1 | PASS | rules: CROSSED x48000 | +0 | 63 fail (CROSSED, TAKES) |
| C2 rep 2 | 2 | PASS | rules: CROSSED x48000 | +0 | 63 fail (CROSSED, TAKES) |
| C2 rep 3 | 1 | PASS | rules: CROSSED x48000 | +0 | 63 fail (CROSSED, TAKES) |
| C2 rep 3 | 2 | PASS | rules: CROSSED x48000 | +0 | 63 fail (CROSSED, TAKES) |
| C3 rep 1 | 1 | PASS | rules: CROSSED x48000 | +0 | 63 fail (CROSSED, TAKES) |
| C3 rep 1 | 2 | PASS | rules: CROSSED x48000 | +0 | 63 fail (CROSSED, TAKES) |
| C3 rep 2 | 1 | PASS | rules: CROSSED x48000 | +0 | 63 fail (CROSSED, TAKES) |
| C3 rep 2 | 2 | PASS | rules: CROSSED x48000 | +0 | 63 fail (CROSSED, TAKES) |
| C3 rep 2 | 3 | PASS | profit lcb -171 | -171 | 144 fail (INVENTORY_LIMIT) |
| C3 rep 3 | 1 | PASS | rules: CROSSED x48000 | +0 | 63 fail (CROSSED, TAKES) |
| C4 rep 1 | 1 | PASS | PASS | +195 | PASS |
| C4 rep 1 | 2 | PASS | PASS | +172 | PASS |
| C4 rep 2 | 1 | PASS | PASS | +282 | PASS |
| C4 rep 3 | 1 | PASS | PASS | +195 | PASS |
| C4 rep 3 | 2 | PASS | PASS | +172 | PASS |
| C6r rep 2 | 1 | PASS | PASS | +282 | PASS |
| C7 rep 1 | 3 | PASS | PASS | +169 | PASS |
| C7r rep 1 | 1 | PASS | PASS | +160 | PASS |
| C7r rep 2 | 1 | PASS | PASS | +160 | PASS |
| C7r rep 2 | 2 | PASS | PASS | +66 | PASS |
