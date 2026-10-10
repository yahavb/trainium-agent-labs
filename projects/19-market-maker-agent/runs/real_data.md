# Real-data replay (sanity check, not part of the claim)

### MSFT, 2012-06-21, 14 episodes of 1500 s

| strategy | PnL per episode (ticks) | lcb | episodes won | fills | markout (10 s) | obligation | rule violations (side withdrawn) |
|---|---|---|---|---|---|---|---|
| baseline.py | +17 ± 22 | -28 | 71% | 137 | +0.09 | 100% | 0 |
| reference.py | +16 ± 8 | +0 | 86% | 54 | +0.40 | 98% | 0 |
| A rep 1 level 1 (unsolved) | -11 ± 7 | -25 | 0% | 2 | -0.74 | 89% | TAKES 2300 |
| A rep 1 level 2 (unsolved) | -11 ± 7 | -25 | 0% | 2 | -0.74 | 89% | TAKES 2300 |
| A rep 1 level 3 (unsolved) | -11 ± 7 | -25 | 0% | 2 | -0.74 | 89% | TAKES 2300 |
| A rep 2 level 1 (unsolved) | -11 ± 7 | -25 | 0% | 2 | -0.74 | 89% | TAKES 2300 |
| A rep 2 level 2 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41688 |
| A rep 2 level 3 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41688 |
| A rep 3 level 1 (unsolved) | -11 ± 7 | -25 | 0% | 2 | -0.74 | 89% | TAKES 2300 |
| A rep 3 level 2 (unsolved) | -11 ± 7 | -25 | 0% | 2 | -0.74 | 89% | TAKES 2300 |
| A rep 3 level 3 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41688 |
| B rep 1 level 1 (unsolved) | +4 ± 4 | -4 | 50% | 3 | -0.31 | 1% | TAKES 20844 |
| B rep 1 level 2 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41688 |
| B rep 1 level 3 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 100% | 0 |
| B rep 2 level 1 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41688 |
| B rep 2 level 2 (unsolved) | +4 ± 4 | -4 | 50% | 3 | -0.31 | 1% | TAKES 20844 |
| B rep 2 level 3 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41688 |
| B rep 3 level 1 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41688 |
| B rep 3 level 2 (unsolved) | +4 ± 4 | -4 | 50% | 3 | -0.31 | 1% | TAKES 20844 |
| B rep 3 level 3 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 100% | 0 |
| C rep 1 level 1 (solved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C rep 1 level 2 (solved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C rep 1 level 3 (unsolved) | -10 ± 5 | -21 | 0% | 2 | -0.55 | 100% | 0 |
| C rep 2 level 1 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41688 |
| C rep 2 level 2 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41688 |
| C rep 2 level 3 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41688 |
| C rep 3 level 1 (solved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C rep 3 level 2 (solved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C rep 3 level 3 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41688 |
| C2 rep 1 level 1 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41688 |
| C2 rep 1 level 2 (solved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C2 rep 1 level 3 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41688 |
| C2 rep 2 level 1 (solved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C2 rep 2 level 2 (solved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C2 rep 2 level 3 (unsolved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C2 rep 3 level 1 (solved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C2 rep 3 level 2 (solved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C2 rep 3 level 3 (unsolved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C3 rep 1 level 1 (solved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C3 rep 1 level 2 (solved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C3 rep 1 level 3 (unsolved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C3 rep 2 level 1 (solved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C3 rep 2 level 2 (solved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C3 rep 2 level 3 (solved) | -21 ± 8 | -37 | 29% | 135 | -0.11 | 100% | INVENTORY_LIMIT 12 |
| C3 rep 3 level 1 (solved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C3 rep 3 level 2 (unsolved) | -34 ± 24 | -83 | 50% | 216 | -0.10 | 100% | 0 |
| C3 rep 3 level 3 (unsolved) | -10 ± 5 | -21 | 0% | 2 | -0.55 | 100% | 0 |
| C4 rep 1 level 1 (solved) | -21 ± 8 | -36 | 29% | 136 | -0.11 | 100% | 0 |
| C4 rep 1 level 2 (solved) | -21 ± 8 | -36 | 29% | 136 | -0.11 | 100% | 0 |
| C4 rep 1 level 3 (unsolved) | -34 ± 24 | -83 | 50% | 216 | -0.10 | 100% | 0 |
| C4 rep 2 level 1 (solved) | -34 ± 24 | -83 | 50% | 216 | -0.10 | 100% | 0 |
| C4 rep 2 level 2 (unsolved) | -34 ± 24 | -83 | 50% | 216 | -0.10 | 100% | 0 |
| C4 rep 2 level 3 (unsolved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C4 rep 3 level 1 (solved) | -20 ± 8 | -37 | 29% | 135 | -0.11 | 100% | 0 |
| C4 rep 3 level 2 (solved) | -20 ± 8 | -37 | 29% | 135 | -0.11 | 100% | 0 |
| C4 rep 3 level 3 (unsolved) | -10 ± 5 | -21 | 0% | 2 | -0.55 | 100% | 0 |
| C4b rep 1 level 3 (unsolved) | +2 ± 4 | -6 | 57% | 3 | -0.13 | 1% | CROSSED 11786, TAKES 9058 |
| C4b rep 2 level 3 (unsolved) | -34 ± 24 | -83 | 50% | 216 | -0.10 | 100% | 0 |
| C5 rep 1 level 3 (unsolved) | -34 ± 24 | -83 | 50% | 216 | -0.10 | 100% | 0 |
| C5 rep 2 level 3 (unsolved) | -34 ± 24 | -83 | 50% | 216 | -0.10 | 100% | 0 |
| C5 rep 3 level 3 (unsolved) | -34 ± 24 | -83 | 50% | 216 | -0.10 | 100% | 0 |
| C5 rep 4 level 3 (unsolved) | -34 ± 24 | -83 | 50% | 216 | -0.10 | 100% | 0 |
| C5 rep 5 level 3 (unsolved) | -34 ± 24 | -83 | 50% | 216 | -0.10 | 100% | 0 |
| C6 rep 1 level 3 (unsolved) | +9 ± 7 | -4 | 50% | 2 | +0.27 | 100% | 0 |
| C6 rep 2 level 3 (unsolved) | -1 ± 12 | -25 | 43% | 3 | -0.08 | 100% | 0 |
| C6 rep 3 level 3 (unsolved) | +9 ± 7 | -4 | 50% | 2 | +0.27 | 100% | 0 |
| C6 rep 4 level 3 (unsolved) | -1 ± 12 | -25 | 43% | 3 | -0.08 | 100% | 0 |
| C6 rep 5 level 3 (unsolved) | -1 ± 12 | -25 | 43% | 3 | -0.08 | 100% | 0 |
| C6r rep 1 level 1 (unsolved) | -1 ± 12 | -25 | 43% | 3 | -0.08 | 100% | 0 |
| C6r rep 1 level 2 (unsolved) | +1 ± 3 | -4 | 36% | 8 | +0.05 | 100% | 0 |
| C6r rep 2 level 1 (solved) | -34 ± 24 | -83 | 50% | 216 | -0.10 | 100% | 0 |
| C6r rep 2 level 2 (unsolved) | +1 ± 3 | -4 | 36% | 8 | +0.05 | 100% | 0 |
| C7 rep 1 level 3 (solved) | -12 ± 29 | -71 | 57% | 69 | +0.26 | 100% | 0 |
| C7 rep 2 level 3 (unsolved) | -1 ± 12 | -25 | 43% | 3 | -0.08 | 100% | 0 |
| C7 rep 3 level 3 (unsolved) | +9 ± 7 | -4 | 50% | 2 | +0.27 | 100% | 0 |
| C7 rep 4 level 3 (unsolved) | -1 ± 12 | -25 | 43% | 3 | -0.08 | 100% | 0 |
| C7 rep 5 level 3 (unsolved) | -1 ± 12 | -25 | 43% | 3 | -0.08 | 100% | 0 |
| C7r rep 1 level 1 (solved) | -12 ± 29 | -71 | 57% | 69 | +0.26 | 100% | 0 |
| C7r rep 1 level 2 (unsolved) | +1 ± 3 | -4 | 36% | 8 | +0.05 | 100% | 0 |
| C7r rep 2 level 1 (solved) | -12 ± 29 | -71 | 57% | 69 | +0.26 | 100% | 0 |
| C7r rep 2 level 2 (solved) | -18 ± 5 | -29 | 21% | 73 | -0.17 | 55% | 0 |

### INTC, 2012-06-21, 14 episodes of 1500 s

| strategy | PnL per episode (ticks) | lcb | episodes won | fills | markout (10 s) | obligation | rule violations (side withdrawn) |
|---|---|---|---|---|---|---|---|
| baseline.py | +12 ± 9 | -6 | 64% | 126 | +0.17 | 100% | 0 |
| reference.py | +23 ± 5 | +13 | 100% | 47 | +0.47 | 99% | 0 |
| A rep 1 level 1 (unsolved) | +4 ± 4 | -4 | 7% | 0 | -0.04 | 96% | TAKES 757 |
| A rep 1 level 2 (unsolved) | +4 ± 4 | -4 | 7% | 0 | -0.04 | 96% | TAKES 757 |
| A rep 1 level 3 (unsolved) | +4 ± 4 | -4 | 7% | 0 | -0.04 | 96% | TAKES 757 |
| A rep 2 level 1 (unsolved) | +4 ± 4 | -4 | 7% | 0 | -0.04 | 96% | TAKES 757 |
| A rep 2 level 2 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41602 |
| A rep 2 level 3 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41602 |
| A rep 3 level 1 (unsolved) | +4 ± 4 | -4 | 7% | 0 | -0.04 | 96% | TAKES 757 |
| A rep 3 level 2 (unsolved) | +4 ± 4 | -4 | 7% | 0 | -0.04 | 96% | TAKES 757 |
| A rep 3 level 3 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41602 |
| B rep 1 level 1 (unsolved) | +5 ± 3 | -2 | 50% | 4 | -0.13 | 1% | TAKES 20801 |
| B rep 1 level 2 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41602 |
| B rep 1 level 3 (unsolved) | -3 ± 3 | -9 | 0% | 0 | -0.25 | 97% | TAKES 597 |
| B rep 2 level 1 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41602 |
| B rep 2 level 2 (unsolved) | +5 ± 3 | -2 | 50% | 4 | -0.13 | 1% | TAKES 20801 |
| B rep 2 level 3 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41602 |
| B rep 3 level 1 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41602 |
| B rep 3 level 2 (unsolved) | +5 ± 3 | -2 | 50% | 4 | -0.13 | 1% | TAKES 20801 |
| B rep 3 level 3 (unsolved) | -3 ± 3 | -9 | 0% | 0 | -0.25 | 97% | TAKES 597 |
| C rep 1 level 1 (solved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C rep 1 level 2 (solved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C rep 1 level 3 (unsolved) | +1 ± 1 | -1 | 7% | 1 | -0.18 | 100% | 0 |
| C rep 2 level 1 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41602 |
| C rep 2 level 2 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41602 |
| C rep 2 level 3 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41602 |
| C rep 3 level 1 (solved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C rep 3 level 2 (solved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C rep 3 level 3 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41602 |
| C2 rep 1 level 1 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41602 |
| C2 rep 1 level 2 (solved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C2 rep 1 level 3 (unsolved) | +0 ± 0 | +0 | 0% | 0 | +0.00 | 1% | OFF_TICK 41602 |
| C2 rep 2 level 1 (solved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C2 rep 2 level 2 (solved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C2 rep 2 level 3 (unsolved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C2 rep 3 level 1 (solved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C2 rep 3 level 2 (solved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C2 rep 3 level 3 (unsolved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C3 rep 1 level 1 (solved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C3 rep 1 level 2 (solved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C3 rep 1 level 3 (unsolved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C3 rep 2 level 1 (solved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C3 rep 2 level 2 (solved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C3 rep 2 level 3 (solved) | -7 ± 6 | -19 | 50% | 110 | +0.08 | 100% | INVENTORY_LIMIT 99 |
| C3 rep 3 level 1 (solved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C3 rep 3 level 2 (unsolved) | -18 ± 12 | -42 | 36% | 185 | +0.07 | 100% | 0 |
| C3 rep 3 level 3 (unsolved) | +1 ± 1 | -1 | 7% | 1 | -0.18 | 100% | 0 |
| C4 rep 1 level 1 (solved) | -7 ± 6 | -19 | 43% | 111 | +0.07 | 100% | 0 |
| C4 rep 1 level 2 (solved) | -7 ± 6 | -19 | 43% | 111 | +0.07 | 100% | 0 |
| C4 rep 1 level 3 (unsolved) | -18 ± 12 | -42 | 36% | 185 | +0.07 | 100% | 0 |
| C4 rep 2 level 1 (solved) | -18 ± 12 | -42 | 36% | 185 | +0.07 | 100% | 0 |
| C4 rep 2 level 2 (unsolved) | -18 ± 12 | -42 | 36% | 185 | +0.07 | 100% | 0 |
| C4 rep 2 level 3 (unsolved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C4 rep 3 level 1 (solved) | -6 ± 6 | -17 | 50% | 109 | +0.08 | 100% | 0 |
| C4 rep 3 level 2 (solved) | -6 ± 6 | -17 | 50% | 109 | +0.08 | 100% | 0 |
| C4 rep 3 level 3 (unsolved) | +1 ± 1 | -1 | 7% | 1 | -0.18 | 100% | 0 |
| C4b rep 1 level 3 (unsolved) | +5 ± 3 | -2 | 57% | 4 | +0.08 | 1% | CROSSED 9283, TAKES 11518 |
| C4b rep 2 level 3 (unsolved) | -18 ± 12 | -42 | 36% | 185 | +0.07 | 100% | 0 |
| C5 rep 1 level 3 (unsolved) | -18 ± 12 | -42 | 36% | 185 | +0.07 | 100% | 0 |
| C5 rep 2 level 3 (unsolved) | -18 ± 12 | -42 | 36% | 185 | +0.07 | 100% | 0 |
| C5 rep 3 level 3 (unsolved) | -18 ± 12 | -42 | 36% | 185 | +0.07 | 100% | 0 |
| C5 rep 4 level 3 (unsolved) | -18 ± 12 | -42 | 36% | 185 | +0.07 | 100% | 0 |
| C5 rep 5 level 3 (unsolved) | -18 ± 12 | -42 | 36% | 185 | +0.07 | 100% | 0 |
| C6 rep 1 level 3 (unsolved) | +9 ± 7 | -6 | 29% | 1 | +0.34 | 100% | 0 |
| C6 rep 2 level 3 (unsolved) | +8 ± 8 | -7 | 21% | 2 | +0.52 | 100% | 0 |
| C6 rep 3 level 3 (unsolved) | +9 ± 7 | -6 | 29% | 1 | +0.34 | 100% | 0 |
| C6 rep 4 level 3 (unsolved) | +8 ± 8 | -7 | 21% | 2 | +0.52 | 100% | 0 |
| C6 rep 5 level 3 (unsolved) | +8 ± 8 | -7 | 21% | 2 | +0.52 | 100% | 0 |
| C6r rep 1 level 1 (unsolved) | +8 ± 8 | -7 | 21% | 2 | +0.52 | 100% | 0 |
| C6r rep 1 level 2 (unsolved) | +4 ± 2 | +1 | 43% | 5 | +0.48 | 100% | 0 |
| C6r rep 2 level 1 (solved) | -18 ± 12 | -42 | 36% | 185 | +0.07 | 100% | 0 |
| C6r rep 2 level 2 (unsolved) | +4 ± 2 | +1 | 43% | 5 | +0.48 | 100% | 0 |
| C7 rep 1 level 3 (solved) | +16 ± 20 | -24 | 50% | 58 | +0.37 | 100% | 0 |
| C7 rep 2 level 3 (unsolved) | +8 ± 8 | -7 | 21% | 2 | +0.52 | 100% | 0 |
| C7 rep 3 level 3 (unsolved) | +9 ± 7 | -6 | 29% | 1 | +0.34 | 100% | 0 |
| C7 rep 4 level 3 (unsolved) | +8 ± 8 | -7 | 21% | 2 | +0.52 | 100% | 0 |
| C7 rep 5 level 3 (unsolved) | +8 ± 8 | -7 | 21% | 2 | +0.52 | 100% | 0 |
| C7r rep 1 level 1 (solved) | +16 ± 20 | -24 | 50% | 58 | +0.37 | 100% | 0 |
| C7r rep 1 level 2 (unsolved) | +4 ± 2 | +1 | 43% | 5 | +0.48 | 100% | 0 |
| C7r rep 2 level 1 (solved) | +16 ± 20 | -24 | 50% | 58 | +0.37 | 100% | 0 |
| C7r rep 2 level 2 (solved) | -9 ± 5 | -20 | 43% | 57 | -0.01 | 58% | 0 |
