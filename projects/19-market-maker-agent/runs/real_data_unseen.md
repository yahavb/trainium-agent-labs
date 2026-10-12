# Real-data replay (sanity check, not part of the claim)

### GOOG, 2012-06-21, 14 episodes of 1500 s

| strategy | PnL per episode (ticks) | lcb | episodes won | fills | markout (10 s) | obligation | rule violations (side withdrawn) |
|---|---|---|---|---|---|---|---|
| baseline.py | +240 ± 216 | -192 | 79% | 109 | +3.81 | 100% | 0 |
| reference.py | -362 ± 204 | -769 | 43% | 37 | -6.27 | 100% | 0 |
| A rep 1 level 1 (unsolved) | +108 ± 99 | -89 | 57% | 174 | +1.15 | 100% | 0 |
| A rep 1 level 2 (unsolved) | +108 ± 99 | -89 | 57% | 174 | +1.15 | 100% | 0 |
| A rep 1 level 3 (unsolved) | +108 ± 99 | -89 | 57% | 174 | +1.15 | 100% | 0 |
| A rep 2 level 1 (unsolved) | +108 ± 99 | -89 | 57% | 174 | +1.15 | 100% | 0 |
| A rep 2 level 2 (unsolved) | -424 ± 164 | -751 | 21% | 8 | -12.27 | 51% | OFF_TICK 20710 |
| A rep 2 level 3 (unsolved) | -424 ± 164 | -751 | 21% | 8 | -12.27 | 51% | OFF_TICK 20710 |
| A rep 3 level 1 (unsolved) | +108 ± 99 | -89 | 57% | 174 | +1.15 | 100% | 0 |
| A rep 3 level 2 (unsolved) | +108 ± 99 | -89 | 57% | 174 | +1.15 | 100% | 0 |
| A rep 3 level 3 (unsolved) | -424 ± 164 | -751 | 21% | 8 | -12.27 | 51% | OFF_TICK 20710 |
| B rep 1 level 1 (unsolved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| B rep 1 level 2 (unsolved) | -424 ± 164 | -751 | 21% | 8 | -12.27 | 51% | OFF_TICK 20710 |
| B rep 1 level 3 (unsolved) | +108 ± 99 | -89 | 57% | 174 | +1.15 | 100% | 0 |
| B rep 2 level 1 (unsolved) | -424 ± 164 | -751 | 21% | 8 | -12.27 | 51% | OFF_TICK 20710 |
| B rep 2 level 2 (unsolved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| B rep 2 level 3 (unsolved) | -424 ± 164 | -751 | 21% | 8 | -12.27 | 51% | OFF_TICK 20710 |
| B rep 3 level 1 (unsolved) | -424 ± 164 | -751 | 21% | 8 | -12.27 | 51% | OFF_TICK 20710 |
| B rep 3 level 2 (unsolved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| B rep 3 level 3 (unsolved) | +108 ± 99 | -89 | 57% | 174 | +1.15 | 100% | 0 |
| C rep 1 level 1 (solved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C rep 1 level 2 (solved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C rep 1 level 3 (unsolved) | -69 ± 115 | -298 | 50% | 10 | -10.41 | 100% | 0 |
| C rep 2 level 1 (unsolved) | -424 ± 164 | -751 | 21% | 8 | -12.27 | 51% | OFF_TICK 20710 |
| C rep 2 level 2 (unsolved) | -424 ± 164 | -751 | 21% | 8 | -12.27 | 51% | OFF_TICK 20710 |
| C rep 2 level 3 (unsolved) | -424 ± 164 | -751 | 21% | 8 | -12.27 | 51% | OFF_TICK 20710 |
| C rep 3 level 1 (solved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C rep 3 level 2 (solved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C rep 3 level 3 (unsolved) | -424 ± 164 | -751 | 21% | 8 | -12.27 | 51% | OFF_TICK 20710 |
| C2 rep 1 level 1 (unsolved) | -424 ± 164 | -751 | 21% | 8 | -12.27 | 51% | OFF_TICK 20710 |
| C2 rep 1 level 2 (solved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C2 rep 1 level 3 (unsolved) | -424 ± 164 | -751 | 21% | 8 | -12.27 | 51% | OFF_TICK 20710 |
| C2 rep 2 level 1 (solved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C2 rep 2 level 2 (solved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C2 rep 2 level 3 (unsolved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C2 rep 3 level 1 (solved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C2 rep 3 level 2 (solved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C2 rep 3 level 3 (unsolved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C3 rep 1 level 1 (solved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C3 rep 1 level 2 (solved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C3 rep 1 level 3 (unsolved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C3 rep 2 level 1 (solved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C3 rep 2 level 2 (solved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C3 rep 2 level 3 (solved) | +131 ± 64 | +2 | 79% | 114 | +1.16 | 99% | INVENTORY_LIMIT 236 |
| C3 rep 3 level 1 (solved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C3 rep 3 level 2 (unsolved) | +259 ± 163 | -67 | 79% | 175 | +1.94 | 100% | 0 |
| C3 rep 3 level 3 (unsolved) | -69 ± 115 | -298 | 50% | 10 | -10.41 | 100% | 0 |
| C4 rep 1 level 1 (solved) | +106 ± 99 | -91 | 57% | 174 | +1.14 | 100% | 0 |
| C4 rep 1 level 2 (solved) | +106 ± 99 | -91 | 57% | 174 | +1.14 | 100% | 0 |
| C4 rep 1 level 3 (unsolved) | +259 ± 163 | -67 | 79% | 175 | +1.94 | 100% | 0 |
| C4 rep 2 level 1 (solved) | +259 ± 163 | -67 | 79% | 175 | +1.94 | 100% | 0 |
| C4 rep 2 level 2 (unsolved) | +259 ± 163 | -67 | 79% | 175 | +1.94 | 100% | 0 |
| C4 rep 2 level 3 (unsolved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C4 rep 3 level 1 (solved) | +114 ± 66 | -17 | 71% | 113 | +1.01 | 100% | 0 |
| C4 rep 3 level 2 (solved) | +114 ± 66 | -17 | 71% | 113 | +1.01 | 100% | 0 |
| C4 rep 3 level 3 (unsolved) | -69 ± 115 | -298 | 50% | 10 | -10.41 | 100% | 0 |
| C4b rep 1 level 3 (unsolved) | +105 ± 99 | -92 | 57% | 174 | +1.13 | 100% | 0 |
| C4b rep 2 level 3 (unsolved) | +259 ± 163 | -67 | 79% | 175 | +1.94 | 100% | 0 |
| C5 rep 1 level 3 (unsolved) | +259 ± 163 | -67 | 79% | 175 | +1.94 | 100% | 0 |
| C5 rep 2 level 3 (unsolved) | +259 ± 163 | -67 | 79% | 175 | +1.94 | 100% | 0 |
| C5 rep 3 level 3 (unsolved) | +259 ± 163 | -67 | 79% | 175 | +1.94 | 100% | 0 |
| C5 rep 4 level 3 (unsolved) | +259 ± 163 | -67 | 79% | 175 | +1.94 | 100% | 0 |
| C5 rep 5 level 3 (unsolved) | +259 ± 163 | -67 | 79% | 175 | +1.94 | 100% | 0 |
| C6 rep 1 level 3 (unsolved) | -1600 ± 285 | -2171 | 0% | 215 | -6.35 | 100% | 0 |
| C6 rep 2 level 3 (unsolved) | -2694 ± 286 | -3266 | 0% | 184 | -14.24 | 100% | 0 |
| C6 rep 3 level 3 (unsolved) | -1600 ± 285 | -2171 | 0% | 215 | -6.35 | 100% | 0 |
| C6 rep 4 level 3 (unsolved) | -2694 ± 286 | -3266 | 0% | 184 | -14.24 | 100% | 0 |
| C6 rep 5 level 3 (unsolved) | -2694 ± 286 | -3266 | 0% | 184 | -14.24 | 100% | 0 |
| C6r rep 1 level 1 (unsolved) | -2694 ± 286 | -3266 | 0% | 184 | -14.24 | 100% | 0 |
| C6r rep 1 level 2 (unsolved) | -818 ± 143 | -1104 | 14% | 120 | -8.05 | 100% | 0 |
| C6r rep 2 level 1 (solved) | -3426 ± 380 | -4186 | 0% | 372 | -9.37 | 100% | 0 |
| C6r rep 2 level 2 (unsolved) | -818 ± 143 | -1104 | 14% | 120 | -8.05 | 100% | 0 |
| C7 rep 1 level 3 (solved) | -3680 ± 357 | -4394 | 0% | 368 | -10.37 | 100% | 0 |
| C7 rep 2 level 3 (unsolved) | -2694 ± 286 | -3266 | 0% | 184 | -14.24 | 100% | 0 |
| C7 rep 3 level 3 (unsolved) | -1600 ± 285 | -2171 | 0% | 215 | -6.35 | 100% | 0 |
| C7 rep 4 level 3 (unsolved) | -2694 ± 286 | -3266 | 0% | 184 | -14.24 | 100% | 0 |
| C7 rep 5 level 3 (unsolved) | -2694 ± 286 | -3266 | 0% | 184 | -14.24 | 100% | 0 |
| C7r rep 1 level 1 (solved) | -3680 ± 357 | -4394 | 0% | 368 | -10.37 | 100% | 0 |
| C7r rep 1 level 2 (unsolved) | -818 ± 143 | -1104 | 14% | 120 | -8.05 | 100% | 0 |
| C7r rep 2 level 1 (solved) | -3680 ± 357 | -4394 | 0% | 368 | -10.37 | 100% | 0 |
| C7r rep 2 level 2 (solved) | -3495 ± 315 | -4125 | 0% | 370 | -9.90 | 96% | 0 |

### AAPL, 2012-06-21, 14 episodes of 1500 s

| strategy | PnL per episode (ticks) | lcb | episodes won | fills | markout (10 s) | obligation | rule violations (side withdrawn) |
|---|---|---|---|---|---|---|---|
| baseline.py | +913 ± 280 | +354 | 86% | 321 | +2.68 | 100% | 0 |
| reference.py | -209 ± 187 | -583 | 50% | 38 | -4.73 | 100% | 0 |
| A rep 1 level 1 (unsolved) | +284 ± 188 | -92 | 79% | 565 | +0.35 | 100% | TAKES 11 |
| A rep 1 level 2 (unsolved) | +284 ± 188 | -92 | 79% | 565 | +0.35 | 100% | TAKES 11 |
| A rep 1 level 3 (unsolved) | +284 ± 188 | -92 | 79% | 565 | +0.35 | 100% | TAKES 11 |
| A rep 2 level 1 (unsolved) | +284 ± 188 | -92 | 79% | 565 | +0.35 | 100% | TAKES 11 |
| A rep 2 level 2 (unsolved) | -116 ± 192 | -499 | 64% | 12 | -6.05 | 50% | OFF_TICK 20794 |
| A rep 2 level 3 (unsolved) | -116 ± 192 | -499 | 64% | 12 | -6.05 | 50% | OFF_TICK 20794 |
| A rep 3 level 1 (unsolved) | +284 ± 188 | -92 | 79% | 565 | +0.35 | 100% | TAKES 11 |
| A rep 3 level 2 (unsolved) | +284 ± 188 | -92 | 79% | 565 | +0.35 | 100% | TAKES 11 |
| A rep 3 level 3 (unsolved) | -116 ± 192 | -499 | 64% | 12 | -6.05 | 50% | OFF_TICK 20794 |
| B rep 1 level 1 (unsolved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | TAKES 13 |
| B rep 1 level 2 (unsolved) | -116 ± 192 | -499 | 64% | 12 | -6.05 | 50% | OFF_TICK 20794 |
| B rep 1 level 3 (unsolved) | +284 ± 188 | -92 | 79% | 565 | +0.35 | 100% | TAKES 11 |
| B rep 2 level 1 (unsolved) | -116 ± 192 | -499 | 64% | 12 | -6.05 | 50% | OFF_TICK 20794 |
| B rep 2 level 2 (unsolved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | TAKES 13 |
| B rep 2 level 3 (unsolved) | -116 ± 192 | -499 | 64% | 12 | -6.05 | 50% | OFF_TICK 20794 |
| B rep 3 level 1 (unsolved) | -116 ± 192 | -499 | 64% | 12 | -6.05 | 50% | OFF_TICK 20794 |
| B rep 3 level 2 (unsolved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | TAKES 13 |
| B rep 3 level 3 (unsolved) | +284 ± 188 | -92 | 79% | 565 | +0.35 | 100% | TAKES 11 |
| C rep 1 level 1 (solved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C rep 1 level 2 (solved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C rep 1 level 3 (unsolved) | -321 ± 184 | -689 | 36% | 47 | -7.14 | 100% | 0 |
| C rep 2 level 1 (unsolved) | -116 ± 192 | -499 | 64% | 12 | -6.05 | 50% | OFF_TICK 20794 |
| C rep 2 level 2 (unsolved) | -116 ± 192 | -499 | 64% | 12 | -6.05 | 50% | OFF_TICK 20794 |
| C rep 2 level 3 (unsolved) | -116 ± 192 | -499 | 64% | 12 | -6.05 | 50% | OFF_TICK 20794 |
| C rep 3 level 1 (solved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C rep 3 level 2 (solved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C rep 3 level 3 (unsolved) | -116 ± 192 | -499 | 64% | 12 | -6.05 | 50% | OFF_TICK 20794 |
| C2 rep 1 level 1 (unsolved) | -116 ± 192 | -499 | 64% | 12 | -6.05 | 50% | OFF_TICK 20794 |
| C2 rep 1 level 2 (solved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C2 rep 1 level 3 (unsolved) | -116 ± 192 | -499 | 64% | 12 | -6.05 | 50% | OFF_TICK 20794 |
| C2 rep 2 level 1 (solved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C2 rep 2 level 2 (solved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C2 rep 2 level 3 (unsolved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C2 rep 3 level 1 (solved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C2 rep 3 level 2 (solved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C2 rep 3 level 3 (unsolved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C3 rep 1 level 1 (solved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C3 rep 1 level 2 (solved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C3 rep 1 level 3 (unsolved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C3 rep 2 level 1 (solved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C3 rep 2 level 2 (solved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C3 rep 2 level 3 (solved) | +202 ± 134 | -67 | 71% | 363 | +0.43 | 98% | INVENTORY_LIMIT 494 |
| C3 rep 3 level 1 (solved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C3 rep 3 level 2 (unsolved) | +1042 ± 196 | +650 | 100% | 549 | +1.01 | 100% | 0 |
| C3 rep 3 level 3 (unsolved) | -321 ± 184 | -689 | 36% | 47 | -7.14 | 100% | 0 |
| C4 rep 1 level 1 (solved) | +280 ± 188 | -96 | 79% | 566 | +0.33 | 100% | 0 |
| C4 rep 1 level 2 (solved) | +280 ± 188 | -96 | 79% | 566 | +0.33 | 100% | 0 |
| C4 rep 1 level 3 (unsolved) | +1042 ± 196 | +650 | 100% | 549 | +1.01 | 100% | 0 |
| C4 rep 2 level 1 (solved) | +1042 ± 196 | +650 | 100% | 549 | +1.01 | 100% | 0 |
| C4 rep 2 level 2 (unsolved) | +1042 ± 196 | +650 | 100% | 549 | +1.01 | 100% | 0 |
| C4 rep 2 level 3 (unsolved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C4 rep 3 level 1 (solved) | +260 ± 139 | -19 | 79% | 357 | +0.62 | 100% | 0 |
| C4 rep 3 level 2 (solved) | +260 ± 139 | -19 | 79% | 357 | +0.62 | 100% | 0 |
| C4 rep 3 level 3 (unsolved) | -321 ± 184 | -689 | 36% | 47 | -7.14 | 100% | 0 |
| C4b rep 1 level 3 (unsolved) | +281 ± 187 | -93 | 79% | 565 | +0.33 | 100% | CROSSED 2, TAKES 11 |
| C4b rep 2 level 3 (unsolved) | +1042 ± 196 | +650 | 100% | 549 | +1.01 | 100% | 0 |
| C5 rep 1 level 3 (unsolved) | +1042 ± 196 | +650 | 100% | 549 | +1.01 | 100% | 0 |
| C5 rep 2 level 3 (unsolved) | +1042 ± 196 | +650 | 100% | 549 | +1.01 | 100% | 0 |
| C5 rep 3 level 3 (unsolved) | +1042 ± 196 | +650 | 100% | 549 | +1.01 | 100% | 0 |
| C5 rep 4 level 3 (unsolved) | +1042 ± 196 | +650 | 100% | 549 | +1.01 | 100% | 0 |
| C5 rep 5 level 3 (unsolved) | +1042 ± 196 | +650 | 100% | 549 | +1.01 | 100% | 0 |
| C6 rep 1 level 3 (unsolved) | -1552 ± 248 | -2049 | 0% | 505 | -3.13 | 100% | 0 |
| C6 rep 2 level 3 (unsolved) | -3357 ± 305 | -3966 | 0% | 459 | -7.19 | 100% | 0 |
| C6 rep 3 level 3 (unsolved) | -1552 ± 248 | -2049 | 0% | 505 | -3.13 | 100% | 0 |
| C6 rep 4 level 3 (unsolved) | -3357 ± 305 | -3966 | 0% | 459 | -7.19 | 100% | 0 |
| C6 rep 5 level 3 (unsolved) | -3357 ± 305 | -3966 | 0% | 459 | -7.19 | 100% | 0 |
| C6r rep 1 level 1 (unsolved) | -3357 ± 305 | -3966 | 0% | 459 | -7.19 | 100% | 0 |
| C6r rep 1 level 2 (unsolved) | -1293 ± 227 | -1747 | 7% | 379 | -3.89 | 100% | 0 |
| C6r rep 2 level 1 (solved) | -4214 ± 331 | -4876 | 0% | 952 | -4.69 | 100% | 0 |
| C6r rep 2 level 2 (unsolved) | -1293 ± 227 | -1747 | 7% | 379 | -3.89 | 100% | 0 |
| C7 rep 1 level 3 (solved) | -4894 ± 381 | -5656 | 0% | 947 | -5.31 | 100% | 0 |
| C7 rep 2 level 3 (unsolved) | -3357 ± 305 | -3966 | 0% | 459 | -7.19 | 100% | 0 |
| C7 rep 3 level 3 (unsolved) | -1552 ± 248 | -2049 | 0% | 505 | -3.13 | 100% | 0 |
| C7 rep 4 level 3 (unsolved) | -3357 ± 305 | -3966 | 0% | 459 | -7.19 | 100% | 0 |
| C7 rep 5 level 3 (unsolved) | -3357 ± 305 | -3966 | 0% | 459 | -7.19 | 100% | 0 |
| C7r rep 1 level 1 (solved) | -4894 ± 381 | -5656 | 0% | 947 | -5.31 | 100% | 0 |
| C7r rep 1 level 2 (unsolved) | -1293 ± 227 | -1747 | 7% | 379 | -3.89 | 100% | 0 |
| C7r rep 2 level 1 (solved) | -4894 ± 381 | -5656 | 0% | 947 | -5.31 | 100% | 0 |
| C7r rep 2 level 2 (solved) | -4831 ± 364 | -5559 | 0% | 967 | -5.31 | 95% | 0 |
