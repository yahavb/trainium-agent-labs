# Failure taxonomy

40 attempts in 2 level-runs from 2 log file(s); 40 failed. Each attempt is classified by the checker's error, first match wins (`scripts/taxonomy.py`).

## Solve rate per level

| level | solved runs |
|---|---|
| 7 | 0/2 |

## Failure modes

*runs* = level-runs that hit the mode at least once. *stuck* = when the round's carried attempt had this mode, how often the next round's did too.

| family | mode | what it means | L7 | total | % of failures | runs | stuck | example |
|---|---|---|---|---|---|---|---|---|
| silent wrong numbers | `numeric_mismatch` | runs, numbers wrong | 36 | 36 | 90% | 2 | 80% | 2 of 4 shapes passed. On K=256 M=256 N=1024: NUMERICAL MISMATCH: worst error 7.71 of the output's RMS (15.96), |
| silent wrong numbers | `too_much_traffic` | correct but over the level's HBM byte bar | 4 | 4 | 10% | 1 | 100% | 2 of 4 shapes passed. On K=256 M=256 N=1024: CORRECT, BUT TOO MUCH HBM TRAFFIC FOR THIS LEVEL: moving 1.56x th |

By family: **silent wrong numbers** 40 (100%)

## Fixed one thing, broke another

Round-to-round changes in the carried attempt's failure mode (not counting a solve).

| was | became | times |
|---|---|---|
| `numeric_mismatch` | `too_much_traffic` | 1 |

## Where unsolved runs ended

| level | last failure | runs |
|---|---|---|
| 7 | `numeric_mismatch` | 1 |
| 7 | `too_much_traffic` | 1 |

## Held-out check

After the loop, each level's best kernel ran once on shapes and values it never saw (`nkibench.py --eval`). Confidence was stated before that check.

| level | verdict | runs |
|---|---|---|
| 7 | NOT SOLVED | 2 |

Calibration over 2 verdicts: Brier score 0.000; confident (>= 0.5) but wrong 0 time(s).

