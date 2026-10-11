# CALIBRATION — confidence vs held-out truth

Every kernel the agent claimed SOLVED (public reward 1.0), re-run on held-out hostile cases it never saw (`projects/02-kernel-agent/heldout.py`). Confidence is computed from the code and the public result only. All **[sim]**.

**Brier score 0.030** over n=38 claims (0 = perfect, 0.25 = always saying 0.5).

| level | source | confidence | held-out | why that confidence |
|---|---|---|---|---|
| L1 | c10_L1_r0 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L1 | c10_L1_r3 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L1 | c2_L1_r1 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L1 | c9_L1_r1 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L1 | c9_L1_r2 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L2 | c10_L2_r0 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L2 | c9_L2_r0 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L3 | c10_L3_r0 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L4 | c10_L4_r0 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L5 | c10_L5_r0 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L5 | c10_L5_r1 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L5 | c10_L5_r4 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L5 | c11_L5_r0 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L5 | c3L5_L5_r0 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L5 | c3L5_L5_r1 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L5 | c5_L5_r0 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L5 | c5_L5_r1 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L5 | c5_L5_r2 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L5 | c7_L5_r2 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L5 | c9_L5_r1 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L5 | c9_L5_r4 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L6 | c10_L6_r0 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L6 | c10_L6_r2 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L6 | c10_L6_r3 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L6 | c11_L6_r4 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L6 | c8a_L6_r0 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L6 | c8b_L6_r0 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L6 | c8b_L6_r3 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L6 | c9_L6_r0 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L7 | c10_L7_r0 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L7 | c5_L7_r0 | 0.95 | PASS | all public shapes pass; remainder handling present |
| L1 | reference_level1.py | 0.67 | PASS | hard-coded public dimensions [3, 4] |
| L2 | reference_level2.py | 0.95 | PASS | all public shapes pass; remainder handling present |
| L3 | reference_level3.py | 0.38 | PASS | asserts on input shape (refuses shapes outside the public set) |
| L4 | reference_level4.py | 0.38 | FAIL — ragged in K, M and N: raised AssertionError: Expected M, 200, to be a multiple of stationary free-dimension max, 128; M=1, N=512+1: raised AssertionError: Expected M, 1, to be a multiple of stationary | asserts on input shape (refuses shapes outside the public set) |
| L5 | reference_level5.py | 0.38 | FAIL — ragged in K, M and N: raised AssertionError: Expected M, 200, to be a multiple of stationary free-dimension max, 128; M=1, N=512+1: raised AssertionError: Expected M, 1, to be a multiple of stationary | asserts on input shape (refuses shapes outside the public set) |
| L6 | reference_level6.py | 0.38 | FAIL — ragged in K, M and N: raised AssertionError: ; M=1, N=512+1: raised ZeroDivisionError: integer modulo by zero; prime K > 128: NUMERICAL MISMATCH: worst error 1.06 of the output's RMS (11.38), toleranc | asserts on input shape (refuses shapes outside the public set) |
| L7 | reference_level7.py | 0.38 | FAIL — ragged in K, M and N: raised AssertionError: Expected M 200 to be divisible by 128 when there are 1; M=1, N=512+1: raised ZeroDivisionError: integer modulo by zero; prime K > 128: raised AssertionErro | asserts on input shape (refuses shapes outside the public set) |

| confidence bin | n | predicted | observed pass rate |
|---|---|---|---|
| [0.0, 0.5) | 5 | 0.38 | 0.20 |
| [0.5, 0.8) | 1 | 0.67 | 1.00 |
| [0.8, 1.0) | 32 | 0.95 | 1.00 |
