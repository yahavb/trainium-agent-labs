# Task 15: level 1 with the partition and dst-shape gate rules (4090 stand-in, not Trainium)

l1fix (`daykit_l1fix`) and l1base (`daykit`) ran with `GATE=static` on levels 1 and 2, 10 runs each, started together at 14:32:37 EDT on the running vLLM server. They ended at 15:01 and 14:59; `seat.sh check` passed all 11 checks. Logs pushed at 15:02 (bb8363d). Analysis: `analyze15.py` → `analysis15.txt`. Solves are **simulator** solves; the seat full-builds them.

| level | l1fix solved (round) | l1base solved (round) | p |
|---|---|---|---|
| 1 | **4/10** (1, 1, 1, 1) | 1/10 (2) | 0.30 |
| 2 | 6/10 (0, 1, 1, 1, 1, 4) | 8/10 (0, 0, 0, 0, 0, 1, 1, 1) | 0.63 |

- **The partition rule fired on level 1 in 4 runs, each on round 0, and all 4 solved on round 1.**
  - Each held-back kernel reduced one channel per iteration of a channel loop (`tile[c, ...]`), and the message gave the loop rewritten as code.
  - All 16 next-round samples contained every suggested line, ignoring spaces, and all 16 scored 1.0. Literal paste was 2 of 4 runs; the other 2 kept the model's own spacing (`h*p`).
- **On level 2 it fired once, with no code,** on round 2. The next round scored 0.30–0.50, and the run solved on round 4.
- **The dst-shape rule and the old gate forms never fired, and no run ended held back at 0.95,** in either condition.
- **What the seat should see.**
  - l1base's 2 distinct level-1 solves (`d2c82452`, `07c4c380`) are flagged by the partition rule.
  - l1fix's 4 (`7a2ef959`, `8f3f8ac7`, `bb8217cc`, `911e2ad4`) are flagged by neither new rule.
  - None of the 14 level-2 solves is flagged.
- **Level 2's 6 against 8 isn't the rule.** The 4 unsolved l1fix runs never reached 1.0, so the gate never acted on them.
- **Cost per run, l1fix / l1base:** level 1 29k / 35k tokens, 1.9 / 2.2 min, 5.6 / 7.1 rounds; level 2 18k / 9k tokens, 0.9 / 0.5 min, 3.7 / 1.9 rounds.
- **The level-1 wall is now before full simulator correctness, where the gate can't act.** All 15 unsolved runs topped out at 0.50.
  - In 10 of them, the best kernel came on round 0 and raised `nisa.tensor_scalar(dst=m, data=s, ...)`: "input operand has more dimensions than allowed by the axis remapping". Later repairs never beat it.
  - Idea, not tried: give that error code too, as the partition rule does.
