# P3 headline: Qwen3-8B made its own matmul kernel 1.52x faster on Trainium, and the referee proved it

*For the final report, the slides and the demo. Every number here is **chip**: seat 101's Trainium2, judged by
P1's `speedcheck.py` (`referee-timing` 7da33ee). Details and the full history are in `P3_STATUS.md`.*

## The result

**Qwen3-8B, guided only by the referee's one-sentence instructions, rewrote the matmul kernel it runs on
into a version that is 1.52x faster on the real chip.** The referee accepted it as `faster` only after it
passed every check, and an independent second check reproduced the result exactly.

| | Start kernel | Qwen's kernel | Speedup |
|---|---|---|---|
| **Total, both Qwen3 shapes** | **960.9 us** | **633.2 us** | **1.517x** |
| q_proj (4096, 256, 2048) | 269.0 us | 183.8 us | 1.463x |
| gate/up (4096, 256, 6144) | 691.9 us | 449.4 us | 1.540x |
| Throughput | | 27.1 TFLOP/s | |

**How we know it is real:** the referee says `faster` only when all of these hold.
- **Rules:** the static scan of the kernel file is clean.
- **Simulator:** correct on 2 shapes, with the inputs untouched.
- **Chip correctness:** correct at Qwen3's real sizes, with hostile inputs as well as normal ones. Worst error
  0.50 bf16 ulps, against a limit of 4.
- **Timing:** interleaved A/B against the start kernel, 1.517x against a noise band of 0.990 to 1.010x, and
  no shape slower.
- **Held-out:** correct on 3 fresh shapes the kernel never saw.
- **Replication:** a standalone `speedcheck.py --check` on the saved kernel gave `VERDICT: FASTER`, 1.517x.

This is the same referee that caught **10 out of 10** planted cheats on the chip, and that rejected every
broken or noise-level kernel all day.

**Where it is:** run `matmul-referee-v5-212632-0`, attempt 2, in `logs/seat-101/attempts_referee_v5.jsonl`.
Attempt 3 was also verified `faster` at 1.517x. The kernel source is in the record's `code` field.

## The story: how the loop got there (all chip, seat 101)

The project's thesis is that the checker decides whether the loop works. The day showed it twice over: the
referee kept every broken kernel out, and once its feedback became precise, the model improved.

| Version | What the model was told | Outcome |
|---|---|---|
| v1 | the referee's original instruction | 0 faster in 24. The model kept repeating the same mistake. |
| v2 | + P3 Rule A: the crash named exactly | the model applied the named fix, then hit a misleading hint |
| v3 | P1's improved, self-contained feedback | the right restructure, one axis order wrong |
| v4 | + P3 Rules B and C | **two bugs fixed in a row**; one load from correct |
| **v5** | **P1's feedback + P3 Rules A-D** | **`faster` 1.517x at attempt 2, replicated** |

- **What made the difference:** every time an instruction named the exact change, the model applied it.
  P1's referee named the NKI tile layout (`(TILE_K, K // TILE_K, TILE_N)`, indexed `[:, k, :]`), and the next
  attempt was the 1.52x kernel. P3's Rules A-D cover the four mistakes the model kept repeating, each in
  one safe sentence (the model's code is parsed only, never run).
- **The team's contribution:** P1 built the referee and its precise feedback. P3 built the loop, the red
  team, the failure taxonomy and the rules for repeated mistakes. P2 provided the shapes and the expert
  ceiling. P4 built the dashboard.

## One-liners for the report

- "Qwen3-8B made the matmul kernel it runs on **1.52x faster on Trainium2** (960.9 to 633.2 us). The result
  was verified on the chip, on held-out shapes, and in an independent re-run."
- "The referee that accepted it caught **10/10 planted cheats** and rejected every broken attempt."
- "The turning point was feedback that **names the exact change**. With generic error reports the model
  repeated itself; with precise instructions it fixed one bug after another until it beat the baseline."

## Context to keep it honest (one line each in the report)

- **One verified run** (v5, 1 run x 8 attempts). It was reproduced by an independent check, but it is not a
  rate over many runs.
- **The gain is not fewer bytes:** the simulator counts the same HBM traffic as the start kernel. The
  speedup comes from how the work is scheduled on the chip, which is consistent with P1's measurement that
  extra loads are nearly free there.
- **AWS's expert kernel reaches 2.49x** (P2), and tuning its block sizes reaches about 3.4x (P2's random
  search). Qwen's 1.52x comes from the plain start kernel, without that design: a separate claim.
