# One-Page Run Note

Team Ultratech (52), seats 260-263: Vedant, Devesh, Dev, Khushboo.

Objective: a bounded, checker-guided Qwen agent optimizes distinct Trainium physics calculations. Qwen chooses operation graphs from equations, source and measured feedback; trusted lowering emits NKI. No weight training or unrestricted compiler discovery.

Hardware: seat-260 on Trainium2 (trn2.48xlarge), confirmed visible cores 0,1, Python 3.13.7, installed NKI 0.6.0. Qwen/Qwen3-8B is stopped during device timing.

Development: four proposals across spring F=-(k*x+c*v) and net-force sum(F_i). Attempt 0 spring had a double-negation error (0/16); attempt 1 net-force passed; attempt 2 spring corrected its sign after feedback and passed; attempt 3 repeated the net-force graph and was rejected as a duplicate without another benchmark.

| Task | Development Ratio | Independent Repeat | Final Holdout |
| --- | ---: | ---: | ---: |
| spring | 1.039810x | 1.038735x | 32/32 |
| net-force | 1.070428x | 1.075468x | 32/32 |

Timing: five repeats, 20 warmups and 200 device samples per repeat, on development seed 3. Original baseline is measured before and after the candidate; >10% drift invalidates the reward. Both kernels are checked on 16 development cases and before/after timing. Device timing excludes compilation, loading, transfers and readback.

spring development candidate repeat means (min/median/max ms): 0.01555320/0.01555898/0.01556944.
spring independent repeat means (min/median/max ms): 0.01545699/0.01546133/0.01546909.
net-force development candidate repeat means (min/median/max ms): 0.01731500/0.01736040/0.01738581.
net-force independent repeat means (min/median/max ms): 0.01734201/0.01734520/0.01736115.

Final correctness: source/input/checker hashes were frozen before one invocation per held-out case; 32 unique unseen inputs per task passed (64/64), unchanged gates, no Qwen feedback. This tests new values in the same equations, distributions and shapes, not new algorithms.

Checker accepts finite FP32 outputs with exact shape, unchanged inputs and error <= 1e-6 + 2e-6 times the sum of absolute contributing terms, against FP64 equations on saved inputs. It rejects wrong signs/values, altered inputs, invalid shapes/dtypes and nonfinite outputs. Compilation errors and duplicates are unmeasured, not correctness scores.

Evidence: complete four-attempt generation history, requests/replies/hypotheses, lowered sources, input/output snapshots, pre/post checks and raw timing samples are included. Compiled NEFFs are omitted; reproduce with the recorded sources and installed SDK. No statistical-significance, global-optimality or full-simulator claim. Independent benchmark replication is recorded above.

Earlier gripping: six FP32 momentum settings at 1024 updates passed 10,10,10,6,6,4 of 16; no accepted gripping throughput. Time limits prompted the simpler task, without relaxed gates. Separate log and per-case reports are included. Parallel engine work covers integration, spring chains, floor contact and fused rollouts; its PDF reports 51x simulated-traffic reduction and 52x host-to-host speedup, not independently reproduced device-only gains. The broad-track Qwen search did not fully solve those levels.
