# Team Ultratech: NKI-Phys

**Team 52 | Seats 260-263 | Vedant, Devesh, Dev, Khushboo**

A bounded Qwen3-8B agent proposes NKI implementations of two physics equations, receives independent correctness and device-throughput feedback, and retains the fastest correct candidate. This is inference-time agent optimization, not model-weight training.

![Agent workflow: Qwen proposals, trusted NKI lowering, correctness checks, device benchmarking and feedback](submission-assets/agent-flow.png)

## Required Submission Artifacts

1. **Checker and reasoning:** [checker.py](checker.py), [CHECKER.md](CHECKER.md), and the actual equation/error gates in [source/math_tasks.py](source/math_tasks.py).
2. **Every attempt and score:** [All available local attempt records](ALL_ATTEMPTS.jsonl), with original logs and saved grading/performance reports in [history/](history/) and [coverage notes](HISTORY.md). This includes earlier contact, gripping, failed/rejected generations, adaptive solvers and plane-agent runs, not just the final four proposals. For concise experiment views, see the [four-proposal math log](ATTEMPTS.jsonl) and [six-trial gripping log](gripping-evidence/ATTEMPTS.jsonl). Full final math evidence is in [development/](development/), including requests/replies, feedback, graphs, sources, outputs and timings. Unmeasured attempts retain `null`; overlapping generation/execution/case records are not counted as unique proposals. Remote-only and broader-engine raw logs not available locally are explicitly identified as a coverage gap.
3. **One-page results note:** [RUN_NOTE.md](RUN_NOTE.md), with hardware, run counts, timing spread and independent repeat results.

Read [submission.md](submission.md) for the full definition, real input examples, embedded flowcharts, successful results and failures. Our broader physics-engine PDF is a clearly labeled prototype-results appendix, not independently reproduced agent evidence.

### The Complete Agent Loop At A Glance

| Attempt | Task | Correctness | Throughput ratio | Outcome |
| --- | --- | ---: | ---: | --- |
| 0 | Spring | 0/16 | Not timed | Wrong force sign; feedback sent for revision |
| 1 | Net-force | 16/16 | 1.070428x | Correct native-reduction proposal |
| 2 | Spring | 16/16 | 1.039810x | Sign corrected after feedback |
| 3 | Net-force | Unmeasured | Unmeasured | Duplicate executable graph rejected |

For a quick review, read this table, `CHECKER.md` and `RUN_NOTE.md`. The nested raw records are included for auditing, not required reading.

## Physics Questions And Results

- **Spring-damper force:** `F = -(k*x + c*v)`. Saved example: `x=2.0409190655`, `v=0.2716225684`, `k=84.6561889648`, `c=8.9611492157`; reference force `-175.2104804343`. Input shapes: x/v 64x8, k/c 64x1; output 64x8.
- **Net force:** `F_net[p,0] = sum(forces[p,:])`. Eight signed contributions per row, 64 rows; output 64x1. Saved row-0 example sums to `-3.5845816880`.

| Task | Initial throughput ratio | Independent repeat | Frozen unseen checks |
| --- | ---: | ---: | ---: |
| Spring-damper | 1.039810x | 1.038735x | 32/32 |
| Net-force | 1.070428x | 1.075468x | 32/32 |

Ratios compare device-only candidate throughput to paired original-baseline throughput. Five repeats, 20 warmups and 200 timing samples per repeat; Qwen stopped during timing. The final 64 cases test unseen values in the same equations/shapes/distributions, not unseen algorithms. Development has 16 public cases per task. All input values are synthetic.

## Gripping: Failed Cases And The Time-Limit Pivot

We first targeted frictional gripping contact-force solves from MuJoCo-derived snapshots, not just the two simple force equations. Six fixed-momentum trials at 1024 solver updates passed **10, 10, 10, 6, 6 and 4 out of 16** cases. None reached full physics acceptance, so none earned an accepted gripping-throughput result. For example, `grip-000` had a small optimization residual but acceleration error **0.000456848**, above its fixed **0.0001** gate. A small objective error alone was not enough to preserve the physical output.

Because of the hackathon time limit, we narrowed the agent experiment to spring-damper and net-force primitives rather than relaxing the gripping checker. Read [GRIPPING_FAILURES.md](GRIPPING_FAILURES.md) for the six scores, concrete failures and precision caveats. The actual six-attempt log and per-case diagnostic reports are in [gripping-evidence/](gripping-evidence/); this separate record must not be confused with the final four-proposal math-agent log.

## Broader Development: Physics-Engine Prototype

In parallel, we developed a simplified engine covering semi-implicit integration, coupled spring-chain forces, floor-contact projection and fused rollouts. Our engine report describes **four reference certifications, four accepted reference kernels and 12 diagnosed planted bugs**, plus a 128-world, 16-mass validation scenario. It reports approximately **51x less simulation-counted DMA traffic** and **52x host-to-host rollout speedup** from reducing launches.

These are **prototype results reported in our engine document**, with a different timing boundary from our device-only agent benchmarks; they are not added to our 3.87%/7.55% throughput gains. Qwen did not fully solve the broader physics levels in that reported search. The prototype is the intended integration direction, not proof of a fully agent-generated engine. Read [BROADER_DEVELOPMENT.md](BROADER_DEVELOPMENT.md) and our [full engine report](submission-assets/teammate-engine-report.pdf). Its underlying engine source and full logs were not available in this package.

## Verify Without AWS Or A Model

From this folder, install NumPy in your preferred environment, then run:

```bash
python -m pip install -r requirements.txt
python checker.py --replay
python -m unittest -v test_checker.py
```

Replay verifies every packaged file hash, checks the attempt-log copy, independently regrades saved public input/output snapshots, and regrades the 64 frozen final outputs using the original pinned evaluator. It runs no generated Python, model requests, accelerator jobs or new benchmarks. It does not prove timing authenticity; hardware measurements and their raw samples remain separate evidence. Public snapshots preserve original inputs; execution-time mutation checks are in their saved reports. Frozen final snapshots additionally contain input readback.

Historical source tests can also be run with `python -m unittest discover -s source -v`. They additionally require SciPy and MuJoCo; one test skips when the separately generated certified gripping suite is absent. The included old gripping candidate is only a test fixture, not a passing submission result.

To grade a new standalone snapshot containing `actual` and the task's named input arrays:

```bash
python checker.py --task spring --snapshot path/to/output.npz
```

This standalone mode checks the equation against those supplied inputs. It cannot attest that a kernel left inputs unchanged without independently captured before/after arrays; use `source/math_harness.py` for execution-time checks.

## Agent And Hardware Reproduction

[source/math_agent_loop.py](source/math_agent_loop.py) is the local controller; [source/qwen_math.py](source/qwen_math.py) generates bounded operation graphs; [source/math_program.py](source/math_program.py) validates and lowers them; [source/math_harness.py](source/math_harness.py) grades and benchmarks. Historical controllers use `/workspace/projects/03-contact-physics` on the seat; this path is intentionally retained to preserve recorded sources and hashes. They require `kubectl`, authorized access to your own seat, the workshop's installed NKI 0.6.0/nrtpy runtime and `/workspace/serve.sh`. Do not use another participant's seat.

```bash
python source/math_agent_loop.py --seat seat-260 --cores 0,1 --minutes 30 --attempts 4
```

Only run this on authorized, exclusively available cores. Fresh experiments write under `source/data/`; they do not replace the frozen submission evidence. The assessment lock prevents rerunning the original final evaluation as an optimization loop. Regrading saved outputs is safe and is not a new device evaluation.

## Scope And Provenance

The agent selects from a finite arithmetic/reduction graph language. We do not claim unrestricted NKI synthesis, superiority to a scripted optimizer, statistical significance, global optimality or full-simulator acceleration. Earlier contact/gripping work and unsuccessful proposals are documented in `source/AWS_PROGRESS.md` and `submission.md`; their available local logs and grading/performance reports are collected separately in `history/`, with coverage limits in `HISTORY.md`.

`SHA256SUMS.json` covers the portable submission files, excluding itself and generated caches. Original source and frozen input/output bytes are preserved. Paths/hostnames in historical logs describe the original environment. No credentials or compiled NEFFs are included.
