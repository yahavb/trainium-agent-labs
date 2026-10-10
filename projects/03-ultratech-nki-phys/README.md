# Team Ultratech: NKI-Phys

**Team 52 | Seats 260-263 | Vedant, Devesh, Dev, Khushboo**

A bounded Qwen3-8B agent proposes NKI implementations of two physics equations, receives independent correctness and device-throughput feedback, and retains the fastest correct candidate. This is inference-time agent optimization, not model-weight training.

## Required Submission Artifacts

1. **Checker and reasoning:** [checker.py](checker.py), [CHECKER.md](CHECKER.md), and the actual equation/error gates in [source/math_tasks.py](source/math_tasks.py).
2. **Every attempt and score:** [ATTEMPTS.jsonl](ATTEMPTS.jsonl). The complete four-proposal math experiment is in [development/](development/), including model requests/replies, wrong-sign feedback, generated graphs and sources, outputs and timings. A rejected duplicate is unmeasured, not a zero correctness score.
3. **One-page results note:** [RUN_NOTE.md](RUN_NOTE.md), with hardware, run counts, timing spread and independent repeat results.

Read [submission.md](submission.md) for the full definition, real input examples, embedded flowcharts, successful results and failures. The broader physics-engine PDF is a clearly labeled teammate-reported appendix, not independently reproduced agent evidence.

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

The agent selects from a finite arithmetic/reduction graph language. We do not claim unrestricted NKI synthesis, superiority to a scripted optimizer, statistical significance, global optimality or full-simulator acceleration. Earlier contact/gripping work and unsuccessful proposals are documented in `source/AWS_PROGRESS.md` and `submission.md`; their full raw histories are not the four-attempt log submitted here.

`SHA256SUMS.json` covers the portable submission files, excluding itself and generated caches. Original source and frozen input/output bytes are preserved. Paths/hostnames in historical logs describe the original environment. No credentials or compiled NEFFs are included.
