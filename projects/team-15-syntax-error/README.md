# Team 15 — Syntax Error

Hack the Chip · NYU × Annapurna Labs · Project 2: the kernel agent.

We built a write → check → repair loop around Qwen3-8B. The model implements and revises NKI kernels; a programmed checker turns failures into specific repair instructions. Our team improved the feedback, templates, checker rules, and controller without fine-tuning the model.

## Start here

| Deliverable | Location |
|---|---|
| Agentic workflow and delegated tasks | [AGENTIC.md](AGENTIC.md) |
| Checker and acceptance rationale | [artifact/CHECKER.md](artifact/CHECKER.md) |
| Agent and NKI checker | [agent.py](artifact/projects/02-kernel-agent/agent.py), [nkibench.py](artifact/projects/02-kernel-agent/nkibench.py), [verdicts.py](artifact/projects/02-kernel-agent/verdicts.py) |
| Every scored attempt | [artifact/ATTEMPTS.md](artifact/ATTEMPTS.md), [attempts_all.csv](artifact/attempts_all.csv), [raw logs](artifact/runs/) |
| Experiment note and reproduction instructions | [artifact/NOTE.md](artifact/NOTE.md), [artifact/REPRODUCE.md](artifact/REPRODUCE.md) |
| Results, failures, and development decisions | [RESULTS.md](artifact/RESULTS.md), [FAILURES.md](artifact/FAILURES.md), [LOG.md](artifact/LOG.md) |
| Token instrumentation | [TOKENS.md](artifact/TOKENS.md) |
| Presentation material | [SCRIPT.md](artifact/SCRIPT.md), [demo](artifact/demo/) |
| English project guide | [artifact/README_en.md](artifact/README_en.md) |

## Results and their scope

- **NKI, final C10 configuration:** L1–L7 each passed 5/5 repeated runs; L8 attention passed 0/5. These are results under our modified checker using the NKI CPU simulator, with Qwen inference on Trainium2.
- **Matched L3/L4 comparison:** the same model, starting templates, and numerical checks gave 0/5 with control feedback and 5/5 with targeted indexing feedback.
- **NumPy, final A6 configuration:** the checker accepted nine of ten levels; A5 softmax remained unsolved. A7 used whole-matrix multiplication instead of the required tiling, so checker acceptance does not establish full rule compliance.
- The current attempt-log index reports **1,941 attempts across 516 runs**, drawn from 500 log files. Repeated outputs are correlated; five successful repeats are not five independent demonstrations of general reliability.

Important qualifications for the preserved historical reports:

- L5/L6 traffic thresholds were changed during development. Static inspection gives the final C10 kernels a traffic ratio of 1.0, below the original limits, but a complete rerun with the original checker has not been established.
- The 31 distinct agent kernels in the additional evaluation passed team-designed **numerical** cases. Those are not the organizers' hidden tests or a complete recheck of every constraint. The confidence score is a host-side heuristic.
- Regrading covered 354 selected attempts behind key decisions, not the full archive.
- Device checks are reported in the development log for selected kernels. The archived machine-readable device timing record is for a reference L4 kernel; full device validation of every final kernel remains incomplete. We claim no measured kernel speedup.
- Claude supported the team's supervised development process. The deployed repair loop uses Qwen, a deterministic checker, and a Python controller.
- Changes extend beyond feedback wording to templates, prompts, checker rules, and harness/controller behavior. The pooled L1 17/20 figure combines configurations and is not a single C10 measurement.

## Use the snapshot

After cloning this repository, enter the preserved artifact root:

```bash
cd projects/team-15-syntax-error/artifact
```

Follow [REPRODUCE.md](artifact/REPRODUCE.md) for the model setup, final configuration, and experiment commands. Kernel-agent commands run from `projects/02-kernel-agent/` relative to that artifact root. Access only your own assigned seat. This submission packaging did not rerun experiments or selftests; all reported measurements come from the archived runs.

## Provenance and credit

`artifact/` is a complete, unchanged file snapshot of [qz2930-crypto/hack-the-chip-seat73](https://github.com/qz2930-crypto/hack-the-chip-seat73) at commit [`f2bda82f2fd406211cf6e9f3286b36e728b7177b`](https://github.com/qz2930-crypto/hack-the-chip-seat73/tree/f2bda82f2fd406211cf6e9f3286b36e728b7177b): **3,941 tracked files, 41,316,024 bytes**. The original file hierarchy and recorded seat identifiers are retained. Team 15 used seats 70–74; seat 73 in historical document titles identifies the source artifact, not the team number.

The experiments used the historical organizer baseline `8f1ca41`; this submission branch is based on organizer commit `b93f3c4bb00a73809f18c34a6102402c1570f587`. It adds this independent project directory. Workshop sample code and model licenses retain their original terms and attribution in the snapshot.
