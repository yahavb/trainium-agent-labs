# Bounded Automatic Feedback Loop

Run from the laptop repository root with the existing physics environment:

```bash
.venv-physics-current/bin/python projects/03-contact-physics/grip_loop.py --minutes 20 --attempts 6 --steps 1024
```

The Qwen server must already be running on seat-260 at localhost:8000/v1.
The laptop must have working kubectl credentials. The trusted gripping suite and
the pinned beta=0.9 candidate must be present locally. The controller uploads
the generation dependencies itself. It does not upload physics references.

1. Evaluate the reviewed beta=0.9 seed at the selected fixed iteration budget.
2. Grade every public case locally against the frozen certified references.
3. Send measured failures and natural-language checker guidance to Qwen.
4. Download its proposal; reject any AST change except the beta numeric literal.
5. Simulate an accepted proposal remotely, retrieve outputs and grade locally.
6. Keep the highest passing-case count as the next revision's starting point.

Stop after all public cases pass, the attempt limit, or the wall-clock budget.
Six attempts includes the seed, so at most five Qwen revisions are requested.
The deadline covers generation, simulation, copying and grading. Remote work
uses timeout with a forced termination grace period; interrupted attempts are
recorded as errors with unmeasured physics, never as a physics failure score.

Each unique data/grip-loop-* directory retains attempts.jsonl, best.json,
best-candidate.py, command logs, Qwen request/response/source artifacts,
simulation outputs, grading reports and a run-note.md. Every grade reports
failure metrics and next-prompt.txt. Scores are accepted-case fractions; the
score range in the loop note is not a latency or throughput spread.

This is inference-time, checker-guided parameter search, not weight training
or unrestricted algorithm discovery. Only momentum varies; solver algorithm,
imports, tensor layout, objective and checker thresholds remain frozen. AST
comparison against the SHA-pinned, manually reviewed template prevents this
controller from importing a general generated program. It is not a general
sandbox. Broader code edits still require manual review or real isolation.

The simulator produces no Trainium performance measurement. A 1024-step result
must not be advertised as faster than a 256-step result. Once correctness passes,
measure accepted solutions/second on device against a correct baseline, with
repeats, equal workloads and unchanged gates. Unseen private gripping tests are
still needed before any generalization claim.

## Partial Pass Handling and Recovery

The grader exits 1 when any physics case fails, even if grading succeeded and
wrote a complete report. The controller accepts that exit only when exactly
one fresh trusted report exists, uses its measured score and sends its feedback
to Qwen. A genuine grading failure without a fresh report stops the loop.

An initial controller bug confused this exit code with a grading failure. The
three saved attempts in grip-loop-19e86832a7614d5aa65ed283fa5dccff were regraded
locally: beta 0.9, 0.95 and 0.98 each passed 10/16 at 1024 updates. Original
logs are unchanged; recovery-203c71c1521e42be9c65c481a9aed26f records corrections.
Those generations did not receive the actual physics feedback because of the
bug. They must not be presented as successful checker-guided revisions.

To recover saved outputs from another interrupted loop without remote work:

```bash
.venv-physics-current/bin/python projects/03-contact-physics/grip_loop.py --recover projects/03-contact-physics/data/grip-loop-RUN_ID
```
