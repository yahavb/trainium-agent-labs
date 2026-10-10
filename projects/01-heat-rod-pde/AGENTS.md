# Project 1 working instructions

Read PROJECT_BASELINE.md before changing this project's code, experiments or submission.
It records the owner's fixed requirements; later explicit owner instructions take precedence.

Owner workflow: reuse `challenge1-runtime-resilience` as the single maintained
Project 1 development branch. Commit changes there and provide the existing PR
targeting `master` (or create its replacement if it has closed). Do not create
additional feature branches or merge master without explicit owner authorization.
The current Checker/consolidation work is explicitly implementation-only: do not
run tests, benchmarks or inference unless the owner authorizes them again.

Preserve the real-model candidate/checker/feedback loop, the model-directed calculator,
official scoring and problem definitions. Never use hidden answers as a solving shortcut.
Keep official deliverables separate from additional engineering standards.

Before an optimization, identify affected requirements, expected benefit, cost and regression
risk. After code changes, run official selftests and appropriate regression tests and report
actual evidence. A documentation-only update can reference unchanged tested code without
pretending to rerun tests.

Keep original checker scores separate from any additional validation. Never present offline
tests, controlled responses or candidate replay as real-model performance. Match budgets and
grading policies in comparisons. Preserve unsuccessful and interrupted logs; unknown data stays
unknown. Do not resume an explicitly paused live experiment merely to fill a report.

Update submission/READINESS.md when evidence changes. Missing official content means NOT READY;
unresolved engineering checks must be named separately. Do not auto-merge main/master.
