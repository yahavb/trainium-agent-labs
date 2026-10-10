# Static checker misses NumPy assignment aliases

The static checker recognizes `import numpy as np`, but does not track
`xp = np`. A forbidden reduction accessed through `xp.max` is therefore
missed when its branch is not exercised by the numerical tests.

Reproduce:
python3 -m kagent.harness 3 kernels/redteam/l3_unreached_alias_max.py -v --fixes

Observed:
PASS, zero rule violations, 24/24 numerical cases correct.

Controlled comparison:
static.check(source) returns [].
Replacing `reduce_rows = xp.max` with `reduce_rows = np.max` produces
a max-like violation.

Impact:
Rule-enforcement false pass on an untested branch. This does not
demonstrate incorrect numerical results on the tested inputs or a
bypass of the runtime guard when the branch executes.

Suggested fix:
Track NumPy assignment aliases with scope and reassignment handling.
Add regression coverage for direct and aliased references.