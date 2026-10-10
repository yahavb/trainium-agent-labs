# Final gate passes when the organizers' checker is missing

`kagent/judge.py` is the v4 final gate: a kernel is reported verified only
if it also passes the organizers' `kernelbench.py`. When `kernelbench.py`
is not found at `KERNELBENCH_DIR`, `judge.check` returns `True` with the
message "gate skipped". `agent.py` discards that message and reports the
kernel as `verified` with confidence 0.95.

Reproduce (main c013fd5):
KERNELBENCH_DIR=/nonexistent python3 -c "from kagent import judge; print(judge.check(8, open('kernels/hand/l8_layernorm_f32_two_pass.py').read()))"

Observed:
(True, "organizers' checker not found; gate skipped")

Controlled comparison:
With KERNELBENCH_DIR pointing at trainium-agent-labs/projects/02-kernel-agent,
the same kernel returns False: worst relative error 0.000385 on
"normal input(128, 512)" (tolerance 0.0001).

Impact:
On any machine where KERNELBENCH_DIR is wrong or the organizers' repo is
missing (the default path only exists on the seat pods), the final gate
silently does nothing. Kernels that fail the organizers' checker, such as
hand/l8_layernorm_f32_two_pass.py, can be reported as verified with 0.95
confidence. Results logs do not record that the gate was skipped, so the
error cannot be detected afterwards. This is the calibration failure the
challenge scores most harshly.

Suggested fix:
Fail closed: when the checker is missing, return False or cap the status
at dev-only. At minimum, record the gate outcome (passed / failed /
skipped) in every Result so skipped runs are visible in the analysis.
Add a regression test that runs judge.check with a missing directory.
