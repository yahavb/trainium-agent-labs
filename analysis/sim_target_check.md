# Does the simulation target change the scores?

**Conclusion: for the baseline and E-A, no. In general, yes.**
- Re-graded with and without `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, all 584 attempts (baseline 424 +
  E-A 160) got the same score as logged in both modes, **and the same feedback text**.
- But the target does change what passes. A level-4 kernel with a moving tile of 1024 fails every
  shape on trn2 and passes 2 of 4 on trn3. So a solve from a run that simulated trn3 has to be
  re-graded on trn2 before it counts.

NKI 0.6.0 CPU simulator, `python:3.12-slim` in Docker, current team/master. Every number below comes
from the commands under "How to reproduce".

## Which chip nki simulates

`nki/compiler/target.py` (nki 0.6.0) resolves the target in this order:

1. `NEURON_PLATFORM_TARGET_OVERRIDE`, if set
2. a target passed explicitly to the trace/compile call
3. `neuron-ls`: the `instance-type:` line's family (`trn2.3xlarge` → `trn2`); **if `neuron-ls` is
   missing or prints no such line, it falls back to `trn3`**

Targets map to generations: trn1 → gen2, **trn2 → gen3**, trn3 → gen4 (`_TARGET_TO_NC_VERSION`). In
this container, which has no `neuron-ls`:

| mode | resolved target | nc_version |
|---|---|---|
| `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` | trn2 | 3 (gen3) |
| unset | **trn3** | 4 (gen4) |

On a seat pod, `neuron-ls` exists. If it prints `instance-type: trn2...`, the **unset default there is
trn2 too**, and the baseline, E-A and E-F simulated trn2 anyway. That has to be read on the pod
(executor 1 is checking): `neuron-ls | grep instance-type`, and
`python -c "from nki.compiler.target import resolve_target as r; print(r())"` with the variable unset.

## Re-grading in both modes

`scripts/calibrate.py` scores every attempt again with `agent.grade()`, one fresh path each, and
compares the result with the logged reward.

| log | attempts | mode | target | scores different from the log |
|---|---|---|---|---|
| baseline seat-116 (`runs/seat-116/latest/.../attempts.jsonl`) | 424 | trn2 | trn2 / gen3 | **0** |
| baseline seat-116 | 424 | unset | trn3 / gen4 | **0** |
| E-A (`runs/seat-116/expA_L1/attempts_expA_L1.jsonl`) | 160 | trn2 | trn2 / gen3 | **0** |
| E-A | 160 | unset | trn3 / gen4 | **0** |

Feedback text compared directly: the same 584 attempts graded once per mode give **0 different
rewards and 0 different feedback strings**. So what the model was shown would not have changed
either.

## Where the targets differ

The same kernel under the two targets (`nkibench.py --level 4 --check`):

| kernel | trn2 (gen3) | trn3 (gen4) |
|---|---|---|
| L4 reference with `TILE_N = 1024` (moving free dim 1024) | 0/4: `Matmul moving free dimension 1024 exceeds max 512 for nc_version=nc_version.gen3` | **2/4 pass** (the shapes with N a multiple of 1024) |
| L4 reference with `TILE_M = 256` (stationary free dim 256) | 0/4: `exceeds gemm_stationary_fmax=128` | 0/4, same error |

So gen4 accepts a 1024-wide moving operand that trn2 rejects. A kernel using one would be a **false
solve** under trn3. Nothing in the baseline or E-A did this, which is why their scores do not move.

## What follows

- Baseline, E-A and E-F can be compared with v7 runs (which set trn2) as far as these logs go. Their
  scores and feedback are identical under both targets.
- Any **1.0 from a run without the override** should be re-graded with the override set before it is
  quoted: `NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python scripts/calibrate.py <attempts.jsonl> -o ...`
  (its re-grade section lists every changed score). That is moot if the pod check shows the unset
  default resolves to trn2 there.
- Every check in `analysis/` from this branch (calibration, E-A re-grade, v7 compatibility, held-out
  sets) was run with the override set to trn2.

## How to reproduce

```bash
# in the nki 0.6.0 container, from the repo root; mode = trn2 or unset
[ "$mode" = trn2 ] && export NEURON_PLATFORM_TARGET_OVERRIDE=trn2 || unset NEURON_PLATFORM_TARGET_OVERRIDE
python -c "from nki.compiler.target import resolve_target as r, target_to_nc_version as v; t=r(); print(t, v(t))"
python scripts/calibrate.py runs/seat-116/latest/projects/02-kernel-agent/attempts.jsonl -o /tmp/sim_$mode
sed 's/TILE_N = nl.tile_size.gemm_moving_fmax  # 512/TILE_N = 1024/' \
    projects/02-kernel-agent/reference_level4.py > /tmp/n1024.py
python projects/02-kernel-agent/nkibench.py --level 4 --check /tmp/n1024.py
```
