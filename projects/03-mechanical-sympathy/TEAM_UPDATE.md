# Samudra inference status

Date: 2026-10-10

## Experiment rule

CPU defines correctness. The first correct Trainium run defines Trainium v0.
Later Trainium attempts compare with Trainium v0.

```text
CPU reference -> correctness gate -> correct Trainium result -> performance
```

Rollout throughput is the primary performance metric. Single-step latency is
the fallback after two documented full-rollout failures.

## Current CPU status

- The synthetic checkpoint forward passed on CPU.
- One real OM4 forward passed on CPU.
- The normalized one-step RMSE against the real target was about `0.162`.
- The full 299-step CPU forward rollout passed on pod `seat-213` on 2026-10-10.
- It wrote 598 forecast records to `rollouts/cpu_8_year/predictions.zarr`.
- The Zarr store opens successfully. It covers 2014-10-20 through 2022-12-24,
  has dimensions `time=598`, `lev=19`, `y=180`, `x=360`, and is about 12 GB.
- The rollout did not calculate forecast metrics or create plots.
- The one-step CPU fixture passed twice with byte-identical output.
- The fixture manifest remains in draft state until the team freezes numeric
  tolerance for its selected Trainium precision.
- See `results/cpu-baseline.md` for the timing record. Total wall time was
  about 54 minutes. Median model-forward latency was about 6.48 seconds per
  call. Input reads and Zarr writes are reported separately.

Do not use CPU timing as the Trainium performance baseline. It is a reference
for seat 213 only.

## Current Trainium status

- No correct Trainium result exists yet.
- No Trainium performance baseline exists yet.
- The official Samudra checkout does not yet contain the team Trainium port.
- The first task is a correct one-step forward with the frozen CPU fixture.

## Repository status

The pod has the official hackathon repository at `/workspace` and the official
Samudra repository under `.scratch/Samudra`. Both repositories still point to
their official origins.

The shared forks now exist:

- `pranoyghosh/trainium-agent-labs`
- `pranoyghosh/Samudra`

The pod remote migration is the next repository step. See `GIT_WORKFLOW.md`.

## Artifact policy

Commit code, small fixtures, manifests, results tables, and reports. Do not
commit checkpoints, datasets, full Zarr stores, compiler caches, or secrets.
The report is small and can go in the team repo. The 12 GB Zarr store remains on
seat 213. Other pods cannot read it unless the team copies it to shared storage.
