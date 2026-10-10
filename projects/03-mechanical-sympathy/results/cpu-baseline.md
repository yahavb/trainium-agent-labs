# Samudra eight-year CPU rollout

Status: Passed on 2026-10-10 on pod `seat-213`.

This was a forward-only autoregressive run. It used the real Samudra 2 one-
degree checkpoint and OM4 one-degree data. It did not calculate forecast
metrics or make visualizations. It wrote the predicted fields to Zarr.

## Run inputs

- Time window: 2014-10-10 through 2022-12-24.
- Model steps: 299. Output time records: 598.
- Grid: 180 by 360, with 19 depth levels in the output.
- Output variables: `so`, `thetao`, `uo`, `vo`, `zos`.
- Data source: public OSN `Samudra/v2026-09/om4_onedeg/OM4.zarr`.
- Checkpoint: the existing Hugging Face Samudra 2 one-degree EMA checkpoint.
- Checkpoint SHA-256:
  `ff6160cb7e2a2b5b8ee2c5731c16bd114a1bb5a267245cc5ed766fb00c7f797a`.
- Samudra commit: `6b88f2c0204828ef77d41b3ab2733679615f53e8`.

## Timing

| Measure | Time |
| --- | ---: |
| Total wall time, including setup | 3,250.47 s (54 min 10 s) |
| Setup, including remote source and checkpoint load | 458.67 s |
| Initial state read | 7.27 s |
| Boundary reads during rollout | 589.62 s |
| Model forward calls, total | 1,947.91 s |
| Model forward latency, median per call | 6,476.64 ms |
| Model forward latency, p90 per call | 6,750.05 ms |
| Model forward latency, maximum | 8,334.65 ms |
| Zarr writes | 165.23 s |
| Full rollout loop | 2,784.53 s (46 min 25 s) |

Model-forward timing includes `forward_once` and prediction assembly. It does not
include boundary reads, autoregressive state updates, or Zarr writes. These
values are one CPU run on seat 213. They are not a Trainium performance result.

## Prediction store

Path on seat 213:
`/workspace/projects/03-mechanical-sympathy/rollouts/cpu_8_year/predictions.zarr`

The store opens successfully. Its dimensions are `time=598`, `lev=19`, `y=180`,
and `x=360`. Forecast times run from `2014-10-20 12:00:00` through
`2022-12-24 12:00:00`. The store uses 11,975,169,211 bytes, about 12 GB. It
stays outside Git. Other pods cannot read it unless it is copied to shared
storage.

## One-step correctness fixture

The one-step CPU fixture was generated twice. Both 43,714,881-byte files were
byte-identical and had SHA-256
`c6c43040fc96178a72d118c820acaa046f0e1c4429aa598a6b58c2751a2c9571`.
The fixture manifest remains in draft state until the team selects Trainium
precision and freezes numeric tolerances.

## Trainium status

Trainium was not tested in this run. We have no Trainium latency or speedup
result. Use this run to confirm the full rollout path and provide a CPU
correctness reference. Do not compare its CPU timing with a Trainium result as
if they were measured under the same hardware conditions.
