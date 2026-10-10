# Run queue

`autopilot.sh` on the seat starts the next `*.job` file here that it has not started before, in name
order, one at a time, and only when nothing else is running. A job file is four settings:

    OUT=results-example
    MODES="directed3 enriched"
    REPEAT=1
    AGENT_ARGS="--levels 2,3,4 --rounds 8 --samples 4 --context 8192"

`OUT` is the results folder (must start with `results-`). `MODES` are the feedback modes to run, in
order. The file is parsed, never executed.

To drop jobs that are waiting or running, a job file can instead hold:

    CANCEL="results-repeat2 results-baselines"

## What is queued and why

Five pairs, each one run of `directed3` and one of `enriched` (the repo's original feedback) on
levels 2, 3 and 4, eight rounds, four samples. The order inside a pair alternates, so neither mode
always runs first. Eight rounds matches the README's baseline command and the five-repeat `located`
run already on record.

Level 1 is left to a single pair at the end: in four of four `located` runs it stayed at 0.30 for all
eight rounds, so it costs the most chip time and separates the modes least.

`directed3` is frozen for these pairs. A message added after they start goes into a new mode, so every
`directed3` run in the comparison used the same checker.
