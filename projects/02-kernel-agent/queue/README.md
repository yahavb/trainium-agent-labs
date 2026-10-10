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

Two kinds of run, kept apart.

**Probes** (`results-probe*`) are development runs. A probe runs the newest feedback mode on the
levels its new messages target, we read where the model stops, and write the next message. Messages
are tuned on these runs, so their scores are not evidence that the messages work.

**Evaluation runs** are the evidence: the repo's original feedback (`enriched`, `results-p1`,
`results-e*`) against the final mode with its messages frozen, on levels 2, 3 and 4, eight rounds, four
samples. The final mode's evaluation runs are queued only after its last message is written, so none
of them was used to write a message.

`results-p1` is the one pair run with `directed3` before it was superseded: the model got past both
old level 3 walls in one round each and stopped at a third, which `directed4` addresses.

Level 1 is run once, last: in five of five `located` runs it stayed at 0.30 and produced the identical
kernel for the first six rounds, so it costs the most chip time and separates the modes least.
