# Run queue

`autopilot.sh` on the seat starts the next `*.job` file here that it has not started before, in name
order, one at a time, and only when nothing else is running. A job file is four settings:

    OUT=results-example
    MODES="directed3 enriched"
    REPEAT=1
    AGENT_ARGS="--levels 3,4 --rounds 12 --samples 4 --context 8192"

`OUT` is the results folder (must start with `results-`). `MODES` are the feedback modes to run, in
order. The file is parsed, never executed.

To drop jobs that are waiting or running, a job file can instead hold:

    CANCEL="results-repeat2 results-baselines"
