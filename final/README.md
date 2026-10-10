# Single-file Verilog checker and agent

Only checker.py is needed to start a fresh copy. It contains the checker, agent,
taxonomy lookup, feedback construction, embedded selftests, all JSON
defaults, and the existing specs/reference designs/testbenches. Python 3.11+
and Icarus Verilog (iverilog and vvp) must be installed separately.

On first execution it creates missing resources beside itself. JSON defaults
are Python variables named KNOWLEDGE, TAXONOMY, PRINCIPLES,
EXPERIMENT_MANIFEST, and REFERENCE_METADATA. Existing runtime JSON files are
kept, and knowledge/taxonomy are read again for each repair prompt. Edit those
runtime files to change retrieval dynamically. Only selected rules enter the
model prompt. Reference source stays on the grading side.

Initialize files without calling a model:

    python checker.py --init

Generate one candidate and check it (local model must already be serving):

    python checker.py --agent --problem lfsr --repeat 1 --rounds 1 --explain --principles

Run the measured LFSR setting or try the CPU:

    python checker.py --agent --problem lfsr --repeat 5 --rounds 5 --explain --principles
    python checker.py --agent --problem cpu8 --repeat 1 --rounds 20 --explain --principles

Other prepared agent tasks: sequence_generator, traffic_light, dice_roller.
Use --base-url URL and --model NAME to select another model endpoint.

Grade an existing candidate or use an arbitrary spec:

    python checker.py --spec prompts/specs/LFSR.md --dut candidate.v
    python checker.py --agent --spec design.md --tb design_tb.v --ref reference.v --repeat 1 --rounds 5
    python checker.py --spec design.md --tb design_tb.v --ref reference.v --dut candidate.v

Run just a supplied testbench, with no reference comparison:

    python checker.py --run-tb --tb TestBench/cpu8_tb.v --dut candidate.v

Batch and offline selftests:

    python checker.py --run-all --mode checker
    python checker.py --run-all --mode agent --repeat 1 --rounds 1
    python checker.py
    python checker.py --selftest

The agent feedback and dynamic --principles behavior are unchanged. Expected
reference values and reference source are not included in repair prompts.

Plain output: log/PROBLEM.log (replaced by the next run of that problem).
Complete attempt history: log/runs/<unique-run>/events.jsonl.
Exact model-facing messages: prompts/runs/<unique-run>/.
Generated designs: candidates/gen/<unique-run>/.
The existing logs and candidates have been copied without rewriting records.
Historical paths/hashes in those records describe the original experiment.
