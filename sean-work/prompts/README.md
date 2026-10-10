# Prompts

- `specs/`: all eight Markdown problem specifications.
- `runs/`: exact model messages exported from logs, grouped by run, then round.
  New agent requests are saved here automatically as JSON and readable text.
- `regression/before/` and `regression/after/`: preserved prompt-move snapshots.
  Historical replay enables detector text to retain their original wording.
- `catalog.md`: the original prompt collection.
- `testbench.md`: the original testbench-generation prompt.

Runtime file requirements and commands are in [the project README](../README.md).
