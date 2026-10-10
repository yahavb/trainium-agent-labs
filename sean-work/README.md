# Sean's Verilog workspace

Run commands from `/workspace/sean-work`. Python 3 and Icarus Verilog (`iverilog`
and `vvp`) are required. Generation also requires a running model endpoint,
selected with `--base-url` or `KERNEL_AGENT_BASE_URL`.

Generate **one attempt and test it**:

```sh
cd /workspace/sean-work
python agent.py --problem lfsr --repeat 1 --rounds 1 --principles-v2 --explain
cat log/lfsr.log
```

The first prompt has no previous failures to retrieve principles for. Use
`--rounds 2` to allow a correction prompt after the first attempt fails.
Generated Verilog is in `candidates/gen/`. Exact requests are in `prompts/runs/`;
full responses, scores, diagnostics, selected keys/signatures and feedback are
in the JSONL files under `log/`.

Grade a file you already have:

```sh
python checker.py --spec prompts/specs/LFSR.md --dut candidate.v --detectors --principles
```

Run all four prepared references without model requests:

```sh
python checker.py --run-all --mode checker
```

Generate and grade one attempt for each of the four problems:

```sh
python checker.py --run-all --mode agent --repeat 1 --rounds 1 --principles-v2
```

Results appear in `log/lfsr.log`, `log/sequence_generator.log`,
`log/traffic_light.log`, and `log/dice_roller.log`. These plain files contain
console output without an added header; the latest run replaces each file.
Batch metadata and historical copies are in `log/runs/`.

For an arbitrary Markdown spec, supply its independent reference:

```sh
python agent.py --spec design.md --ref reference.v --tb design_tb.v --repeat 1 --rounds 1
python checker.py --spec design.md --ref reference.v --tb design_tb.v --dut candidate.v
```

`--tb` is optional when ports can be inferred from the reference. Same-stem
references and prepared testbenches are discovered automatically. An official
testbench alone can run with `python checker.py --tb design_tb.v --dut candidate.v`.

## Exactly which files are needed

| Use | Files |
| --- | --- |
| Generate and grade a Markdown problem | `agent.py`, `checker.py`, `taxonomy.py`, `taxonomy.json`, `knowledge.json`, that spec's `.md`, and its reference `.v` |
| Grade only, without model requests | The same files except `agent.py` |
| Prepared behavior clarifications/contracts | The reference's adjacent `.json`; keep these for all four prepared problems, especially dice grading |
| Official testbench validation | That problem's `TestBench/*_tb.v` |
| Batch runs | Additionally `experiments/md_generic_v4_dynamic_taxonomy.json` |
| Legacy named-problem principle feedback | Additionally `principles.json` |
| Selftests and LFSR regression | Keep `tests/`, `baseline_lfsr.txt`, `candidates/lfsr.v`, `candidates/bad/`, `reference/seq/`, `principles.json`, prepared specs/testbenches/references, and `prompts/regression/` for prompt regression tests |
| Replay historical prompts | Additionally `prompt_regression.py` and `prompts/regression/` |

The four prepared problem file sets are:

| Problem | Spec | Reference and metadata | Official testbench |
| --- | --- | --- | --- |
| LFSR | `prompts/specs/LFSR.md` | `verilog/reference/lfsr.v`, `lfsr.json` | `TestBench/lfsr_tb.v` |
| Sequence generator | `prompts/specs/sequence_generator.md` | `verilog/reference/sequence_generator.v`, `sequence_generator.json` | `TestBench/sequence_generator_tb.v` |
| Traffic light | `prompts/specs/traffic_light.md` | `verilog/reference/traffic_light.v`, `traffic_light.json` | `TestBench/traffic_light_tb.v` |
| Dice roller | `prompts/specs/dice_roller.md` | `verilog/reference/dice_roller.v`, `dice_roller.json` | `TestBench/dice_roller_tb.v` |

Other specs/testbenches are available for your own references. Logs, archived
records, generated candidates, and build output are not prerequisites for a
fresh run. Keep `knowledge_original.json` only if you want the original-file
comparison. Old unused files remain in the original workspace's `archive/`.

## Dynamic rules and knowledge comparison

The checker reloads both JSON files on every round, then selects principles for
current failure signatures. Category keywords match whole words/phrases.
Rules without principles are skipped. Default model-facing rule text contains
only `Principle: <rule>`; add `--show-detect-text` to also send signature names
and detection descriptions. Log records retain selected keys and signatures.

Put the original at `/workspace/sean-work/knowledge_original.json`, then run:

```sh
python checker.py --knowledge-report
```

This read-only report lists every added, removed and changed key with its text,
plus full MD5 hashes. The original's expected hash prefix is `a38ea61b`.
The captured LFSR round-1 comparison is in
`log/history/taxonomy_rules_update/round1_rules_before.txt` and
`log/history/taxonomy_rules_update/round1_rules_after.txt`.
