**Verilog generation with checker feedback — recorded results, 10 October 2026**

The best recorded LFSR setting, `--explain --principles`, solved **5/5 runs**. An 8-bit CPU experiment with the same feedback setting solved **0/1 runs after 20 attempts**, ending at **0.75**. These are saved experiment results; no new model calls were made for this report.

Both experiments used `Qwen/Qwen3-8B` at `http://localhost:8000/v1`, temperature 0.7, thinking disabled, a 1,500-token output limit, and an 8,000-token configured context. Checking uses Icarus Verilog on the CPU. Each run starts with empty history.

**How the loop works**

1. `checker.py` builds a prompt from the Markdown specification, the module interface, and any public clarification stored in the adjacent reference JSON. `agent.py` sends these messages to Qwen. The initial LFSR prompt explicitly states immediate reset, the reset value, and the recurrence.
2. Qwen returns Verilog. The checker extracts the module and the agent saves it under `candidates/gen/`. The checker compiles a generated testbench containing both the candidate and an independent reference design. It drives identical stimulus, changes inputs while the clock is low, and samples outputs 1 ns after rising edges. Separate observations test immediate reset and reset during execution. These runs used seed 0, 256 primary edges, and 128 additional cycles in the random segment.
3. The checker compares candidate outputs against reference outputs and returns the first mismatch for each failed requirement, including undefined outputs, differing bits, timing context, and applicable facts about the candidate's own code. A score is the sum of weights for passing check categories, rather than the proportion of cycles that pass. A score of 1.0 also requires the supplied official testbench to pass.
4. Before the next attempt, the checker reloads `taxonomy.json` and `knowledge.json`, matches current failure signatures and applicable categories, and selects at most two relevant principles. It sends the specification, latest candidate, feedback, and selected principles; `--explain` asks for a diagnosis before the revised module. The full knowledge table is never inserted into the prompt. The loop ends at 1.0 or the attempt limit.

The reference `.v` supplies the grading comparison; its source and sampled expected outputs are not sent to Qwen in these recorded runs. Repair guidance comes from observed failures, candidate code facts, and retrieved principles. Public specification values remain in the prompt.

**Best LFSR result**

Five runs allowed up to five attempts each. All initial candidates scored 0.70 and used reset logic that waited for a clock edge. The final candidates handled reset immediately and passed the generic checks and official testbench. Four runs needed one repair; the fifth repeated its initial candidate once and needed two repairs.

| Run (log index) | Scores by attempt | Attempts | Official testbench on final candidate |
|---|---|---:|---|
| 0 | 0.70 → 1.00 | 2 | Pass |
| 1 | 0.70 → 1.00 | 2 | Pass |
| 2 | 0.70 → 1.00 | 2 | Pass |
| 3 | 0.70 → 1.00 | 2 | Pass |
| 4 | 0.70 → 0.70 → 1.00 | 3 | Pass |

Total: **11 attempts**, averaging **2.2 attempts per solve**. Final scores were all 1.00. This reports the best recorded condition only; it does not isolate the effect of principles from `--explain` or establish performance on unseen tasks. LFSR-specific knowledge keys were enabled.

The selected rules were:

> Principle: An output required before the first clock edge must become defined independently of a clocked transition.
>
> Principle: A reset specified to act independently of the clock must restore the required state between clock edges.

Evidence: [LFSR console log](log/lfsr.log) and [complete prompts, replies, candidates, settings, and scores](log/runs/md_generic_v4_dynamic_taxonomy_lfsr_20261010T200026Z_1791662426884934708/events.jsonl).

**Failed 8-bit CPU experiment**

The [CPU specification](prompts/specs/cpu8.md) describes an accumulator core with externally supplied instructions, arithmetic, jumps, output, halt, enable, and reset. This was one run with 20 attempts, rather than a five-run success-rate measurement.

| Attempts (zero-based rounds) | Score | Recorded outcome |
|---|---:|---|
| 0 | 0.00 | Compilation failed: instruction array/interface errors. |
| 1–2 | 0.20 | Compiled, but failed reset and execution checks; multiple output drivers were reported. |
| 3–14 | 0.00 | Repeated compilation syntax errors. |
| 15 | 0.00 | No complete module returned. |
| 16 | 0.00 | Instruction array/interface compilation errors returned. |
| 17–19 | 0.75 | Compiled, but failed random-input and input-toggle checks. |

There were **14 compilation failures**, one missing-module response, and five compiled attempts with functional failures. No candidate passed the official testbench. The final feedback reports `pc` undefined at cycle 259 and `acc` undefined at cycle 387, two cycles after enable went low. The final attempt repeated the preceding candidate. Thus **0.75 is a failed result**, not a solved CPU.

Evidence: [CPU console log](log/cpu8.log), [complete attempt records](log/runs/md_generic_v4_dynamic_taxonomy_cpu8_20261010T201926Z_1791663566973007868/events.jsonl), and [final candidate](candidates/gen/md_generic_v4_dynamic_taxonomy_cpu8_20261010T201926Z_1791663566973007868/cpu8/cpu8_0_19.v).

**Commands for the installed project**

With the local model already serving, reproduce the experiment settings:

```bash
cd /workspace/final
python checker.py --agent --problem lfsr --repeat 5 --rounds 5 --explain --principles
python checker.py --agent --problem cpu8 --repeat 1 --rounds 20 --explain --principles
```

Run the other prepared problems individually:

```bash
cd /workspace/final
python checker.py --agent --problem sequence_generator --repeat 5 --rounds 5 --explain --principles
python checker.py --agent --problem traffic_light --repeat 5 --rounds 5 --explain --principles
python checker.py --agent --problem dice_roller --repeat 5 --rounds 5 --explain --principles
```

Use `--repeat 1 --rounds 1` for one prompt followed by one check. For a run that survives disconnection:

```bash
cd /workspace/final
nohup python checker.py --agent --problem sequence_generator --repeat 5 --rounds 5 --explain --principles > log/sequence_generator.launch.log 2>&1 < /dev/null &
tail -f log/sequence_generator.log
```

Check the four prepared references without calling Qwen:

```bash
cd /workspace/final
python checker.py --run-all --mode checker
```

Run an existing official testbench on your own saved design without a reference comparison:

```bash
cd /workspace/final
python checker.py --run-tb --tb TestBench/cpu8_tb.v --dut /path/to/your_cpu8.v
```

View results with `tail -f log/PROBLEM.log`. Each new run replaces that problem's short log; complete attempt records remain under uniquely named `log/runs/` directories. Saved model-facing prompts are under `prompts/`. The checker has been updated since the recorded experiments, so the commands above start new runs with the installed version. Integration of the newly merged TB branch remains stopped.
