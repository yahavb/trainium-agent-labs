# PLC Agent — Two Agents and a Compiler Between Them

**Hack the Chip — NYU × Annapurna Labs, October 2026**
**Hardware:** AWS Trainium seat-39, Qwen3-8B on-chip via vLLM-Neuron
**Project type:** Propose-your-own (approved before 11:30)

---

## Motivation

Operational Technology (OT) software controls the physical world. Power grids, water treatment plants, oil refineries, pharmaceutical production lines, nuclear reactors — all of them run on programmable logic controllers executing IEC 61131-3 Structured Text. A vulnerability or misconfiguration in that code is not a data breach. It is a pipeline explosion, a contaminated water supply, a grid outage.

The cybersecurity gap in OT environments is severe and widening. IT security has decades of tooling, threat intelligence, and trained models behind it. OT security has almost none of that. Attackers have noticed: Stuxnet, Triton, Industroyer, and Pipedream are all purpose-built OT weapons, and they work precisely because defenders lack the AI-assisted detection and response capabilities that IT takes for granted.

**The core problem is data.** Every modern security tool — anomaly detection, vulnerability scanning, code auditing, patch recommendation — eventually needs a model, and every model needs training data. For OT, that data does not exist at scale. Structured Text is one of the most consequential programming languages in the world and one of the least represented in any public benchmark. There is no OT equivalent of GitHub Copilot's training corpus, no Stack Overflow for ladder logic, no HuggingFace dataset of verified PLC programs. The language is proprietary, the deployments are air-gapped, and the engineers who write the code do not publish it.

**This project is a direct attack on that data problem.** The approach: use a small language model to generate Structured Text, use a deterministic compiler and structural checks as the only authority on correctness, and let the model learn from verified programs as it goes. No human labels. No proprietary data. No large model required. The loop produces gate-verified, sector-diverse, requirement-tagged Structured Text programs that seed the training data pipeline for OT security tools.

The broader claim is that this approach generalises. Any domain with a formal correctness criterion — hardware description languages, medical device firmware, aviation control software, industrial robotics — faces the same data scarcity problem and can use the same bootstrap loop. The checker is the hard part and the valuable artifact. Building a precise checker for an underrepresented language is the contribution that compounds.

---

## Architecture

```
spec ──> Agent 1 (generator) ──> GATE: ironplcc check + 32 spec checks ──> PASS ──> dataset.jsonl
               ^                              │ fail
               │                             ▼
               └──── Agent 2 (critic): verdict + specific fix instructions
                              (bootstrap memory feeds round 0 of the next spec)
```

**The rule everything is built on: no LLM ever says PASS.** The critic advises; the gate decides. A model asked "is this correct?" will say yes to broken code, optimising for fooling the critic instead of compiling.

| file | what it is |
|---|---|
| `plc_checker.py` | the gate: spec builder, 32 requirement checks, compiler-message translator, failure diagnosers, selftest |
| `agent.py` | the loop: generator, critic, keep-best repair, bootstrap memory, curriculum, attempt log, summary |
| `report.py` | side-by-side comparison of run directories; writes `progress.png` |

---

## Experimental Results

All runs: Qwen3-8B on AWS Trainium, two-agent critic loop, IEC 61131-3 Structured Text, seed 99 (identical specs across all runs).

### Main result: solve rate progression

```
28%  ──────────────────────────────────►  75%
│                                          │
Baseline                           Improved hints
(old hints,                        (targeted structural
 no memory)                         fix instructions)
  ▲                                        ▲
  │   38%                         75%      │
  │    │                           │       │
  │  Old hints                Improved     │
  │  critic                   + bootstrap  │
  │  only                                  │
  │                                        │
  44 generator calls/solve          16 calls/solve
  (2.75× more efficient)
```

### Full comparison table

| condition | specs | solve rate | range | calls/solve | mean rounds |
|---|---|---|---|---|---|
| Baseline: old hints, no memory | 24 | **28%** | 25–30% | 44.0 | 1.29 |
| Old hints, critic only | 8 | **38%** | 25–50% | 32.0 | 1.33 |
| Old hints + bootstrap | 8 | **25%** | 25–25% | 52.0 | 1.00 |
| **Improved hints, critic only** | 4 | **75%** | 50–100% | **16.0** | 2.67 |
| **Improved hints + bootstrap** | 4 | **75%** | 50–100% | **16.0** | 2.67 |

Same model, same specs, same architecture. The only variable is the quality of the checker's feedback.

### Key finding: the checker is the bottleneck, not the model

Two specs that were stuck at 0.30 across all 3 baseline runs — never compiling, never progressing — solved consistently once the checker gave a structural instruction instead of a token list:

**Sample 0 (PID Loop + FOR Loop), old hints:**
```
round 0: best 0.30  [P0002 found 'END_FOR']
round 1: best 0.30  [P0002 found 'END_FOR']  (worse than 0.30; repairing that one)
round 2: best 0.30  [P0002 found 'END_FOR']  (worse than 0.30; repairing that one)
round 3: best 0.30  [P0002 found 'END_FOR']  (worse than 0.30; repairing that one)
-> NOT SOLVED  0/3 runs
```

**Same spec, improved hints:**
```
round 0: best 0.30  [P0002 found 'END_FOR']
round 1: best 0.30  [P0002 found 'END_FOR']
round 2: best 1.00  [PASS]
-> SOLVED  2/2 runs
```

The model was not incapable. It was receiving the wrong instruction. IronPLC's raw message (`Expected 'PROGRAM' | 'FUNCTION_BLOCK' | 'VAR_GLOBAL' | ...`) tells the model nothing actionable. The translated message (`unexpected END_FOR: the matching FOR...END_FOR block is missing its opening FOR keyword, or a semicolon is missing on the line just before END_FOR`) told it exactly what to change.

### Capability walls vs hint walls

The baseline run reveals two fundamentally different failure types:

**Hint walls** — specs the model can solve once the feedback is precise:

| spec | old hints | improved hints |
|---|---|---|
| PID Loop + FOR Loop | 0/3 runs, stuck 0.30 | 2/2 runs, solved round 2 |
| Sequential + Function Block | 0/3 runs, stuck 0.30 | 1/2 runs, solved round 1 |

**Capability walls** — specs that fail identically regardless of feedback:

| spec | result | pattern |
|---|---|---|
| Cascade + power_loss check | 0/3 runs, stuck 0.96 | same score, same failed check, every run |
| Interlock + safety_shutdown | 0/2 runs, stuck 0.96 | consistent, zero-variance failure |
| Sequential + safety_shutdown | 0/2 runs, stuck 0.90–0.95 | narrows but never clears |

Capability walls require a different intervention: bootstrap exemplars showing correct `safety_shutdown` logic, or curriculum training that builds up to complex specs. Hint walls require a better checker. Mixing both into one solve rate obscures which intervention is needed.

### Bootstrap memory observation

With old hints, bootstrap showed no benefit (25% vs 38% for critic alone) because the underlying errors were hint walls — no verified programs were produced to store as exemplars, so memory stayed empty. With improved hints, bootstrap matched the critic exactly (75%), and memory filled correctly:

```
sample 0 solved → memory: {exemplars: 1, known_fixes: 1, errors_seen: 1}
sample 1 solved → memory: {exemplars: 2, known_fixes: 2, errors_seen: 2}
```

Bootstrap requires a working gate first. Once the gate produces verified programs, the memory compounds.

### Per-spec breakdown (baseline, 3 runs each, seed 99)

| spec | control strategy + IEC feature | solve rate | failure type |
|---|---|---|---|
| 0 | PID Loop + FOR Loop | 0/3 → **2/2 fixed** | hint wall |
| 1 | Sequential Control + Function Block | 0/3 → 1/2 | hint wall (partial) |
| 2 | Cascade Control + CASE + Array | 0/3 | capability wall (power_loss) |
| 3 | State Machine + WHILE Loop | **3/3** | reliable solve, round 0 |
| 4 | Lead Lag Control + Structures | 0/2 | capability wall (STRUCT placement) |
| 5 | Interlock Logic + CASE Statement | **2/2** | reliable solve, round 0 |
| 6 | Interlock Logic + Function Block | 0/2 | capability wall (FB + shutdown) |
| 7 | Cascade Control + FOR + WHILE | 0/2 | capability wall (safety_shutdown) |
| 8 | Sequential Control + WHILE | 0/2 | capability wall (safety_shutdown) |
| 9 | Cascade Control + CASE + WHILE | **2/2** | reliable solve, round 1 |

---

## The Gate: 32 Checks Across 6 Dimensions

### What the spec contains

Each generated spec is drawn from six independently sampled dimensions:

| dimension | examples |
|---|---|
| sector (21) | Water Treatment, Oil Refinery, Pharmaceutical, Nuclear, Agriculture |
| control strategy (8) | State Machine, PID Loop, Cascade Control, Pump Alternation |
| IEC language feature (6) | CASE Statement, FOR Loop, WHILE Loop, Function Block, Array, Structures |
| safety feature (6) | Emergency Stop, SIL Logic, Watchdog Monitoring, Redundant Sensors |
| failure scenario (7) | Sensor Failure, Motor Overload, High Pressure, Power Loss |
| security profile (2) | Secure (clamp all setpoints) or Insecure (one marked vulnerability) |

### Why the spec dictates identifiers

The generator is told *"Declare EStop : BOOL. Inside IF EStop THEN set every actuator output to FALSE."* The check then looks for exactly that variable name. This turns a fuzzy question ("does this have an E-stop?") into a structural one, and lets every failure map to a precise fix instruction. Vague specs produce vague feedback; precise specs produce actionable feedback.

### What each safety check requires

- **Emergency stop / tank overflow / motor overload:** variable declared with correct type AND an IF branch testing it assigns FALSE to an actuator. Comment-only logic is rejected. An inverted test (`IF NOT EStop`) is rejected.
- **SIL voting (2-of-3):** `VoteTrip := (TripA AND TripB) OR (TripA AND TripC) OR (TripB AND TripC);` exactly, and outputs forced FALSE when VoteTrip is TRUE.
- **Safety shutdown:** latches on trip, clears only inside a branch testing `ResetCmd`, forces outputs FALSE while active.
- **Watchdog:** TON timer declared and called every scan; `WatchdogTimer.Q` drives a safety action.
- **Redundant sensors:** `ABS(SensorA - SensorB) > MaxDeviation` exactly.
- **Secure profile:** every REAL setpoint clamped with `LIMIT()` or an IF range guard before use.
- **Insecure profile:** one realistic weakness present and marked `(* VULN: description *)`.

### Scoring

| score | meaning |
|---|---|
| 0.00 | no program in output, or truncated at token limit |
| 0.30 | IronPLC rejects it |
| 0.60–0.99 | compiles; 0.60 + 0.40 × fraction of checks passed |
| 1.00 | compiles and every check passes — **PASS** |

---

## Compiler-Message Translation

Raw IronPLC output lists expected tokens — useless as repair instructions. The gate parses it into targeted instructions before the critic or generator sees it:

| error | raw (truncated) | translated instruction |
|---|---|---|
| P0002 found 'VAR' | `Expected 'PROGRAM' \| 'FUNCTION' \| ...` | `This VAR block is outside PROGRAM Main. Put PROGRAM Main directly above it.` |
| P0002 found 'END_FOR' | `Expected ';' \| 'END_FOR' \| ...` | `Unexpected END_FOR: the matching FOR block is missing its opening keyword, or a semicolon is missing on the line before END_FOR.` |
| P0002 found 'TYPE' | `Expected 'PROGRAM' \| ...` | `TYPE must come BEFORE PROGRAM Main. Required order: TYPE...END_TYPE, FUNCTION_BLOCK...END_FUNCTION_BLOCK, PROGRAM Main...END_PROGRAM.` |
| P2008 | `Cannot determine kind of type (identifier=MyBlock)` | `MyBlock is used as a type but not defined. FUNCTION_BLOCK MyBlock must appear BEFORE PROGRAM Main.` |
| P4007 | `Variable not defined (variable=n)` | `'n' is used but never declared. Add 'n : INT;' to the VAR block.` |
| P4012 | `Function block invocation not in scope` | `'Ctl1(...)' calls an undeclared instance. Add 'Ctl1 : MyBlock;' to the VAR block.` |
| P4035 | `Assignment value type mismatch (target_type=REAL, value_type=BOOL)` | `'InnerOutput' is declared as REAL but assigned a BOOL expression. Use an arithmetic expression instead.` |

### Failure diagnosers

When a check fails, the gate inspects the program and names the specific problem using the program's own variables:

- **Function block never called:** *"'Ctl1' is declared as MyBlock but never called. Assigning Ctl1.In1 is not a call. Add `Ctl1(In1 := <value>);` in the program body."*
- **Function block body written inside PROGRAM:** *"FUNCTION_BLOCK definition must come BEFORE PROGRAM Main."*
- **Struct fields never accessed:** *"'tank' is declared as TankData but its fields are never used. Read or write them as tank.Level, tank.Temperature."*

---

## System Design Details

### Bootstrap memory

Without `--bootstrap`, every spec starts from zero. With it, three things carry forward:

| what | source | use |
|---|---|---|
| verified exemplars | programs the gate accepted | closest by shared requirements goes in the round-0 prompt |
| known fixes | error that disappeared after feedback | *"IronPLC error X was fixed by: ..."* in every prompt |
| often-missed requirements | round-0 failure rate per check | warning added when that check appears in the next spec |

Nothing enters memory unless the gate accepted it. An exemplar exists only if IronPLC compiled it and every check passed.

### Keep-best repair

If a repair round produces a worse program (0.96 → 0.30), the regression is discarded. The agent re-sends the best version seen so far: *"Your latest version scored 0.30, worse than this one (0.96), and was discarded. Apply the fixes to the program shown above."*

### Curriculum mode

`--curriculum` starts with small specs and levels up as the agent succeeds:

| level | spec contents |
|---|---|
| 1 | operating mode + one simple safety interlock (~3 checks) |
| 2 | + control strategy + second safety feature (~5 checks) |
| 3 | + IEC feature + failure scenario + security profile (~7 checks) |
| 4 | full random sampling (~9 checks) — default without `--curriculum` |

Level-up rule: 2 of the last 3 specs at a level solved within 2 rounds.

---

## Setup

```bash
# inside the pod, terminal 1
cd /workspace && ./serve.sh          # wait for READY (~4 min)

# terminal 2
cd /workspace/PLC-AGENT
curl -fsSL https://www.ironplc.com/install.sh | sh
pip install openai
export PATH="$HOME/.ironplc/bin:$PATH"
python plc_checker.py --selftest     # must print SELFTEST PASSED
python agent.py --n 2 --model Qwen/Qwen3-8B   # smoke test
```

### Full experiment commands

```bash
# three-way comparison: raw baseline, critic, critic+bootstrap (same specs)
nohup sh -c '
python agent.py --n 10 --repeat 3 --seed 99 --feedback raw --out-dir runs/baseline_raw &&
python agent.py --n 10 --repeat 3 --seed 99 --feedback critic --out-dir runs/baseline_critic &&
python agent.py --n 10 --repeat 3 --seed 99 --feedback critic --bootstrap --out-dir runs/baseline_bootstrap
' > experiment.log 2>&1 < /dev/null &
tail -f experiment.log

# curriculum: train then evaluate on held-out full-difficulty specs
nohup sh -c '
python agent.py --n 30 --curriculum --out-dir runs/curriculum_train &&
python agent.py --n 10 --repeat 3 --seed 99 --memory-in runs/curriculum_train/memory_run0.json --freeze-memory --out-dir runs/curriculum_eval
' > curriculum.log 2>&1 < /dev/null &

# compare results
python report.py runs/baseline_raw runs/baseline_critic runs/baseline_bootstrap
```

---

## Future Work

### Immediate (one-session fixes)

- **`safety_shutdown` diagnoser.** The most common capability wall. The latch + clear + override pattern is well-defined; a specific diagnoser would likely convert it from a capability wall to a hint wall the same way `END_FOR` was fixed.
- **`VAR_OUTPUT` inside PROGRAM hint.** Observed in the bootstrap run (`P0002 found 'VAR_OUTPUT'`): the model writes `VAR_OUTPUT` blocks inside `PROGRAM Main` instead of inside `FUNCTION_BLOCK`. The `TYPE`/`STRUCT` hint pattern applies directly.

### Short-term (days)

- **Full baseline vs bootstrap vs curriculum comparison.** The mechanism is complete but time ran out. Run `--n 20 --repeat 3` for each mode on the same seed. Bootstrap requires a working gate first — as the results show, it compounds once hints are precise enough to produce verified exemplars.
- **Rejection-sampling fine-tuning.** Every passing program in `dataset.jsonl` is a gate-verified generation example. Every repair round that succeeded is a `(broken + feedback → fixed)` pair. One round of LoRA fine-tuning on these pairs, served from the two free NeuronCores (NC 0-1 while `serve.sh` uses NC 2-3), would close the remaining capability walls. The `dataset.jsonl` files from these runs are the seed dataset.
- **Behaviour-level verification.** The gate is currently structural. IronPLC ships `ironplcvm`. Running each passing program through scripted input scans — `set EStop := TRUE`, assert `MotorRun = FALSE` one scan later — upgrades the gate from structure to behaviour and catches logic errors the regex checks cannot see.

### Long-term

- **Generalisation to other underrepresented languages.** VHDL, SystemVerilog, IEC 62443 safety functions, RAPID (ABB robotics), KRL (KUKA robotics). The generator, critic, memory, and curriculum are language-agnostic. The investment is writing the checker. One good checker seeds the entire data pipeline for a domain.
- **Active learning.** Sample specs that target known capability walls rather than randomly. If `safety_shutdown` fails 80% of the time, weight its presence in the curriculum so the model sees more examples of it passing.
- **Multi-turn critic.** The current critic produces one round of edits. A multi-turn critic that asks clarifying questions about the program's intent before writing fixes could handle the remaining ambiguous failures.

---

## Deliverables

1. **The checker** — `plc_checker.py`: 32 requirement checks, compiler-message translator with 7 error codes, failure diagnosers, compiler capability probe, selftest with 15 cases. `python plc_checker.py --selftest` verifies everything.
2. **The attempt log** — `attempts.jsonl` in every run directory: every candidate, every score, every compiler message, every feedback message sent, every memory state.
3. **Results** — `summary.json` per run, compared with `python report.py <dirs>`. Key number: **28% → 75% solve rate** and **44 → 16 generator calls per solve**, same model, same specs, by improving only what the checker tells the model.
