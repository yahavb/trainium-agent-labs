# Verified RTL Agent: Technical Design

| | |
|---|---|
| Document version | 1.6.0 |
| Date | 2026-10-10 |
| Status | Draft for team sign-off. Organiser approval for an own project is pending (`STATUS.md`, ORG-1) |
| Folder | `projects/19-verified-rtl-agent/` |
| Companion files | [`STATUS.md`](STATUS.md) is the task board. [`CHANGELOG.md`](CHANGELOG.md) holds versions, decisions and the git workflow |

This is the reference for every person and coding agent working in this folder. If code and this document disagree, fix one of them in the same commit and log it in `CHANGELOG.md`.

---

## 1. Purpose

Build and measure an agent loop. Qwen3-8B, served on AWS Trainium2 seats, writes Verilog for VerilogEval spec-to-RTL problems. A deterministic checker grades every attempt and reports where the design first fails. A translator turns that report into one instruction for the next attempt.

**Primary claim to measure:**

> On a fixed held-out set, the same model on the same hardware passes more problems when the feedback names a located fix (run C) than when it returns the raw tool output (run B), at an equal attempt budget.

**Secondary claim:** at an equal attempt budget, compare one seat repairing in sequence (depth, run C) against three seats with different strategies (breadth, run D). Optionally compare both with the shared gpt-oss-20b (run E).

## 2. Scope

**In scope:**
- Spec-to-RTL generation on VerilogEval.
- A layered deterministic checker: static, compile, simulate, formal.
- Rule-based feedback translation.
- Orchestration across seats.
- Attempt logging, calibration audit, failure taxonomy, and a report.

**Out of scope:** fine-tuning, any LLM-based grading, languages other than Verilog/SystemVerilog, synthesis timing, power or area optimisation (area is a stretch goal only), and frontier models anywhere inside the loop.

## 3. Hard constraints

These are fixed by the event or were measured on the seats on 2026-10-10. Code must respect every row.

| # | Constraint | Value | Consequence for code |
|---|---|---|---|
| C1 | Model in the loop | `Qwen/Qwen3-8B` on a Trainium seat. Run E only: `gpt-oss-20b` on the shared Trainium endpoint | Frontier models may help write this code; they never generate attempts |
| C2 | Endpoints | Own pod: `http://localhost:8000/v1`. A teammate's pod: `http://seat-<N>.seat:8000/v1` (cluster DNS) | Use only seats belonging to this team, with the owner's OK |
| C3 | Context | `MAX_MODEL_LEN=8192` in the seat pods | Prompt tokens plus `max_tokens` must stay under 8192. Always check `finish_reason` |
| C4 | Concurrency | `max_num_seqs=4` per seat | At most 4 requests in flight per seat |
| C5 | Thinking mode | Must be off | Every request body carries `"chat_template_kwargs": {"enable_thinking": false}` |
| C6 | `seed` | **Never send it.** It kills the engine (`EngineDeadError`, HTTP 500) | `workers.py` rejects any request that contains `seed` |
| C7 | Parallel identical prompts | Return identical outputs (measured: 4 of 4 identical in every round) | Diversity comes from different prompts or sequential calls, never from repeating one prompt in parallel |
| C8 | Generation speed | About 5.5 tokens/s per request with 4 in flight. Answers ran 33–1,500 tokens, so 7–130 s | Budget runs from this. Cap `max_tokens` at 1,500 |
| C9 | Simulator | Icarus Verilog **12.0** (VerilogEval requires v12; v13 is unsupported) | Install with `apt-get install -y iverilog` on Ubuntu 24.04 |
| C10 | Formal tool | Yosys 0.33 | Install with `apt-get install -y yosys` |
| C11 | Dataset | `NVlabs/verilog-eval`, commit `c498220`, `dataset_spec-to-rtl`, MIT licence. 156 problems: 73 sequential (have `clk`), 83 combinational | Pin the commit in `setup.sh`. Don't copy the dataset into this repo |
| C12 | Credentials | One shared AWS set for all participants | Never write them to any file. CI-style grep before every push (section 12) |
| C13 | Submission | The final PR to the organisers' repo touches only this folder | Every file this project needs lives under `projects/19-verified-rtl-agent/` |

## 4. Architecture

```
eval/heldout.txt (fixed)                  eval/dev.txt
        │                                       │
        ▼                                       ▼
   problems.py  ──  Problem(id, spec, ref, test, kind, ports)
        │
        ▼
   agent.py  (orchestrator + router, plain code, no LLM)
        │  picks strategy per attempt, enforces budget, keeps ledger
        │
        ├──► workers.py ──► seat A  Qwen3-8B   (strategy S1)
        ├──► workers.py ──► seat B  Qwen3-8B   (strategy S2)
        └──► workers.py ──► seat C  Qwen3-8B   (strategy S4, then S3)
        │
        ▼
   checker.py   L0 static → L1 iverilog compile → L2 testbench sim → L3 yosys counterexample (comb)
        │  returns CheckResult (deterministic, no model)
        ▼
   translator.py   mode "raw" (run B)  |  mode "located" (runs C, D, E)
        │  returns one feedback string
        ▼
   router (in agent.py): pass → stop · budget left → next attempt with feedback · cycling → stop
        │
        ▼
   runs/<run>-<rep>.jsonl  ──►  audit.py (formal re-check of passes)  ──►  report.py (tables, chart)
```

Only `workers.py` talks to a model. Everything that grades runs on the CPU and is deterministic: the same code always gives the same `CheckResult`.

## 5. Repository layout and ownership

| Path | Purpose | Owner |
|---|---|---|
| `DESIGN.md` | This document | All, via CHANGELOG |
| `STATUS.md` | Task board | All |
| `CHANGELOG.md` | Versions, decisions, git workflow | All |
| `setup.sh` | Installs iverilog and yosys, clones VerilogEval at `c498220` | P3 |
| `problems.py` | Loads problems; builds the dev/held-out split | P3 |
| `eval/dev.txt`, `eval/heldout.txt` | Problem ids, one per line | P3 |
| `checker.py` | Layers L0–L3, `CheckResult`, scoring, `--selftest` | P1 |
| `translator.py` | Feedback modes and rules | P1 |
| `workers.py` | Model client with guards | P2 |
| `prompts.py` | Strategy templates S1–S4 | P2 |
| `agent.py` | Orchestrator, router, run definitions, CLI | P2 |
| `audit.py` | Calibration audit | P1 |
| `report.py` | Summary tables and chart from `runs/` | P3 |
| `kernel-transfer/kernel_agent.py`, `kernel-transfer/runs/` | The NKI feedback ablation (6.9) | bhaveshgupta01 |
| `runs/` | Append-only run outputs | Whoever ran it |
| `TAXONOMY.md` | Failure modes with counts | P3 |
| `README.md`, `NOTE.md` | Results write-up, one-page note (written last) | P3 |

Owners: **P1** = checker and translator, **P2** = agent and workers, **P3** = evaluation, measurement and docs. Edit another owner's file only after noting it in `STATUS.md`.

## 6. Component specifications

Every component lists its interface and a **Done when** line, which is its acceptance test. `STATUS.md` tracks each one.

### 6.1 `problems.py`

```python
@dataclass(frozen=True)
class Problem:
    id: str            # e.g. "Prob035_count1to10"
    spec: str          # contents of <id>_prompt.txt
    ref_path: str      # <id>_ref.sv  (module RefModule)
    test_path: str     # <id>_test.sv (module tb)
    kind: str          # "seq" if RefModule has a clk input, else "comb"
    ports: list[tuple[str, str, int]]   # (direction, name, width) from RefModule

def load(problem_id: str) -> Problem
def load_list(path: str) -> list[Problem]
def make_split(seed: int = 20261010) -> tuple[list[str], list[str]]   # (dev, heldout)
```

- The root comes from `VE_ROOT`, defaulting to `/root/verilog-eval/dataset_spec-to-rtl`.
- **Split rule:**
  1. Dev = the 12 problems already probed on 2026-10-10 (`Prob001_zero`, `Prob021_mux256to1v`, `Prob027_fadd`, `Prob035_count1to10`, `Prob050_kmap1`, `Prob054_edgedetect`, `Prob071_always_casez`, `Prob086_lfsr5`, `Prob109_fsm1`, `Prob128_fsm_ps2`, `Prob142_lemmings2`, `Prob156_review2015_fancytimer`), plus 8 more drawn with the seed.
  2. Held-out = 40 problems from the remaining pool: sort by problem number, cut into 4 quartiles, and draw 5 comb + 5 seq per quartile with `random.Random(seed)`. If a quartile runs short of one kind, fill from the other.
- **Done when:** `python problems.py --split` writes both files deterministically (the same seed gives the same files), and dev and held-out don't overlap.

### 6.2 `checker.py`

```python
@dataclass
class CheckResult:
    layer_reached: int          # highest layer completed: 0,1,2,3 ; -1 = no code
    passed: bool                # True iff L2 reports 0 mismatches
    score: float                # see scoring
    code: str                   # the extracted candidate
    compile_error: str | None   # first error line from iverilog, verbatim
    mismatches: int | None
    samples: int | None
    per_output: dict[str, dict] # {"q": {"mismatches": 12, "first_time_ps": 210}}
    first_mismatch: dict | None # {"output": "q", "time_ps": 210, "cycle": 21, "inputs": {...}, "dut": "..."}
    tb_hints: list[str]         # testbench "Hint:" lines, e.g. reset hints
    formal: str                 # "equivalent" | "not_equivalent" | "unsupported" | "not_run"
    counterexample: dict | None # {"inputs": {...}, "gold": {...}, "gate": {...}}  (comb only)
    raw: str                    # raw tool text, used by translator mode "raw"
    elapsed_s: float
    l0_error: str | None        # 1.1.0: "no_code" | "no_topmodule" | "missing_port:<name>"
    timed_out: str | None       # 1.1.0: "compile" | "simulate" when that tool ran past 60 s

def check(problem: Problem, reply_text: str, formal: bool = True) -> CheckResult
```

Each attempt runs in its own temporary directory. The layers run in order, and each stops at the first failure.

| Layer | Action | Exact command | Timeout |
|---|---|---|---|
| L0 static | Extract code: the longest fenced block tagged `verilog`, `systemverilog`, `sv` or untagged; if there's no fence, from the first `module` line to the last `endmodule`. Strip stray fence markers. Require `module TopModule`. Require every port name from `problem.ports` | Python only | 1 s |
| L1 compile | Compile the candidate with the testbench and the reference | `iverilog -Wall -Winfloop -Wno-timescale -g2012 -s tb -o t.vvp cand.sv <test.sv> <ref.sv>` | 60 s |
| L2 simulate | Run it; parse `Mismatches: N in M samples`, every `Hint: Output '<o>' has N mismatches. First mismatch occurred at time T.` and every other `Hint:` line | `vvp -n t.vvp` (writes `wave.vcd`) | 60 s |
| L3 formal (comb only, and only when L2 fails or for the audit) | Equivalence miter between RefModule and TopModule; parse the counterexample table | `yosys -p "read_verilog -sv <ref.sv>; read_verilog -sv cand.sv; prep; miter -equiv -flatten -make_outputs RefModule TopModule miter; hierarchy -top miter; sat -prove trigger 0 -show-inputs -show-outputs miter"` | 60 s |

**Testbench facts** (from `Prob035_count1to10_test.sv`):
- `timescale 1 ps/1 ps`, and `clk` toggles every 5 ps, so one clock period is 10 ps.
- Outputs are compared on both clock edges, so there are 2 samples per cycle.
- `cycle = time_ps // 10`.
- The testbench dumps tb-level inputs plus `<out>_ref` and `<out>_dut` to `wave.vcd`.
- **First-mismatch inputs:** read `wave.vcd` and take each input's last value change at or before `time_ps`. Implement a minimal VCD reader; don't add a dependency.

**Raw text includes the counterexample (1.1.0):** when L3 finds one, its table is appended to `raw`, so run B gets the same information as run C, untranslated. The primary claim is about translation, so B must not be missing anything C sees.

**Yosys output:** the counterexample rows have the form `\in_<name>`, `\gold_<out>` and `\gate_<out>`, each with a binary column. `SAT proof finished - model found: FAIL!` means `not_equivalent`. A proof success means `equivalent`. A `read_verilog` or `prep` error means `unsupported`, which is never a failure of the candidate.

**Scoring:** 0.1 if L0 passes, +0.2 if L1 passes, +0.2 if L2 produces a `Mismatches` line, +0.5 × (1 − mismatches / samples). Maximum 1.0 if and only if `passed`.

**Self-test** (`python checker.py --selftest`), which must catch all of these planted bugs:

1. The reference renamed to TopModule passes, with score 1.0.
2. A K-map with one minterm wrong (kmap1: `a | b`) is not passed; L3 returns a counterexample with `c=1`.
3. A `wire` output driven from an `always` block fails L1 with `is not a valid l-value`.
4. A counter missing its reset fails L2 and produces a reset hint.
5. An off-by-one counter limit fails L2 with a `first_mismatch.cycle`.
6. A reply with no code returns `layer_reached = -1`.

**Done when:** the self-test passes in a seat pod, and `check()` handles all 20 dev problems' references without crashing (`python checker.py --refs eval/dev.txt`). `--selftest --parsers-only` runs the tool-free half on a laptop; `--check <id> <file>` grades one file.

### 6.3 `translator.py`

```python
def feedback(problem: Problem, result: CheckResult, mode: str) -> str   # mode: "raw" | "located"
```

- **`raw` (run B):** returns `result.raw`, cut to 1,500 characters. No rewriting.
- **`located` (runs C, D, E):** returns **one** instruction of at most 400 characters, chosen by the first applicable rule:

| Situation | Message pattern |
|---|---|
| L0: no code or no `module TopModule` | "Reply with one ```verilog block containing `module TopModule` with exactly these ports: \<port list\>." |
| L0: a port is missing | "TopModule is missing port `<name>` (`<direction>`, `<width>` bits). Use exactly the ports in the specification." |
| L1: compile error matching a rule | The rule's instruction, plus the candidate's offending line number and text |
| L1: compile error, no rule | "Line \<n\> does not compile: `<iverilog message>`. Fix only that line." |
| L2: a reset hint from the testbench | Turn the hint into an instruction, e.g. "Make the reset synchronous: check `reset` only inside `always @(posedge clk)`." |
| L2: mismatch, comb, with a counterexample | "When \<inputs\>, your `<out>` is \<dut value\>, which is wrong. Re-derive `<out>` for that input combination." |
| L2: mismatch, seq | "Output `<out>` first goes wrong at clock cycle \<k\>. At that cycle the inputs were \<inputs\> and your `<out>` was \<dut value\>. Check the logic that updates `<out>` on that cycle." |

**Never include** the reference module's code, the testbench's code, or the expected value of a multi-bit output. For 1-bit outputs, "is wrong" implies the expected value. That's accepted and noted in the README.

**Rules** live in one list, so coverage and provenance can be audited:

```python
RULES = [
  {"id": "R1", "pattern": r"is not a valid l-value",
   "instruction": "`{signal}` is assigned inside an always block, so declare it as `output logic` (or `output reg`), not a wire.",
   "observed_in": "Prob054_edgedetect, Prob109_fsm1"},
  {"id": "R2", "pattern": r"always process does not have any delay",
   "instruction": "This always block has no trigger. Use `always @(*)` (or `always_comb`) for combinational logic, or `assign` for a constant.",
   "observed_in": "Prob001_zero"},
  {"id": "R3", "pattern": r"break statements not supported",
   "instruction": "Icarus does not support `break`. Replace it with a flag variable or restructure the loop.",
   "observed_in": "Prob071_always_casez"},
  {"id": "R4", "pattern": r"requires an explicit cast",
   "instruction": "Assigning to an enum needs a cast, e.g. `state <= state_t'(next);`, or declare the state as `logic [N:0]`.",
   "observed_in": "Prob142_lemmings2"},
]
```

**1.4.0 (TRN-4), from the dev failures in `runs/dev/C-1.jsonl`:**
- **R1** now names the right declaration: `output reg <name>` for a port, `reg` (or `logic`) for an internal signal. It used to say `output logic` for both, which is wrong for `next_state` (Prob100_fsm3comb).
- **R5** `is not allowed in a constant expression`: "A part-select `[hi:lo]` needs constant bounds. To take W bits starting at a variable position, use the indexed part-select `vector[start +: W]`." Observed in Prob021_mux256to1v.
- **R6** `can not select part of scalar`: "`<name>` is declared as a single bit, but the code selects bits of it. Declare it with its full width: `<declaration from the spec's port list>`." Observed in Prob073_dff16e. The width comes from the specification the model already has, so nothing is revealed.
- A compile error that matches no rule now quotes the offending line, not just its number. Observed in Prob092_gatesv100: a bare `syntax error` came back unchanged.
- **1.5.0:** a combinational module with no inputs gets "Your `<out>` is <value>, which is wrong. Re-read what the specification says `<out>` must be." instead of "wrong for some inputs". Observed in Prob001_zero, which repeated the old message 3 times.
- **1.6.0, active-low resets:** for a reset input named like `resetn`, `reset_n`, `rst_n`, `aresetn` or `nreset`, the reset messages say "low" (and `negedge` for an asynchronous reset). They used to say "high" for every reset. Observed in Prob073_dff16e.
- **1.6.0, R7:** a `syntax error` whose line is a declaration (`integer`, `reg`, `logic`, `wire`, `int`, `bit`, `genvar`) gets "Icarus rejects the declaration `<line>` at this point. Declare `<name>` once at module level, above the always block, and only use it inside the block." Observed in Prob071_always_casez, which repeated the quoted line 4 times. R7 is the only rule that also matches on the offending line.
- **1.6.0, R8** `has already been declared in this scope`: "`<name>` is declared twice. Keep one declaration, `output reg <name>` in the port list, and delete the other." Observed in Prob109_fsm1. **R1 for a port** now adds "Change it in the port list itself; do not add a second declaration.", because in Prob109 R1's advice led the model to add `reg out;` beside the port, which is what R8 then had to catch.
- **1.6.0, X values:** when the simulator's first wrong value has an X or Z bit, the message says so ("your `<out>` is 4'bxxxx: X means it has no value. Some bit … is never assigned on that path, or is driven from two places …"; for sequential outputs, "a register behind it is never given a value, or is assigned in two places"). It takes priority over a Yosys counterexample, whose value is defined and was never what the testbench saw. Observed on dev in Prob093_ece241_2014_q3 and Prob092_gatesv100 (TAXONOMY.md, finding 3). Reset hints still come first.

- New rules may come **only from dev-set failures**, and each must set `observed_in`.

**Done when:** every self-test case produces the expected message pattern, and no message contains a line from any `_ref.sv`. Unit-check this by searching each message for reference lines.

### 6.4 `workers.py`

```python
@dataclass
class Reply:
    text: str; prompt_tokens: int; completion_tokens: int
    finish_reason: str; latency_s: float; seat: str; model: str

def ask(base_url: str, prompt: str, model: str = "Qwen/Qwen3-8B",
        max_tokens: int = 1500, temperature: float = 0.6, top_p: float = 0.95) -> Reply
def healthy(base_url: str) -> bool        # GET <base without /v1>/health
```

- **Request body:** `{"model", "messages": [{"role": "user", "content": prompt}], "max_tokens", "temperature", "top_p", "chat_template_kwargs": {"enable_thinking": false}}`.
  - For gpt-oss (run E): drop `chat_template_kwargs`, set `max_tokens >= 2500`, use base `"$GPTOSS_BASE_URL/agg/v1"` and model `"gpt-oss-20b"`, and set `verify=False` on TLS.
- **Guards:**
  1. Assert that `seed` is not in the body.
  2. Estimate prompt tokens as `len(prompt) // 4` and lower `max_tokens` so the total stays under 8,192 − 64.
  3. Use a per-seat semaphore of 4.
  4. Set an HTTP timeout of 900 s.
- **Errors:**
  - A connection error is retried once after 10 s.
  - **HTTP 500 is never retried.** Call `healthy()`; if the server is unhealthy, raise `SeatDown(seat)` so the orchestrator stops using that seat and logs it.
  - `finish_reason == "length"` is not an error. It's logged, and the checker sees the truncated text.
- **Done when:** a test call to each team seat returns a `Reply`, and a body containing `seed` raises before sending.

### 6.5 `prompts.py`

These templates are deliberately short and positive. Don't add rule lists: measured on these seats, long prohibition lists make models audit themselves and return nothing. `{spec}` is the problem's `_prompt.txt`.

```text
S1 (from spec):
{spec}

Write the complete SystemVerilog module TopModule for this specification.
Reply with one ```verilog code block and nothing else.

S2 (table first):
{spec}

First write the truth table or state-transition table for TopModule as Verilog comments, then the complete module.
Reply with one ```verilog code block and nothing else.

S4 (ports first):
{spec}

Start from the exact port list above, declare every output with its type, then write the complete module TopModule.
Reply with one ```verilog code block and nothing else.

S3 (repair):
Specification:
{spec}

This SystemVerilog module TopModule is not correct yet:

```verilog
{code}
```

A checker reports:
{feedback}

Rewrite the complete module TopModule so that what the checker names is fixed. Your module must differ from the code above.
Reply with one ```verilog code block.
S5 (fresh with a hint, 1.3.0, D-014): used instead of S3 right after a repair that returned the code it was asked to fix, byte for byte.
{spec}
A previous attempt at TopModule failed. A checker reported:
{feedback}

Write the complete SystemVerilog module TopModule for this specification so that this problem does not occur.
Reply with one ```verilog code block and nothing else.
```

**Ledger (1.6.0, D-016).** From the second repair on, S3 and S5 also list up to 4 earlier, distinct findings for this problem (240 characters each), above the latest one: "Earlier versions of this module also failed these checks. Your new module must not fail them again:". It repeats only what the model was already told, so it reveals nothing new. Every repairing run gets it (B with raw text, C, D and E with located messages), so B vs C stays a test of the feedback alone. Each attempt logs `ledger_items`. Evidence: on dev, fixes did not stick (Prob050_kmap1 went 0.82, 0.88, 0.93, 0.87, 0.82, 0.88).

If S3 plus the code exceeds the context, drop the ledger first, then the spec, then the code's comments. Log `prompt_chars`.

### 6.6 `agent.py`

```text
python agent.py --run {A,B,C,D,E} --rep N --problems eval/heldout.txt \
                --seats http://localhost:8000/v1,http://seat-<B>.seat:8000/v1,http://seat-<C>.seat:8000/v1 \
                --out runs/<RUN>-<N>.jsonl [--budget 6] [--max-tokens 1500] [--parallel-problems 4]
```

| Run | Seats | Attempts per problem | Strategy sequence | Feedback mode |
|---|---|---|---|---|
| A | 1 | 1 | S1 | none |
| B | 1 | up to 6 | S1, then S3 × 5 (S5 after a copy) | raw |
| C | 1 | up to 6 | S1, then S3 × 5 (S5 after a copy) | located |
| D | 3 | 6, as 2 rounds × 3 seats | Round 1: S1 on seat A, S2 on seat B, S4 on seat C. Round 2: each seat applies S3 to its own round-1 attempt | located |
| E | gpt-oss-20b | up to 6 | S1, then S3 × 5 (S5 after a copy) | located |
| R | any | up to 6 | S1 × 6, a fresh attempt each time | none |

Run R (1.1.0, D-008) is the control for "more attempts": the same budget as B and C with no feedback at all. Without it, a gain of C over A could be extra tries rather than feedback. For A, B, C, E and R, `--seats` may list several seats: problems spread across them, which changes only wall-clock time, never the attempts a problem gets.

**Copy guard (1.3.0, D-014).** If a repair returns code byte-identical to the code it was asked to fix, the next attempt is S5 (fresh from the spec, with the feedback as a hint) instead of S3. A copy still costs one attempt from the budget, but it doesn't count towards the cycling stop. Evidence: on dev, 30 of 42 repairs in `C-1` and 21 of 50 in `C-s3rewrite-full-1` were copies. B, C and E get identical treatment, so B vs C still measures only the feedback text.

- **Router and stop rules, per problem:**
  1. Stop on the first `passed`.
  2. Stop when the budget is used up.
  3. In B, C and E, stop when the same feedback string repeats 3 times in a row, and log `stop_reason = "cycling"`.
  4. Repair the **latest** attempt, not the best one; the repo measured that repairing the best one freezes the loop.
- **Concurrency:** `--parallel-problems` problems run at once per seat (at most 4). Within one problem, attempts in B, C and E are sequential by definition. In D, the 3 seats run in parallel, each with a different prompt.
- **Claim per problem:**
  - `PASS` if the checker passed.
  - `FAIL` if the budget ran out or the problem stopped on cycling.
  - `UNVERIFIED` if the checker timed out or a seat went down mid-problem.
- **Progress output:** one line per attempt, matching the repo's style, e.g. `Prob050_kmap1  C  r2  seat-93  L2 0.62  "When a=0,b=0,c=1 your out is 0..."`.
- **Output location (1.1.0, D-010):** `--problems eval/heldout.txt` writes `runs/<RUN>-<N>.jsonl`; any other list writes `runs/<list name>/<RUN>-<N>.jsonl`, e.g. `runs/dev/C-1.jsonl`. An existing run file is never overwritten: pick a new `--rep`, or `--resume` to finish a run that was interrupted.
- **Held-out list check:** a held-out run refuses to start unless `eval/heldout.txt` matches the committed file, so every held-out run uses the same 40 problems.
- `--only id1,id2` limits a dev run to a few problems; `--selftest` checks the run shapes and stop rules with a fake model and checker.
- **Done when:** run C on the 20 dev problems completes end to end, writes a valid JSONL, and survives a dropped `kubectl exec` (it's launched with `nohup`).

### 6.7 `audit.py`

For every `PASS` on a comb problem in a run file, re-run L3. Record `formal` as `equivalent`, `not_equivalent` (a **testbench escape**) or `unsupported`. Seq problems are recorded as `not_run`; bounded model checking with `sat -seq` is a stretch goal.

**Output:** `runs/<RUN>-<N>.audit.json` and a table in the report.

**Done when:** on the dev runs, it reports a count for each category, and `equivalent + not_equivalent + unsupported` equals the number of comb passes.

### 6.8 `report.py`

Reads `runs/*.jsonl` (or `--dir runs/dev`) and writes `runs/summary.md` and `runs/chart.svg`. Besides the metrics, it writes (1.1.0) a paired, problem-by-problem comparison for B vs C, R vs C, A vs C, C vs D and R vs B with an exact sign test on the discordant problems, and a failure taxonomy that assigns every failed attempt one mode from the checker's result:
- one row per run and rep, with every metric in section 8;
- the pass rate for A–E with every rep shown as its own point.

Never average across runs without also showing each one.

### 6.9 `kernel-transfer/` (1.2.0, D-012)

A second, smaller experiment on the organisers' own NKI kernel agent (`projects/02-kernel-agent`). The question is whether the checker's message fixes failures when the model has never seen the language, as it should for Verilog, which the model knows.

**Hypothesis:** a located message fixes *mistakes* (Verilog: wire vs. reg, a missing cast), but not *missing knowledge* (NKI: invented `nl.dot`, 1-D tiles, reshaping instead of slicing). Missing knowledge needs the facts themselves, like the heat-rod project's calculator.

`kernel-transfer/kernel_agent.py` imports the organisers' `agent.py` unchanged and switches exactly one thing:

| Setting | `--feedback` | What the model gets back | Patched |
|---|---|---|---|
| K-raw | `raw` | The simulator's error, verbatim | `enrich()` returns its input |
| K-located | `located` | The organisers' `enrich()` instruction, as shipped | Nothing |
| K-docs | `docs` | K-located, plus the signature and first docstring lines of every real NKI call in the failing kernel, from the installed `nki` package, capped at 2,400 characters | `grade()` appends the docs when the kernel is not correct |

- **Settings, identical for all three:** `--all --rounds 8 --samples 1 --context 8192 --repeat 3`, levels 1–4, Qwen3-8B on seat 93. `--samples 1` because parallel identical prompts return identical answers on the seats (C7).
- **K-located is the 2026-10-10 baseline.** It used the same organisers' code at `8f1ca41` and the same settings, so it isn't re-run. Its files are `runs/K-located*`.
- **Run from a scratch directory** (`/tmp/kt-<mode>`), because the NKI simulator writes cache folders into the working directory.
- **Outputs:** `kernel-transfer/runs/K-<mode>.log` (the organisers' per-level summary) and `K-<mode>-attempts.jsonl` (one line per attempt, with the feedback text).
- **Metric:** per level, solved k of 3 plus best, worst and mean reward, as the organisers' `--repeat` summary prints them. Levels 1 and 3 scored 0.30 in every run measured so far (the organisers' 5 and today's 3), so a change there is attributable at 3 reps. Levels 2 and 4 varied today (0.30–0.50 and 0.30–0.62), so read them as rates.
- **Done when:** all three settings have 3 reps on levels 1–4, and `python kernel_agent.py --kt-selftest` passes.

## 7. Data formats

### 7.1 `runs/<RUN>-<N>.jsonl`, one object per attempt

| Field | Type | Notes |
|---|---|---|
| `ts` | ISO 8601 string | UTC |
| `run`, `rep` | str, int | |
| `problem`, `kind` | str, str | `comb` or `seq` |
| `attempt`, `round` | int, int | `attempt` counts from 1 within a problem |
| `seat`, `model`, `strategy` | str | e.g. `seat-93`, `Qwen/Qwen3-8B`, `S3` |
| `prompt_chars`, `prompt_tokens`, `completion_tokens` | int | Token counts from the server's `usage` |
| `finish_reason`, `latency_s` | str, float | |
| `layer_reached`, `passed`, `score` | int, bool, float | From `CheckResult` |
| `mismatches`, `samples`, `first_mismatch`, `formal` | | From `CheckResult` |
| `feedback_mode`, `feedback_sent` | str, str | The text that went into the next prompt |
| `code_sha1`, `code` | str, str | |
| `stop_reason` | str or null | Set on the last attempt of a problem: `passed`, `budget`, `cycling`, `seat_down`, `checker_timeout` |
| `claim` | str or null | Set on the last attempt: `PASS`, `FAIL`, `UNVERIFIED` |
| `prompt_note` | str | 1.1.0: what was dropped to fit the context, or empty |
| `compile_error`, `l0_error`, `tb_hints`, `counterexample`, `timed_out`, `check_s` | | 1.1.0: checker detail for the taxonomy |
| `error` | str or null | 1.1.0: a failed model request (the attempt still counts) |
| `code_version` | str | Git short sha of the last commit that changed a code file (`checker.py`, `translator.py`, `prompts.py`, `agent.py`, `workers.py`, `problems.py`), `+dirty` if one of them has uncommitted changes. Docs and run-file commits don't change it |

### 7.2 `eval/*.txt`

One problem id per line, sorted, with no blank lines. The first line is a comment: `# seed=20261010 created=<UTC>`.

## 8. Metrics, defined

| Metric | Definition |
|---|---|
| Pass rate | Problems with `claim == PASS`, divided by the number of problems in the run. Report each rep separately |
| Message gain | Pass rate of C minus pass rate of B, for each rep pair |
| Depth vs breadth | Pass rate and wall-clock of C vs D |
| Attempts to pass | Median `attempt` of the passing attempt, over solved problems. Unsolved problems are counted separately, never as 6 |
| Tokens per solve | Total `completion_tokens` in the run, divided by problems solved |
| Seconds per solve | The run's wall-clock, divided by problems solved |
| Failure layer | For unsolved problems, the distribution of `layer_reached` on the last attempt |
| Truncation rate | Share of attempts with `finish_reason == "length"` |
| Calibration | Each `claim` against the audit's `formal` result. Report `PASS & not_equivalent` (escapes) explicitly |
| Taxonomy | Every failed attempt assigned to one named mode in `TAXONOMY.md`, with counts per run |

## 9. Experiment protocol

1. Run `setup.sh` on every team seat, then confirm with `python checker.py --selftest`.
2. Create the split with `python problems.py --split` and commit `eval/heldout.txt`, so every run uses the same 40 problems.
3. Develop the translator on dev only (runs B and C on `eval/dev.txt`).
4. Final runs on held-out, 2 reps each: A, B and C in parallel on separate seats, E on the shared endpoint, then D using all three seats.
5. Run `audit.py` on every final run file, then `report.py`.
6. Improve the code whenever it helps (D-015). Every attempt records `code_version`, and `report.py` shows it for each run and flags a run that mixes versions. Compare only runs made with the same code version: after changing `checker.py`, `translator.py` or `prompts.py`, re-run whatever you want to compare.

## 10. Integrity rules (non-negotiable)

- The model in the loop runs on Trainium (C1).
- Translator rules come only from dev failures and carry `observed_in`.
- No held-out problem's spec, attempts or failures are read before its final runs finish.
- Feedback never contains reference code, testbench code, or expected multi-bit values.
- Every run file is committed as written, and corrected only by a new run.
- Every reported number cites a run file. Simulator results and formal results are labelled as such.

## 11. Operations runbook

```bash
# in each seat pod
cd /workspace/projects/19-verified-rtl-agent
./setup.sh                                   # iverilog, yosys, VerilogEval @ c498220
python checker.py --selftest
cd /workspace && ./serve.sh                  # model; wait for READY

# start a run so it survives a dropped connection
cd /workspace/projects/19-verified-rtl-agent
nohup python agent.py --run C --rep 1 --problems eval/heldout.txt --seats "$SEATS" \
      --out runs/C-1.jsonl > runs/C-1.log 2>&1 < /dev/null &
tail -f runs/C-1.log

# health
curl -s localhost:8000/health && echo ok
```

- **Restarting the model:** use `./serve.sh --stop`, then `./serve.sh`. Never run `pkill -f "vllm serve"` from a shell whose own command line contains that text, because it kills itself.
- **Backups, every hour, from the laptop:** `kubectl exec seat-<N> -- tar czf - -C /workspace projects | tar xzf - -C backup`. Commit `runs/` to the team repo from the laptop.
- **Credentials:** they expire around 20:00. On `ExpiredToken`, paste the newest block from the workshop channel, in that terminal only.

## 12. Collaboration protocol (people and coding agents)

- **Before editing:** read sections 3, 6 (your component) and 10. Claim the task in `STATUS.md` (owner and `DOING`, with the time) in its own small commit.
- **Stay inside your files.** Changing another owner's interface means updating section 6 of this document in the same commit, plus a `CHANGELOG.md` entry.
- **Before every push:**

```bash
python checker.py --selftest
git diff --stat origin/master            # only projects/19-verified-rtl-agent/ may appear
grep -rnE 'ASIA[A-Z0-9]{12}|AWS_SECRET|AWS_SESSION_TOKEN' . && echo "STOP: credentials" || true
```

- **Style:** Python 3 standard library plus `httpx`, matching the event repo's agents. Plain functions, `argparse` CLIs, readable progress output, no agent frameworks. Never reformat other people's code.
- **When you finish:** set the task to `DONE` with the commit sha in `STATUS.md`, and add a line under `Unreleased` in `CHANGELOG.md`.

## 13. Submission

The working repo is the private team repo. The organisers' repo is never pushed to directly; locally its push URL is disabled. To submit:
1. Fork `yahavb/trainium-agent-labs`.
2. Copy this folder into `projects/` on a branch.
3. Open one PR that touches only this folder.

Before opening it, `README.md` must start with a real transcript, and `NOTE.md`, `TAXONOMY.md`, `runs/` and `eval/` must be present.

## 14. References

- VerilogEval: https://github.com/NVlabs/verilog-eval (commit `c498220`)
- AutoChip, NYU Tandon: https://arxiv.org/abs/2311.04887, with a VerilogEval follow-up at https://arxiv.org/abs/2411.11856
- RTLFixer, NVIDIA, DAC 2024: https://research.nvidia.com/labs/electronic-design-automation/publication/tsai2023rtlfixer/
- Event repo and its measured findings: https://github.com/yahavb/trainium-agent-labs (`README.md`, `STATE.md`, `projects/02-kernel-agent/README.md`)
