# Verilog Debugging Agent: testbench step, one-page note

**Goal.** A small model on a chip we control writes and fixes Verilog; a checker grades every attempt and says *why*,
and the reason feeds the next attempt. This note covers the checker (testbench / design verification step).

**Hardware.** Qwen3-8B served by vLLM-Neuron 0.24 on one AWS Trainium chip (seat pod, TP=2), thinking mode off,
~14 tokens/s per request. The checker runs on the pod's CPU: Icarus Verilog 12 + Python.

**Loop.** `python vagent.py` → describe a circuit (AI writes it) **or** paste your own Verilog → one JSON hand-off
`{prompt, verilog, testbench?}` → testbench step → FAIL: `for_feedback.json` → AI fix → re-check with the *same*
testbench → PASS: `for_optimize.json`. One command: `python run.py "<description>"`.

## What we ran

A fixed 9-circuit exam (`make_tests.py --run`): each circuit has a **correct** and a **buggy** version (one planted bug).
A circuit is *perfect* only if a testbench was built, the correct version **passes** and the buggy version **fails**.
Levels: large combinational (`mult8`, `barrel8`, `add16`), tricky (`prienc8`, `popcount8`, `bcd7seg`, `scmp4` signed),
clocked (`shreg8`, `det101` FSM). All 18 designs were first checked against hand-written references.

## What came out

| Version | Change | Perfect |
|---|---|---|
| v10 | AI writes a Verilog reference; 3 models, 2 must agree | 3/9 |
| v11 | + clocked template, signed ports | 6/9 |
| v12 | + **golden model in Python** (no width/sign traps) | 7/9 |
| v14 | + judge (design vs golden model) | **7/9, 7/9** (2 runs) |

**Spread.** The two v14 runs are identical, down to the failing inputs (req=4; clock edges 16, 21, 71). The 7 passes
are stable and the 2 failures are systematic, not luck. **Bugs caught:** 9/9 planted bugs in every v14 run.
**Testbench strength:** 45/45 mutants killed on the 7 passing designs.

## What we learned

1. **The checker's own correctness is the bottleneck.** Every false FAIL came from a wrong golden model, never from
   the simulator. A wrong checker sends correct designs to be "fixed".
2. **Ask the model for less, in a language it is good at.** Full AI testbenches mostly failed to compile. Python
   writes the testbench structure; the AI writes only the expected-value function, in Python. 3/9 → 7/9.
3. **Validate the testbench before trusting it:** it must fail an empty design and drive every input bit to 0 and 1
   (an adder testbench that never sets `cin = 1` would let a carry-in bug through).
4. **Voting fixes random mistakes, not shared ones.** On `prienc8` the 3 models *and* the judge agreed on the same wrong
   answer, so the correct design was still blamed. On `det101` the judge correctly blamed the model, but the rebuilt
   model was wrong in a new way. A judge that shares the model's blind spot cannot catch it, hence the next step.

**Next.** An independent, larger judge model (shared gpt-oss-20b) for the same-model blind spot; formal equivalence
(Yosys) to prove the optimize step's smaller circuits identical for all inputs; a third exam run.

*Artifacts:* `CHECKER.md` (rules + reasons), `ATTEMPT_LOG.md` (every attempt with its score; `python attempt_log.py`),
`results/scorecard_*.log` (exam runs).
