# The VeriLoop checker — what it accepts, what it rejects, and why

The checker (`veriloop/checker.py`) grades every Verilog design the model writes. **It is the only
authority on success**: the agent may report "solved" only when the checker passed every test, and the
model never sees the expected answers in full.

## What it does, in order

| stage | what is checked | if it fails | score |
|---|---|---|---|
| 1. **Compile** | the design compiles with Icarus Verilog (`iverilog -g2012`, so the SystemVerilog style the model often writes — `logic`, `always_ff` — is accepted) | the error is translated into plain words **with the line**: a missing `;`, an undeclared signal, a wire assigned in an `always` block, an unclosed `begin`, words before the code | 0.0 |
| 2. **Interface** | the module has **exactly the spec's name, port names and port widths** (a tiny wrapper instantiates it with the spec's ports) | "no port named `sum`, the spec's ports are …", "port `a` is 3 bits, the spec says 4" — a width mismatch is only a *warning* in iverilog (it silently drops bits), so we fail it ourselves | 0.0 |
| 3. **Run** | the simulation finishes and prints every test step | a design that stops the simulation itself (`$finish`) or a loop that never ends (stopped after 10 s) is named as such | 0.2 |
| 4. **Behaviour** | every output, at every test step, equals the Python reference. Clocked designs are sampled 1 ns after each rising edge | the first wrong step is reported (below) | 0.3 + 0.7 × fraction of steps correct |
| **Solved** | all of the above, on every test | — | **1.0** |

An undefined output (`x`) is always wrong — a design whose register is never reset fails even if it
would "settle" later.

## The tests

Each level's `reference.py` lists the test inputs and the correct outputs. Combinational levels are
tested on **every** input combination (mux: 64, adder: 256). Clocked levels use sequences built around the
traps: the counter wraps 255 → 0, holds with enable off, and is reset mid-count (272 cycles); the traffic
light gets the pedestrian button on each green cycle in turn, a held button, a reset mid-phase, and 150
random cycles from a fixed seed (256 cycles). The same tests run every time, so results are comparable.

**Limit, stated plainly:** passing means *passing these tests*, not a proof of correctness. The tests are
chosen to hit the edges where generated designs break, and every level ships deliberately broken designs
that the tests must catch (below).

## The three feedback levels — the experiment

From **one** simulation the checker writes three messages; the experiment changes only which one the model
receives. Each adds information to the one before:

| level | the model is told | example (counter that ignores `en`) |
|---|---|---|
| **A** | only the verdict | `FAIL: the design is not correct.` |
| **B** | how much is wrong | `FAIL: 263 of 272 tests give a wrong output.` |
| **C** | where and how it first goes wrong, like a waveform: the inputs, the signal, got vs expected, and the two cycles before | `cycle 6: inputs reset=0 en=0 -> count = 6, expected 5  <-- first wrong cycle` |

Feedback C names values when the level defines labels (`light = YELLOW (2), expected GREEN (1)`), shows
at most **three** example rows, and never the whole table — the organisers measured that a model handed
the answer copies it instead of fixing the logic.

## How we know the checker is right

| proof | what it shows | result |
|---|---|---|
| `veriloop/tests/test_compile.py` | 12 designs: good ones compile, each kind of mistake gets the right plain-words message | 12/12, on iverilog 12 (seats) and 13 (Mac) |
| `veriloop/tests/test_simulate.py` | 11 designs that compile but behave wrongly in different ways are all caught; good ones pass | 11/11 |
| `veriloop/tests/test_feedback.py` | scores land in the right ranges; A < B < C in information; C never shows more than 3 rows | 9/9 |
| `veriloop/selftest.py` | **per level**: the spec names every port, the reference is consistent, `good.v` scores 1.0, every `bad_*.v` is caught — at least one by simulation, not only by the compiler | 4/4 levels; 17 broken designs, all caught |

The experiment runner refuses any level that fails `selftest.py`, so a broken level can never grade the
model.

## What it deliberately does not check

- **Timing, area, power.** Correct behaviour only — no synthesis.
- **Style.** Any Verilog that behaves correctly passes; we do not reward a "nice" design.
- **Hidden tests.** All tests are in the repo; the model never sees them, only the feedback.
