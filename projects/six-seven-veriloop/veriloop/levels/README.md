# Levels — the contract every level folder follows

One folder per level, named `NN_name` (e.g. `03_counter`). Because every folder has the same files, the
levels and the checker can be written at the same time, by different people.

| file | what it is |
|---|---|
| `spec.txt` | what the model is told: 3–6 plain lines, **including the exact module name and every port** (name, width, input/output; `clk` for clocked levels). No hints about the solution. |
| `reference.py` | the correct behaviour, in Python (below) |
| `good.v` | a correct design you wrote by hand — **must pass** |
| `bad_1.v`, `bad_2.v`, `bad_3.v` | at least three deliberately broken designs, each with a top comment saying what is wrong (e.g. `// carry out is never set`) — **must fail** |

## `reference.py`

```python
MODULE  = "counter8"                     # must match spec.txt and good.v
INPUTS  = {"reset": 1, "en": 1}          # name -> bit width. Do NOT list clk.
OUTPUTS = {"count": 8}                   # name -> bit width
CLOCKED = True                           # True: the test bench drives clk. False: combinational.
LABELS  = {"state": {0: "IDLE", 1: "RUN"}}   # optional: names for output values, used in feedback C

def vectors():
    """The test inputs: a list of dicts, one per step (one per clock cycle if CLOCKED).
    Combinational: every input combination if small, else edge cases + random.
    Clocked: start with reset, then cover enable off, wrap-around, reset mid-run, etc."""
    return [{"reset": 1, "en": 0}, {"reset": 0, "en": 1}, ...]

def reference(vectors):
    """The correct outputs: a list of dicts, one per step.
    CLOCKED: the output value just after that cycle's rising clock edge."""
    return [{"count": 0}, {"count": 1}, ...]
```

The checker builds the Verilog test bench from this automatically. **Level authors never write test
benches.**

## Check your level while you write it

Needs the simulator: on a seat it is already there (see `SETUP.md`); on a Mac, `brew install icarus-verilog`.

```bash
python veriloop/checker.py veriloop/levels/03_counter veriloop/levels/03_counter/good.v
```

It checks that the design **compiles with exactly the spec's module name and ports**, then simulates it
on your `vectors()` and compares every output with your `reference()`: `PASS`, or `FAIL` with the
first wrong steps (inputs, signal, got vs expected). `good.v` must pass; every `bad_*.v` must fail — and
at least one `bad_*.v` should compile but behave wrongly, so it tests the simulation and not only the
compiler. Working examples: `veriloop/tests/sim_fixtures/` (an adder and a counter).

## Prove your level is ready

```bash
python veriloop/selftest.py veriloop/levels/03_counter
```

It checks every file is there, the spec names the module and every port, `reference.py` is consistent,
`good.v` scores 1.0, and every `bad_*.v` is caught — at least one by simulation. **A level is done when
this says PASS.** `python veriloop/selftest.py` with no argument checks every level.

## Spec style — keep the model honest

- Say **what** the block does, never **how** to build it.
- Name the module and every port exactly; say which edge (rising) and whether reset is synchronous.
- Keep it short — the organisers measured that long prompts make the model think instead of answer.
