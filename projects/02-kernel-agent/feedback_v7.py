"""Feedback v7: a better FIRST prompt, behind one switch, plus three more held-out operations.

Run from the event repo's projects/02-kernel-agent directory, with that directory and this file's
directory on PYTHONPATH, and run THIS file from this folder so Python finds the copies next to it:

    MESSAGES=v5 CARD=category SAMPLING=qwen REPAIR_PROMPT=restructure PROMPT1=theirs python feedback_v7.py ...
    MESSAGES=v5 CARD=category SAMPLING=qwen REPAIR_PROMPT=restructure PROMPT1=v2     python feedback_v7.py ...

Task 07 showed that repair messages fix one error at a time, and that on a new operation the model
starts too far from a working kernel for 8 rounds to close. Its two largest failures were gaps in the
FIRST prompt, not in the messages:
  * the signature: their prompt names the entry function but never gives its parameters, and 317 of
    320 level-11 samples wrote `def nki_swiglu_(a)` for a two-input operation;
  * the API: the card never mentions nisa.activation, so the model used exp and sqrt as two-value ops
    (`tensor_scalar(op0=nl.exp)`), the largest error class on levels 8-10.

PROMPT1=v2 changes the first prompt in two ways, on every level:
  1. The entry point line gives the exact def line, with the reference's parameters (their grade()
     already does this, but only after a "no function named" failure).
  2. The card gets a short block of real functions: nisa.activation and its one-value ops,
     tensor_tensor, tensor_scalar with a (rows, 1) operand for one number per row, reciprocal for
     division, and nl.max/sum/min/mean with keepdims. Every line was checked to simulate and lower for
     trn2. CARD=category's worked example is kept after it, without its own (now duplicate) list.
PROMPT1=theirs leaves the first prompt exactly as CARD=category builds it. Repair prompts are never
changed.

Importing this also registers levels 12-14 (ops08.py), on top of feedback_v6's levels 9-11, timeout
and grade fix. MESSAGES must be v4 or v5: task 07's v6 didn't help and isn't part of this test.
"""
import inspect
import os
import sys

PROMPT1 = os.environ.get("PROMPT1", "theirs")
if PROMPT1 not in ("theirs", "v2"):
    sys.exit(f"PROMPT1 must be theirs or v2, got {PROMPT1!r}")
if os.environ.get("MESSAGES", "v5") not in ("v4", "v5"):
    sys.exit("MESSAGES must be v4 or v5 for task 08")
import feedback_v6 as v6   # noqa: E402  (levels 9-11, timeout, grade fix; imports v5, v4, v3, v2)
import ops08               # noqa: E402,F401  levels 12-14
v5, agent, nkibench = v6.v5, v6.agent, v6.nkibench

API_V2 = """
More real functions. Each writes into dst, a tile you allocate first with nl.ndarray:

  nisa.activation(dst=, op=nl.exp, data=)           op(data) on every element; op is a function of one
                                                    value: nl.exp, nl.log, nl.sqrt, nl.rsqrt, nl.sigmoid,
                                                    nl.silu, nl.tanh, nl.gelu
  nisa.tensor_tensor(dst=, data1=, data2=, op=nl.add)            two tiles of the same shape
  nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=x)   a tile and x, where x is a number, or a
                                                    (rows, 1) tile that gives each row its own number
  nisa.reciprocal(dst=, data=)                      1 / data; divide by multiplying with it
  m = nl.max(t, axis=[1], keepdims=True)            each row's maximum as a (rows, 1) tile; also nl.sum,
                                                    nl.min, nl.mean
  The two-value ops are nl.add, nl.subtract, nl.multiply, nl.maximum and nl.minimum. Python operators
  (+ - * /) do not work on tiles, and one-value functions such as nl.exp go in nisa.activation.
"""


def example_only(addition):
    """CARD=category's addition without its 'More real functions' list: just the worked example."""
    i = addition.find("\nMean ")
    return "\n" + addition[i + 1:] if i >= 0 else ""


def def_line(level):
    s = nkibench.LEVELS[level]
    params = ", ".join(inspect.signature(s["ref"]).parameters)
    return f"def {s['entry']}({params}):"


_category_first_prompt = agent.first_prompt     # v5 installed first_prompt_category (CARD=category)


def first_prompt_v2(level, terse=0):
    if v5.CARD == "category":
        agent.API_CARD = v5.BASE_CARD + API_V2 + example_only(v5.ADDITION[v5.category(level)])
    else:
        agent.API_CARD = v5.BASE_CARD + API_V2
    p = _their_first_prompt(level, terse)
    s = nkibench.LEVELS[level]
    old = f"Entry point: a function named `{s['entry']}`, decorated with `@nki.jit`.\n"
    new = (f"Entry point: a function named `{s['entry']}`, decorated with `@nki.jit`, with exactly the "
           f"reference's inputs:\n\n    @nki.jit\n    {def_line(level)}\n\n")
    return p.replace(old, new, 1)


_their_first_prompt = v5._their_first_prompt     # their agent.first_prompt, unwrapped

if PROMPT1 == "v2":
    agent.first_prompt = first_prompt_v2          # solve() looks it up at call time

GATE = os.environ.get("GATE", "")
if GATE in ("static", "compile"):                # a simulator solve the trn2 compiler rejects is not solved
    import gate_nki
    gate_nki.install(agent, GATE)

if os.environ.get("NKI_VERDICTS"):                # a verdict and confidence after every level (verdict_nki.py)
    import verdict_nki
    verdict_nki.install(agent, os.environ["NKI_VERDICTS"])

if __name__ == "__main__":
    print(f"feedback v7: PROMPT1={PROMPT1} MESSAGES={v6.MESSAGES} CARD={v5.CARD} SAMPLING={v5.SAMPLING} "
          f"REPAIR_PROMPT={v5.v3.PROMPT} GRADE_TIMEOUT={v6.GRADE_TIMEOUT} GATE={GATE or 'off'}"
          + (f" USAGE_LOG={v5.USAGE_LOG}" if v5.USAGE_LOG else ""))
    agent.main()
