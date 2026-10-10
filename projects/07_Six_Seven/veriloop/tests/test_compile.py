"""C1 test: the checker must compile good designs and explain broken ones in plain words.

    python veriloop/tests/test_compile.py

Each case is a design in compile_cases/, the spec it is checked against, whether it should compile, and a
phrase the explanation must contain. Run it after any change to checker.py; it passes on iverilog 12 (the
seats) and 13 (Homebrew), whose messages are worded differently.
"""

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import checker  # noqa: E402


def spec(module, inputs, outputs, clocked):
    return checker.ports_of(types.SimpleNamespace(MODULE=module, INPUTS=inputs, OUTPUTS=outputs,
                                                  CLOCKED=clocked))


COUNTER = ("counter8", spec("counter8", {"reset": 1, "en": 1}, {"count": 8}, True))
ADDER = ("add4", spec("add4", {"a": 4, "b": 4}, {"sum": 4, "cout": 1}, False))

CASES = [
    # design file       spec     compiles  the explanation must mention
    ("good_counter.v",  COUNTER, True,  ""),
    ("good_sv.v",       COUNTER, True,  ""),                 # SystemVerilog style is accepted
    ("good_add.v",      ADDER,   True,  ""),
    ("semi.v",          ADDER,   False, "missing `;`"),
    ("undecl.v",        ADDER,   False, "never declared"),
    ("wire_always.v",   COUNTER, False, "declared as a wire"),
    ("noend.v",         COUNTER, False, "`begin`"),
    ("prose.v",         ADDER,   False, "not Verilog"),
    ("wrongname.v",     ADDER,   False, "There is no module named `add4`"),
    ("wrongport.v",     ADDER,   False, "no port named `sum`"),
    ("narrow.v",        ADDER,   False, "spec says 4"),       # only a warning in iverilog; we fail it
    ("unknownmod.v",    ADDER,   False, "`full_adder`"),
]


def main():
    failed = 0
    for fname, (module, ports), should_compile, phrase in CASES:
        src = open(os.path.join(HERE, "compile_cases", fname)).read()
        r = checker.compile_verilog(src, module, ports)
        text = " ".join(r.messages)
        ok = r.ok == should_compile and phrase in text
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {fname:<18} {'compiles' if r.ok else text[:90]}")
        if not ok:
            print(f"         expected compiles={should_compile} and a mention of {phrase!r}\n"
                  f"         raw iverilog output:\n{r.raw}")
    print(f"\nC1 compile test: {len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
