"""Level 5 -- multiply-accumulate (MAC) cell, the unit a matrix engine like Trainium's is built from.

Traps, each covered by the tests: the product is SIGNED (an unsigned multiply is the classic bug);
the 16-bit product must be sign-extended into the 20-bit sum; the sum wraps at 20 bits; clear and reset
both zero it, reset first; en = 0 holds; clear wins over en.
"""
import random

MODULE = "mac"
INPUTS = {"clear": 1, "en": 1, "a": 8, "b": 8, "reset": 1}
OUTPUTS = {"acc": 20}
CLOCKED = True
SIGNED = {"acc"}          # feedback C shows acc as a signed number


def vectors():
    v = []
    def step(a=0, b=0, en=1, clear=0, reset=0):
        v.append({"clear": clear, "en": en, "a": a, "b": b, "reset": reset})
    step(reset=1, en=0)                                   # start from reset
    for a, b in [(3, 4), (5, -2), (-7, 6), (-8, -9), (127, 1), (-128, 1)]:
        step(a, b)                                        # mixed signs
    step(9, 9, en=0)                                      # en off: hold
    step(9, 9, en=0)
    step(10, 10, clear=1)                                 # clear wins over en
    for _ in range(40):
        step(-128, -128)                                  # +16384 each: passes the 20-bit maximum and wraps
    step(5, 5, clear=1, reset=1)                          # reset with clear
    for _ in range(40):
        step(-128, 127)                                   # -16256 each: wraps the other way
    step(reset=1, en=1, a=7, b=7)                         # reset wins over en
    rng = random.Random(5)                                # fixed seed: the same tests every time
    for _ in range(150):
        r = rng.random()
        step(rng.randint(-128, 127), rng.randint(-128, 127), en=int(rng.random() < 0.8),
             clear=int(r < 0.05), reset=int(0.05 <= r < 0.07))
    return v


def reference(vectors):
    out, acc = [], None
    for v in vectors:
        if v["reset"] or v["clear"]:
            if acc is None and not v["reset"]:
                raise ValueError("vectors must start with reset")
            acc = 0
        elif acc is None:
            raise ValueError("vectors must start with reset")
        elif v["en"]:
            acc = acc + v["a"] * v["b"]
        acc = (acc + (1 << 19)) % (1 << 20) - (1 << 19)   # keep it a 20-bit signed number
        out.append({"acc": acc & ((1 << 20) - 1)})       # the checker compares raw 20-bit patterns
    return out
