"""Level 2 -- 4-bit adder with carry out. All 256 input pairs are tested; the trap is the carry out."""
MODULE = "add4"
INPUTS = {"a": 4, "b": 4}
OUTPUTS = {"sum": 4, "cout": 1}
CLOCKED = False


def vectors():
    return [{"a": a, "b": b} for a in range(16) for b in range(16)]   # all 256 combinations


def reference(vectors):
    return [{"sum": (v["a"] + v["b"]) & 15, "cout": (v["a"] + v["b"]) >> 4} for v in vectors]
