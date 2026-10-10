"""Level 1 -- 4-to-1 multiplexer. The warm-up: proves the loop works. All 64 input combinations tested."""
MODULE = "mux4"
INPUTS = {"d": 4, "sel": 2}
OUTPUTS = {"y": 1}
CLOCKED = False


def vectors():
    return [{"d": d, "sel": s} for d in range(16) for s in range(4)]


def reference(vectors):
    return [{"y": (v["d"] >> v["sel"]) & 1} for v in vectors]
