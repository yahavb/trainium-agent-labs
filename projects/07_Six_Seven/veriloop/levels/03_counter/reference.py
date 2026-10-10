"""Level 3 -- 8-bit counter with enable and synchronous reset. Traps: wrap 255 -> 0, enable off holds,
reset has priority over enable, reset mid-count."""
MODULE = "counter8"
INPUTS = {"reset": 1, "en": 1}
OUTPUTS = {"count": 8}
CLOCKED = True


def vectors():
    v = [{"reset": 1, "en": 0}]                      # start from reset
    v += [{"reset": 0, "en": 1}] * 5                 # count up
    v += [{"reset": 0, "en": 0}] * 3                 # enable off: hold
    v += [{"reset": 0, "en": 1}] * 260               # long enough to wrap 255 -> 0
    v += [{"reset": 1, "en": 1}]                     # reset wins over enable
    v += [{"reset": 0, "en": 1}] * 2
    return v


def reference(vectors):
    out, count = [], 0
    for v in vectors:
        if v["reset"]:
            count = 0
        elif v["en"]:
            count = (count + 1) & 255
        out.append({"count": count})
    return out
