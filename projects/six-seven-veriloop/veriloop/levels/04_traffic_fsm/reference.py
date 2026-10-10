"""Level 4 -- traffic-light state machine with a pedestrian button. See spec.txt.

The tricky parts, each covered by the test inputs below: exact phase lengths (off-by-one cycle counts),
reset counting as RED's first cycle, the pedestrian cutting GREEN short on any of its cycles, the button
being ignored in YELLOW and RED, a button held down for a long time, and reset in the middle of a phase.
"""
import random

MODULE = "traffic_light"
INPUTS = {"reset": 1, "ped": 1}
OUTPUTS = {"light": 2, "walk": 1}
CLOCKED = True
# Optional: names for output values, shown in feedback C ("YELLOW (2)" rather than "2").
LABELS = {"light": {0: "RED", 1: "GREEN", 2: "YELLOW"}}

RED, GREEN, YELLOW = 0, 1, 2
CYCLES = {RED: 3, GREEN: 4, YELLOW: 2}
NEXT = {RED: GREEN, GREEN: YELLOW, YELLOW: RED}


def vectors():
    v = []
    def run(n, ped=0):
        v.extend({"reset": 0, "ped": ped} for _ in range(n))
    v.append({"reset": 1, "ped": 0})          # start from reset
    run(20)                                    # two full cycles, no pedestrian
    # press ped on each cycle of GREEN in turn (1st, 2nd, 3rd, 4th green cycle), one press per full cycle
    for k in range(4):
        v.append({"reset": 1, "ped": 0})      # resync: RED cycle 1
        run(2)                                 # RED cycles 2-3
        run(k)                                 # k GREEN cycles without the button
        run(1, ped=1)                          # press on GREEN cycle k+1
        run(6)                                 # let it finish YELLOW and come round
    run(3, ped=1)                              # button during YELLOW/RED: must be ignored
    run(25, ped=1)                             # button held down: GREEN lasts one cycle each time
    run(5)
    v.append({"reset": 1, "ped": 0})          # reset in the middle of a phase
    run(4)
    v.append({"reset": 1, "ped": 1})          # reset wins over the button
    rng = random.Random(4)                     # fixed seed: the same tests every time
    for _ in range(150):
        v.append({"reset": int(rng.random() < 0.02), "ped": int(rng.random() < 0.25)})
    return v


def reference(vectors):
    out, state, shown = [], None, 0
    for v in vectors:
        if v["reset"]:
            state, shown = RED, 1
        elif state is None:
            raise ValueError("vectors must start with reset")
        elif state == GREEN and v["ped"]:
            state, shown = YELLOW, 1
        elif shown < CYCLES[state]:
            shown += 1
        else:
            state, shown = NEXT[state], 1
        out.append({"light": state, "walk": int(state == RED)})
    return out
