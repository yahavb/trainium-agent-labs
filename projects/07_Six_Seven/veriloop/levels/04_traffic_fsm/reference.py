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


NAMES = {RED: "RED", GREEN: "GREEN", YELLOW: "YELLOW"}


def _phases(lights, upto=21):
    """[(colour, cycles)] of the complete phases in lights[:upto]; None marks an undefined light."""
    out, cur, n = [], lights[0], 0
    for x in lights[:upto]:
        if x == cur:
            n += 1
        else:
            out.append((cur, n))
            cur, n = x, 1
    return out


def diagnose(rows):
    """Feedback D for this level: name the CAUSE of the failure in terms of the spec, never the code.

    The first 21 test cycles are a reset followed by no button presses, so the phase lengths the design
    shows there can be measured and compared with the spec directly. If the timing is right, the button
    and `walk` are checked. At most three sentences, or None.
    """
    got = [g["light"] for _, _, g in rows]
    exp = [e["light"] for _, e, _ in rows]
    found = []
    if None in got[:21]:
        return ("Your light is undefined (x) after reset: on a rising edge with reset = 1 every register, "
                "including the light and any counter, must get a defined value.")
    gp, ep = _phases(got), _phases(exp)
    if gp[:4] != ep[:4]:
        order_ok = all(gp[k][0] == ep[k][0] for k in range(min(len(gp), len(ep), 4)))
        if not order_ok:
            seq = " -> ".join(NAMES.get(c, "?") for c, _ in gp[:4])
            return (f"With no button pressed, your light goes {seq}. The order must be RED -> GREEN -> YELLOW -> RED.")
        mine = ", ".join(f"{NAMES[c]} {n}" for c, n in gp[:4])
        want = ", ".join(f"{NAMES[c]} {n}" for c, n in ep[:4])
        found.append(f"Starting from reset with no button pressed, your phases last {mine} cycles; the spec says "
                     f"{want} (the reset cycle counts as RED's first).")
        diffs = [g[1] - e[1] for g, e in zip(gp[:4], ep[:4])]
        if all(d == 1 for d in diffs):
            found.append("Every phase is exactly one cycle too long: the light changes one cycle late each time, so "
                         "whatever decides when a phase ends allows one cycle too many.")
        elif diffs[0] == 1 and all(d == 0 for d in diffs[1:]):
            found.append("Only the first RED after reset is too long: the cycle in which reset is 1 must itself count "
                         "as RED's first cycle.")
        elif all(d == -1 for d in diffs):
            found.append("Every phase is exactly one cycle too short: the light changes one cycle early each time.")
        return " ".join(found[:3])
    # Timing right: report the FIRST button mistake only (after it the design is out of step with the spec,
    # so later differences are symptoms, not causes), plus walk if it is wrong while the light is right.
    for i in range(1, len(rows)):
        v, e, g = rows[i]
        _, ep_, gp_ = rows[i - 1]
        if v["reset"] or gp_["light"] != ep_["light"] or g["light"] == e["light"]:
            continue
        if v["ped"] and ep_["light"] == GREEN:
            found.append(f"On a rising edge where the light is GREEN and ped = 1, the light must turn YELLOW at that "
                         f"edge (green ends early); yours shows {NAMES.get(g['light'], g['light'])}.")
        elif v["ped"]:
            found.append("ped must be ignored while the light is YELLOW or RED; yours reacts to it.")
        elif ep_["light"] == YELLOW:
            found.append("After the button cuts GREEN short, YELLOW must still last its full 2 cycles.")
        break
    if any(g["light"] == e["light"] and g["walk"] != e["walk"] for _, e, g in rows):
        found.append("walk must be 1 exactly when the light is RED, and 0 otherwise.")
    return " ".join(found[:3]) if found else None
