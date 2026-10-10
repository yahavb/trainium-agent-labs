"""Level 6 -- 4-entry FIFO with full / empty / count. See spec.txt.

Traps, each covered by the tests: writes ignored when full, reads ignored when empty, read + write in the
same cycle (in the middle, at full, at empty), dout showing the oldest byte straight after the edge (not a
cycle late) and 0 when empty, wrap-around of the internal storage, reset in the middle.
"""
import random
from collections import deque

MODULE = "fifo4"
INPUTS = {"reset": 1, "wr_en": 1, "rd_en": 1, "din": 8}
OUTPUTS = {"dout": 8, "full": 1, "empty": 1, "count": 3}
CLOCKED = True


def vectors():
    v = []
    def step(wr=0, rd=0, din=0, reset=0):
        v.append({"reset": reset, "wr_en": wr, "rd_en": rd, "din": din})
    step(reset=1)
    for d in (0x11, 0x22, 0x33, 0x44):
        step(wr=1, din=d)                         # fill up
    step(wr=1, din=0x55)                          # write when full: ignored
    step(wr=1, rd=1, din=0x66)                    # both when full: only the read
    for _ in range(4):
        step(rd=1)                                # empty it
    step(rd=1)                                    # read when empty: ignored
    step(wr=1, rd=1, din=0x77)                    # both when empty: only the write
    step(wr=1, din=0x88)
    for d in (0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE):
        step(wr=1, rd=1, din=d)                   # both in the middle: count stays, storage wraps round
    step(reset=1, wr=1, din=0xFF)                 # reset wins
    step(wr=1, din=0x01)
    step(wr=1, din=0x02)
    step(reset=1)
    rng = random.Random(6)                        # fixed seed: the same tests every time
    for _ in range(150):
        r = rng.random()
        step(wr=int(rng.random() < 0.55), rd=int(rng.random() < 0.45), din=rng.randint(0, 255),
             reset=int(r < 0.02))
    return v


def reference(vectors):
    out, q = [], None
    for v in vectors:
        if v["reset"]:
            q = deque()
        elif q is None:
            raise ValueError("vectors must start with reset")
        else:
            full, empty = len(q) == 4, len(q) == 0
            do_rd = v["rd_en"] and not empty
            do_wr = v["wr_en"] and not full
            if do_rd:
                q.popleft()
            if do_wr:
                q.append(v["din"])
        out.append({"dout": q[0] if q else 0, "full": int(len(q) == 4), "empty": int(len(q) == 0),
                    "count": len(q)})
    return out


def diagnose(rows):
    """Feedback D for this level: name the CAUSE of the failure in terms of the spec, never the code.

    `rows` is the simulation: one (inputs, expected, got) per clock cycle. Looks for the mistakes the model
    made most (results/TAXONOMY.md): dout that only updates on a read, full/empty a cycle late, count that
    changes when a read and a write happen together, writes accepted when full, reads accepted when empty.
    A rule only fires on a cycle where the design's count was still right just before the edge, so a
    symptom of an earlier bug is not reported as a cause. At most three sentences, or None.
    """
    found = []

    def say(msg):
        if msg not in found:
            found.append(msg)

    for i in range(1, len(rows)):
        v, exp, got = rows[i]
        _, exp_prev, got_prev = rows[i - 1]
        if v["reset"] or got_prev["count"] != exp_prev["count"]:
            continue
        before, wr, rd = exp_prev["count"], v["wr_en"], v["rd_en"]
        count_ok = got["count"] == exp["count"]
        if before == 4 and wr and not rd and not count_ok:
            say("A write while the queue is full must be ignored (count stays 4). Yours accepts it.")
        if before == 4 and wr and rd and not count_ok:
            say("When the queue is full and both wr_en and rd_en are 1, only the read happens (count goes to 3). "
                "Yours also accepts the write.")
        if before == 0 and rd and not wr and not count_ok:
            say("A read while the queue is empty must be ignored (count stays 0). Yours changes the queue.")
        if 0 < before < 4 and wr and rd and not count_ok:
            say("When a read and a write happen in the same cycle (queue neither full nor empty), both take "
                "effect: count stays the same. Yours changes count, so one of the two is lost.")
        if count_ok:
            for flag in ("full", "empty"):
                if got[flag] != exp[flag] and got[flag] == exp_prev[flag]:
                    say(f"`{flag}` is one cycle late: it must describe the queue AFTER this clock edge's read "
                        f"and write (follow the new count), not the count from before the edge.")
            if before == 0 and wr and got["dout"] != exp["dout"]:
                say("dout must show the oldest stored byte at all times, including right after a byte is written "
                    "into an empty queue -- no read is needed for dout to change. Yours does not show it.")
            if exp["count"] == 0 and got["dout"] != 0:
                say("When the queue is empty, dout must be 0.")
        if len(found) >= 3:
            break
    return " ".join(found[:3]) if found else None
