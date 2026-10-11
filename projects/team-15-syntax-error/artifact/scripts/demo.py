#!/usr/bin/env python3
"""
demo.py -- the <1-minute live demo. Replays REAL logged runs (committed under runs/), no pod needed.

    python scripts/demo.py            # press Enter between the three screens
    python scripts/demo.py --fast     # no pauses

Screen 1  the same bug, two checkers: what upstream said vs what ours says, and what happened next
Screen 2  the cleanest A/B of the day: level 5, control vs treatment, round by round
Screen 3  the scoreboard: baseline vs final agent, from the logs
"""

import glob
import json
import re
import sys

FAST = "--fast" in sys.argv
B, D, G, R, Y, X = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def load(tag):
    path = sorted(glob.glob(f"runs/[0-9]*/{tag}.jsonl"))[0]
    return [json.loads(l) for l in open(path, encoding="utf-8") if '"claim"' not in l]


def pause():
    if not FAST:
        input(f"{D}  [Enter]{X}")


def wrap(text, width=96, indent="    "):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) > width:
            out.append(indent + line)
            line = ""
        line += w + " "
    out.append(indent + line)
    return "\n".join(out)


def screen1():
    print(f"\n{B}1. SAME BUG, TWO CHECKERS{X}  (level 4: tiled matmul, Qwen3-8B, [sim])\n")
    up = load("skel_L4_r0")[0]
    ours = load("c1_L4_r0")
    line = next(l for l in up["code"].splitlines() if "lhsT[" in l and "dma_copy" in l)
    print(f"  the model wrote:  {R}{line.strip()}{X}")
    print(f"  {D}lhsT has shape [K, M] -- the axes are swapped. On square tiles this does NOT crash.{X}\n")
    up_fb = up["feedback"].split(" On ", 1)[-1]
    up_fb = re.sub(r"at index .*?\n", "", up_fb)
    print(f"  {Y}upstream's message{X} {D}(same skeleton -- only the message differs):{X}")
    print(wrap(up_fb.replace("\n", " ")[:330]))
    print(f"  {D}-> it rewrote the same line, 5 runs of 5, 0 solves{X}\n")
    cause = ours[0]["feedback"][ours[0]["feedback"].index("CAUSE"):]
    print(f"  {G}our checker (reads the code):{X}")
    print(wrap(cause[:330]))
    solved = ours[1]
    print(f"\n  {B}next round: reward {solved['reward']:.2f} -- SOLVED{X}   {D}(5 runs of 5, every one on round 2){X}")


def screen2():
    print(f"\n{B}2. ONE SENTENCE, A/B{X}  (level 5: same result, must move <= 1.90x the minimum HBM bytes)\n")
    for tag, name, col in (("c2L5_L5_r0", "control  (upstream hint)", Y), ("c3L5_L5_r0", "treatment (our hint)   ", G)):
        rows = load(tag)
        traj = "  ".join(f"{r['reward']:.2f}" for r in rows)
        print(f"  {col}{name}{X}  rounds: {traj}")
    c = load("c2L5_L5_r0")[1]["feedback"]
    t = load("c3L5_L5_r0")[1]["feedback"]
    print(f"\n  {Y}upstream:{X} " + re.search(r"Hoist the operand loads[^.]*\.", c).group(0))
    ours = re.search(r"the lhsT load is inside the n loop.*?re-reads the same lhsT tiles\.", t, re.S).group(0)
    print(f"  {G}ours:{X}     " + " ".join(ours.split()) + " Move it OUT of the n loop.")
    print(f"\n  {B}control 0/5 (stuck at 2.00x the floor)   treatment 5/5{X}   {D}same seat, interleaved runs{X}")


def rate(prefix, lv):
    best = {}
    for p in glob.glob(f"runs/[0-9]*/{prefix}_L{lv}*.jsonl"):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if "claim" in r:
                continue
            k = (r.get("tag"), r.get("run", 0))
            best[k] = max(best.get(k, 0), r["reward"])
    if not best:
        return "  -  "
    k = sum(1 for v in best.values() if v >= 0.999)
    return f"{k}/{len(best)}"


def screen3():
    print(f"\n{B}3. THE SCOREBOARD{X}  (NKI ladder, n=5 per level, [sim]; same model all day)\n")
    print("               " + "  ".join(f"L{l:<4}" for l in range(1, 9)))
    print("  upstream     0/5    4/5    0/5    0/5    -      -      -      -     (STATE.md)")
    print("  baseline     " + "  ".join(f"{rate('base', l):<5}" for l in range(1, 5)) + "  -      -      -      -")
    print(f"  {G}final (C10){X}  " + "  ".join(f"{rate('c10', l):<5}" for l in range(1, 9)))
    print(f"\n  {D}+ Stage A NumPy ladder 9/10 · agent L3/L4/L5 kernels correct on the Trainium2 chip [device]"
          f" · calibration Brier 0.030{X}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    screen1(); pause(); screen2(); pause(); screen3()
