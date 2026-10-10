"""Draws the charts in RESULTS.md from the measured numbers (run logs, held-out check, attempt logs).

    python analysis/make_charts.py        # writes assets/chart_*.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(OUT, exist_ok=True)
GREY, GREEN, RED, BLUE = "#b0b0b0", "#2e9e5b", "#d9534f", "#3b7dd8"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150, facecolor="white")
    plt.close(fig)


# 0. The loop, as it runs now --------------------------------------------------------------------
from matplotlib.patches import FancyBboxPatch


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor=color, edgecolor="#333", linewidth=1.2))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=11.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.72, body, ha="center", va="top", fontsize=9, linespacing=1.35)


def arrow(ax, a, b, text="", color="#333", rad=0.0):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=1.6, color=color,
                                                     connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.18, text, ha="center", fontsize=8.5, color=color)


fig, ax = plt.subplots(figsize=(15, 7.2))
ax.set_xlim(0, 15); ax.set_ylim(0, 7.2); ax.axis("off")
box(ax, 0.2, 4.1, 2.7, 2.3, "1. Task", "NumPy reference\n+ test shapes\n(nkibench.py, 11 levels)", "#eef3fb")
box(ax, 3.3, 4.1, 2.9, 2.3, "2. Prompt", "level hints, the agent's own\nsolved kernels as blocks,\nreal-API card; level 8:\n13-stage plan", "#eef3fb")
box(ax, 6.6, 4.1, 2.6, 2.3, "3. Model", "Qwen3-8B on Trainium2\nvLLM, whole chip (TP=4)\n4 samples per round\n~39 tokens/s each", "#fff4e0")
box(ax, 9.6, 4.1, 2.5, 2.3, "4. Static check", "lint.py: EVERY memory,\nshape and API mistake\nat once, with the\nAWS doc rule", "#fdecea")
box(ax, 12.4, 4.1, 2.4, 2.3, "5. Checker", "NKI simulator: shapes,\nnumbers (0.02 x RMS),\ndata-movement bar,\nhardware hazards", "#fdecea")
arrow(ax, (2.9, 5.25), (3.3, 5.25)); arrow(ax, (6.2, 5.25), (6.6, 5.25)); arrow(ax, (9.2, 5.25), (9.6, 5.25))
arrow(ax, (12.1, 5.25), (12.4, 5.25), "clean")
box(ax, 5.2, 1.6, 5.4, 1.75, "6. Feedback, then retry (up to 8 rounds)",
    "names the fix, with the error's own numbers - names the wrong step\n(scale inverted, P^T V ...) - picks the sample with the fewest problems\n- says so when a reply changed nothing or was cut off", "#f3f0fb")
arrow(ax, (10.85, 4.1), (9.6, 3.35), "problems found", RED, rad=0.0)
arrow(ax, (13.6, 4.1), (10.6, 2.9), "wrong", RED, rad=-0.15)
arrow(ax, (5.2, 2.5), (4.75, 4.1), "retry", "#6a4fb3", rad=-0.3)
box(ax, 11.3, 0.15, 3.5, 1.7, "7. SOLVED only if every shape passes",
    "then re-checked: 228 unseen / hostile\ninputs, 13 seeded bugs, and the\nreal chip (37/37)", "#e6f4ea")
arrow(ax, (14.3, 4.1), (13.6, 1.85), "all shapes pass", GREEN, rad=0.0)
ax.text(0.2, 6.85, "The kernel agent loop (team 20)", fontsize=15, weight="bold")
save(fig, "agent_loop.png")

# 1. Before vs after: share of runs solved per level -----------------------------------------
levels = ["1\npooling", "2\ntranspose", "3\nmatmul\n1 tile", "4\nmatmul\ntiled", "5\nless\ntraffic",
          "6\nless\ntraffic", "7\nleast\ntraffic", "8\nattention", "9\ntranspose\n(new)", "10\nsoftmax\n(new)",
          "11\nscores\n(new)"]
before = [0, 3 / 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # level 2: the repo measured 2/5 to 4/5 with the original agent
after = [2 / 2, 5 / 6, 2 / 2, 5 / 6, 2 / 2, 2 / 2, 2 / 2, 2 / 2, 2 / 2, 2 / 2, 4 / 4]
after_lbl = ["2/2", "5/6", "2/2", "5/6", "2/2", "2/2", "2/2", "2/2*", "2/2", "2/2", "4/4"]
fig, ax = plt.subplots(figsize=(13, 5))
x = range(len(levels))
ax.bar([i - 0.2 for i in x], before, 0.4, color=GREY, label="before our changes")
ax.bar([i + 0.2 for i in x], after, 0.4, color=GREEN, label="after")
for i, t in enumerate(after_lbl):
    ax.text(i + 0.2, after[i] + 0.03, t, ha="center", fontsize=11)
for i in range(len(levels)):
    if i in (4, 5, 6):
        ax.text(i - 0.2, 0.03, "never\nreached", ha="center", fontsize=8, color="#666")
    elif i == 1:
        ax.text(i - 0.2, before[i] + 0.03, "2-4/5", ha="center", fontsize=10, color="#555")
    else:
        ax.text(i - 0.2, 0.02, "0", ha="center", fontsize=10, color="#555")
ax.set_xticks(list(x)); ax.set_xticklabels(levels, fontsize=10)
ax.set_ylim(0, 1.35); ax.set_ylabel("share of runs solved")
ax.set_title("Before vs after: the same model now solves all 11 levels  (* level 8 with an explicit stage plan)")
ax.legend(loc="upper center", ncol=2, frameon=False)
save(fig, "chart_1_before_after.png")

# 2. How fast: round of the first solve, per solved run ---------------------------------------
solves = {"1": [4, 4], "2": [0, 2, 0, 0, 1], "3": [1, 2], "4": [1, 1], "5": [0, 0], "6": [0, 1], "7": [1, 1], "8": [0, 0],
          "9": [0, 0], "10": [0, 0], "11": [1, 1, 1, 1]}
fig, ax = plt.subplots(figsize=(10, 4.5))
for i, (lv, rounds) in enumerate(solves.items()):
    for j, r in enumerate(rounds):
        ax.scatter(i + (j - (len(rounds) - 1) / 2) * 0.12, r + 1, s=90, color=BLUE)
ax.set_xticks(range(len(solves))); ax.set_xticklabels([f"level {k}" for k in solves])
ax.set_ylabel("attempt number that solved it"); ax.set_ylim(0, 6); ax.set_yticks(range(1, 6))
ax.set_title("How quickly each level was solved (each dot = one run; lower is faster)")
save(fig, "chart_2_speed.png")

# ---- everything below is computed from the attempt log in this repo -------------------------------
import collections, io, json, re, statistics, tarfile
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "logs", "attempts.tar.gz")
L8SUM = json.load(open(os.path.join(HERE, "..", "logs", "level8_failures_summary.json")))
FILES = {}
with tarfile.open(LOG) as tf:
    for m in tf.getmembers():
        if m.isfile() and m.name.endswith(".jsonl") and not os.path.basename(m.name).startswith("._"):
            rows = []
            for line in io.TextIOWrapper(tf.extractfile(m), encoding="utf-8"):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
            FILES[m.name] = rows
ALL = [r for rows in FILES.values() for r in rows]


def rounds_of(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r.get("session"), r.get("run", 0), r["round"])].append(r)
    return by


# 3. Where the attempts were wasted ------------------------------------------------------------
same = rnds = echo = repairs = invented = repeat = sel_rounds = 0
for rows in FILES.values():
    by = rounds_of(rows)
    for key, rs in by.items():
        if len(rs) >= 2:
            rnds += 1; same += len({r["code"].strip() for r in rs}) == 1
    sel = sorted(((k, r) for k, rs in by.items() for r in rs if r.get("selected")), key=lambda x: (str(x[0][0]), x[0][1], x[0][2]))
    for (k0, a), (k1, b) in zip(sel, sel[1:]):
        if k0[:2] == k1[:2]:
            sel_rounds += 1
            norm = lambda f: re.sub(r"\d+", "N", f or "")[:200]
            repeat += norm(a["feedback"]) == norm(b["feedback"])
            others = by[k1]
            repairs += len(others); echo += sum(o["code"].strip() == a["code"].strip() for o in others)
fails = [r for r in ALL if r.get("reward", 0) < 0.999]
invented = sum(bool(re.search(r"has no attribute|unexpected keyword|does not exist|has no argument", r.get("feedback", "")))
               for r in fails)
waste = [("all samples in a round\nwere the same code", 100 * same / max(rnds, 1)),
         ("a retry returned the code\nit was given, unchanged", 100 * echo / max(repairs, 1)),
         ("round failed exactly like\nthe previous one", 100 * repeat / max(sel_rounds, 1)),
         ("failure was an invented\nAPI call", 100 * invented / max(len(fails), 1))]
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.barh([w[0] for w in waste][::-1], [w[1] for w in waste][::-1], color=RED)
for i, (_, v) in enumerate(waste[::-1]):
    ax.text(v + 1, i, f"{v:.0f}%", va="center")
ax.set_xlim(0, 100); ax.set_xlabel("% of rounds / retries / failed attempts")
ax.set_title(f"Where the effort went ({len(ALL):,} logged attempts, failed level-8 ones excluded)")
save(fig, "chart_3_waste.png")
print("waste:", [(w[0].replace(chr(10), " "), round(w[1])) for w in waste], "| rounds", rnds, "repairs", repairs)

# 4. Failure taxonomy per level -----------------------------------------------------------------
CATS = [("wrong shape / layout", r"same number of elements|WRONG SHAPE|partition dimension|exceeds maximum|^copying|"
         r"contraction|transposing|sums over the FIRST|stationary\^T|is 1-D|rows; an on-chip"),
        ("invented API call", r"has no attribute|unexpected keyword|does not exist|has no argument|not callable|is a dtype"),
        ("data in the wrong memory", r"must be in \[|is in (psum|sbuf|hbm), must be|cannot touch psum"),
        ("out of bounds / reshape", r"Out-of-bound|cannot reshape|broadcast"),
        ("too much data movement", r"TOO MUCH HBM TRAFFIC|MORE THAN NECESSARY|more HBM traffic"),
        ("wrong numbers", r"NUMERICAL MISMATCH|NON-FINITE|DIAGNOSIS"),
        ("other", r"")]
L8MAP = {"contraction": 0, "copy between": 0, "never returns": 6, "wrong memory": 2, "AssertionError": 6,
         "ValueError": 3, "TypeError": 1, "NameError": 1, "other": 6}


def cat_of(fb):
    first = fb.split("\n- ", 1)[1].split(" -- ", 1)[-1] if fb.startswith("A static check") and "\n- " in fb else fb
    for i, (_, pat) in enumerate(CATS):
        if pat and re.search(pat, first, re.M):
            return i
    return len(CATS) - 1


per = collections.defaultdict(lambda: [0] * len(CATS))
for r in fails:
    if r.get("level") != 8:
        per[r["level"]][cat_of(r.get("feedback", ""))] += 1
for name, n in L8SUM["by_failure"].items():
    per[8][next((v for k, v in L8MAP.items() if k in name), len(CATS) - 1)] += n
levels_sorted = sorted(per)
colors = ["#f0ad4e", "#d9534f", "#5bc0de", "#9370db", "#3b7dd8", "#5cb85c", "#cccccc"]
fig, ax = plt.subplots(figsize=(11, 5))
bottom = [0] * len(levels_sorted)
for c, ((name, _), col) in enumerate(zip(CATS, colors)):
    vals = [per[lv][c] for lv in levels_sorted]
    ax.bar([f"level {lv}" for lv in levels_sorted], vals, bottom=bottom, color=col, label=name)
    bottom = [b + v for b, v in zip(bottom, vals)]
for i, t in enumerate(bottom):
    ax.text(i, t + 8, f"{t:,}", ha="center", fontsize=9)
ax.set_ylabel("failed attempts"); ax.legend(frameon=False, fontsize=9)
ax.set_title(f"Failure taxonomy: what each failed attempt got wrong first ({sum(bottom):,} failures)")
save(fig, "chart_4_failures.png")
print("taxonomy:", {lv: dict(zip([c[0] for c in CATS], per[lv])) for lv in levels_sorted})

# 6. Speed of the loop: model server on half the chip (TP=2) vs the whole chip (TP=4), level 11 ---
def speed(match):
    tps, rnd = [], []
    for name, rows in FILES.items():
        if re.search(match, name):
            t = [r for r in rows if r.get("completion_tokens") and r.get("seconds")]
            tps += [r["completion_tokens"] / r["seconds"] for r in t]
            for rs in rounds_of(t).values():
                rnd.append(max(r["seconds"] for r in rs))
    return statistics.median(tps), statistics.median(rnd), len(tps)
before = speed(r"attempts-l11-lint\d")
after = speed(r"attempts-l11-(tp4|device-safe)")
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, i, lab in ((axs[0], 0, "tokens / s per request"), (axs[1], 1, "seconds per round")):
    vals = [before[i], after[i]]
    ax.bar(["half the chip\n(TP=2)", "whole chip\n(TP=4)"], vals, color=[GREY, GREEN])
    for j, v in enumerate(vals):
        ax.text(j, v * 1.02, f"{v:.1f}" if i == 0 else f"{v:.0f} s", ha="center")
    ax.set_title(lab)
fig.suptitle(f"Same model, same level 11: the model server on the whole chip (median of {before[2]} and {after[2]} requests)")
save(fig, "chart_6_inference.png")
print("speed before/after:", before, after)

# 5. Unseen tests: share of held-out cases passed per level -----------------------------------
hold = {"1": (28, 29), "2": (24, 25), "3": (17, 17), "4": (22, 26), "5": (22, 26), "6": (22, 26),
        "7": (22, 26), "9": (23, 23), "10": (30, 30)}
fig, ax = plt.subplots(figsize=(10, 4.5))
names = [f"level {k}" for k in hold]
passed = [p for p, t in hold.values()]; failed = [t - p for p, t in hold.values()]
ax.bar(names, passed, color=GREEN, label="passed")
ax.bar(names, failed, bottom=passed, color=RED, label="failed (shape not supported)")
for i, (p, t) in enumerate(hold.values()):
    ax.text(i, t + 0.5, f"{p}/{t}", ha="center", fontsize=10)
ax.set_ylim(0, max(t for _, t in hold.values()) + 4)
ax.set_ylabel("test cases never seen before"); ax.legend(frameon=True, loc="lower right")
ax.set_title("Tested on 228 cases the agent never saw: 210 pass, none gave wrong numbers")
save(fig, "chart_5_unseen.png")
print("charts written to", os.path.abspath(OUT))
