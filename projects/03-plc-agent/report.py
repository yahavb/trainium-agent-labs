"""report.py - compare runs side by side and draw the learning curve.

    python report.py runs/<baseline-dir> runs/<bootstrap-dir> [more dirs...]

Prints one row per run directory and, if matplotlib is installed, writes progress.png:
the running share of specs solved on the FIRST attempt, by position in the spec sequence.
A memoryless agent's curve stays flat; a bootstrapping agent's curve should climb.
"""
import csv
import json
import os
import sys


def load(d):
    with open(os.path.join(d, "summary.json")) as f:
        s = json.load(f)
    with open(os.path.join(d, "learning_curve.csv")) as f:
        curve = list(csv.DictReader(f))
    return s, curve


def pct(x):
    return "-" if x is None else f"{x:.0%}"


def main(dirs):
    rows = [(d, *load(d)) for d in dirs]
    hdr = f"{'mode':<26}{'runs':>6}{'solved':>19}{'round-1':>9}{'rounds':>8}{'calls/solve':>13}{'round-1, 1st->2nd half':>26}"
    print(hdr)
    print("-" * len(hdr))
    for d, s, _ in rows:
        solved = f"{pct(s['solve_rate_mean'])} ({pct(s['solve_rate_min'])}-{pct(s['solve_rate_max'])})"
        halves = f"{pct(s['first_half']['first_round_rate'])} -> {pct(s['second_half']['first_round_rate'])}"
        print(f"{s['mode']:<26}{s['runs']:>6}{solved:>19}{pct(s['first_round_solve_rate']):>9}"
              f"{str(s['mean_rounds_when_solved']):>8}{str(s['generator_calls_per_solve']):>13}{halves:>26}")
    specs = {(s["seed"], s["specs_per_run"]) for _, s, _ in rows}
    if len(specs) > 1:
        print("\nNote: these runs used different seeds or spec counts, so they did not see the same specs.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: skipping progress.png; learning_curve.csv has the numbers)")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for d, s, curve in rows:
        xs, ys, total = [], [], 0.0
        for k, row in enumerate(curve, 1):
            total += float(row["first_round_rate"])
            xs.append(k)
            ys.append(total / k)
        ax.plot(xs, ys, marker="o", markersize=3, label=f"{s['mode']} ({s['runs']} runs)")
    ax.set_xlabel("spec number in the sequence")
    ax.set_ylabel("running share solved on first attempt")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("progress.png", dpi=150)
    print("\nwrote progress.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])x
