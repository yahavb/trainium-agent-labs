# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot single-trajectory forward progress, including spatial model parallelism."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    trials = json.loads(args.manifest.read_text())
    rows = []
    best = float("inf")
    for index, trial in enumerate(trials):
        reports = [
            json.loads((args.manifest.parent / p).read_text()) for p in trial["reports"]
        ]
        report = reports[0]
        seconds = max(r["eval_range"]["total_forward_wall_seconds"] for r in reports)
        record = seconds < best
        best = min(best, seconds)
        years = report["forecast_timesteps"] * 5 / 365.25
        rows.append(
            {
                "experiment": index,
                "label": trial["label"],
                "logical_cores": trial["logical_cores"],
                "cpu_threads_per_rank": report["cpu_threads"],
                "forward_seconds": seconds,
                "running_best_seconds": best,
                "years_per_minute": years * 60 / seconds,
                "running_best_years_per_minute": years * 60 / best,
                "speed_record": record,
                "calls": report["eval_range"]["calls"],
                "forecast_records": report["forecast_timesteps"],
                "reports": ";".join(trial["reports"]),
            }
        )
    out = args.output_directory
    out.mkdir(parents=True, exist_ok=True)
    with (out / "karpathy_trials.tsv").open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    for stem, metric, best_metric, ylabel in [
        (
            "karpathy_seconds",
            "forward_seconds",
            "running_best_seconds",
            "Single eight-year forward time, seconds (lower is better)",
        ),
        (
            "karpathy_years_per_minute",
            "years_per_minute",
            "running_best_years_per_minute",
            "Simulated years per forward minute (higher is better)",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(18, 8))
        fig.subplots_adjust(left=0.085, right=0.98, bottom=0.29, top=0.87)
        for kept, color, label in [
            (False, "#c9cdd0", "Slower trials"),
            (True, "#2cc978", "Record improvements"),
        ]:
            selected = [r for r in rows if r["speed_record"] == kept]
            if selected:
                ax.scatter(
                    [r["experiment"] for r in selected],
                    [r[metric] for r in selected],
                    s=75 if kept else 45,
                    color=color,
                    edgecolors="#367550" if kept else "none",
                    label=label,
                    zorder=3,
                )
        x = [r["experiment"] for r in rows]
        y = [r[best_metric] for r in rows]
        ax.step(
            x + [x[-1] + 0.45],
            y + [y[-1]],
            where="post",
            color="#60c58f",
            lw=2.5,
            label="Running best",
        )
        for row in rows:
            value = row[metric]
            score = (
                f"{value:.2f} s"
                if metric == "forward_seconds"
                else f"{value:.2f} yr/min"
            )
            offset = (0, -42) if not row["speed_record"] else (0, 14)
            ax.annotate(
                score,
                (row["experiment"], value),
                xytext=offset,
                textcoords="offset points",
                ha="center",
                color="#367550" if row["speed_record"] else "#72777a",
                fontsize=12,
                fontweight="bold",
            )
        values = [r[metric] for r in rows]
        ax.set_ylim(min(values) * 0.76, max(values) * 1.19)
        ax.set_xlim(-0.35, x[-1] + 0.55)
        ax.set_xticks(x, [r["label"] for r in rows], fontsize=10)
        ax.tick_params(axis="x", pad=12)
        ax.set_xlabel("Experiment #", labelpad=12)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"Autoresearch Progress: {len(rows)} Experiments, {sum(r['speed_record'] for r in rows[1:])} Speed Improvements",
            fontsize=18,
            pad=20,
        )
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(
            loc="upper right" if metric == "forward_seconds" else "upper left",
            fontsize=10,
        )
        fig.text(
            0.085,
            0.035,
            "One trajectory throughout · 299 calls / 598 five-day records · 365.25 days/year\nTrials 0–4 and 7–8: 1 core, 8 CPU threads. Trials 5–6: 2 threads/rank; 1 → 4 cores.\nTrial 6 splits two blocks; trial 8 segments 17 UNet layers on ONE core. Combined multicore + segmentation untested.\nCompilation/warmup excluded. Accuracy unverified.",
            fontsize=10,
            color="#59606b",
            linespacing=1.5,
        )
        for ext in ("png", "svg", "pdf"):
            fig.savefig(out / f"{stem}.{ext}", dpi=180)
        svg = out / f"{stem}.svg"
        svg.write_text(
            "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
        )
        plt.close(fig)
    print(
        json.dumps(
            {
                "final_forward_seconds": rows[-1]["forward_seconds"],
                "final_years_per_minute": rows[-1]["years_per_minute"],
                "previous_trial_forward_seconds": rows[-2]["forward_seconds"],
                "speedup_over_previous_trial": rows[-2]["forward_seconds"]
                / rows[-1]["forward_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
