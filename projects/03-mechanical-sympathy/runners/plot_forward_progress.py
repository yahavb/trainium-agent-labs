# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot completed forward-only experiments and their running best wall time."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiments = json.loads(args.manifest.read_text())
    rows = []
    baseline = None
    best = float("inf")
    pending = []
    for index, experiment in enumerate(experiments):
        path = args.manifest.parent / experiment["report"]
        if not path.exists():
            pending.append(f"#{index}: {experiment['label']}")
            continue
        report = json.loads(path.read_text())
        if not report["complete_eval_range"] or report["forecast_metrics_computed"]:
            raise ValueError(f"Not a complete forward-only run: {path}")
        if baseline is None:
            baseline = report
        else:
            for key in (
                "backend",
                "checkpoint_sha256",
                "prepared_case_sha256",
                "torch_version",
                "dtype",
                "cpu_threads",
                "warmup_calls",
                "input_prognostic_shape",
                "input_boundary_shape",
                "output_shape",
                "target_time_start",
                "target_time_end",
                "forecast_timesteps",
                "timing_contract",
            ):
                if report[key] != baseline[key]:
                    raise ValueError(f"Workload mismatch on {key}: {path}")
            for section in ("single_call", "eval_range"):
                if report[section]["calls"] != baseline[section]["calls"]:
                    raise ValueError(f"Call count mismatch: {path}")
            for key in ("visible_cores", "logical_nc_config"):
                if report["neuron"][key] != baseline["neuron"][key]:
                    raise ValueError(f"Hardware allocation mismatch on {key}")
        seconds = report["eval_range"]["total_forward_wall_seconds"]
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"Invalid forward time: {path}")
        improved = seconds < best
        best = min(best, seconds)
        rows.append(
            {
                "experiment": index,
                "label": experiment["label"],
                "forward_seconds": seconds,
                "running_best_seconds": best,
                "speed_record": improved,
                "single_median_ms": 1000 * report["single_call"]["median_wall_seconds"],
                "calls": report["eval_range"]["calls"],
                "host_cpu_seconds": report["eval_range"]["total_host_cpu_seconds"],
                "peak_rss_bytes": report["eval_range"]["process_peak_rss_bytes"],
                "warmup_including_compilation_seconds": report[
                    "warmup_including_compilation_seconds"
                ],
                "compiler_flags": report.get("neuron", {}).get("flags", ""),
                "checkpoint_sha256": report["checkpoint_sha256"],
                "prepared_case_sha256": report["prepared_case_sha256"],
                "source_commit": report["source_commit"],
                "report": str(path),
            }
        )
    if not rows:
        raise ValueError("No completed experiments to plot")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".tsv").open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.subplots_adjust(bottom=0.17, top=0.91, left=0.09, right=0.98)
    for kept, color, label in (
        (False, "#c9cdd0", "Slower trials"),
        (True, "#2cc978", "Record improvements"),
    ):
        selected = [r for r in rows if r["speed_record"] == kept]
        ax.scatter(
            [r["experiment"] for r in selected],
            [r["forward_seconds"] for r in selected],
            s=65 if kept else 28,
            color=color,
            edgecolors="#345447" if kept else "none",
            label=label,
            zorder=3,
        )
    x = [r["experiment"] for r in rows]
    y = [r["running_best_seconds"] for r in rows]
    ax.step(
        x + [max(len(experiments) - 1, x[-1]) + 0.25],
        y + [y[-1]],
        where="post",
        color="#60c58f",
        linewidth=2.5,
        label="Running best",
    )
    for row in rows:
        ax.annotate(
            f"{row['label']} ({row['forward_seconds']:.2f} s)",
            (row["experiment"], row["forward_seconds"]),
            xytext=(9, 9),
            textcoords="offset points",
            fontsize=11,
            rotation=28,
            rotation_mode="anchor",
            ha="left",
            va="bottom",
            color="#367550" if row["speed_record"] else "#666666",
        )
    ax.set_xlim(-0.1, max(len(experiments) - 1, x[-1]) + 0.65)
    spread = max(max(y) - min(y), max(y) * 0.12)
    observed = [r["forward_seconds"] for r in rows]
    ax.set_ylim(max(0, min(observed) - spread * 0.35), max(observed) + spread * 0.6)
    ax.set_xticks(range(len(experiments)))
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Full evaluation forward seconds (lower is better)")
    ax.set_title(
        f"Autoresearch Progress: {len(rows)} Experiments, "
        f"{sum(r['speed_record'] for r in rows[1:])} Speed Improvement(s)",
        fontsize=16,
        pad=12,
    )
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    footer = (
        f"Matched workload: {rows[0]['calls']} calls · device-resident inputs · "
        "compilation/warmup excluded\nSpeed records only: BF16 changes arithmetic; "
        "prediction accuracy has not been evaluated."
    )
    if pending:
        footer += "\nPending (no invented measurements): " + "; ".join(pending)
    fig.text(0.09, 0.035, footer, fontsize=10, color="#59606b", linespacing=1.6)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=180)
    svg = args.output.with_suffix(".svg")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
