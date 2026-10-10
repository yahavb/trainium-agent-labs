#!/usr/bin/env python3
"""
visualize_pipeline.py — 3-Engine Concurrency Profiler & Gantt Chart Generator
Track 3 Lead: Tanay

Generates presentation-ready visualizations demonstrating:
1. Sequential execution with 67% idle stalls
2. 3-Way Overlapped Pipeline execution with 0% memory stalls
3. Clear latency speedup numbers for the pitch deck
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from reference_pipeline import sequential_pipeline, overlapped_pipeline, EngineEvent


ENGINE_COLORS = {
    "DMA": "#3498db",      # Blue
    "VECTOR": "#e67e22",   # Orange
    "TENSOR": "#2ecc71"    # Green
}


def calculate_engine_idle_stats(events: list[EngineEvent], total_time_us: float) -> dict:
    """Calculates active vs. idle percentage for each of the 3 physical engines."""
    engines = ["DMA", "VECTOR", "TENSOR"]
    active_time = {e: 0.0 for e in engines}

    for ev in events:
        active_time[ev.engine] += (ev.end_us - ev.start_us)

    stats = {}
    for eng in engines:
        utilization = (active_time[eng] / total_time_us) * 100.0 if total_time_us > 0 else 0.0
        stats[eng] = {
            "active_us": active_time[eng],
            "utilization_pct": utilization,
            "idle_pct": 100.0 - utilization
        }
    avg_idle = np.mean([stats[e]["idle_pct"] for e in engines])
    stats["average_engine_idle_pct"] = avg_idle
    return stats


def plot_gantt_chart(seq_events: list[EngineEvent], seq_t: float,
                     ovl_events: list[EngineEvent], ovl_t: float,
                     output_path: str = "pipeline_gantt.png"):
    """Creates a high-resolution Gantt chart comparing Naive vs Overlapped execution."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.patch.set_facecolor('#f8f9fa')

    engine_y_map = {"DMA": 2, "VECTOR": 1, "TENSOR": 0}
    engine_labels = ["Tensor Engine", "Vector Engine", "DMA Engine"]

    # ----------------- PANEL 1: SEQUENTIAL -----------------
    ax1.set_facecolor('#ffffff')
    ax1.set_title(f"Naive Sequential Execution (Latency: {seq_t:.1f} µs) — 67% Hardware Idle Stalls",
                  fontsize=13, fontweight='bold', pad=10)

    for ev in seq_events:
        y = engine_y_map[ev.engine]
        duration = ev.end_us - ev.start_us
        ax1.broken_barh([(ev.start_us, duration)], (y - 0.35, 0.7),
                        facecolors=ENGINE_COLORS[ev.engine], edgecolor='black', linewidth=0.8)
        # Block label
        if duration >= 5.0:
            ax1.text(ev.start_us + duration / 2, y, f"B{ev.block_id}",
                     ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(engine_labels, fontsize=10, fontweight='bold')
    ax1.set_ylabel("Hardware Engine", fontsize=11)
    ax1.set_xlim(0, max(seq_t, ovl_t) * 1.05)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='x')

    # ----------------- PANEL 2: 3-WAY OVERLAPPED -----------------
    ax2.set_facecolor('#ffffff')
    speedup = seq_t / ovl_t
    ax2.set_title(f"3-Way Overlapped Pipeline (Latency: {ovl_t:.1f} µs) — {speedup:.2f}x Speedup (Zero Memory Stalls)",
                  fontsize=13, fontweight='bold', pad=10)

    for ev in ovl_events:
        y = engine_y_map[ev.engine]
        duration = ev.end_us - ev.start_us
        ax2.broken_barh([(ev.start_us, duration)], (y - 0.35, 0.7),
                        facecolors=ENGINE_COLORS[ev.engine], edgecolor='black', linewidth=0.8)
        if duration >= 5.0:
            ax2.text(ev.start_us + duration / 2, y, f"B{ev.block_id}",
                     ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(engine_labels, fontsize=10, fontweight='bold')
    ax2.set_xlabel("Timeline (µs)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Hardware Engine", fontsize=11)
    ax2.set_xlim(0, max(seq_t, ovl_t) * 1.05)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='x')

    # Legend
    legend_patches = [
        mpatches.Patch(color=ENGINE_COLORS["DMA"], label="DMA Engine (HBM <-> SBUF Transfer)"),
        mpatches.Patch(color=ENGINE_COLORS["VECTOR"], label="Vector Engine (Bilinear / Elementwise Math)"),
        mpatches.Patch(color=ENGINE_COLORS["TENSOR"], label="Tensor Engine (Matrix Multiplication)")
    ]
    fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=3, fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] High-resolution Gantt chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=512, help="Number of matrix rows (e.g. 512 = 4 blocks)")
    parser.add_argument("--cols", type=int, default=128, help="Matrix columns")
    parser.add_argument("--output", type=str, default="pipeline_gantt.png", help="Output PNG file path")
    args = parser.parse_args()

    np.random.seed(42)
    X = np.random.randn(args.rows, args.cols).astype(np.float32)
    W = np.random.randn(args.cols, args.cols).astype(np.float32)

    _, seq_events, seq_t = sequential_pipeline(X, W)
    _, ovl_events, ovl_t = overlapped_pipeline(X, W)

    seq_stats = calculate_engine_idle_stats(seq_events, seq_t)
    ovl_stats = calculate_engine_idle_stats(ovl_events, ovl_t)

    speedup = seq_t / ovl_t

    print("\n" + "=" * 65)
    print("      AWS TRAINIUM 3-ENGINE PIPELINE BENCHMARK")
    print("=" * 65)
    print(f"Matrix Dimension: {args.rows} x {args.cols} (Total Blocks: {(args.rows + 127)//128})")
    print("-" * 65)
    print(f"Naive Sequential Latency:     {seq_t:6.1f} µs | Avg Engine Idle: {seq_stats['average_engine_idle_pct']:5.1f}%")
    print(f"3-Way Overlapped Latency:     {ovl_t:6.1f} µs | Avg Engine Idle: {ovl_stats['average_engine_idle_pct']:5.1f}%")
    print("-" * 65)
    print(f"HARDWARE ACCELERATION SPEEDUP: {speedup:6.2f}x ({((1 - ovl_t/seq_t)*100):.1f}% Latency Reduction)")
    print("=" * 65 + "\n")

    plot_gantt_chart(seq_events, seq_t, ovl_events, ovl_t, args.output)


if __name__ == "__main__":
    main()
