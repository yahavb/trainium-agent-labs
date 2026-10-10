"""Human-readable campaign report rendering."""

from __future__ import annotations

from typing import Any


def render_campaign_report(campaign: dict[str, Any]) -> str:
    baseline = campaign.get("baseline_metrics") or {}
    attempts = campaign.get("attempts") or []
    winner = campaign.get("best_attempt") or {}
    lines = [
        f"# AccelTwin campaign: {campaign.get('id', 'unknown')}",
        "",
        f"Status: **{campaign.get('status', 'unknown')}**",
        f"Objective: minimize predicted p99 TTFT; throughput must meet or exceed baseline.",
        "Evidence labels: request telemetry is measured when available; hardware, power, area, and thermal counterfactuals are modeled.",
        "",
        "## Baseline",
        "",
        f"- Predicted p99 TTFT: {baseline.get('predicted_p99_ttft_ms', baseline.get('ttft_p99_ms', 'n/a'))} ms",
        f"- Throughput: {baseline.get('throughput_tokens_per_sec', baseline.get('throughput_tokens_s', 'n/a'))} tokens/s",
        f"- Memory: {baseline.get('memory_gb', 'n/a')} GB",
        f"- Modeled power: {baseline.get('power_w', 'n/a')} W",
        f"- Modeled peak temperature: {baseline.get('max_temperature_c', 'n/a')} °C",
        f"- Modeled area proxy: {baseline.get('area_mm2', 'n/a')} mm²",
        "",
        "## Attempts",
        "",
        "| Round | Candidate | p99 TTFT (ms) | Throughput (tokens/s) | Gates | Accepted |",
        "|---:|---|---:|---:|---|:---:|",
    ]
    for attempt in attempts:
        metrics = attempt.get("metrics") or {}
        gates = attempt.get("gate_results") or {}
        gate_text = ", ".join(f"{key}={'pass' if value else 'fail'}" for key, value in gates.items()) or "n/a"
        lines.append(
            f"| {attempt.get('round', 'n/a')} | {attempt.get('name', attempt.get('id', 'candidate'))} "
            f"| {metrics.get('predicted_p99_ttft_ms', metrics.get('ttft_p99_ms', 'n/a'))} "
            f"| {metrics.get('throughput_tokens_per_sec', metrics.get('throughput_tokens_s', 'n/a'))} "
            f"| {gate_text} | {'yes' if attempt.get('accepted') else 'no'} |"
        )
        design = attempt.get("design") or {}
        diff = attempt.get("design_diff") or {}
        component_changes = diff.get("components") or {}
        connection_changes = diff.get("connections") or {}
        lines += ["", f"### Round {attempt.get('round', 'n/a')}: {attempt.get('name', 'candidate')}", "",
                  f"Checker feedback: {attempt.get('feedback', 'n/a')}"]
        if component_changes:
            moved = [entry.get("id", "component") for entry in component_changes.get("modified", [])
                     if (entry.get("from") or {}).get("x") != (entry.get("to") or {}).get("x")
                     or (entry.get("from") or {}).get("y") != (entry.get("to") or {}).get("y")]
            added = [entry.get("id", "component") for entry in component_changes.get("added", [])]
            removed = [entry.get("id", "component") for entry in component_changes.get("removed", [])]
            layout = []
            if moved:
                layout.append("moved " + ", ".join(moved))
            if added:
                layout.append("added " + ", ".join(added))
            if removed:
                layout.append("removed " + ", ".join(removed))
            if layout:
                lines.append("Layout changes: " + "; ".join(layout) + ".")
        if connection_changes and any(connection_changes.values()):
            lines.append("Interconnect changes: " + "; ".join(
                f"{kind} {', '.join(item.get('id', 'link') for item in connection_changes.get(kind, []))}"
                for kind in ("added", "removed", "modified") if connection_changes.get(kind)) + ".")
        scalar_changes = diff.get("scalars") or {}
        if scalar_changes:
            lines.append("Architecture parameter changes: " + "; ".join(
                f"{key} {change.get('from')} → {change.get('to')}" for key, change in scalar_changes.items()) + ".")
        components = design.get("components") or []
        if components:
            lines += ["", "Complete proposed chip graph:"]
            for component in components:
                pos = f"({component.get('x', '?')}, {component.get('y', '?')})"
                size = f"{component.get('width', '?')}×{component.get('height', '?')}"
                details = [f"{component.get('type', 'component')} {component.get('id', '?')} at {pos}, {size}"]
                for key in ("compute_tflops", "capacity_gb", "bandwidth_tb_s", "dma_engines"):
                    if component.get(key) is not None:
                        details.append(f"{key}={component[key]}")
                lines.append("- " + "; ".join(details))
        connections = design.get("connections") or []
        if connections:
            lines.append("Data paths:")
            for connection in connections:
                lines.append(f"- {connection.get('id', '?')}: {connection.get('source', '?')} → "
                             f"{connection.get('target', '?')} ({connection.get('bandwidth_tb_s', 'n/a')} TB/s)")
    if not attempts:
        lines.append("| — | No candidates evaluated | — | — | — | — |")
    winner_metrics = winner.get("metrics") or {}
    winner_design = winner.get("design") or campaign.get("baseline_design") or {}
    lines += [
        "",
        "## Result",
        "",
        f"Best candidate: **{winner.get('name', 'baseline')}**",
        f"Predicted p99 TTFT: {winner_metrics.get('predicted_p99_ttft_ms', winner_metrics.get('ttft_p99_ms', baseline.get('predicted_p99_ttft_ms', baseline.get('ttft_p99_ms', 'n/a'))))} ms",
        f"Throughput: {winner_metrics.get('throughput_tokens_s', 'n/a')} tokens/s",
        f"Modeled power / peak temperature / area: {winner_metrics.get('power_w', 'n/a')} W / {winner_metrics.get('max_temperature_c', 'n/a')} °C / {winner_metrics.get('area_mm2', 'n/a')} mm²",
        f"Rounds evaluated: {len(attempts)}",
        "",
    ]
    if winner_design.get("components"):
        lines += ["## Selected chip layout", ""]
        for component in winner_design["components"]:
            lines.append(f"- {component.get('id', '?')} ({component.get('type', 'component')}): "
                         f"position ({component.get('x', '?')}, {component.get('y', '?')}), "
                         f"size {component.get('width', '?')}×{component.get('height', '?')}")
        lines.append("")
    return "\n".join(lines)

