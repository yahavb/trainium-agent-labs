#!/usr/bin/env python3
"""make_charts.py — build SVG charts from the benchmark outputs in results/.

Reads results/bench_baseline.txt and results/bench_dflash2*.txt (output of bench_dflash2.py),
writes results/throughput.svg and results/acceptance.svg, and prints a markdown table.
Standard library only.
"""
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def parse(path):
    text = open(path, encoding="utf-8").read()
    m = re.search(r"overall: (\d+) tokens in ([\d.]+)s = ([\d.]+) tok/s \(per-request median ([\d.]+)", text)
    if not m:
        return None
    k = re.search(r"(\d+) draft tokens", text.split("\n", 3)[1] if "\n" in text else "")
    pos = {int(p): int(float(v)) for p, v in re.findall(r'position="(\d+)"\} = ([\d.]+)', text)}
    acc = re.search(r"acceptance rate: ([\d.]+)% \((\d+)/(\d+)", text)
    drafts = re.search(r"spec_decode_num_drafts_total\{[^}]*\} = ([\d.]+)", text)
    return {
        "file": os.path.basename(path),
        "k": int(k.group(1)) if k else (0 if "baseline" in path else None),
        "tps": float(m.group(3)),
        "median": float(m.group(4)),
        "acc_rate": float(acc.group(1)) if acc else None,
        "accepted": int(acc.group(2)) if acc else None,
        "drafted": int(acc.group(3)) if acc else None,
        "steps": int(float(drafts.group(1))) if drafts else None,
        "per_pos": [pos[i] for i in sorted(pos)],
    }


def bar_chart(rows, base):
    w, left, bar_h, gap, top = 680, 200, 34, 18, 80
    h = top + len(rows) * (bar_h + gap) + 50
    vmax = max(r["tps"] for r in rows) * 1.25
    scale = (w - left - 140) / vmax
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        'font-family="Helvetica, Arial, sans-serif" role="img" aria-labelledby="t d">',
        '<title id="t">Qwen3-8B decode throughput on Trainium 2 vs number of DFlash2 draft tokens</title>',
        '<desc id="d">' + "; ".join(
            f'{("baseline" if r["k"] == 0 else str(r["k"]) + " draft tokens")}: {r["tps"]:.1f} tokens per second'
            for r in rows) + "</desc>",
        f'<rect width="{w}" height="{h}" fill="#fff"/>',
        f'<text x="{w/2}" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#111">'
        'Qwen3-8B on Trainium 2 — decode throughput</text>',
        f'<text x="{w/2}" y="52" text-anchor="middle" font-size="12" fill="#444">'
        '1 request, greedy, 128 new tokens, TP=2 on one trn2 chip, same server config</text>',
        f'<line x1="{left}" y1="{top - 8}" x2="{left}" y2="{h - 40}" stroke="#333"/>',
    ]
    for i, r in enumerate(rows):
        y = top + i * (bar_h + gap)
        label = "Baseline (no drafts)" if r["k"] == 0 else f'DFlash2, {r["k"]} draft tokens'
        color = "#6b7280" if r["k"] == 0 else "#2563eb"
        speed = r["tps"] / base
        bw = r["tps"] * scale
        out.append(f'<text x="{left - 10}" y="{y + bar_h * 0.65}" text-anchor="end" font-size="14" fill="#111">{label}</text>')
        out.append(f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="{bar_h}" fill="{color}"/>')
        weight = ' font-weight="bold"' if r["k"] else ""
        out.append(f'<text x="{left + bw + 8:.1f}" y="{y + bar_h * 0.65}" font-size="14" fill="#111"'
                   f'{weight}>{r["tps"]:.1f} tok/s ({speed:.2f}x)</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def acceptance_chart(rows):
    rows = [r for r in rows if r["k"] and r["per_pos"] and r["steps"]]
    if not rows:
        return None
    kmax = max(r["k"] for r in rows)
    w, h, left, right, top, bottom = 680, 340, 70, 150, 70, 60
    pw, ph = w - left - right, h - top - bottom
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c"]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        'font-family="Helvetica, Arial, sans-serif" role="img" aria-labelledby="t d">',
        '<title id="t">Share of DFlash2 draft steps where draft position N was accepted</title>',
        '<desc id="d">' + "; ".join(
            f'{r["k"]} drafts: ' + ", ".join(f'position {i+1} {100*v/r["steps"]:.0f}%' for i, v in enumerate(r["per_pos"]))
            for r in rows) + "</desc>",
        f'<rect width="{w}" height="{h}" fill="#fff"/>',
        f'<text x="{w/2}" y="28" text-anchor="middle" font-size="18" font-weight="bold" fill="#111">'
        'Draft acceptance by position</text>',
        f'<text x="{w/2}" y="48" text-anchor="middle" font-size="12" fill="#444">'
        'share of draft steps in which the target accepted draft token N</text>',
    ]
    for t in range(0, 101, 20):
        y = top + ph * (1 - t / 100)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#444">{t}%</text>')
    for n in range(1, kmax + 1):
        x = left + pw * (n - 1) / max(kmax - 1, 1)
        out.append(f'<text x="{x:.1f}" y="{top + ph + 18}" text-anchor="middle" font-size="11" fill="#444">{n}</text>')
    out.append(f'<text x="{left + pw/2}" y="{h - 14}" text-anchor="middle" font-size="12" fill="#444">draft position</text>')
    for j, r in enumerate(rows):
        c = colors[j % len(colors)]
        pts = []
        for i, v in enumerate(r["per_pos"]):
            x = left + pw * i / max(kmax - 1, 1)
            y = top + ph * (1 - v / r["steps"])
            pts.append(f"{x:.1f},{y:.1f}")
        out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{c}" stroke-width="2.5"/>')
        for p in pts:
            px, py = p.split(",")
            out.append(f'<circle cx="{px}" cy="{py}" r="3.5" fill="{c}"/>')
        ly = top + 10 + j * 22
        out.append(f'<rect x="{left + pw + 18}" y="{ly}" width="14" height="4" fill="{c}"/>')
        out.append(f'<text x="{left + pw + 38}" y="{ly + 6}" font-size="12" fill="#111">{r["k"]} drafts</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    rows = [r for r in (parse(p) for p in sorted(glob.glob(os.path.join(RES, "bench_*.txt")))) if r]
    rows = [r for r in rows if r["k"] is not None]
    rows.sort(key=lambda r: r["k"])
    base = next(r["tps"] for r in rows if r["k"] == 0)
    open(os.path.join(RES, "throughput.svg"), "w", encoding="utf-8").write(bar_chart(rows, base))
    acc = acceptance_chart(rows)
    if acc:
        open(os.path.join(RES, "acceptance.svg"), "w", encoding="utf-8").write(acc)
    print("| config | tokens/s | per-request median | speedup | draft acceptance | tokens per target step |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        name = "Qwen3-8B (baseline)" if r["k"] == 0 else f'Qwen3-8B + DFlash2, {r["k"]} draft tokens'
        acc_s = f'{r["acc_rate"]:.1f}% ({r["accepted"]}/{r["drafted"]})' if r["acc_rate"] is not None else "n/a"
        tps_step = f'{1 + r["accepted"] / r["steps"]:.2f}' if r["steps"] else "1.00"
        print(f'| {name} | {r["tps"]:.1f} | {r["median"]:.1f} | {r["tps"] / base:.2f}x | {acc_s} | {tps_step} |')


if __name__ == "__main__":
    main()
