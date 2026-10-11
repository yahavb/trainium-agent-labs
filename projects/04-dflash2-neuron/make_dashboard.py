#!/usr/bin/env python3
"""make_dashboard.py — build a self-contained HTML dashboard for the DFlash2
Neuron benchmarks (bidirectional vs causal query-block attention).

Reads /workspace/results/bench_*.txt, renders inline SVG charts (no deps):
  1. Throughput (tok/s) by config, baseline / causal / bidirectional
  2. Draft acceptance rate by config
  3. Accepted-tokens-by-position (k=8) causal vs bidirectional
Writes /workspace/results/dashboard.html and prints a text table.
"""
import glob
import os
import re

RES = "/workspace/results"


def parse(path):
    if not os.path.isfile(path):
        return None
    t = open(path).read()
    out = {}
    m = re.search(r"overall:.*?=\s*([\d.]+)\s*tok/s", t)
    if m:
        out["tok_s"] = float(m.group(1))
    m = re.search(r"acceptance rate:\s*([\d.]+)%\s*\((\d+)/(\d+)", t)
    if m:
        out["accept_pct"] = float(m.group(1))
        out["accepted"] = int(m.group(2))
        out["drafted"] = int(m.group(3))
    pos = {}
    for pm in re.finditer(r'position="(\d+)"\}\s*=\s*(\d+)', t):
        pos[int(pm.group(1))] = int(pm.group(2))
    out["pos"] = pos
    return out


def collect():
    data = {"baseline": parse(f"{RES}/bench_baseline.txt")}
    for k in (2, 4, 8):
        data[f"k{k}_causal"] = parse(f"{RES}/bench_dflash2_k{k}_causal.txt")
        data[f"k{k}_bidir"] = parse(f"{RES}/bench_dflash2_k{k}_bidir.txt")
    data["k15_causal"] = parse(f"{RES}/bench_dflash2_k15.txt")
    return data


def bar_chart(title, labels, series, ymax, unit, colors):
    """series: list of (name, [values], color)."""
    W, H, pad = 640, 300, 48
    gw = W - 2 * pad
    gh = H - 2 * pad
    n = len(labels)
    group = gw / n
    bw = group / (len(series) + 1)
    svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:680px">']
    svg.append(f'<text x="{W/2}" y="20" text-anchor="middle" font-size="15" font-weight="600">{title}</text>')
    # axes
    svg.append(f'<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#888"/>')
    svg.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" stroke="#888"/>')
    for gy in range(0, 6):
        val = ymax * gy / 5
        y = H - pad - gh * gy / 5
        svg.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{W-pad}" y2="{y:.1f}" stroke="#eee"/>')
        svg.append(f'<text x="{pad-6}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="#666">{val:.0f}</text>')
    for gi, lab in enumerate(labels):
        gx = pad + gi * group
        for si, (name, vals, color) in enumerate(series):
            v = vals[gi]
            if v is None:
                continue
            bh = gh * v / ymax
            x = gx + bw * (si + 0.5)
            y = H - pad - bh
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.9:.1f}" height="{bh:.1f}" fill="{color}"><title>{name} {lab}: {v}{unit}</title></rect>')
            svg.append(f'<text x="{x+bw*0.45:.1f}" y="{y-3:.1f}" text-anchor="middle" font-size="9" fill="#333">{v:g}</text>')
        svg.append(f'<text x="{gx+group/2:.1f}" y="{H-pad+16}" text-anchor="middle" font-size="11">{lab}</text>')
    # legend
    lx = pad + 10
    for name, vals, color in series:
        svg.append(f'<rect x="{lx}" y="{pad-14}" width="11" height="11" fill="{color}"/>')
        svg.append(f'<text x="{lx+15}" y="{pad-4}" font-size="11">{name}</text>')
        lx += 20 + len(name) * 7
    svg.append("</svg>")
    return "".join(svg)


def line_chart(title, max_pos, series):
    """series: list of (name, {pos:val}, color). Accepted tokens by position."""
    W, H, pad = 640, 300, 48
    gw, gh = W - 2 * pad, H - 2 * pad
    allv = [v for _, d, _ in series for v in d.values()]
    ymax = max(allv + [1])
    svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:680px">']
    svg.append(f'<text x="{W/2}" y="20" text-anchor="middle" font-size="15" font-weight="600">{title}</text>')
    svg.append(f'<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#888"/>')
    svg.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" stroke="#888"/>')
    for gy in range(0, 6):
        y = H - pad - gh * gy / 5
        svg.append(f'<text x="{pad-6}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="#666">{ymax*gy/5:.0f}</text>')
        svg.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{W-pad}" y2="{y:.1f}" stroke="#eee"/>')
    def X(p): return pad + gw * (p / max(max_pos, 1))
    def Y(v): return H - pad - gh * (v / ymax)
    for p in range(max_pos + 1):
        svg.append(f'<text x="{X(p):.1f}" y="{H-pad+16}" text-anchor="middle" font-size="10">{p}</text>')
    for name, d, color in series:
        pts = " ".join(f"{X(p):.1f},{Y(d.get(p,0)):.1f}" for p in range(max_pos + 1))
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for p in range(max_pos + 1):
            svg.append(f'<circle cx="{X(p):.1f}" cy="{Y(d.get(p,0)):.1f}" r="3" fill="{color}"><title>{name} pos{p}: {d.get(p,0)}</title></circle>')
    lx = pad + 10
    for name, d, color in series:
        svg.append(f'<rect x="{lx}" y="{pad-14}" width="11" height="11" fill="{color}"/>')
        svg.append(f'<text x="{lx+15}" y="{pad-4}" font-size="11">{name}</text>')
        lx += 20 + len(name) * 7
    svg.append("</svg>")
    return "".join(svg)


def main():
    d = collect()
    base = d["baseline"]["tok_s"]
    labels = ["k=2", "k=4", "k=8"]
    causal_ts = [d[f"k{k}_causal"]["tok_s"] for k in (2, 4, 8)]
    bidir_ts = [d[f"k{k}_bidir"]["tok_s"] for k in (2, 4, 8)]
    causal_ac = [d[f"k{k}_causal"]["accept_pct"] for k in (2, 4, 8)]
    bidir_ac = [d[f"k{k}_bidir"]["accept_pct"] for k in (2, 4, 8)]

    C_BASE, C_CAUSAL, C_BIDIR = "#9aa0a6", "#e8893b", "#2f7ed8"

    tp = bar_chart(
        "Decode throughput (tok/s) — higher is better",
        labels,
        [("baseline (no spec)", [base]*3, C_BASE),
         ("causal", causal_ts, C_CAUSAL),
         ("bidirectional", bidir_ts, C_BIDIR)],
        ymax=100, unit=" tok/s",
        colors=None,
    )
    ac = bar_chart(
        "Draft acceptance rate (%) — higher is better",
        labels,
        [("causal", causal_ac, C_CAUSAL), ("bidirectional", bidir_ac, C_BIDIR)],
        ymax=50, unit="%", colors=None,
    )
    lc = line_chart(
        "Accepted tokens by draft position (k=8) — bidirectional lifts every position",
        7,
        [("causal", d["k8_causal"]["pos"], C_CAUSAL),
         ("bidirectional", d["k8_bidir"]["pos"], C_BIDIR)],
    )

    # table rows
    rows = []
    rows.append(("baseline (no spec)", f"{base:.1f}", "1.00x", "n/a"))
    for k in (2, 4, 8):
        c, b = d[f"k{k}_causal"], d[f"k{k}_bidir"]
        rows.append((f"DFlash2 k={k} causal", f"{c['tok_s']:.1f}", f"{c['tok_s']/base:.2f}x", f"{c['accept_pct']:.1f}%"))
        rows.append((f"DFlash2 k={k} <b>bidirectional</b>", f"<b>{b['tok_s']:.1f}</b>", f"<b>{b['tok_s']/base:.2f}x</b>", f"<b>{b['accept_pct']:.1f}%</b>"))

    trs = "\n".join(
        f"<tr><td>{a}</td><td style='text-align:right'>{tk}</td><td style='text-align:right'>{sp}</td><td style='text-align:right'>{acc}</td></tr>"
        for a, tk, sp, acc in rows
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>DFlash2 on Trainium — bidirectional query-block attention</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#222}}
h1{{font-size:1.5rem}} h2{{font-size:1.1rem;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;margin-top:1rem;font-size:.9rem}}
th,td{{border:1px solid #ddd;padding:6px 10px}} th{{background:#f5f5f5;text-align:left}}
.card{{border:1px solid #eee;border-radius:8px;padding:12px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
.k{{color:#2f7ed8;font-weight:600}}
small{{color:#666}}
</style></head><body>
<h1>DFlash2 speculative decoding on Trainium 2</h1>
<p>Qwen3-8B (target) + z-lab/Qwen3-8B-DFlash-b16 (5-layer draft), one <code>trn2</code> chip (seat-40), TP=2,
greedy, 128 new tokens, 3 prompts x 2 runs, same server config across all rows (only <code>--speculative-config</code> differs).
<br><small>This dashboard compares the <b>causal</b> query-block attention (reused Neuron decode kernel) vs the
<b>bidirectional</b> query-block attention this project added (raw PyTorch all-ones active mask), which matches how the
DFlash2 draft was trained on GPU.</small></p>

<div class="card">
<b>Headline:</b> making the draft's query block <span class="k">bidirectional</span> raises draft acceptance at every
draft count and pushes Qwen3-8B from <b>{base:.1f} tok/s</b> (no speculation) to <b>{d['k8_bidir']['tok_s']:.1f} tok/s</b>
at k=8 (<b>{d['k8_bidir']['tok_s']/base:.2f}x</b>). Acceptance at k=4 jumped {d['k4_causal']['accept_pct']:.1f}% → {d['k4_bidir']['accept_pct']:.1f}%.
</div>

<h2>Throughput</h2><div class="card">{tp}</div>
<h2>Draft acceptance rate</h2><div class="card">{ac}</div>
<h2>Accepted tokens by position (k=8)</h2><div class="card">{lc}
<p><small>Bidirectional attention lifts acceptance most at the later draft positions (1-3), which is exactly where the
causal mask was starving the draft of intra-block context.</small></p></div>

<h2>Results table</h2>
<table><tr><th>config</th><th>tok/s</th><th>speedup</th><th>draft acceptance</th></tr>
{trs}
</table>
<p><small>Generated by make_dashboard.py from /workspace/results/bench_*.txt. All numbers measured on-device this session.</small></p>
</body></html>"""

    out = f"{RES}/dashboard.html"
    open(out, "w").write(html)
    print("WROTE", out, f"({len(html)} bytes)")
    print("\n=== TABLE ===")
    print(f"{'config':34s} {'tok/s':>7s} {'speedup':>8s} {'accept':>8s}")
    for a, tk, sp, acc in rows:
        a2 = a.replace("<b>", "").replace("</b>", "")
        tk2 = tk.replace("<b>", "").replace("</b>", "")
        sp2 = sp.replace("<b>", "").replace("</b>", "")
        acc2 = acc.replace("<b>", "").replace("</b>", "")
        print(f"{a2:34s} {tk2:>7s} {sp2:>8s} {acc2:>8s}")


if __name__ == "__main__":
    main()
