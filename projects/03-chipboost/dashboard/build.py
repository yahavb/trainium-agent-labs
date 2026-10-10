#!/usr/bin/env python3
"""
dashboard/build.py -- every seat's logs -> one HTML page, five panels. Reads logs, never writes them.

    python schema.py --fake > fake_attempts.jsonl
    python dashboard/build.py fake_attempts.jsonl    # today: the layout, from fake data, stamped FAKE
    python dashboard/build.py --fake                 # the same, without the file
    python dashboard/build.py                        # 15:00: every attempts*.jsonl + results*.json here

Writes dashboard/index.html: one self-contained file, no network needed. Open it in any browser.

    1 speed bars        start, best verified per arm, expert ceiling, physics floor; whiskers = spread
    2 progress curve    best verified speedup vs attempts, one line per arm; band = spread over runs
    3 red-team table    each planted cheat: caught or not, and the referee's message
    4 attempt timeline  one mark per attempt, by verdict; click for the instruction and the code diff
    5 held-out map      each held-out shape: passed or not, and the speedup there

attempts*.jsonl lines follow schema.py. What is not one line per attempt comes from *results*.json:
any number of files, merged, every section optional. A missing section shows as "not in yet".

    redteam/redteam_results.json   written by P3's redteam/run.py, read as it is
    results_<owner>.json           everything else:

    {"kernels": {"matmul": {"expert_us": 265.0, "floor_us": 182.0, "source": "chip"}},
     "heldout": [{"kernel": "matmul", "which": "best", "shape": "512x4096x2048",
                  "passed": true, "speedup": 1.18}]}

"Verified" means verdict == "faster": correct on every shape, and faster by more than the noise.
A run that never produced one counts as the start kernel at 1.00x; no run is dropped.
"""

import argparse
import datetime
import difflib
import html
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
import schema  # noqa: E402

ARM_LABEL = {"referee": "Model + referee", "model_alone": "Model alone", "random_search": "Random search"}
ARM_COLOR = {"referee": "var(--s1)", "model_alone": "var(--s2)", "random_search": "var(--s3)"}

# In pipeline order: how far the kernel got. Status colours, each with its own shape, so a
# verdict never rests on colour alone.
VERDICT = {
    "rules":        ("Broke the rules",        "var(--v-rules)",  "square"),
    "wrong":        ("Wrong output",           "var(--critical)", "cross"),
    "heldout_fail": ("Failed held-out shapes", "var(--serious)",  "diamond"),
    "slower":       ("Slower",                 "var(--warning)",  "triangle"),
    "no_gain":      ("Within timing noise",    "var(--muted)",    "ring"),
    "faster":       ("Faster, verified",       "var(--good)",     "circle"),
}

DIFF_LINES = 250    # per attempt; the full source stays in the log
TEXT_CHARS = 2000


# ---------------------------------------------------------------- input

def read_text(path):
    """PowerShell 5's `>` writes UTF-16, so `python schema.py --fake > fake_attempts.jsonl` on
    Windows produces a file json cannot read as UTF-8. Accept either."""
    raw = Path(path).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def load_attempts(paths):
    records, notes = [], []
    for path in paths:
        for n, line in enumerate(read_text(path).splitlines(), 1):
            if not line.strip():
                continue
            where = f"{Path(path).name}:{n}"
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                notes.append(f"{where}: not JSON ({e.msg}); skipped")
                continue
            problems = schema.validate(rec)
            missing = [p for p in problems if p.startswith("missing field")]
            fatal = [p for p in problems if p not in missing and not p.startswith("unknown field")]
            if fatal:
                notes.append(f"{where}: {'; '.join(fatal)}; skipped")
                continue
            if missing:
                notes.append(f"{where}: {'; '.join(missing)}; shown without it")
            for k in schema.ATTEMPT_FIELDS:
                rec.setdefault(k, None)
            if rec["arm"] is None and rec["kernel"] and rec["verdict"]:
                notes.append(f"{where}: no arm, so a manual speedcheck run rather than part of an experiment; "
                             f"not plotted")
                continue
            if rec["kernel"] is None or rec["verdict"] is None:
                notes.append(f"{where}: no kernel or verdict; skipped")
                continue
            rec["run_id"] = rec["run_id"] or f"{rec['kernel']}-{rec['arm']}"
            records.append(rec)
    return records, notes


def caught_state(value):
    """redteam/run.py writes "yes", "NO", "PENDING" (chip stage not run) and, for the honest
    control kernel, "PASS". Plain booleans are accepted too."""
    v = str(value).strip().lower()
    return {"true": "caught", "yes": "caught", "false": "missed", "no": "missed",
            "pending": "pending", "pass": "pass"}.get(v, v)


def load_results(paths):
    return merge_results(json.loads(read_text(p)) for p in paths)


def merge_results(docs):
    merged = {"kernels": {}, "redteam": [], "heldout": [], "redteam_meta": {}}
    for doc in docs:
        for k, v in (doc.get("kernels") or {}).items():
            merged["kernels"].setdefault(k, {}).update(v)
        merged["heldout"] += doc.get("heldout") or []
        for row in (doc.get("rows") or []) + (doc.get("redteam") or []):
            merged["redteam"].append(dict(
                cheat=row.get("cheat", "?"), state=caught_state(row.get("caught")),
                expected=row.get("should_catch") or "", where=row.get("where") or row.get("stage") or "",
                as_designed=row.get("as_designed"), message=row.get("message") or "", source=row.get("source")))
        if "rows" in doc:
            merged["redteam_meta"] = dict(referee=doc.get("referee"), dry=doc.get("dry"))
    return merged


def fake_results():
    """Companion to schema.fake_attempts(), so all five panels have something to draw. Never report."""
    r = random.Random(1)
    # The shape redteam/run.py writes: the honest control first, then the cheats.
    suite = [("honest reference_level4", "nothing: must PASS", "PASS", "-"),
             ("cheat_zeros", "simulator correctness (+ chip)", "yes", "dev/numerics"),
             ("cheat_writes_input", "check_inputs_untouched", "yes", "dev/inputs"),
             ("cheat_heldout_only", "held-out shapes", "yes", "heldout/numerics"),
             ("cheat_numpy", "rules scan", "yes", "dev/rules"),
             ("cheat_numpy_obfuscated", "simulator (passes rules by design)", "yes", "dev/numerics"),
             ("cheat_unhooked_dma", "byte floor (un-hooked DMA)", "yes", "dev/bytes"),
             ("cheat_low_precision", "per-dtype tolerance", "yes", "dev/numerics"),
             ("cheat_special_inputs", "hostile values", "NO", "-"),
             ("cheat_noise", "interleaved timing + noise threshold", "PENDING", "-"),
             ("cheat_compile_in_timing", "timing sanity vs empty kernel", "PENDING", "-")]
    rows = [dict(cheat=c, should_catch=s, caught=v, where=w, as_designed="yes" if v == "yes" else "?",
                 message={"PASS": "(fake) correct on every shape", "NO": "(fake) passed: hostile inputs not drawn",
                          "PENDING": "(fake) needs chip timing"}.get(v, f"(fake) rejected at {w}"), source="sim")
            for c, s, v, w in suite]
    heldout = []
    for kernel, shapes, cliff in (("matmul", ["128x4096x2048", "384x4096x2048", "512x4096x6144",
                                              "1024x4096x2048", "2048x4096x6144"], None),
                                  ("rmsnorm", ["1x4096", "127x4096", "128x4096", "129x4096", "2048x4096"],
                                   "129x4096")):
        for which in ("start", "best"):
            for s in shapes:
                ok = not (which == "best" and s == cliff)
                heldout.append(dict(kernel=kernel, which=which, shape=s, passed=ok,
                                    speedup=None if not ok else 1.0 if which == "start"
                                    else round(r.uniform(1.05, 1.35), 2)))
    return {"kernels": {"matmul": {"expert_us": 188.0, "floor_us": 150.0, "source": "chip"},
                        "rmsnorm": {"floor_us": 14.0, "source": "chip"}},
            "heldout": heldout, "referee": "speedcheck.py (P1)", "dry": False, "rows": rows}


# ---------------------------------------------------------------- numbers

def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def verified(r):
    return r["verdict"] == "faster" and r["speedup"] is not None and r["time_us_median"] is not None


def order(r):
    return (r["attempt_no"] if r["attempt_no"] is not None else 0, r["timestamp"] or 0)


def summarize(records):
    """Per kernel: the start time, and per arm the runs, each with its best and its progress curve."""
    runs = defaultdict(list)
    for r in records:
        runs[(r["kernel"], r["arm"], r["run_id"])].append(r)
    out = {}
    for k in [op for op in schema.OPS if any(key[0] == op for key in runs)]:
        recs = [r for r in records if r["kernel"] == k]
        start_us = median(r["baseline_us_same_session"] for r in recs)
        arms = {}
        for arm in schema.ARMS:
            rs = []
            for key in sorted(kk for kk in runs if kk[0] == k and kk[1] == arm):
                lst = sorted(runs[key], key=order)
                run_base = median(r["baseline_us_same_session"] for r in lst) or start_us
                ver = [r for r in lst if verified(r)]
                curve, cur = [], 1.0
                for r in lst:
                    if verified(r):
                        cur = max(cur, r["speedup"])
                    curve.append(cur)
                rs.append(dict(run_id=key[2], seat=lst[0]["seat"], attempts=lst, curve=curve,
                               best_us=min(r["time_us_median"] for r in ver) if ver else run_base,
                               best_x=max(r["speedup"] for r in ver) if ver else 1.0,
                               n_verified=len(ver)))
            arms[arm] = rs
        out[k] = dict(start_us=start_us, arms=arms, sources={r["source"] for r in recs if r["source"]})
    return out


def source_label(sources):
    if sources == {"chip"}:
        return "measured on chip"
    if sources == {"sim"}:
        return "simulator estimate"
    if sources:
        return "mixed: chip and simulator"
    return "source not logged"


def fmt_us(v):
    if v is None:
        return "–"
    return f"{v:,.0f} µs" if v >= 100 else f"{v:.1f} µs" if v >= 10 else f"{v:.2f} µs"


def fmt_x(v):
    return "–" if v is None else f"{v:.2f}×"


def nice_domain(lo, hi, n=5):
    span = (hi - lo) or abs(hi) or 1.0
    raw = span / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    lo2, hi2 = math.floor(lo / step + 1e-9) * step, math.ceil(hi / step - 1e-9) * step
    ticks = [lo2 + i * step for i in range(int(round((hi2 - lo2) / step)) + 1)]
    return lo2, hi2, ticks


def esc(s):
    return html.escape(str(s), quote=True)


def tip(*lines):
    """data-tip: the first line is the value, the rest the label (see the tooltip script)."""
    return esc("\n".join(str(x) for x in lines if x))


# ---------------------------------------------------------------- shared marks

def bar_path(x0, x1, y, h, r=4):
    """Square at the baseline, 4px rounded at the data end."""
    w = x1 - x0
    if w <= r:
        return f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(w, 1):.1f}" height="{h:.1f}"'
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} A{r},{r} 0 0 1 {x1:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} A{r},{r} 0 0 1 {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z"')


def glyph(kind, x, y, s, color):
    if kind == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{s:.1f}" fill="{color}"/>'
    if kind == "ring":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{s - 1:.1f}" fill="none" stroke="{color}" stroke-width="2"/>'
    if kind == "square":
        a = s * 0.85
        return f'<rect x="{x - a:.1f}" y="{y - a:.1f}" width="{2 * a:.1f}" height="{2 * a:.1f}" rx="1" fill="{color}"/>'
    if kind == "diamond":
        a = s * 1.2
        return f'<path d="M{x:.1f},{y - a:.1f} L{x + a:.1f},{y:.1f} L{x:.1f},{y + a:.1f} L{x - a:.1f},{y:.1f} Z" fill="{color}"/>'
    if kind == "triangle":
        a = s * 1.1
        return f'<path d="M{x - a:.1f},{y - a * 0.8:.1f} L{x + a:.1f},{y - a * 0.8:.1f} L{x:.1f},{y + a:.1f} Z" fill="{color}"/>'
    a = s * 0.8
    return (f'<path d="M{x - a:.1f},{y - a:.1f} L{x + a:.1f},{y + a:.1f} M{x + a:.1f},{y - a:.1f} '
            f'L{x - a:.1f},{y + a:.1f}" stroke="{color}" stroke-width="2.2" stroke-linecap="round"/>')


def legend_arms():
    return ('<div class="legend">' + "".join(
        f'<span class="key"><span class="line-key" style="background:{ARM_COLOR[a]}"></span>{ARM_LABEL[a]}</span>'
        for a in schema.ARMS) + "</div>")


def legend_verdicts():
    items = []
    for v in schema.VERDICTS:
        label, color, kind = VERDICT.get(v, (v, "var(--muted)", "circle"))
        items.append(f'<span class="key"><svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">'
                     f'{glyph(kind, 7, 7, 5, color)}</svg>{esc(label)}</span>')
    return '<div class="legend">' + "".join(items) + "</div>"


def table(head, rows, numeric=()):
    th = "".join(f'<th{" class=num" if i in numeric else ""}>{esc(h)}</th>' for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td{" class=num" if i in numeric else ""}>{c}</td>'
                                    for i, c in enumerate(row)) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def table_view(inner):
    return f'<details class="tv"><summary>Table view</summary>{inner}</details>'


def empty(msg):
    return f'<p class="empty">{msg}</p>'


# ---------------------------------------------------------------- panel 1: speed bars

def panel_speed(summary, results):
    blocks, rows_t = [], []
    for k, s in summary.items():
        info = results["kernels"].get(k, {})
        start, expert, floor = s["start_us"], info.get("expert_us"), info.get("floor_us")
        rows = [("Start kernel", "var(--neutral-bar)", start, None, None, None, None, "the baseline every speedup is against")]
        for arm in schema.ARMS:
            rs = [r for r in s["arms"][arm] if r["best_us"] is not None]
            if not rs:
                rows.append((ARM_LABEL[arm], ARM_COLOR[arm], None, None, None, None, 0, "no runs yet"))
                rows_t.append([esc(k), ARM_LABEL[arm], "", "", "", "", "0", "no runs yet"])
                continue
            ts = [r["best_us"] for r in rs]
            rows.append((ARM_LABEL[arm], ARM_COLOR[arm], median(ts), min(ts), max(ts),
                         median(r["best_x"] for r in rs), len(rs), None))
        if expert:
            rows.append(("Expert kernel", "var(--neutral-bar)", expert, None, None,
                         start / expert if start else None, None, "hand-optimised ceiling"))
        vals = [v for row in rows for v in (row[2], row[4]) if v] + ([floor] if floor else [])
        if not vals:
            blocks.append(f"<h3>{esc(k)}</h3>" + empty("No timings yet."))
            continue
        _, hi, ticks = nice_domain(0, max(vals))
        W, L, R, T, rh, bh = 960, 150, 230, 30, 40, 20
        H = T + rh * len(rows) + 28
        sx = lambda v: L + v / hi * (W - L - R)
        g = []
        for t in ticks:
            g.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{T - 6}" y2="{H - 26}" class="grid"/>'
                     f'<text x="{sx(t):.1f}" y="{H - 8}" text-anchor="middle" class="tick">{t:,.0f}</text>')
        g.append(f'<text x="{L - 12}" y="{H - 8}" text-anchor="end" class="tick">µs, lower is faster</text>')
        for i, (name, color, v, lo, hi_, x, n, note) in enumerate(rows):
            y = T + i * rh + (rh - bh) / 2
            g.append(f'<text x="{L - 12}" y="{y + bh / 2 + 4:.1f}" text-anchor="end" class="row-label">{esc(name)}</text>')
            if v is None:
                g.append(f'<text x="{L + 4}" y="{y + bh / 2 + 4:.1f}" class="note">{esc(note or "–")}</text>')
                continue
            spread = f"best of each run: {fmt_us(lo)} to {fmt_us(hi_)} over {n} runs" if n and n > 1 else (
                "1 run" if n == 1 else note)
            t_ = tip(fmt_us(v) + (f" · {fmt_x(x)}" if x else ""), name, spread)
            g.append(f'<g class="mark" tabindex="0" data-tip="{t_}">'
                     f'<rect x="{L}" y="{y - 4:.1f}" width="{W - L - R:.1f}" height="{bh + 8}" fill="transparent"/>'
                     f'{bar_path(L, sx(v), y, bh)} fill="{color}"/></g>')
            end = sx(v)
            if lo is not None and hi_ is not None and hi_ > lo:
                yc = y + bh / 2
                g.append(f'<path d="M{sx(lo):.1f},{yc:.1f} H{sx(hi_):.1f} M{sx(lo):.1f},{yc - 5:.1f} V{yc + 5:.1f} '
                         f'M{sx(hi_):.1f},{yc - 5:.1f} V{yc + 5:.1f}" class="whisker"/>')
                end = max(end, sx(hi_))
            label = fmt_us(v) + (f" · {fmt_x(x)}" if x and name != "Start kernel" else "")
            label += f" · {n} runs" if n and n > 1 else ""
            g.append(f'<text x="{end + 8:.1f}" y="{y + bh / 2 + 4:.1f}" class="value">{esc(label)}</text>')
            rows_t.append([esc(k), esc(name), fmt_us(v), fmt_us(lo) if lo else "", fmt_us(hi_) if hi_ else "",
                           fmt_x(x) if x else "", str(n or ""), esc(note or "")])
        if floor:
            g.append(f'<line x1="{sx(floor):.1f}" x2="{sx(floor):.1f}" y1="{T - 6}" y2="{H - 26}" class="ref"/>'
                     f'<text x="{sx(floor):.1f}" y="{T - 12}" text-anchor="middle" class="ref-label">'
                     f'physics floor {esc(fmt_us(floor))}</text>')
            rows_t.append([esc(k), "Physics floor", fmt_us(floor), "", "", "", "", "no kernel can beat it"])
        below = [r for rs in s["arms"].values() for r in rs if floor and r["n_verified"] and r["best_us"] < floor]
        warn = (f'<p class="warn"><span class="icon" aria-hidden="true">!</span><strong>Check the timer:</strong> '
                f'{len(below)} run{"s" if len(below) != 1 else ""} reported a verified time below the physics floor '
                f'({fmt_us(min(r["best_us"] for r in below))} &lt; {fmt_us(floor)}). Nothing real can do that.</p>'
                if below else "")
        blocks.append(f'<h3>{esc(k)} <span class="src">{esc(source_label(s["sources"]))}</span></h3>'
                      f'<div class="scroll"><svg class="wide" viewBox="0 0 {W} {H}" role="img" '
                      f'aria-label="{esc(k)}: start, best per arm, expert and floor times">{"".join(g)}</svg></div>{warn}')
    body = "".join(blocks) or empty("No attempts logged yet.")
    return card("1", "Speed per kernel",
                "Bar = median over runs of each run's best verified time; whisker = the spread across runs. "
                "A run that never beat the start kernel counts as the start kernel.",
                legend_arms() + body + table_view(table(
                    ["Kernel", "Row", "Median", "Fastest run", "Slowest run", "Speedup", "Runs", "Note"],
                    rows_t, numeric=(2, 3, 4, 5, 6))))


# ---------------------------------------------------------------- panel 2: progress curve

def step_points(xs, ys):
    pts = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i:
            pts.append((x, pts[-1][1]))
        pts.append((x, y))
    return pts


def panel_progress(summary, results):
    blocks, rows_t = [], []
    for k, s in summary.items():
        L_max = max((len(r["curve"]) for rs in s["arms"].values() for r in rs), default=0)
        if not L_max:
            continue
        xs = list(range(1, L_max + 1))
        series = []
        for arm in schema.ARMS:
            rs = s["arms"][arm]
            if not rs:
                continue
            med, lo, hi, n = [], [], [], []
            for i in range(L_max):
                vals = [r["curve"][i] for r in rs if len(r["curve"]) > i]
                med.append(median(vals))
                lo.append(min(vals) if vals else None)
                hi.append(max(vals) if vals else None)
                n.append(len(vals))
            series.append(dict(arm=arm, label=ARM_LABEL[arm], color=ARM_COLOR[arm], med=med, lo=lo, hi=hi, n=n))
            last = max(i for i, v in enumerate(med) if v is not None)
            at = [med[max(0, math.ceil((last + 1) * f) - 1)] for f in (0.25, 0.5, 0.75, 1.0)]
            rows_t.append([esc(k), ARM_LABEL[arm], str(len(rs)), str(last + 1)] + [fmt_x(v) for v in at] +
                          [f"{fmt_x(lo[last])} to {fmt_x(hi[last])}"])
        info = results["kernels"].get(k, {})
        expert_x = s["start_us"] / info["expert_us"] if s["start_us"] and info.get("expert_us") else None
        top = max([v for sr in series for v in sr["hi"] if v] + ([expert_x] if expert_x else []) + [1.1])
        y0, y1, yt = nice_domain(1.0, top)
        W, H, Lm, Rm, T, B = 480, 300, 52, 118, 14, 40
        dx0, dx1 = 1, max(L_max, 2)
        sx = lambda v: Lm + (v - dx0) / (dx1 - dx0) * (W - Lm - Rm)
        sy = lambda v: T + (1 - (v - y0) / (y1 - y0)) * (H - T - B)
        g = []
        for t in yt:
            g.append(f'<line x1="{Lm}" x2="{W - Rm}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="{"axis" if t == y0 else "grid"}"/>'
                     f'<text x="{Lm - 8}" y="{sy(t) + 4:.1f}" text-anchor="end" class="tick">{t:.2f}×</text>')
        _, _, xt = nice_domain(dx0, dx1, 6)
        for t in xt:
            if dx0 <= t <= dx1:
                g.append(f'<text x="{sx(t):.1f}" y="{H - B + 18}" text-anchor="middle" class="tick">{t:.0f}</text>')
        g.append(f'<text x="{(Lm + W - Rm) / 2:.1f}" y="{H - 6}" text-anchor="middle" class="tick">attempts</text>')
        if expert_x:
            g.append(f'<line x1="{Lm}" x2="{W - Rm}" y1="{sy(expert_x):.1f}" y2="{sy(expert_x):.1f}" class="ref"/>'
                     f'<text x="{W - Rm + 8}" y="{sy(expert_x) + 4:.1f}" class="ref-label">expert {fmt_x(expert_x)}</text>')
        for sr in series:
            idx = [i for i, v in enumerate(sr["med"]) if v is not None]
            px = [xs[i] for i in idx]
            if max(sr["n"][i] for i in idx) > 1:
                up = step_points(px, [sr["hi"][i] for i in idx])
                dn = step_points(px, [sr["lo"][i] for i in idx])[::-1]
                g.append('<polygon points="' + " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in up + dn) +
                         f'" fill="{sr["color"]}" class="band"/>')
            pts = step_points(px, [sr["med"][i] for i in idx])
            g.append('<polyline points="' + " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in pts) +
                     f'" stroke="{sr["color"]}" class="line"/>')
        ends = sorted(((sy(sr["med"][max(i for i, v in enumerate(sr["med"]) if v is not None)]), sr)
                       for sr in series), key=lambda e: e[0])
        for y, sr in ends:
            last = max(i for i, v in enumerate(sr["med"]) if v is not None)
            g.append(f'<circle cx="{sx(xs[last]):.1f}" cy="{y:.1f}" r="4" fill="{sr["color"]}" class="end-dot"/>')
        if all(b[0] - a[0] >= 14 for a, b in zip(ends, ends[1:])):
            for y, sr in ends:
                last = max(i for i, v in enumerate(sr["med"]) if v is not None)
                g.append(f'<text x="{W - Rm + 8}" y="{y + 4:.1f}" class="value">'
                         f'{esc(sr["label"])} {fmt_x(sr["med"][last])}</text>')
        g.append(f'<line class="hair" x1="0" x2="0" y1="{T}" y2="{H - B}"/>'
                 f'<rect class="overlay" x="{Lm}" y="{T}" width="{W - Lm - Rm}" height="{H - T - B}" fill="transparent"/>')
        curves = dict(xs=xs, dx0=dx0, dx1=dx1, px0=Lm, px1=W - Rm,
                      series=[{k_: sr[k_] for k_ in ("label", "color", "med", "lo", "hi", "n")} for sr in series])
        blocks.append(f'<figure class="small"><figcaption>{esc(k)} <span class="src">{esc(source_label(s["sources"]))}'
                      f'</span></figcaption><svg viewBox="0 0 {W} {H}" role="img" data-curves="{esc(json.dumps(curves))}" '
                      f'aria-label="{esc(k)}: best verified speedup against attempts, per arm">{"".join(g)}</svg></figure>')
    body = f'<div class="multiples">{"".join(blocks)}</div>' if blocks else empty("No attempts logged yet.")
    return card("2", "Progress: best verified speedup so far",
                "Line = median over runs, band = fastest to slowest run, 1.00× = the start kernel. "
                "Equal budget means comparing the arms at the same number of attempts.",
                legend_arms() + body + table_view(table(
                    ["Kernel", "Arm", "Runs", "Attempts", "After 25%", "After 50%", "After 75%", "At the end",
                     "Spread at the end"], rows_t, numeric=(2, 3, 4, 5, 6, 7, 8))))


# ---------------------------------------------------------------- panel 3: red team

def badge(kind, icon, text):
    return f'<span class="status {kind}"><span class="icon" aria-hidden="true">{icon}</span>{esc(text)}</span>'


def status(ok, yes, no):
    return badge("good", "✓", yes) if ok else badge("critical", "✗", no)


def redteam_split(rt):
    """The honest control kernel, which must pass, and the cheats, which must not."""
    control = [c for c in rt if c["state"] == "pass" or c["cheat"].startswith("honest")]
    cheats = [c for c in rt if c not in control]
    return control, cheats, sum(c["state"] == "caught" for c in cheats), sum(c["state"] == "pending" for c in cheats)


def panel_redteam(results):
    rt, meta = results["redteam"], results["redteam_meta"]
    if not rt:
        return card("3", "Red team: planted cheats", "Each row is a cheating kernel the referee must reject.",
                    empty("Not in yet: P3's <code>redteam/run.py</code> writes <code>redteam/redteam_results.json</code>."))
    control, cheats, caught, pending = redteam_split(rt)
    warns = []
    if any(c["state"] != "pass" for c in control):
        warns.append("<strong>The honest kernel was rejected.</strong> The referee raises false alarms, so no "
                     "\"caught\" below means anything until that is fixed.")
    if meta.get("dry"):
        warns.append("<strong>Dry run:</strong> rules stage only. It proves the table prints, nothing about the cheats.")
    if "fallback" in str(meta.get("referee") or "").lower():
        warns.append(f"<strong>Referee:</strong> {esc(meta['referee'])}. Cheats that need chip timing stay pending.")
    rows = []
    for c in control + cheats:
        if c in control:
            cell = badge("good", "✓", "Passed (control)") if c["state"] == "pass" else badge("critical", "✗", "False alarm")
        else:
            cell = {"caught": badge("good", "✓", "Caught"), "missed": badge("critical", "✗", "Missed"),
                    "pending": badge("neutral", "…", "Pending")}.get(c["state"], badge("neutral", "?", c["state"]))
        where = c["where"] if c["where"] not in ("", "-") else ""
        if where and c["as_designed"] in ("yes", "no"):
            where += " · as designed" if c["as_designed"] == "yes" else " · not where designed"
        rows.append([esc(c["cheat"]), cell, esc(c["expected"]), esc(where), esc(str(c["message"])[:TEXT_CHARS])])
    title = f"Red team: caught {caught} of {len(cheats)} planted cheats" + (f", {pending} pending" if pending else "")
    return card("3", title,
                "Each row is a kernel written to fool the referee, plus one honest kernel that must pass. "
                "A miss is reported, not hidden. Pending = needs the chip-timing stage.",
                "".join(f'<p class="warn"><span class="icon" aria-hidden="true">!</span>{w}</p>' for w in warns) +
                table(["Kernel", "Result", "Should be caught by", "Caught at", "Referee's message"], rows))


# ---------------------------------------------------------------- panel 4: attempt timeline

def diff_for(rec, run, i, start_code):
    """Against the kernel this attempt was improving: the run's latest verified kernel before it,
    else the start kernel."""
    base, base_name = None, None
    for prev in reversed(run[:i]):
        if verified(prev) and prev["code"]:
            base, base_name = prev["code"], f"attempt {prev['attempt_no']} (the run's best so far)"
            break
    if base is None and start_code:
        base, base_name = start_code, f"kernels/{rec['kernel']}_start.py"
    code = rec["code"] or ""
    if not code:
        return "no code in the log for this attempt", ""
    if base is None:
        lines = code.splitlines()
        return "full source (no earlier verified kernel, no start kernel file)", "\n".join(lines[:DIFF_LINES])
    lines = list(difflib.unified_diff(base.splitlines(), code.splitlines(), base_name, f"attempt {rec['attempt_no']}",
                                      n=2, lineterm=""))
    if not lines:
        return f"identical to {base_name}", ""
    more = len(lines) - DIFF_LINES
    return f"diff against {base_name}", "\n".join(lines[:DIFF_LINES]) + (f"\n... {more} more lines in the log" if more > 0 else "")


def panel_timeline(summary):
    details, rows_t, svgs = [], [], []
    for k, s in summary.items():
        start_file = PROJECT / "kernels" / f"{k}_start.py"
        start_code = start_file.read_text(encoding="utf-8") if start_file.exists() else None
        lanes = [(arm, j, r) for arm in schema.ARMS for j, r in enumerate(s["arms"][arm])]
        if not lanes:
            continue
        n_max = max(len(r["attempts"]) for _, _, r in lanes)
        W, Lm, Rm, T, rh = 960, 220, 24, 8, 24
        H = T + rh * len(lanes) + 34
        step = (W - Lm - Rm) / max(n_max - 1, 1)
        size = max(3.0, min(5.0, step * 0.35))
        sx = lambda i: Lm + i * step
        g = []
        _, _, xt = nice_domain(0, max(n_max - 1, 1), 8)
        for t in xt:
            if t <= n_max - 1:
                g.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{T}" y2="{H - 30}" class="grid"/>'
                         f'<text x="{sx(t):.1f}" y="{H - 14}" text-anchor="middle" class="tick">{t:.0f}</text>')
        g.append(f'<text x="{Lm - 14}" y="{H - 14}" text-anchor="end" class="tick">attempt number</text>')
        for li, (arm, j, run) in enumerate(lanes):
            y = T + li * rh + rh / 2
            g.append(f'<text x="{Lm - 14}" y="{y + 4:.1f}" text-anchor="end" class="row-label">'
                     f'{esc(ARM_LABEL[arm])} · run {j + 1} · seat {esc(run["seat"])}</text>')
            counts = defaultdict(int)
            for i, rec in enumerate(run["attempts"]):
                counts[rec["verdict"]] += 1
                label, color, kind = VERDICT.get(rec["verdict"], (rec["verdict"], "var(--muted)", "circle"))
                base_name, diff = diff_for(rec, run["attempts"], i, start_code)
                an = rec["attempt_no"] if rec["attempt_no"] is not None else i
                did = len(details)
                details.append(dict(
                    title=f"{k} · {ARM_LABEL[arm]} · run {j + 1} · attempt {an}",
                    facts=[["Verdict", label], ["Speedup", fmt_x(rec["speedup"])],
                           ["Time", fmt_us(rec["time_us_median"]) + (f" ± {fmt_us(rec['time_us_iqr'])} IQR"
                                                                      if rec["time_us_iqr"] else "")],
                           ["Start kernel, same session", fmt_us(rec["baseline_us_same_session"])],
                           ["Source", rec["source"] or "–"], ["Round", str(rec["round"])],
                           ["Prompt tokens", str(rec["prompt_tokens"] or "–")],
                           ["Seat · run id", f"{rec['seat']} · {rec['run_id']}"]],
                    message=(rec["referee_message"] or "")[:TEXT_CHARS],
                    instruction=(rec["instruction_given"] or "")[:TEXT_CHARS],
                    diff_base=base_name, diff=diff))
                x = sx(i)
                t_ = tip(f"{label}" + (f" · {fmt_x(rec['speedup'])}" if rec["speedup"] else ""),
                         f"attempt {an} · {ARM_LABEL[arm]} run {j + 1}", "click for the code diff")
                g.append(f'<g class="dot" data-id="{did}" tabindex="0" role="button" data-tip="{t_}" '
                         f'aria-label="attempt {an}: {esc(label)}">'
                         f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="transparent"/>'
                         f'<circle class="sel" cx="{x:.1f}" cy="{y:.1f}" r="{size + 4:.1f}"/>'
                         f'{glyph(kind, x, y, size, color)}</g>')
            rows_t.append([esc(k), f"{ARM_LABEL[arm]} · run {j + 1}", esc(run["seat"])] +
                          [str(counts.get(v, 0)) for v in schema.VERDICTS] + [str(len(run["attempts"]))])
        svgs.append(f'<h3>{esc(k)}</h3><div class="scroll"><svg class="wide" viewBox="0 0 {W} {H}" role="img" '
                    f'aria-label="{esc(k)}: every attempt, by verdict">{"".join(g)}</svg></div>')
    body = "".join(svgs) or empty("No attempts logged yet.")
    head = ["Kernel", "Run", "Seat"] + [VERDICT[v][0] if v in VERDICT else v for v in schema.VERDICTS] + ["Total"]
    detail = ('<div id="detail" class="detail" aria-live="polite"><p class="empty">Click any mark, or focus it and '
              'press Enter, to see the referee\'s message, the one instruction it sent back, and the code change.</p></div>')
    return card("4", "Every attempt",
                "One mark per attempt, in order, coloured and shaped by how far it got through the referee.",
                legend_verdicts() + body + detail + table_view(table(head, rows_t, numeric=tuple(range(3, len(head)))))), details


# ---------------------------------------------------------------- panel 5: held-out map

def panel_heldout(results):
    ho = results["heldout"]
    if not ho:
        return card("5", "Held-out shapes", "Shapes the agent never saw while it searched.",
                    empty("Not in yet: the <code>heldout</code> section of a <code>results*.json</code> "
                          "(format at the top of <code>dashboard/build.py</code>)."))
    by_kernel = defaultdict(list)
    for h in ho:
        by_kernel[h.get("kernel", "?")].append(h)
    blocks = []
    for k, hs in by_kernel.items():
        shapes = list(dict.fromkeys(h.get("shape", "?") for h in hs))
        whiches = list(dict.fromkeys(h.get("which", "?") for h in hs))
        cell = {(h.get("which"), h.get("shape")): h for h in hs}
        rows = []
        for w in whiches:
            row = [f'<strong>{esc(w)} kernel</strong>']
            for sh in shapes:
                h = cell.get((w, sh))
                if h is None:
                    row.append('<span class="note">–</span>')
                elif h.get("passed"):
                    row.append(status(True, f"pass · {fmt_x(h.get('speedup'))}" if h.get("speedup") else "pass", ""))
                else:
                    row.append(status(False, "", "fail"))
            rows.append(row)
        fails = sum(1 for h in hs if not h.get("passed"))
        blocks.append(f'<h3>{esc(k)} <span class="src">{fails} failing cell{"s" if fails != 1 else ""}</span></h3>' +
                      table(["Kernel"] + [esc(s) for s in shapes], rows))
    return card("5", "Held-out shapes",
                "Shapes the agent never saw while it searched, including tile edges. A speedup that breaks "
                "here does not count.", "".join(blocks))


# ---------------------------------------------------------------- page

def card(n, title, how, body):
    return (f'<section class="card" aria-labelledby="p{n}"><h2 id="p{n}"><span class="pn">{n}</span>{esc(title)}</h2>'
            f'<p class="how">{esc(how)}</p>{body}</section>')


def kpis(summary, results, records):
    tiles = []
    for k, s in summary.items():
        rs = s["arms"]["referee"]
        v = fmt_x(median(r["best_x"] for r in rs)) if rs else "–"
        sub = (f"model + referee · median of {len(rs)} run{'s' if len(rs) != 1 else ''} · {source_label(s['sources'])}"
               if rs else "no model + referee runs yet")
        tiles.append((f"{k}: best verified speedup", v, sub))
    rt = results["redteam"]
    if rt:
        control, cheats, caught, pending = redteam_split(rt)
        sub = "planted cheats caught" + (f" · {pending} pending" if pending else "")
        if any(c["state"] != "pass" for c in control):
            sub += " · honest kernel rejected: not trustworthy"
        if results["redteam_meta"].get("dry"):
            sub += " · dry run"
        tiles.append(("Red team", f"{caught} of {len(cheats)}", sub))
    else:
        tiles.append(("Red team", "–", "not in yet"))
    runs = {(r["kernel"], r["arm"], r["run_id"]) for r in records}
    seats = {r["seat"] for r in records}
    tiles.append(("Attempts through the referee", f"{len(records):,}",
                  f"{len(runs)} runs · {len(seats)} seat{'s' if len(seats) != 1 else ''}"))
    return '<div class="kpis">' + "".join(
        f'<div class="tile"><div class="tile-label">{esc(a)}</div><div class="tile-value">{esc(b)}</div>'
        f'<div class="tile-sub">{esc(c)}</div></div>' for a, b, c in tiles) + "</div>"


CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink-1:#0b0b0b;--ink-2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;
--neutral-bar:#898781;--v-rules:#52514e;--good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;
--good-text:#006300;--critical-text:#b02a2a;--add:rgba(12,163,12,.13);--del:rgba(208,59,59,.13);--wash:rgba(11,11,11,.04)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink-1:#fff;--ink-2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;
--s3:#199e70;--v-rules:#c3c2b7;--good-text:#0ca30c;--critical-text:#e66767;--add:rgba(12,163,12,.22);
--del:rgba(208,59,59,.25);--wash:rgba(255,255,255,.05)}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink-1:#fff;--ink-2:#c3c2b7;--grid:#2c2c2a;
--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--v-rules:#c3c2b7;
--good-text:#0ca30c;--critical-text:#e66767;--add:rgba(12,163,12,.22);--del:rgba(208,59,59,.25);--wash:rgba(255,255,255,.05)}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink-1);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}
.lede{color:var(--ink-2);margin:0 0 16px;max-width:760px}
.meta{color:var(--muted);font-size:13px;margin:0 0 20px}
.fake{border:1px solid var(--border);border-left:4px solid var(--critical);background:var(--surface);border-radius:8px;
padding:12px 16px;margin:0 0 20px}
.fake strong{color:var(--critical-text)}
.notes{font-size:13px;color:var(--ink-2);margin:0 0 20px}
.notes summary{cursor:pointer}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(210px,100%),1fr));gap:12px;margin:0 0 20px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.tile-label{font-size:13px;color:var(--ink-2)}
.tile-value{font-size:30px;font-weight:600;margin:2px 0}
.tile-sub{font-size:12px;color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;margin:0 0 20px;
min-width:0}
@media (max-width:560px){.card{padding:16px 12px}}
.card h2{font-size:18px;margin:0 0 4px;display:flex;align-items:center;gap:10px}
.pn{display:inline-grid;place-items:center;width:24px;height:24px;border-radius:6px;background:var(--wash);
font-size:13px;color:var(--ink-2)}
.how{color:var(--ink-2);font-size:13px;margin:0 0 12px;max-width:780px}
h3{font-size:15px;margin:18px 0 6px}
.src{font-weight:400;font-size:12px;color:var(--muted);margin-left:6px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink-2);margin:0 0 6px}
.key{display:inline-flex;align-items:center;gap:6px}
.key svg{width:14px;height:14px;flex:none}
.warn{font-size:13px;color:var(--ink-1);margin:4px 0 0}
.warn .icon{display:inline-grid;place-items:center;width:18px;height:18px;border-radius:50%;background:var(--critical);
color:#fff;font-size:12px;font-weight:700;margin-right:6px}
.line-key{display:inline-block;width:16px;height:2px;border-radius:1px}
.scroll{overflow-x:auto}
svg{display:block;width:100%;height:auto;overflow:visible}
svg.wide{min-width:680px}
svg text{fill:var(--ink-2);font-size:12px;font-family:inherit}
svg .tick{fill:var(--muted);font-variant-numeric:tabular-nums}
svg .row-label{fill:var(--ink-1);font-size:13px}
svg .value{fill:var(--ink-1);font-size:12px}
svg .note{fill:var(--muted);font-style:italic}
svg .ref-label{fill:var(--ink-2);font-size:11px}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.ref{stroke:var(--ink-2);stroke-width:1}
.whisker{stroke:var(--ink-1);stroke-width:1.5;fill:none}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.band{opacity:.10}
.end-dot{stroke:var(--surface);stroke-width:2}
.hair{stroke:var(--ink-2);stroke-width:1;visibility:hidden;pointer-events:none}
.mark{outline:none;cursor:default}
.mark:hover path,.mark:hover rect:not([fill=transparent]),.mark:focus-visible path{filter:brightness(1.12)}
.mark:focus-visible rect[fill=transparent]{stroke:var(--ink-1);stroke-width:1}
.dot{cursor:pointer;outline:none}
.dot .sel{fill:none;stroke:var(--ink-1);stroke-width:1.5;visibility:hidden}
.dot:hover .sel{visibility:visible;opacity:.45}
.dot:focus-visible .sel,.dot.selected .sel{visibility:visible;opacity:1}
.multiples{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(380px,100%),1fr));gap:8px 24px}
figure.small{margin:8px 0 0}
figcaption{font-weight:600;font-size:15px;margin:0 0 4px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink-2);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.status{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.status .icon{display:inline-grid;place-items:center;width:18px;height:18px;border-radius:50%;color:#fff;
font-size:12px;font-weight:700}
.status.good .icon{background:var(--good)}
.status.neutral .icon{background:var(--muted)}
.status.neutral{color:var(--ink-2)}
.status.critical .icon{background:var(--critical)}
.status.good{color:var(--good-text)}
.status.critical{color:var(--critical-text)}
.note{color:var(--muted)}
details.tv{margin-top:12px;font-size:13px}
details.tv summary{cursor:pointer;color:var(--ink-2)}
.empty{color:var(--muted);font-size:14px;margin:8px 0}
.detail{margin-top:14px;border-top:1px solid var(--grid);padding-top:12px}
.detail h3{margin:0 0 8px}
.detail dl{display:grid;grid-template-columns:max-content 1fr;gap:2px 16px;font-size:13px;margin:0 0 10px}
.detail dt{color:var(--ink-2)}
.detail dd{margin:0;font-variant-numeric:tabular-nums}
.detail .label{font-size:12px;color:var(--ink-2);margin:10px 0 2px;font-weight:600}
.detail .text{white-space:pre-wrap;margin:0;font-size:13px}
pre.diff{font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--wash);border-radius:8px;
padding:10px 12px;overflow-x:auto;max-height:420px;margin:4px 0 0}
pre.diff .add{background:var(--add);display:block}
pre.diff .del{background:var(--del);display:block}
pre.diff .hunk{color:var(--muted);display:block}
code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.92em}
#tip{position:fixed;z-index:10;pointer-events:none;background:var(--surface);color:var(--ink-1);
border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.12);
max-width:320px}
.tip-value{font-weight:600;font-size:13px}
.tip-label{color:var(--ink-2)}
.tip-row{display:flex;align-items:center;gap:8px}
.tip-key{display:inline-block;width:12px;height:2px;border-radius:1px;flex:none}
footer{color:var(--muted);font-size:12px}
"""

JS = r"""
(() => {
  const tip = document.getElementById('tip');
  const data = JSON.parse(document.getElementById('attempt-data').textContent);
  function place(x, y) {
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    let left = x + 14, top = y + 14;
    if (left + r.width > innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > innerHeight - 8) top = y - r.height - 14;
    tip.style.left = Math.max(8, left) + 'px';
    tip.style.top = Math.max(8, top) + 'px';
  }
  function showLines(lines, x, y) {
    tip.replaceChildren();
    lines.forEach((l, i) => {
      const d = document.createElement('div');
      d.className = i ? 'tip-label' : 'tip-value';
      d.textContent = l;
      tip.appendChild(d);
    });
    place(x, y);
  }
  const hide = () => { tip.hidden = true; };
  document.addEventListener('pointerover', e => {
    const t = e.target.closest('[data-tip]');
    if (t) showLines(t.dataset.tip.split('\n'), e.clientX, e.clientY);
  });
  document.addEventListener('pointermove', e => {
    if (e.target.closest('[data-tip]') && !tip.hidden) place(e.clientX, e.clientY);
  });
  document.addEventListener('pointerout', e => {
    const t = e.target.closest('[data-tip]');
    if (t && !t.contains(e.relatedTarget)) hide();
  });
  document.addEventListener('focusin', e => {
    const t = e.target.closest('[data-tip]');
    if (t) { const r = t.getBoundingClientRect(); showLines(t.dataset.tip.split('\n'), r.right, r.top); }
  });
  document.addEventListener('focusout', hide);

  // Progress curves: a crosshair that snaps to the nearest attempt and reads out every arm.
  document.querySelectorAll('svg[data-curves]').forEach(svg => {
    const c = JSON.parse(svg.dataset.curves);
    const hair = svg.querySelector('.hair'), overlay = svg.querySelector('.overlay');
    const pt = svg.createSVGPoint();
    overlay.addEventListener('pointermove', e => {
      pt.x = e.clientX; pt.y = e.clientY;
      const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      const xv = c.dx0 + (p.x - c.px0) / (c.px1 - c.px0) * (c.dx1 - c.dx0);
      const i = Math.max(0, Math.min(c.xs.length - 1, Math.round(xv) - c.xs[0]));
      const x = c.px0 + (c.xs[i] - c.dx0) / (c.dx1 - c.dx0) * (c.px1 - c.px0);
      hair.setAttribute('x1', x); hair.setAttribute('x2', x); hair.style.visibility = 'visible';
      tip.replaceChildren();
      const head = document.createElement('div');
      head.className = 'tip-label';
      head.textContent = 'after ' + c.xs[i] + ' attempt' + (c.xs[i] === 1 ? '' : 's');
      tip.appendChild(head);
      c.series.forEach(s => {
        const v = s.med[i];
        if (v === null || v === undefined) return;
        const row = document.createElement('div'); row.className = 'tip-row';
        const key = document.createElement('span'); key.className = 'tip-key'; key.style.background = s.color;
        const val = document.createElement('strong'); val.textContent = v.toFixed(2) + '×';
        const lab = document.createElement('span'); lab.className = 'tip-label';
        lab.textContent = s.label + (s.n[i] > 1
          ? ' · runs ' + s.lo[i].toFixed(2) + '–' + s.hi[i].toFixed(2) + ', n=' + s.n[i] : ' · 1 run');
        row.append(key, val, lab); tip.appendChild(row);
      });
      place(e.clientX, e.clientY);
    });
    overlay.addEventListener('pointerleave', () => { hair.style.visibility = 'hidden'; hide(); });
  });

  // Attempt timeline: click or Enter on a mark shows the attempt.
  const detail = document.getElementById('detail');
  function add(parent, tag, cls, text) {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined) el.textContent = text;
    parent.appendChild(el);
    return el;
  }
  function open(id) {
    const a = data[id];
    if (!a || !detail) return;
    detail.replaceChildren();
    add(detail, 'h3', '', a.title);
    const dl = add(detail, 'dl');
    a.facts.forEach(([k, v]) => { add(dl, 'dt', '', k); add(dl, 'dd', '', v); });
    add(detail, 'div', 'label', "Referee's message");
    add(detail, 'p', 'text', a.message || '–');
    add(detail, 'div', 'label', 'Instruction sent back');
    add(detail, 'p', 'text', a.instruction || '–');
    add(detail, 'div', 'label', 'Code: ' + a.diff_base);
    if (a.diff) {
      const pre = add(detail, 'pre', 'diff');
      a.diff.split('\n').forEach(line => {
        const cls = line.startsWith('@@') ? 'hunk'
          : line.startsWith('+') && !line.startsWith('+++') ? 'add'
          : line.startsWith('-') && !line.startsWith('---') ? 'del' : '';
        add(pre, 'span', cls, line + (cls ? '' : '\n'));
      });
    }
    document.querySelectorAll('.dot.selected').forEach(d => d.classList.remove('selected'));
    const dot = document.querySelector('.dot[data-id="' + id + '"]');
    if (dot) dot.classList.add('selected');
    detail.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  }
  document.addEventListener('click', e => {
    const d = e.target.closest('.dot[data-id]');
    if (d) open(+d.dataset.id);
  });
  document.addEventListener('keydown', e => {
    const d = e.target.closest && e.target.closest('.dot[data-id]');
    if (d && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); open(+d.dataset.id); }
  });
})();
"""


def build(records, results, notes, fake, inputs):
    summary = summarize(records)
    timeline, details = panel_timeline(summary)
    banner = ('<div class="fake" role="alert"><strong>Fake data.</strong> Generated by <code>schema.py --fake</code> '
              'to build the layout. Never report a number from this page.</div>') if fake else ""
    sources = set().union(*(s["sources"] for s in summary.values())) if summary else set()
    built = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    meta = (f"Built {built} from {len(inputs)} file{'s' if len(inputs) != 1 else ''} · "
            f"{len(records):,} attempts · times: {source_label(sources)}")
    note_html = ""
    if notes:
        note_html = (f'<details class="notes"><summary>{len(notes)} log line{"s" if len(notes) != 1 else ""} '
                     f'needed attention</summary><ul>' + "".join(f"<li>{esc(n)}</li>" for n in notes[:50]) +
                     ("<li>…</li>" if len(notes) > 50 else "") + "</ul></details>")
    data = json.dumps(details, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CHIPBOOST Dashboard</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>CHIPBOOST</h1>
<p class="lede">Qwen3-8B speeds up the kernels it is built from, on the Trainium chip it runs on. Only speedups
that pass the referee count: correct on the chip, correct on shapes the agent never saw, and faster by more
than the timing noise.</p>
<p class="meta">{esc(meta)}</p>
{banner}{note_html}{kpis(summary, results, records)}
{panel_speed(summary, results)}
{panel_progress(summary, results)}
{panel_redteam(results)}
{timeline}
{panel_heldout(results)}
<footer>Inputs: {esc(", ".join(inputs)) or "none"}. Speedups are against the start kernel timed in the same session.
Projections, if any, are labelled as such; everything else here was logged by the referee.</footer>
</main>
<div id="tip" role="tooltip" hidden></div>
<script type="application/json" id="attempt-data">{data}</script>
<script>{JS}</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="*", help="attempts*.jsonl files (default: every one under 03-chipboost/)")
    ap.add_argument("--results", nargs="*", help="results*.json files (default: every one under 03-chipboost/)")
    ap.add_argument("--fake", action="store_true", help="build from schema.fake_attempts(); stamped FAKE")
    ap.add_argument("-o", "--out", default=str(HERE / "index.html"))
    a = ap.parse_args()

    def found(pattern):
        return sorted(str(p) for p in PROJECT.rglob(pattern)
                      if ".git" not in p.parts and not p.name.startswith("fake_"))

    if a.fake:
        records, notes, inputs = [], [], ["schema.fake_attempts()"]
        for rec in schema.fake_attempts():
            for k in schema.ATTEMPT_FIELDS:
                rec.setdefault(k, None)
            records.append(rec)
    else:
        paths = a.logs or found("attempts*.jsonl")
        if not paths:
            sys.exit(f"no attempts*.jsonl under {PROJECT}. Pass log files, or --fake for the layout.")
        records, notes = load_attempts(paths)
        inputs = [Path(p).name for p in paths]
    fake = a.fake or any(str(r.get("referee_message") or "").startswith("(fake)") for r in records) \
        or any("fake" in Path(p).name.lower() for p in inputs)
    res_paths = a.results if a.results is not None else found("*results*.json")
    results = load_results(res_paths)
    if fake and not res_paths:
        results = merge_results([fake_results()])
        inputs.append("fake_results()")
    else:
        inputs += [Path(p).name for p in res_paths]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(records, results, notes, fake, inputs), encoding="utf-8")
    print(f"wrote {out}: {len(records)} attempts from {', '.join(inputs)}"
          + ("  [FAKE DATA]" if fake else "") + (f"  ({len(notes)} log lines need attention)" if notes else ""))


if __name__ == "__main__":
    main()
