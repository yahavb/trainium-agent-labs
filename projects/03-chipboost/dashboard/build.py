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
any number of files, merged, every section optional. A missing section shows as "not in yet". With no
logs at all (before the loops run) the page is built from the results files alone.

    results_p1.json                P1: red-team rows {cheat, description, honest, caught, ok, stage, message}
    redteam/redteam_results.json   P3's redteam/run.py: {referee, dry, rows: [...]}, read as it is
    results_p2.json                P2: start, expert and floor times per kernel
    results_heldout_<op>.json      P2's heldout_grid.py: the end-of-run held-out grid

    {"kernels": {"matmul": {"start_us": 960.5, "expert_us": 385.1, "floor_us": null, "source": "chip"}},
     "heldout": [{"kernel": "matmul", "which": "start" | "expert" | <arm>, "shape": "256x4096x6144",
                  "passed": true, "speedup": 1.18, "time_us": 512.3, "message": ""}]}

start_us is checked against the start kernel the logs timed: a mismatch means the expert and floor were
measured at other shapes, and the page says so.

"Verified" means verdict == "faster": correct on every shape, and faster by more than the noise.
A run that never produced one counts as the start kernel at 1.00x; no run is dropped.

A log line whose referee_message starts with "(fake)" (schema.py --fake) or "(stub)" (search.py --stub)
stamps the whole page, and names the files those lines came from.
"""

import argparse
import datetime
import difflib
import html
import json
import math
import random
import re
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
MARKERS = ("(fake)", "(stub)")   # how schema.py --fake and search.py --stub label what they invent


def marked(rec):
    return str(rec.get("referee_message") or "").startswith(MARKERS)


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
            rec["_file"] = str(path)
            records.append(rec)
    return records, notes


def caught_state(value):
    """redteam/run.py writes "yes", "NO", "PENDING" (chip stage not run) and, for the honest
    control kernel, "PASS". results_p1.json writes booleans."""
    v = str(value).strip().lower()
    return {"true": "caught", "yes": "caught", "false": "missed", "no": "missed",
            "pending": "pending", "pass": "pass"}.get(v, v)


def redteam_rows(doc):
    """Both writers' red-team rows, as one shape.

    redteam/run.py (P3): {cheat, should_catch, caught: "yes"/"NO"/"PENDING"/"PASS", where, as_designed,
    message}; its honest control is the row named "honest ...".
    results_p1.json (P1): {cheat, description, honest, caught, ok, verdict, stage, message}; an honest
    kernel has honest=true, and ok=true when the referee accepted it."""
    out = []
    for row in (doc.get("rows") or []) + (doc.get("redteam") or []):
        honest = str(row.get("honest")).lower() == "true" or str(row.get("cheat", "")).startswith("honest")
        state = caught_state(row.get("caught"))
        ok = str(row.get("ok")).lower() == "true" if row.get("ok") is not None else None
        if honest:
            accepted = ok if ok is not None else state in ("pass", "missed")
            state = "pass" if accepted else "false alarm"
        elif state == "missed" and ok:
            # P1: not caught, but not faster either (verdict slower), so cheating bought nothing.
            state = "no gain"
        what = row.get("description") or (f"should be caught by {row['should_catch']}"
                                          if row.get("should_catch") and not honest else "")
        out.append(dict(cheat=row.get("cheat", "?"), honest=honest, state=state, what=what,
                        where=row.get("where") or row.get("stage") or "", as_designed=row.get("as_designed"),
                        message=row.get("message") or ""))
    return out


def load_results(paths, name=lambda p: Path(p).name):
    return merge_results((name(p), json.loads(read_text(p))) for p in paths)


def merge_results(named_docs):
    """kernels merge key by key, held-out rows concatenate, and each file's red-team rows stay one
    suite: P1's and P3's are run against different referees, so their counts are not one number."""
    merged = {"kernels": {}, "heldout": [], "suites": []}
    for name, doc in named_docs:
        for k, v in (doc.get("kernels") or {}).items():
            merged["kernels"].setdefault(k, {}).update(v)
        merged["heldout"] += doc.get("heldout") or []
        rows = redteam_rows(doc)
        if rows:
            merged["suites"].append(dict(name=name, rows=rows, dry=doc.get("dry"),
                                         referee=doc.get("referee") or doc.get("referee_version")))
    merged["redteam"] = [r for s in merged["suites"] for r in s["rows"]]
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
    for r in records:   # by seat too: two seats can pick the same run_id
        runs[(r["kernel"], r["arm"], str(r["seat"]), r["run_id"])].append(r)
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
                rs.append(dict(run_id=key[3], seat=lst[0]["seat"], attempts=lst, curve=curve,
                               best_us=min(r["time_us_median"] for r in ver) if ver else run_base,
                               best_x=max(r["speedup"] for r in ver) if ver else 1.0,
                               n_verified=len(ver)))
            arms[arm] = rs
        out[k] = dict(start_us=start_us, arms=arms, sources={r["source"] for r in recs if r["source"]})
    return out


CAP_RE = [re.compile(rf"^{n}\s*=\s*(\d+)", re.M) for n in ("TILES_IN_BLOCK_M", "TILES_IN_BLOCK_N", "TILES_IN_BLOCK_K")]
SHIPPED_CAPS = (16, 2, 8)   # kernels/matmul_expert.py as AWS ships it: search.py's attempt 0


def caps_of(code):
    """search.py's three block caps, read back from the candidate's source; None if they are not there."""
    got = [r.search(code or "") for r in CAP_RE]
    return tuple(int(m.group(1)) for m in got) if all(got) else None


def tuning(summary, sweep):
    """Random search tunes the expert's block sizes and starts FROM the expert, so it is measured against
    the expert as shipped -- each run's attempt 0, timed in that run's session -- not the start kernel.
    The sweep (search.py --exhaustive, sweep-*.jsonl) times every legal setting once: its best is the
    ceiling for this arm, and each run's best is ranked among its settings."""
    timed = sorted((x for x in sweep if x["speedup"] and x["verdict"] in ("faster", "no_gain", "slower")),
                   key=lambda x: -x["speedup"])
    ref = next((x["speedup"] for x in timed if caps_of(x["code"]) == SHIPPED_CAPS), None)
    order = [caps_of(x["code"]) for x in timed]
    sw = (dict(n=len(timed), best=timed[0]["speedup"] / ref, best_caps=caps_of(timed[0]["code"]))
          if timed and ref else None)
    runs = []
    for r in (summary.get("matmul") or {}).get("arms", {}).get("random_search", []):
        a0 = r["attempts"][0] if r["attempts"] else None
        base = a0["speedup"] if a0 and a0["speedup"] and caps_of(a0["code"]) == SHIPPED_CAPS else None
        if not base:
            continue
        best = max((a for a in r["attempts"] if verified(a)), key=lambda a: a["speedup"], default=a0)
        caps = caps_of(best["code"])
        runs.append(dict(run_id=r["run_id"], seat=r["seat"], curve=[c / base for c in r["curve"]],
                         best=r["curve"][-1] / base, caps=caps,
                         rank=order.index(caps) + 1 if sw and caps in order else None))
    return dict(runs=runs, sweep=sw) if runs or sw else None


def kernels_of(summary, results):
    """Every op with logs or with numbers in a results file, in schema order: before the loops run, the
    page still shows P2's start, expert and floor times."""
    return [op for op in schema.OPS if op in summary or op in results["kernels"]]


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
    """Square at the baseline, rounded at the data end."""
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


def table_view(inner, label="Table view"):
    return f'<details class="tv"><summary>{esc(label)}</summary>{inner}</details>' if inner else ""


def empty(msg):
    return f'<p class="empty">{msg}</p>'


def badge(kind, icon, text):
    return f'<span class="status {kind}"><span class="icon" aria-hidden="true">{icon}</span>{esc(text)}</span>'


def status(ok, yes, no):
    return badge("good", "✓", yes) if ok else badge("critical", "✗", no)


def warn(html_text):
    return f'<p class="warn"><span class="icon" aria-hidden="true">!</span>{html_text}</p>'


def card(title, how, body, cls=""):
    return (f'<section class="card {cls}"><h2>{esc(title)}</h2>'
            + (f'<p class="how">{esc(how)}</p>' if how else "") + body + "</section>")


def shape_text(shape):
    return str(shape).replace("x", "×")


# ---------------------------------------------------------------- screen 1: speed

def start_mismatch(k, s, info):
    """results_*.json's start_us against the start kernel the logs timed. Expert and floor times are only
    comparable to the logs if they were measured at the same shapes, and the start time is the one number
    both sides have. A shape change in shapes.py after the results were written shows up here."""
    logged, written = s["start_us"] if s else None, info.get("start_us")
    if not logged or not written or 0.8 <= logged / written <= 1.25:
        return ""
    return warn(f"<strong>Different shapes?</strong> The logs timed the start kernel at {fmt_us(logged)}, "
                f"the results file says {fmt_us(written)}: re-measure the expert and floor before comparing.")


def kernel_view(k, summary, results):
    """One kernel's numbers from wherever they exist: the logs first, the results files otherwise."""
    info = results["kernels"].get(k, {})
    s = summary.get(k) or dict(start_us=None, arms={a: [] for a in schema.ARMS},
                               sources={info["source"]} if info.get("source") else set())
    start = s["start_us"] or info.get("start_us")
    expert, floor = info.get("expert_us"), info.get("floor_us")
    return s, info, start, (start / expert if start and expert else None), (start / floor if start and floor else None)


def panel_speed(summary, results, tune):
    blocks, rows_t = [], []
    tuned = median(r["best"] for r in tune["runs"]) if tune and tune["runs"] else None
    for k in kernels_of(summary, results):
        s, info, start, expert_x, limit_x = kernel_view(k, summary, results)
        rows = []
        for arm in schema.ARMS:
            rs = [r for r in s["arms"][arm] if r["best_us"] is not None]
            if rs:
                xs = [r["best_x"] for r in rs]
                rows.append((ARM_LABEL[arm], ARM_COLOR[arm], median(xs), min(xs), max(xs), len(rs),
                             median(r["best_us"] for r in rs)))
            else:
                rows.append((ARM_LABEL[arm], ARM_COLOR[arm], None, None, None, 0, None))
        if expert_x:
            rows.append(("Expert kernel", "var(--neutral-bar)", expert_x, None, None, None, info["expert_us"]))
        top = max([v for row in rows for v in (row[2], row[4]) if v] + [1.0] + ([limit_x] if limit_x else []))
        _, hi, ticks = nice_domain(0, top * 1.04, 5)
        W, L, R, T, rh, bh = 540, 112, 70, 20, 26, 12
        H = T + rh * len(rows) + 20
        sx = lambda v: L + v / hi * (W - L - R)
        g = []
        for t in ticks:
            g.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{T - 2}" y2="{H - 18}" class="grid"/>'
                     f'<text x="{sx(t):.1f}" y="{H - 5}" text-anchor="middle" class="tick">{t:g}×</text>')
        g.append(f'<line x1="{sx(1):.1f}" x2="{sx(1):.1f}" y1="{T - 6}" y2="{H - 18}" class="ref"/>'
                 f'<text x="{sx(1) - 4:.1f}" y="{T - 9}" text-anchor="end" class="ref-label">start 1×</text>')
        if limit_x:
            right = sx(limit_x) + 112 <= W   # the label goes right of the line unless it would run off
            g.append(f'<line x1="{sx(limit_x):.1f}" x2="{sx(limit_x):.1f}" y1="{T - 6}" y2="{H - 18}" class="ref"/>'
                     f'<text x="{sx(limit_x) + (4 if right else -4):.1f}" y="{T - 9}" '
                     f'text-anchor="{"start" if right else "end"}" class="ref-label">floor: {fmt_x(limit_x)} at most</text>')
        for i, (name, color, x, lo, hi_, n, us) in enumerate(rows):
            y = T + i * rh + (rh - bh) / 2
            g.append(f'<text x="{L - 10}" y="{y + bh / 2 + 4:.1f}" text-anchor="end" class="row-label">{esc(name)}</text>')
            if x is None:
                g.append(f'<text x="{L + 4}" y="{y + bh / 2 + 4:.1f}" class="note">no runs yet</text>')
                rows_t.append([esc(k), esc(name), "", "", "", "", "0"])
                continue
            spread = (f"runs ranged {fmt_x(lo)} to {fmt_x(hi_)}, n={n}" if n and n > 1 else
                      "1 run" if n == 1 else "hand-optimised ceiling")
            g.append(f'<g class="mark" tabindex="0" data-tip="{tip(fmt_x(x), name, fmt_us(us), spread)}">'
                     f'<rect x="{L}" y="{y - 6:.1f}" width="{W - L - R:.1f}" height="{bh + 12}" fill="transparent"/>'
                     f'{bar_path(L, sx(x), y, bh)} fill="{color}"/></g>')
            end = sx(x)
            if lo is not None and hi_ is not None and hi_ > lo:
                yc = y + bh / 2
                g.append(f'<path d="M{sx(lo):.1f},{yc:.1f} H{sx(hi_):.1f} M{sx(lo):.1f},{yc - 4:.1f} V{yc + 4:.1f} '
                         f'M{sx(hi_):.1f},{yc - 4:.1f} V{yc + 4:.1f}" class="whisker"/>')
                end = max(end, sx(hi_))
            over = (f" · expert {(tuned - 1) * 100:+.0f}%" if tuned and k == "matmul"
                    and name == ARM_LABEL["random_search"] else "")
            g.append(f'<text x="{end + 6:.1f}" y="{y + bh / 2 + 4:.1f}" class="value strong">{fmt_x(x)}{over}</text>')
            rows_t.append([esc(k), esc(name), fmt_x(x), fmt_x(lo) if lo else "", fmt_x(hi_) if hi_ else "",
                           fmt_us(us), str(n or "")])
        below = [r for rs in s["arms"].values() for r in rs
                 if info.get("floor_us") and r["n_verified"] and r["best_us"] < info["floor_us"]]
        warns = (warn(f"<strong>Below the floor:</strong> {len(below)} run(s) beat the copy floor "
                      f"({fmt_us(info['floor_us'])}). Either the timer is wrong, or the floor kernel is not the "
                      f"fastest copy: re-measure before reporting.") if below else "") + start_mismatch(k, summary.get(k), info)
        blocks.append(f'<div class="kblock"><h3>{esc(k)} <span class="src">start {esc(fmt_us(start))} · '
                      f'{esc(source_label(s["sources"]))}</span></h3><svg viewBox="0 0 {W} {H}" role="img" '
                      f'aria-label="{esc(k)}: speedup per arm, expert and floor">{"".join(g)}</svg>{warns}</div>')
    body = "".join(blocks) or empty("No kernels yet.")
    return card("Speedup over the start kernel",
                "Bar: median of each run's best verified speedup; whisker: fastest to slowest run. Random search "
                "starts from the expert, so its bar includes the expert's own speedup.",
                legend_arms() + body + table_view(table(
                    ["Kernel", "Row", "Speedup", "Fastest run", "Slowest run", "Time", "Runs"], rows_t,
                    numeric=(2, 3, 4, 5, 6))))


def step_points(xs, ys):
    pts = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i:
            pts.append((x, pts[-1][1]))
        pts.append((x, y))
    return pts


MODEL_ARMS = ("referee", "model_alone")   # both start from the start kernel; random search starts from the expert


def panel_progress(summary, results, tune):
    blocks, rows_t = [], []
    for k, s in summary.items():
        L_max = max((len(r["curve"]) for arm in MODEL_ARMS for r in s["arms"][arm]), default=0)
        if not L_max:
            continue
        xs = list(range(1, L_max + 1))
        series = []
        for arm in MODEL_ARMS:
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
            series.append(dict(label=ARM_LABEL[arm], color=ARM_COLOR[arm], med=med, lo=lo, hi=hi, n=n))
            last = max(i for i, v in enumerate(med) if v is not None)
            at = [med[max(0, math.ceil((last + 1) * f) - 1)] for f in (0.25, 0.5, 0.75, 1.0)]
            rows_t.append([esc(k), ARM_LABEL[arm], str(len(rs)), str(last + 1)] + [fmt_x(v) for v in at] +
                          [f"{fmt_x(lo[last])} to {fmt_x(hi[last])}"])
        _, _, _, expert_x, _ = kernel_view(k, summary, results)
        top = max([v for sr in series for v in sr["hi"] if v] + ([expert_x] if expert_x else []) + [1.1])
        y0, y1, yt = nice_domain(1.0, top, 4)
        W, H, Lm, Rm, T, B = 540, 168, 44, 104, 10, 30
        dx0, dx1 = 1, max(L_max, 2)
        sx = lambda v: Lm + (v - dx0) / (dx1 - dx0) * (W - Lm - Rm)
        sy = lambda v: T + (1 - (v - y0) / (y1 - y0)) * (H - T - B)
        g = []
        for t in yt:
            g.append(f'<line x1="{Lm}" x2="{W - Rm}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="{"axis" if t == y0 else "grid"}"/>'
                     f'<text x="{Lm - 6}" y="{sy(t) + 4:.1f}" text-anchor="end" class="tick">{t:g}×</text>')
        _, _, xt = nice_domain(dx0, dx1, 6)
        for t in xt:
            if dx0 <= t <= dx1:
                g.append(f'<text x="{sx(t):.1f}" y="{H - B + 15}" text-anchor="middle" class="tick">{t:.0f}</text>')
        g.append(f'<text x="{W - Rm}" y="{H - 2}" text-anchor="end" class="tick">attempts →</text>')
        if expert_x:
            g.append(f'<line x1="{Lm}" x2="{W - Rm}" y1="{sy(expert_x):.1f}" y2="{sy(expert_x):.1f}" class="ref"/>'
                     f'<text x="{W - Rm + 6}" y="{sy(expert_x) + 4:.1f}" class="ref-label">expert {fmt_x(expert_x)}</text>')
        for sr in series:
            idx = [i for i, v in enumerate(sr["med"]) if v is not None]
            px = [xs[i] for i in idx]
            if max(sr["n"][i] for i in idx) > 1:
                up = step_points(px, [sr["hi"][i] for i in idx])
                dn = step_points(px, [sr["lo"][i] for i in idx])[::-1]
                g.append('<polygon points="' + " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in up + dn) +
                         f'" fill="{sr["color"]}" class="band"/>')
            g.append('<polyline points="' + " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in
                                                     step_points(px, [sr["med"][i] for i in idx])) +
                     f'" stroke="{sr["color"]}" class="line"/>')
        ends = sorted(((sy(sr["med"][max(i for i, v in enumerate(sr["med"]) if v is not None)]), sr) for sr in series),
                      key=lambda e: e[0])
        for y, sr in ends:
            last = max(i for i, v in enumerate(sr["med"]) if v is not None)
            g.append(f'<circle cx="{sx(xs[last]):.1f}" cy="{y:.1f}" r="4" fill="{sr["color"]}" class="end-dot"/>')
        if all(b[0] - a[0] >= 13 for a, b in zip(ends, ends[1:])):
            for y, sr in ends:
                last = max(i for i, v in enumerate(sr["med"]) if v is not None)
                g.append(f'<text x="{sx(xs[last]) + 8:.1f}" y="{y + 4:.1f}" class="value strong">{fmt_x(sr["med"][last])}</text>')
        g.append(f'<line class="hair" x1="0" x2="0" y1="{T}" y2="{H - B}"/>'
                 f'<rect class="overlay" x="{Lm}" y="{T}" width="{W - Lm - Rm}" height="{H - T - B}" fill="transparent"/>')
        curves = dict(xs=xs, dx0=dx0, dx1=dx1, px0=Lm, px1=W - Rm, series=series)
        blocks.append(f'<div class="kblock"><h3>{esc(k)} <span class="src">{esc(source_label(s["sources"]))}</span></h3>'
                      f'<svg viewBox="0 0 {W} {H}" role="img" data-curves="{esc(json.dumps(curves))}" '
                      f'aria-label="{esc(k)}: best verified speedup against attempts, per arm">{"".join(g)}</svg></div>')
    body = "".join(blocks) or empty("No model-arm attempts logged yet: this fills in as the loops run.")
    legend = ('<div class="legend">' + "".join(
        f'<span class="key"><span class="line-key" style="background:{ARM_COLOR[a]}"></span>{ARM_LABEL[a]}</span>'
        for a in MODEL_ARMS) + "</div>")
    return card("Progress: the model improves the start kernel",
                "Best verified speedup so far, from the start kernel (1×). Line: median over runs; band: fastest to "
                "slowest run. Same x = same budget.",
                legend + body + tuning_block(tune) + table_view(table(
                    ["Kernel", "Arm", "Runs", "Attempts", "After 25%", "After 50%", "After 75%", "At the end",
                     "Spread at the end"], rows_t, numeric=(2, 3, 4, 5, 6, 7))))


def tuning_block(tune):
    """Random search's own chart: speedup over the expert as shipped, against the sweep's best of every
    legal block setting. It starts from AWS's design, so it is a different claim from the model arms'."""
    if not tune:
        return ""
    runs, sw = tune["runs"], tune["sweep"]
    head = ('<h3 class="sep">Random search: tuning the expert\'s block sizes <span class="src">'
            'speedup over the expert as shipped (1×), so not comparable with the curves above</span></h3>')
    if not runs:
        return head + empty(f"No random-search runs yet. The sweep's best of all {sw['n']} settings is "
                            f"{fmt_x(sw['best'])} the expert.")
    L_max = max(len(r["curve"]) for r in runs)
    xs = list(range(1, L_max + 1))
    med, lo, hi, n = [], [], [], []
    for i in range(L_max):
        vals = [r["curve"][i] for r in runs if len(r["curve"]) > i]
        med.append(median(vals))
        lo.append(min(vals))
        hi.append(max(vals))
        n.append(len(vals))
    top = max(hi + ([sw["best"]] if sw else []) + [1.05])
    bottom = min(lo + [1.0])
    y0, y1, yt = nice_domain(bottom, top, 4)
    W, H, Lm, Rm, T, B = 540, 150, 44, 104, 10, 28
    dx0, dx1 = 1, max(L_max, 2)
    sx = lambda v: Lm + (v - dx0) / (dx1 - dx0) * (W - Lm - Rm)
    sy = lambda v: T + (1 - (v - y0) / (y1 - y0)) * (H - T - B)
    color = ARM_COLOR["random_search"]
    g = []
    for t in yt:
        g.append(f'<line x1="{Lm}" x2="{W - Rm}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="{"axis" if t == y0 else "grid"}"/>'
                 f'<text x="{Lm - 6}" y="{sy(t) + 4:.1f}" text-anchor="end" class="tick">{t:g}×</text>')
    _, _, xt = nice_domain(dx0, dx1, 6)
    for t in xt:
        if dx0 <= t <= dx1:
            g.append(f'<text x="{sx(t):.1f}" y="{H - B + 15}" text-anchor="middle" class="tick">{t:.0f}</text>')
    g.append(f'<line x1="{Lm}" x2="{W - Rm}" y1="{sy(1):.1f}" y2="{sy(1):.1f}" class="ref"/>'
             f'<text x="{W - Rm + 6}" y="{sy(1) + 4:.1f}" class="ref-label">expert as shipped</text>')
    if sw:
        g.append(f'<line x1="{Lm}" x2="{W - Rm}" y1="{sy(sw["best"]):.1f}" y2="{sy(sw["best"]):.1f}" class="ref"/>'
                 f'<text x="{W - Rm + 6}" y="{sy(sw["best"]) + 4:.1f}" class="ref-label">best of all {sw["n"]}: '
                 f'{fmt_x(sw["best"])}</text>')
    if max(n) > 1:
        up, dn = step_points(xs, hi), step_points(xs, lo)[::-1]
        g.append('<polygon points="' + " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in up + dn) +
                 f'" fill="{color}" class="band"/>')
    g.append('<polyline points="' + " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in step_points(xs, med)) +
             f'" stroke="{color}" class="line"/>'
             f'<circle cx="{sx(xs[-1]):.1f}" cy="{sy(med[-1]):.1f}" r="4" fill="{color}" class="end-dot"/>'
             f'<line class="hair" x1="0" x2="0" y1="{T}" y2="{H - B}"/>'
             f'<rect class="overlay" x="{Lm}" y="{T}" width="{W - Lm - Rm}" height="{H - T - B}" fill="transparent"/>')
    curves = dict(xs=xs, dx0=dx0, dx1=dx1, px0=Lm, px1=W - Rm,
                  series=[dict(label="Random search, over the expert", color=color, med=med, lo=lo, hi=hi, n=n)])
    ranks = [r["rank"] for r in runs if r["rank"]]
    note = (f'{len(runs)} run{"s" if len(runs) != 1 else ""}; median best <strong>{fmt_x(median(r["best"] for r in runs))}'
            f'</strong> the expert')
    if ranks and sw:
        note += f'; each run\'s best ranks #{", #".join(str(x) for x in ranks)} of the sweep\'s {sw["n"]} settings'
    return (head + f'<svg viewBox="0 0 {W} {H}" role="img" data-curves="{esc(json.dumps(curves))}" '
            f'aria-label="random search: best speedup over the expert against attempts">{"".join(g)}</svg>'
            f'<p class="how">{note}.</p>')


# ---------------------------------------------------------------- screen 2: trust

def redteam_split(rt):
    """The honest control kernels, which must pass, and the cheats, which must not."""
    control = [c for c in rt if c["honest"]]
    cheats = [c for c in rt if not c["honest"]]
    return control, cheats, sum(c["state"] == "caught" for c in cheats), sum(c["state"] == "pending" for c in cheats)


def redteam_tail(cheats, pending):
    """What the 'caught N of M' count leaves out, so a reader never takes a gap for a miss or the reverse."""
    no_gain = sum(c["state"] == "no gain" for c in cheats)
    missed = sum(c["state"] == "missed" for c in cheats)
    return "".join(f", {n} {what}" for n, what in ((no_gain, "not caught but gained nothing"),
                                                    (missed, "missed"), (pending, "pending")) if n)


STAGES = ("Rules scan", "Inputs untouched", "Simulator", "Byte count", "Chip", "Held-out shapes", "Timing",
          "Timing (pending)", "Not stopped", "Other", "Honest kernels")
CHIP = {"caught": ("caught", "✓", "Caught"), "missed": ("missed", "✗", "Missed"),
        "no gain": ("nogain", "–", "Not caught, gained nothing"), "pending": ("pending", "…", "Pending"),
        "pass": ("pass", "✓", "Honest, accepted"), "false alarm": ("alarm", "✗", "Honest, rejected: false alarm")}


def stage_of(c):
    if c["honest"]:
        return "Honest kernels"
    if c["state"] == "pending":
        return "Timing (pending)"
    if c["state"] in ("missed", "no gain"):
        return "Not stopped"
    w = c["where"].lower()
    for key, name in (("rules", "Rules scan"), ("input", "Inputs untouched"), ("byte", "Byte count"),
                      ("heldout", "Held-out shapes"), ("held-out", "Held-out shapes"), ("timing", "Timing"),
                      ("chip", "Chip"), ("simulat", "Simulator"), ("numerics", "Simulator"), ("crash", "Simulator")):
        if key in w:
            return name
    return "Other"


def suite_name(s):
    return {"results_p1.json": "P1 red team", "redteam_results.json": "P3 red team"}.get(
        Path(s["name"]).name, s["name"])


def panel_redteam(results):
    suites = results["suites"]
    if not suites:
        return card("Red team: can the referee be fooled?", "",
                    empty("Not in yet: <code>results_p1.json</code> (P1) and <code>redteam/redteam_results.json</code> "
                          "(P3) hold the red-team rows."))
    legend = '<div class="legend">' + "".join(
        f'<span class="key"><span class="chip {cls}{" round" if st in ("pass", "false alarm") else ""}" '
        f'aria-hidden="true">{icon}</span>{esc(label)}</span>' for st, (cls, icon, label) in CHIP.items()) + "</div>"
    blocks, rows_t = [], []
    for s in suites:
        control, cheats, caught, pending = redteam_split(s["rows"])
        honest_ok = sum(c["state"] == "pass" for c in control)
        groups = defaultdict(list)
        for c in s["rows"]:
            groups[stage_of(c)].append(c)
            where = c["where"] if c["where"] not in ("", "-") else ""
            rows_t.append([esc(suite_name(s)), esc(c["cheat"]), esc(CHIP.get(c["state"], ("", "", c["state"]))[2]),
                           esc(c["what"]), esc(where), esc(str(c["message"])[:TEXT_CHARS])])
        lines = []
        for stage in STAGES:
            cs = groups.get(stage)
            if not cs:
                continue
            chips = "".join(
                f'<span class="chip {CHIP.get(c["state"], ("other", "?", ""))[0]}'
                f'{" round" if c["honest"] else ""}" tabindex="0" data-tip="'
                f'{tip(c["cheat"], CHIP.get(c["state"], ("", "", c["state"]))[2], c["what"], c["where"], str(c["message"]).strip().splitlines()[0][:200] if str(c["message"]).strip() else "")}">'
                f'{CHIP.get(c["state"], ("", "?", ""))[1]}</span>' for c in cs)
            lines.append(f'<div class="wall-row"><span class="wall-label">{esc(stage)}</span>'
                         f'<span class="chips">{chips}</span><span class="wall-n">{len(cs)}</span></div>')
        warns = []
        if any(c["state"] != "pass" for c in control):
            warns.append("<strong>An honest kernel was rejected:</strong> no \"caught\" in this suite counts until "
                         "that false alarm is fixed.")
        if s["dry"]:
            warns.append("<strong>Dry run:</strong> rules stage only; says nothing about the cheats.")
        if "fallback" in str(s["referee"] or "").lower():
            warns.append("<strong>Run against the temporary fallback referee</strong> (rules + simulator), not the "
                         "hardened one.")
        head = (f'{esc(suite_name(s))} <span class="src">caught {caught} of {len(cheats)}'
                f'{redteam_tail(cheats, pending)}'
                + (f' · {honest_ok}/{len(control)} honest accepted' if control else "") + "</span>")
        blocks.append(f'<div class="suite"><h3>{head}</h3>{"".join(warn(w) for w in warns)}{"".join(lines)}</div>')
    _, cheats, caught, pending = redteam_split(results["redteam"])
    tail = redteam_tail(cheats, pending).lstrip(", ")
    return card(f"Red team: {caught} of {len(cheats)} cheats caught",
                (f"Also {tail}. " if tail else "") + "One square per cheating kernel, by the stage that stopped it; "
                "circles are honest kernels that must be accepted. Hover for the referee's message.",
                legend + "".join(blocks) + table_view(table(
                    ["Suite", "Kernel", "Result", "What it does", "Stopped at", "Referee's message"], rows_t)))


HEAT = 7    # buckets of the blue ramp; CSS .hc0-.hc6 hold the light and dark steps


def heat_bucket(s, top):
    if s is None or s <= 1.0:
        return 0
    return max(0, min(HEAT - 1, int(round(math.log(s) / math.log(max(top, 1.5)) * (HEAT - 1)))))


def heldout_from_logs(summary):
    """Per arm, from the attempt logs alone: how many would-be-faster kernels the referee tested on
    held-out shapes, and how many those shapes rejected."""
    items = []
    for k, s in summary.items():
        for arm in schema.ARMS:
            atts = [a for r in s["arms"][arm] for a in r["attempts"]]
            if not atts:
                continue
            fails = sum(a["verdict"] == "heldout_fail" for a in atts)
            checked = sum(a["verdict"] in ("heldout_fail", "faster") for a in atts)
            items.append(f'<li><span class="line-key" style="background:{ARM_COLOR[arm]}"></span>{esc(k)} · '
                         f'{esc(ARM_LABEL[arm])}: <strong>{fails}</strong> of {checked} would-be-faster kernels '
                         f'rejected at held-out shapes</li>')
    return f'<ul class="mini">{"".join(items)}</ul>' if items else ""


def panel_heldout(results, summary):
    ho = results["heldout"]
    from_logs = heldout_from_logs(summary)
    how = ("Each kernel re-checked on shapes the search never saw, with hostile inputs. Colour = speedup over "
           "the start kernel at that shape; red = wrong there, so its speedup does not count.")
    if not ho:
        return card("Held-out shapes: does the speedup survive?", how,
                    empty("End-of-run grid not in yet: P2's <code>heldout_grid.py</code> writes it.") + from_logs)
    by_kernel = defaultdict(list)
    for h in ho:
        by_kernel[h.get("kernel", "?")].append(h)
    blocks, rows_t = [], []
    for k, hs in by_kernel.items():
        shapes = list(dict.fromkeys(h.get("shape", "?") for h in hs))
        whiches = list(dict.fromkeys(h.get("which", "?") for h in hs))
        cell = {(h.get("which"), h.get("shape")): h for h in hs}
        top = max([h["speedup"] for h in hs if h.get("passed") and h.get("speedup")] + [1.5])
        W, L, T, rh = 540, 132, 64, 26
        cw = (W - L - 40) / max(len(shapes), 1)   # room on the right for the last slanted label
        H = T + rh * len(whiches) + 4
        g = []
        for j, sh in enumerate(shapes):
            x = L + j * cw + cw / 2
            g.append(f'<text transform="translate({x:.1f},{T - 8}) rotate(-35)" class="tick col">{esc(shape_text(sh))}</text>')
        for i, w in enumerate(whiches):
            name = {"start": "Start kernel", "expert": "Expert kernel", "aws as published": "AWS as published"}.get(w) or (
                f"{ARM_LABEL[w]}: best" if w in ARM_LABEL else str(w)[:1].upper() + str(w)[1:])
            y = T + i * rh
            g.append(f'<text x="{L - 8}" y="{y + rh / 2 + 4:.1f}" text-anchor="end" class="row-label">{esc(name)}</text>')
            for j, sh in enumerate(shapes):
                h = cell.get((w, sh))
                x = L + j * cw
                if h is None:
                    continue
                ok, sp = bool(h.get("passed")), h.get("speedup")
                text = (fmt_x(sp) if sp else "pass") if ok else "✗ wrong"
                extra = [fmt_us(h["time_us"]) if h.get("time_us") else "", str(h.get("message") or "")[:240]]
                cls = f"hc{heat_bucket(sp, top)}" if ok else "hfail"
                tcls = f"ht{heat_bucket(sp, top)}" if ok else "htfail"
                g.append(f'<g class="mark" tabindex="0" data-tip="{tip(text, f"{name} at {shape_text(sh)}", *extra)}">'
                         f'<rect x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cw - 2:.1f}" height="{rh - 2}" rx="3" class="{cls}"/>'
                         f'<text x="{x + cw / 2:.1f}" y="{y + rh / 2 + 4:.1f}" text-anchor="middle" class="cell {tcls}">'
                         f'{esc(text)}</text></g>')
                rows_t.append([esc(k), esc(name), esc(shape_text(sh)), "pass" if ok else "WRONG", fmt_x(sp) if sp else "",
                               fmt_us(h.get("time_us")) if h.get("time_us") else "", esc(str(h.get("message") or "")[:300])])
        fails = sum(1 for h in hs if not h.get("passed"))
        blocks.append(f'<div class="kblock"><h3>{esc(k)} <span class="src">{len(hs) - fails} of {len(hs)} cells correct'
                      f'</span></h3><svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(k)}: held-out grid">'
                      f'{"".join(g)}</svg></div>')
    scale = ('<div class="legend"><span class="key"><span class="ramp" aria-hidden="true">'
             + "".join(f'<span class="swatch hc{i}"></span>' for i in range(HEAT))
             + '</span>1× → faster</span><span class="key"><span class="swatch hfail" aria-hidden="true">'
             '</span>wrong at that shape</span></div>')
    return card("Held-out shapes: does the speedup survive?", how,
                scale + "".join(blocks) + from_logs + table_view(table(
                    ["Kernel", "Row", "Shape", "Correct", "Speedup", "Time", "Message"], rows_t, numeric=(4, 5))))


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
        return "full source (no earlier verified kernel, no start kernel file)", "\n".join(code.splitlines()[:DIFF_LINES])
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
        W, Lm, Rm, T, rh = 540, 150, 10, 4, 17
        H = T + rh * len(lanes) + 22
        step = (W - Lm - Rm) / max(n_max - 1, 1)
        size = max(2.5, min(4.5, step * 0.35))
        sx = lambda i: Lm + i * step
        g = []
        _, _, xt = nice_domain(0, max(n_max - 1, 1), 6)
        for t in xt:
            if t <= n_max - 1:
                g.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{T}" y2="{H - 18}" class="grid"/>'
                         f'<text x="{sx(t):.1f}" y="{H - 5}" text-anchor="middle" class="tick">{t:.0f}</text>')
        for li, (arm, j, run) in enumerate(lanes):
            y = T + li * rh + rh / 2
            g.append(f'<text x="{Lm - 10}" y="{y + 4:.1f}" text-anchor="end" class="row-label small">'
                     f'{esc(ARM_LABEL[arm])} · {j + 1} · seat {esc(run["seat"])}</text>')
            counts = defaultdict(int)
            for i, rec in enumerate(run["attempts"]):
                counts[rec["verdict"]] += 1
                label, color, kind = VERDICT.get(rec["verdict"], (rec["verdict"], "var(--muted)", "circle"))
                if rec["verdict"] == "slower" and rec["speedup"] and rec["speedup"] >= 1.0:
                    # A referee without no_gain calls a 1.03x kernel "slower": say why, keep its verdict.
                    label += " (faster, but inside the noise threshold)"
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
                         f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="transparent"/>'
                         f'<circle class="sel" cx="{x:.1f}" cy="{y:.1f}" r="{size + 3.5:.1f}"/>'
                         f'{glyph(kind, x, y, size, color)}</g>')
            rows_t.append([esc(k), f"{ARM_LABEL[arm]} · run {j + 1}", esc(run["seat"])] +
                          [str(counts.get(v, 0)) for v in schema.VERDICTS] + [str(len(run["attempts"]))])
        svgs.append(f'<div class="kblock"><h3>{esc(k)}</h3><svg viewBox="0 0 {W} {H}" role="img" '
                    f'aria-label="{esc(k)}: every attempt, by verdict">{"".join(g)}</svg></div>')
    head = ["Kernel", "Run", "Seat"] + [VERDICT[v][0] if v in VERDICT else v for v in schema.VERDICTS] + ["Total"]
    if not svgs:
        return card("Every attempt", "", empty("No attempts logged yet: one mark per attempt appears here as the "
                                               "loops run; click one for the referee's message and the code diff.")), []
    body = (legend_verdicts() + f'<div class="pair">{"".join(svgs)}</div>' +
            '<div id="detail" class="detail" aria-live="polite" hidden></div>' +
            table_view(table(head, rows_t, numeric=tuple(range(3, len(head))))))
    return card("Every attempt", "One mark per attempt, by how far it got through the referee. Click one for the "
                                 "message, the instruction sent back and the code change.", body, "wide"), details


# ---------------------------------------------------------------- page

def meter(frac):
    return f'<div class="meter" aria-hidden="true"><span style="width:{max(0.0, min(1.0, frac)) * 100:.0f}%"></span></div>'


def kpis(summary, results, records, tune):
    tiles = []
    for k in kernels_of(summary, results):
        s, info, start, expert_x, limit_x = kernel_view(k, summary, results)
        rs = s["arms"]["referee"]
        best = median(r["best_x"] for r in rs) if rs else None
        target, tname = (expert_x, "expert") if expert_x else (limit_x, "floor limit")
        if best:
            sub = f"model + referee · {len(rs)} run{'s' if len(rs) != 1 else ''}"
            sub += f" · {best / target:.0%} of the {tname}'s {fmt_x(target)}" if target else ""
        else:
            sub = "no runs yet" + (f" · target: {tname} {fmt_x(target)}" if target else "")
        tiles.append((f"{k} speedup", fmt_x(best) if best else "–", sub,
                      meter((best - 1) / (target - 1)) if best and target and target > 1 else meter(0)))
    if tune:
        sw, runs = tune["sweep"], tune["runs"]
        best = median(r["best"] for r in runs) if runs else None
        sub = (f"random search over AWS's default · {len(runs)} run{'s' if len(runs) != 1 else ''}" if runs
               else "random search: no runs yet")
        if sw:
            sub += f" · best of all {sw['n']}: {(sw['best'] - 1) * 100:+.0f}%"
        tiles.append(("Block tuning", f"{(best - 1) * 100:+.0f}%" if best else "–", sub,
                      meter((best - 1) / (sw["best"] - 1)) if best and sw and sw["best"] > 1 else meter(0)))
    rt = results["redteam"]
    if rt:
        control, cheats, caught, pending = redteam_split(rt)
        honest_ok = sum(c["state"] == "pass" for c in control)
        sub = f"cheats caught · {honest_ok}/{len(control)} honest accepted" + redteam_tail(cheats, pending).replace(", ", " · ")
        tiles.append(("Red team", f"{caught}/{len(cheats)}", sub, meter(caught / len(cheats) if cheats else 0)))
    else:
        tiles.append(("Red team", "–", "not in yet", meter(0)))
    ho = results["heldout"]
    if ho:
        ok = sum(1 for h in ho if h.get("passed"))
        bad = [h for h in ho if not h.get("passed")]
        sub = "held-out cells correct" + (f" · wrong: {bad[0].get('which')} at {shape_text(bad[0].get('shape'))}"
                                          + (f" +{len(bad) - 1}" if len(bad) > 1 else "") if bad else "")
        tiles.append(("Held-out", f"{ok}/{len(ho)}", sub, meter(ok / len(ho))))
    else:
        tiles.append(("Held-out", "–", "end-of-run grid not in yet", meter(0)))
    runs = {(r["kernel"], r["arm"], r["seat"], r["run_id"]) for r in records}
    seats = {r["seat"] for r in records}
    tiles.append(("Attempts", f"{len(records):,}", f"{len(runs)} runs · {len(seats)} seat{'s' if len(seats) != 1 else ''}", ""))
    return '<div class="kpis">' + "".join(
        f'<div class="tile"><div class="tile-label">{esc(a)}</div><div class="tile-value">{esc(b)}</div>'
        f'{m}<div class="tile-sub">{esc(c)}</div></div>' for a, b, c, m in tiles) + "</div>"


CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink-1:#0b0b0b;--ink-2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--track:#cde2fb;
--neutral-bar:#898781;--v-rules:#52514e;--good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;
--good-text:#006300;--critical-text:#b02a2a;--add:rgba(12,163,12,.13);--del:rgba(208,59,59,.13);--wash:rgba(11,11,11,.04);
--h0:#cde2fb;--h1:#9ec5f4;--h2:#6da7ec;--h3:#3987e5;--h4:#256abf;--h5:#184f95;--h6:#0d366b;
--t0:#0b0b0b;--t1:#0b0b0b;--t2:#0b0b0b;--t3:#0b0b0b;--t4:#fff;--t5:#fff;--t6:#fff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink-1:#fff;--ink-2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;
--s3:#199e70;--track:#184f95;--v-rules:#c3c2b7;--good-text:#0ca30c;--critical-text:#e66767;--add:rgba(12,163,12,.22);
--del:rgba(208,59,59,.25);--wash:rgba(255,255,255,.05);
--h0:#104281;--h1:#184f95;--h2:#1c5cab;--h3:#2a78d6;--h4:#3987e5;--h5:#6da7ec;--h6:#9ec5f4;
--t0:#fff;--t1:#fff;--t2:#fff;--t3:#0b0b0b;--t4:#0b0b0b;--t5:#0b0b0b;--t6:#0b0b0b}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink-1:#fff;--ink-2:#c3c2b7;--grid:#2c2c2a;
--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--track:#184f95;--v-rules:#c3c2b7;
--good-text:#0ca30c;--critical-text:#e66767;--add:rgba(12,163,12,.22);--del:rgba(208,59,59,.25);--wash:rgba(255,255,255,.05);
--h0:#104281;--h1:#184f95;--h2:#1c5cab;--h3:#2a78d6;--h4:#3987e5;--h5:#6da7ec;--h6:#9ec5f4;
--t0:#fff;--t1:#fff;--t2:#fff;--t3:#0b0b0b;--t4:#0b0b0b;--t5:#0b0b0b;--t6:#0b0b0b}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink-1);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1240px;margin:0 auto;padding:18px 16px 32px}
.top{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:4px 24px;margin:0 0 12px}
h1{font-size:24px;margin:0;letter-spacing:-.01em}
.lede{color:var(--ink-2);margin:2px 0 0;font-size:13px;max-width:720px}
.meta{color:var(--muted);font-size:12px;margin:0}
.fake{border:1px solid var(--border);border-left:4px solid var(--critical);background:var(--surface);border-radius:8px;
padding:8px 12px;margin:0 0 12px;font-size:13px}
.fake strong{color:var(--critical-text)}
.notes{font-size:12px;color:var(--ink-2);margin:0 0 12px}
.notes summary{cursor:pointer}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(190px,100%),1fr));gap:12px;margin:0 0 14px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 14px}
.tile-label{font-size:12px;color:var(--ink-2)}
.tile-value{font-size:30px;font-weight:600;line-height:1.15;margin:2px 0 0}
.tile-sub{font-size:11.5px;color:var(--muted);margin-top:6px}
.meter{height:6px;border-radius:3px;background:var(--track);overflow:hidden;margin-top:8px}
.meter span{display:block;height:100%;background:var(--s1);border-radius:3px}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(520px,100%),1fr));gap:14px;margin:0 0 14px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px;min-width:0}
.card.wide{margin:0 0 14px}
.card h2{font-size:16px;margin:0 0 2px}
.how{color:var(--ink-2);font-size:12px;margin:0 0 8px}
h3{font-size:13px;margin:8px 0 2px}
.src{font-weight:400;font-size:11.5px;color:var(--muted);margin-left:4px}
.kblock+.kblock{margin-top:4px}
h3.sep{margin-top:12px;padding-top:10px;border-top:1px solid var(--grid)}
.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(480px,100%),1fr));gap:4px 20px}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--ink-2);margin:0 0 4px}
.key{display:inline-flex;align-items:center;gap:5px}
.key svg{width:14px;height:14px;flex:none}
.line-key{display:inline-block;width:14px;height:2px;border-radius:1px;vertical-align:middle;margin-right:2px}
.scroll{overflow-x:auto}
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{fill:var(--ink-2);font-size:11px;font-family:inherit}
svg .tick{fill:var(--muted);font-variant-numeric:tabular-nums;font-size:10.5px}
svg .col{text-anchor:start}
svg .row-label{fill:var(--ink-1);font-size:12px}
svg .row-label.small{font-size:11px}
svg .value{fill:var(--ink-1);font-size:11.5px}
svg .strong{font-weight:600}
svg .note{fill:var(--muted);font-style:italic}
svg .ref-label{fill:var(--ink-2);font-size:10.5px}
svg .cell{font-size:11px;font-weight:600;font-variant-numeric:tabular-nums}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.ref{stroke:var(--ink-2);stroke-width:1}
.whisker{stroke:var(--ink-1);stroke-width:1.3;fill:none}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.band{opacity:.10}
.end-dot{stroke:var(--surface);stroke-width:2}
.hair{stroke:var(--ink-2);stroke-width:1;visibility:hidden;pointer-events:none}
.mark{outline:none;cursor:default}
.mark:hover path,.mark:hover rect:not([fill=transparent]),.mark:focus-visible path,.mark:focus-visible rect{filter:brightness(1.12)}
.dot{cursor:pointer;outline:none}
.dot .sel{fill:none;stroke:var(--ink-1);stroke-width:1.5;visibility:hidden}
.dot:hover .sel{visibility:visible;opacity:.45}
.dot:focus-visible .sel,.dot.selected .sel{visibility:visible;opacity:1}
.hc0{fill:var(--h0)}.hc1{fill:var(--h1)}.hc2{fill:var(--h2)}.hc3{fill:var(--h3)}.hc4{fill:var(--h4)}.hc5{fill:var(--h5)}.hc6{fill:var(--h6)}
.ht0{fill:var(--t0)}.ht1{fill:var(--t1)}.ht2{fill:var(--t2)}.ht3{fill:var(--t3)}.ht4{fill:var(--t4)}.ht5{fill:var(--t5)}.ht6{fill:var(--t6)}
.hfail{fill:var(--critical)}.htfail{fill:#fff}
.ramp{display:inline-flex;border-radius:2px;overflow:hidden}
.swatch{display:inline-block;width:12px;height:10px}
.swatch.hfail{border-radius:2px}
.swatch.hc0{background:var(--h0)}.swatch.hc1{background:var(--h1)}.swatch.hc2{background:var(--h2)}
.swatch.hc3{background:var(--h3)}.swatch.hc4{background:var(--h4)}.swatch.hc5{background:var(--h5)}
.swatch.hc6{background:var(--h6)}.swatch.hfail{background:var(--critical)}
.suite+.suite{margin-top:6px}
.wall-row{display:grid;grid-template-columns:118px 1fr 24px;gap:8px;align-items:center;padding:3px 0;
border-top:1px solid var(--grid)}
.wall-label{font-size:12px;color:var(--ink-2)}
.wall-n{font-size:11.5px;color:var(--muted);text-align:right;font-variant-numeric:tabular-nums}
.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{display:inline-grid;place-items:center;width:18px;height:18px;border-radius:4px;font-size:11px;font-weight:700;
color:#fff;line-height:1;cursor:default;outline:none;flex:none}
.chip.round{border-radius:50%}
.chip.caught,.chip.pass{background:var(--good)}
.chip.missed,.chip.alarm{background:var(--critical)}
.chip.nogain{background:var(--muted)}
.chip.pending,.chip.other{background:transparent;box-shadow:inset 0 0 0 2px var(--muted);color:var(--muted)}
.chip:hover,.chip:focus-visible{filter:brightness(1.15);box-shadow:0 0 0 2px var(--ink-1)}
.legend .chip{width:14px;height:14px;font-size:9px}
ul.mini{list-style:none;padding:0;margin:8px 0 0;font-size:12px;color:var(--ink-2)}
ul.mini li{margin:2px 0}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink-2);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.status{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.status .icon{display:inline-grid;place-items:center;width:16px;height:16px;border-radius:50%;color:#fff;font-size:11px;font-weight:700}
.status.good .icon{background:var(--good)}.status.critical .icon{background:var(--critical)}
.status.neutral .icon{background:var(--muted)}
.status.good{color:var(--good-text)}.status.critical{color:var(--critical-text)}.status.neutral{color:var(--ink-2)}
details.tv{margin-top:8px;font-size:12px}
details.tv summary{cursor:pointer;color:var(--ink-2)}
.empty{color:var(--muted);font-size:13px;margin:6px 0}
.warn{font-size:12px;color:var(--ink-1);margin:4px 0}
.warn .icon{display:inline-grid;place-items:center;width:16px;height:16px;border-radius:50%;background:var(--critical);
color:#fff;font-size:11px;font-weight:700;margin-right:6px}
.detail{margin-top:10px;border-top:1px solid var(--grid);padding-top:10px}
.detail h3{margin:0 0 6px}
.detail dl{display:grid;grid-template-columns:max-content 1fr;gap:1px 14px;font-size:12px;margin:0 0 8px}
.detail dt{color:var(--ink-2)}
.detail dd{margin:0;font-variant-numeric:tabular-nums}
.detail .label{font-size:11.5px;color:var(--ink-2);margin:8px 0 2px;font-weight:600}
.detail .text{white-space:pre-wrap;margin:0;font-size:12px}
pre.diff{font:11.5px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--wash);border-radius:8px;
padding:8px 10px;overflow-x:auto;max-height:360px;margin:4px 0 0}
pre.diff .add{background:var(--add);display:block}
pre.diff .del{background:var(--del);display:block}
pre.diff .hunk{color:var(--muted);display:block}
code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.92em}
#tip{position:fixed;z-index:10;pointer-events:none;background:var(--surface);color:var(--ink-1);border:1px solid var(--border);
border-radius:8px;padding:7px 9px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.14);max-width:340px}
.tip-value{font-weight:600;font-size:13px}
.tip-label{color:var(--ink-2)}
.tip-row{display:flex;align-items:center;gap:8px}
.tip-key{display:inline-block;width:12px;height:2px;border-radius:1px;flex:none}
footer{color:var(--muted);font-size:11.5px;margin-top:4px}
@media (max-width:560px){.card{padding:12px}.tile-value{font-size:26px}}
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
    const t = e.target.closest && e.target.closest('[data-tip]');
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
    detail.hidden = false;
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


def stamp(records, fake):
    """The banner for invented data: all of it (the layout build), or some lines mixed into real logs."""
    if not fake:
        return ""
    bad = [r for r in records if marked(r)]
    if not records or len(bad) == len(records):
        what = ("Generated by <code>schema.py --fake</code> or <code>search.py --stub</code> to build the "
                "layout. Never report a number from this page.")
    else:
        files = sorted({r.get("_file") or "?" for r in bad})
        what = (f"{len(bad):,} of {len(records):,} attempts are marked (fake) or (stub), in "
                f"{esc(', '.join(files))}. Remove those lines before reporting any number from this page.")
    return f'<div class="fake" role="alert"><strong>Fake or stub data.</strong> {what}</div>'


def build(records, results, notes, fake, inputs, sweep=()):
    summary = summarize(records)
    tune = tuning(summary, list(sweep))
    timeline, details = panel_timeline(summary)
    sources = set().union(*(s["sources"] for s in summary.values())) if summary else set()
    sources |= {v["source"] for v in results["kernels"].values() if v.get("source")}
    built = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    meta = (f"Built {built} · {len(inputs)} file{'s' if len(inputs) != 1 else ''} · "
            f"{len(records):,} attempts · {source_label(sources)}")
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
<header class="top">
<div><h1>CHIPBOOST</h1><p class="lede">Qwen3-8B speeds up the kernels it is built from, on the Trainium chip it runs on.
A speedup counts only if the referee verifies it: correct on the chip and on unseen shapes, and faster than the noise.</p></div>
<p class="meta">{esc(meta)}</p>
</header>
{stamp(records, fake)}{note_html}{kpis(summary, results, records, tune)}
<div class="row">{panel_speed(summary, results, tune)}{panel_progress(summary, results, tune)}</div>
<div class="row">{panel_redteam(results)}{panel_heldout(results, summary)}</div>
{timeline}
<footer>Inputs: {esc(", ".join(inputs)) or "none"}. Speedups are against the start kernel timed in the same session;
projections, if any, are labelled as such.</footer>
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
    ap.add_argument("--sweep", nargs="*", help="search.py --exhaustive logs (default: every sweep*.jsonl here)")
    ap.add_argument("--fake", action="store_true", help="build from schema.fake_attempts(); stamped FAKE")
    ap.add_argument("-o", "--out", default=str(HERE / "index.html"))
    a = ap.parse_args()

    def found(pattern):
        # dashboard/collected/ is collect.py's copy of other branches and pods: it passes those files
        # explicitly, so finding them here too would count every line twice.
        return sorted(str(p) for p in PROJECT.rglob(pattern)
                      if ".git" not in p.parts and "collected" not in p.parts and not p.name.startswith("fake_"))

    def rel(p):
        """logs/seat-100/attempts.jsonl, not attempts.jsonl: every seat's file has the same name."""
        try:
            return Path(p).resolve().relative_to(PROJECT).as_posix()
        except ValueError:
            return Path(p).name

    if a.fake:
        records, notes, inputs = [], [], ["schema.fake_attempts()"]
        for rec in schema.fake_attempts():
            for k in schema.ATTEMPT_FIELDS:
                rec.setdefault(k, None)
            rec["_file"] = "schema.fake_attempts()"
            records.append(rec)
    res_paths = a.results if a.results is not None else found("*results*.json")
    if a.fake:
        records, notes, inputs = [], [], ["schema.fake_attempts()"]
        for rec in schema.fake_attempts():
            for k in schema.ATTEMPT_FIELDS:
                rec.setdefault(k, None)
            rec["_file"] = "schema.fake_attempts()"
            records.append(rec)
    else:
        paths = a.logs or found("attempts*.jsonl")
        if not paths and not res_paths:
            sys.exit(f"no attempts*.jsonl or *results*.json under {PROJECT}. Pass files, or --fake for the layout.")
        if not paths:   # before the loops run: P1's and P2's results alone are still real numbers
            print("no attempts*.jsonl yet: building from the results files only")
        records, notes = load_attempts(paths)
        for r in records:
            r["_file"] = rel(r["_file"])
        inputs = [rel(p) for p in paths]
    fake = a.fake or any(marked(r) for r in records) or any("fake" in Path(p).name.lower() for p in inputs)
    # Fake panels 3 and 5 only for a page that is fake through and through, never mixed into real logs.
    all_fake = a.fake or (bool(records) and all(marked(r) for r in records))
    results = load_results(res_paths, name=rel)
    if all_fake and not res_paths:
        results = merge_results([("fake_results()", fake_results())])
        inputs.append("fake_results()")
    else:
        inputs += [rel(p) for p in res_paths]

    sweep_paths = [] if a.fake else (a.sweep if a.sweep is not None else found("sweep*.jsonl"))
    sweep, sweep_notes = load_attempts(sweep_paths)
    notes += sweep_notes
    inputs += [rel(p) for p in sweep_paths]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(records, results, notes, fake, inputs, sweep), encoding="utf-8")
    print(f"wrote {out}: {len(records)} attempts from {', '.join(inputs)}"
          + ("  [FAKE DATA]" if fake else "") + (f"  ({len(notes)} log lines need attention)" if notes else ""))


if __name__ == "__main__":
    main()
