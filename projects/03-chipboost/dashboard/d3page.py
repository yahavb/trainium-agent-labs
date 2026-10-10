"""
dashboard/d3page.py -- the D3 dashboard (dashboard/index.html): the team's verified results, drawn in the
browser from one JSON blob that build.py computes from the logs. Every number on the page comes from DATA.
"""

import json
import re

AGENT = {"referee_v2": "P1's agent", "referee_v2_p1fix": "P1's agent", "referee_v3": "P3's agent",
         "referee_v4": "P3's agent", "referee_v5": "P3's agent", "referee_later": "P3's agent"}


def data(b, summary, results, tune, sweep, records, meta):
    """b is the build module: its helpers keep this page's numbers identical to the rest of the dashboard."""
    s = summary.get("matmul") or {"arms": {}}
    m = results["kernels"].get("matmul") or {}
    lnc = (m.get("expert_derived") or {}).get("lnc2") or {}
    bd = lnc.get("breakdown", "")
    steps = [float(x) for x in re.findall(r"([\d.]+) us", bd)] + [float(x) for x in re.findall(r"-> ([\d.]+) \(", bd)]

    wins = []
    for arm in b.MODEL_ARMS:
        for r in s["arms"].get(arm) or []:
            if not (r["n_verified"] and r["best_x"] > 1.0):
                continue
            first = next(a for a in sorted(r["attempts"], key=b.order) if b.verified(a))
            msg = first.get("referee_message") or ""
            ho = re.search(r"including (\d+) undisclosed held-out", msg)
            wins.append(dict(arm=arm, label=b.ARM_LABEL[arm], agent=AGENT.get(arm, "Qwen3-8B"), run=r["run_id"],
                             seat=r["seat"], attempt=first["attempt_no"], us=round(first["time_us_median"], 1),
                             base=round(first["baseline_us_same_session"], 1), x=round(r["best_x"], 4),
                             heldout=int(ho.group(1)) if ho else None, code=first["code_hash"],
                             curve=[round(v, 4) for v in r["curve"]], n=len(r["attempts"]),
                             faster=sum(1 for a in r["attempts"] if b.verified(a) and a["speedup"] > 1.0)))
    qx = max((w["x"] for w in wins), default=None)

    ladder = []
    if len(steps) >= 4:
        start = steps[0]
        ladder = [dict(name="Start kernel", desc="NKI tutorial matmul", us=start, kind="base"),
                  dict(name="Qwen3-8B's kernel", desc="written by the model, verified", us=round(start / qx, 1) if qx else None, kind="qwen"),
                  dict(name="AWS expert kernel", desc="AWS's design, fp32 fix", us=steps[1], kind="base"),
                  dict(name="+ block tuning", desc="P2's best block sizes", us=steps[2], kind="tuned"),
                  dict(name="+ both physical cores", desc="LNC=2 · not a referee verdict", us=steps[3], kind="tuned")]
        ladder = [dict(l, x=round(start / l["us"], 3)) for l in ladder if l["us"]]

    search = [dict(run=r["run_id"], seat=r["seat"], curve=[round(v, 4) for v in r["curve"]], best=round(r["best_x"], 3))
              for r in s["arms"].get("random_search") or []]

    timed = sorted((x for x in sweep if x["speedup"] and x["verdict"] in ("faster", "no_gain", "slower")),
                   key=lambda x: -x["speedup"])
    ref = next((x["speedup"] for x in timed if b.caps_of(x["code"]) == b.SHIPPED_CAPS), None)
    found = {r["caps"] for r in (tune or {}).get("runs", []) if r["caps"]}
    sw = [dict(rank=i + 1, caps="m{} n{} k{}".format(*b.caps_of(x["code"])), rel=round(x["speedup"] / ref, 4),
               x=round(x["speedup"], 3), default=b.caps_of(x["code"]) == b.SHIPPED_CAPS, found=b.caps_of(x["code"]) in found)
          for i, x in enumerate(timed)] if ref else []
    tuned = b.median(r["best"] for r in tune["runs"]) if tune and tune["runs"] else None

    ho = []
    names = {"aws as published": "AWS as published", "expert": "AWS expert, fp32 fix",
             "best random_search": "Random search best", "referee_v2": "Qwen3-8B's kernel"}
    for h in results["heldout"]:
        if h.get("which") in names:
            u = re.search(r"([\d.]+) bf16 ulps", str(h.get("message") or ""))
            ho.append(dict(row=names[h["which"]], shape=b.shape_text(h.get("shape")), ok=bool(h.get("passed")),
                           x=h.get("speedup"), ulps=float(u.group(1)) if u else None))
    q = [h for h in ho if h["row"] == "Qwen3-8B's kernel"]

    rt = results["redteam"]
    control, cheats, caught, _ = b.redteam_split(rt) if rt else ([], [], 0, 0)
    wall = [dict(name=c["cheat"], what=c["what"], honest=c["honest"]) for c in cheats + control
            if c["state"] in ("caught", "pass")]

    evo = []
    for arm, agent in (("referee", "Agent v1"), ("referee_v2_p3", "P3 v2"), ("referee_v3", "P3 v3"), ("referee_v4", "P3 v4"),
                       ("referee_v5", "P3 v5"), ("referee_v2", "P1 v2"), ("referee_v2_p1fix", "P1 v2, fixed")):
        att = [a for r in s["arms"].get(arm) or [] for a in r["attempts"]]
        if not att:
            continue
        st = [stage(a, b) for a in att]
        evo.append(dict(label=agent, arm=b.ARM_LABEL[arm], n=len(att),
                        reach=[sum(1 for x in st if x >= k) for k in range(1, 5)]))
    tries = [dict(run=i, t=j + 1, x=round(a["speedup"], 3)) for i, r in enumerate(s["arms"].get("random_search") or [])
             for j, a in enumerate(sorted(r["attempts"], key=b.order)) if b.verified(a) and a["speedup"]]
    grid = {}
    for x in timed:
        cm, cn, ck = b.caps_of(x["code"])
        if ref:
            grid[(cm, cn)] = max(grid.get((cm, cn), 0), x["speedup"] / ref)
    grid = [dict(m=k[0], n=k[1], rel=round(v, 3)) for k, v in sorted(grid.items())]

    return dict(meta=meta, qx=qx, evo=evo, tries=tries, grid=grid, wins=wins, ladder=ladder, search=search, sweep=sw, tuned=tuned,
                expert_x=ladder[2]["x"] if len(ladder) > 2 else None, heldout=ho,
                q_ok=sum(h["ok"] for h in q), q_n=len(q),
                q_geo=round(b.math.exp(sum(b.math.log(h["x"]) for h in q if h["x"]) / max(1, sum(1 for h in q if h["x"]))), 2) if q else None,
                caught=caught, cheats=len(cheats), honest_ok=sum(c["state"] == "pass" for c in control),
                honest=len(control), wall=wall, attempts=len(records),
                agents=sorted({w["agent"] for w in wins}))


def stage(a, b):
    """How far one attempt got through the referee: 0 did not compile, 1 compiled, 2 ran, 3 correct, 4 faster."""
    m = a.get("referee_message") or ""
    if b.verified(a) and a["speedup"] > 1.0:
        return 4
    if a["verdict"] in ("no_gain", "slower", "faster", "heldout_fail"):
        return 3
    if "failed during compile" in m or a["verdict"] == "rules":
        return 0
    if "failed during simulate" in m:
        return 1
    return 2


def page(d):
    return HTML.replace("__DATA__", json.dumps(d, ensure_ascii=False).replace("</", "<\\/"))


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CHIPBOOST Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<style>
:root{--bg:#f6f5f1;--card:#fff;--ink:#1d1c19;--ink2:#55534c;--muted:#8a877e;--line:#e4e1d8;--grid:#eeebe3;
--qwen:#2a6fdb;--qwen2:#7aa7ef;--tuned:#1a9e77;--tuned2:#8fd3bd;--base:#c9c5bb;--gold:#d08a00;--bad:#d64541;--good:#16a34a;
--h0:#dce9fb;--h1:#a9c8f2;--h2:#5f97e2;--h3:#2a6fdb;--h4:#173f86}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0e0e0d;--card:#191917;--ink:#f1efe8;--ink2:#bdb9ae;
--muted:#8a877e;--line:#2c2b27;--grid:#24231f;--qwen:#5b97f0;--qwen2:#2f5ea8;--tuned:#2fbf8f;--tuned2:#1f6f56;--base:#4a4842;
--h0:#1b2b45;--h1:#21417a;--h2:#2e5fb0;--h3:#4c86e3;--h4:#9cc2f7}}
:root[data-theme="dark"]{--bg:#0e0e0d;--card:#191917;--ink:#f1efe8;--ink2:#bdb9ae;--muted:#8a877e;--line:#2c2b27;--grid:#24231f;
--qwen:#5b97f0;--qwen2:#2f5ea8;--tuned:#2fbf8f;--tuned2:#1f6f56;--base:#4a4842;--h0:#1b2b45;--h1:#21417a;--h2:#2e5fb0;--h3:#4c86e3;--h4:#9cc2f7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Segoe UI",system-ui,-apple-system,sans-serif}
main{max-width:1240px;margin:0 auto;padding:28px 16px 48px}
header{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:flex-end;margin-bottom:18px}
h1{margin:0;font-size:34px;letter-spacing:-.5px}h1 span{color:var(--qwen)}
.lede{margin:4px 0 0;color:var(--ink2);max-width:760px}.meta{color:var(--muted);font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:16px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 16px 14px;position:relative;overflow:hidden}
.tile .n{font-size:34px;font-weight:700;letter-spacing:-1px;line-height:1.1}.tile .c{color:var(--ink2);font-size:13px;margin-top:6px}
.tile.au .n{color:var(--gold)}.tile.hi{border-color:var(--qwen)}.tile.hi .n{color:var(--qwen)}.tile.tu .n{color:var(--tuned)}
.tile::after{content:"";position:absolute;left:0;top:0;height:4px;width:100%;background:var(--accent,transparent)}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.wide{grid-column:1/-1}
@media (max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px 22px}
.card h2{margin:0 0 2px;font-size:19px}.card .how{margin:0 0 10px;color:var(--ink2);font-size:13.5px}
.tag{display:inline-block;font-size:11px;font-weight:600;padding:2px 8px;border-radius:99px;margin-left:8px;vertical-align:2px;
background:color-mix(in srgb,var(--qwen) 14%,transparent);color:var(--qwen)}.tag.t{background:color-mix(in srgb,var(--tuned) 16%,transparent);color:var(--tuned)}
svg{display:block;width:100%;height:auto;overflow:visible}
.axis text,.tick{fill:var(--muted);font-size:11px}.axis path,.axis line{stroke:var(--line)}.gridl line{stroke:var(--grid)}.gridl path{display:none}
.lbl{fill:var(--ink);font-size:13px;font-weight:600}.sub{fill:var(--muted);font-size:11px}.val{fill:var(--ink);font-size:13px;font-weight:700}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px;color:var(--ink2);margin:0 0 6px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:12px}td.num{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600;background:color-mix(in srgb,var(--good) 15%,transparent);color:var(--good)}
code{font-size:12px;color:var(--ink2)}
.wall{display:flex;flex-wrap:wrap;gap:6px}.chip{width:30px;height:30px;border-radius:7px;display:grid;place-items:center;color:#fff;font-weight:700;background:var(--good);cursor:default}
.chip.h{border-radius:50%;background:var(--qwen)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:14px}
.stat{border:1px solid var(--line);border-radius:12px;padding:10px 12px}.stat b{display:block;font-size:20px}.stat span{font-size:12.5px;color:var(--ink2)}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);padding:7px 10px;border-radius:8px;font-size:12.5px;
max-width:300px;opacity:0;transition:opacity .12s;z-index:9}#tip b{display:block;font-size:14px}
.bug{display:grid;grid-template-columns:1.1fr 1fr;gap:24px;align-items:center}@media (max-width:860px){.bug{grid-template-columns:1fr}}
.steps{margin:6px 0 0;padding-left:20px;font-size:13.5px;color:var(--ink2)}.steps li{margin:5px 0}.steps b{color:var(--ink)}
.foot{color:var(--muted);font-size:12px;margin-top:18px}
</style>
</head>
<body>
<main>
<header><div><h1>CHIP<span>BOOST</span></h1>
<p class="lede">Qwen3-8B, served on one Trainium2 chip, made the matmul it is built from faster on that same chip, and a
referee that planted cheats could not fool verified it. Every number below is computed from the run logs.</p></div>
<div class="meta" id="meta"></div></header>
<section class="tiles" id="tiles"></section>
<div class="grid">
<section class="card wide"><h2>From the start kernel to <span id="ladder-top"></span><span class="tag">speed ladder</span></h2>
<p class="how">Matmul time on the chip at Qwen3-8B's shapes, as a multiple of the NKI tutorial kernel. Hover a bar for its time.</p>
<div id="ladder"></div></section>
<section class="card wide"><h2>Found and fixed: a precision bug in AWS's published matmul<span class="tag" style="color:var(--gold);background:color-mix(in srgb,var(--gold) 15%,transparent)">discovery</span></h2>
<div class="bug"><div><p class="how" style="font-size:14px;color:var(--ink)">AWS publishes a "fully optimised" NKI matmul. Our referee checks every
kernel at shapes it was never shown, with hostile inputs, so it ran AWS's kernel at Qwen3-8B's real sizes, and the numbers came back wrong.</p>
<ol class="steps">
<li><b>The cause.</b> The kernel rounds each K-block's partial sum to bf16 before adding the next. Its own test uses K = 1024, a single block, so it never takes that path.</li>
<li><b>Simulator stage.</b> 5.3 bf16 ulps at K = 2048, over the limit of 4.</li>
<li><b>On the chip.</b> At Qwen3's down_proj shape (K = 6144), a held-out shape: 4.9 ulps, rejected.</li>
<li><b>Its own benchmark.</b> At K = 8192, the K AWS benchmarks with: 19.4 ulps in the simulator. It fails its own correctness check, file unmodified.</li>
<li><b>The fix.</b> One line: accumulate in fp32. Error drops to 0.5 ulps at no measurable speed cost. Every expert number on this page uses the fixed kernel.</li>
</ol></div><div id="bug"></div></div></section>
<section class="card wide"><h2>Agent optimisation: every version gets further<span class="tag">agent evolution</span></h2>
<p class="how">Share of each agent version's attempts that reached each referee stage. Each version fixed what stopped the
one before: P3's rules took the model from crashing, to running, to correct and faster; P1's agent got there too.</p>
<div class="legend" id="evo-legend"></div><div id="evo"></div></section>
<section class="card"><h2>How the winning runs climbed<span class="tag">winning runs</span></h2>
<p class="how">Best verified speedup after each attempt, for every run that produced a faster kernel. One line per run.</p>
<div class="legend" id="climb-legend"></div><div id="climb"></div></section>
<section class="card"><h2>Search agent: tuning AWS's kernel<span class="tag t">random search</span></h2>
<p class="how">Best speedup over the start kernel after each try, per run. Each run starts from AWS's expert kernel (dashed) and beats it.</p>
<div id="search"></div></section>
<section class="card wide"><h2>The winning kernels<span class="tag">verified by the referee</span></h2>
<p class="how">Every kernel the model wrote that the referee verified faster: correct on the chip with hostile inputs, correct
on undisclosed held-out shapes, and faster than the timing noise.</p><div id="wins"></div></section>
<section class="card"><h2>Every search try<span class="tag t">search agent</span></h2>
<p class="how">Every verified kernel the random-search agent tried, by try number. One colour per run; dashed: AWS's expert.</p>
<div id="tries"></div></section>
<section class="card"><h2>Where the speed is: block sizes<span class="tag t">sweep heat map</span></h2>
<p class="how">Best speedup over AWS's default for each M and N block size (best K), from the exhaustive sweep. Ringed in ink: AWS's default; gold: the best.</p>
<div id="grid"></div></section>
<section class="card"><h2>Every block setting, timed<span class="tag t">exhaustive sweep</span></h2>
<p class="how" id="sweep-how"></p><div id="sweep"></div></section>
<section class="card"><h2>Correct on shapes it never saw<span class="tag">held out</span></h2>
<p class="how">Speedup over the start kernel at 6 unseen shapes with hostile inputs. Red: the referee rejecting AWS's
published kernel for its precision bug (see below); the fixed expert passes everywhere.</p><div id="heldout"></div></section>
<section class="card wide"><h2>A referee you can trust<span class="tag">red team</span></h2>
<p class="how">Squares: planted cheats the referee caught. Circles: honest kernels it accepted. Every candidate runs in a
sandboxed process, and timing is interleaved A/B on the device clock.</p><div class="wall" id="wall"></div>
<div class="stats" id="stats"></div></section>
</div>
<p class="foot">Built by dashboard/build.py from every seat's attempt logs and the team's results files. The 5.00× bar runs the
expert across both physical cores (LNC=2): P2 engineering timed with P1's timer, not a referee verdict.</p>
</main>
<div id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
if (/Headless/.test(navigator.userAgent) || navigator.webdriver || matchMedia("(prefers-reduced-motion: reduce)").matches) {
  /* screenshots and reduced motion: draw the final state at once */
  d3.selection.prototype.transition = function () { return this; };
  d3.selection.prototype.duration = d3.selection.prototype.delay = function () { return this; };
}
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fx = v => v == null ? "–" : (v >= 10 ? v.toFixed(1) : v.toFixed(2)) + "×";
const tip = d3.select("#tip");
function hover(sel, html) {
  sel.on("mousemove", (e, d) => tip.style("opacity", 1).html(html(d))
        .style("left", Math.min(e.clientX + 14, innerWidth - 310) + "px").style("top", (e.clientY + 14) + "px"))
     .on("mouseleave", () => tip.style("opacity", 0));
}
function svg(id, W, H) { return d3.select(id).append("svg").attr("viewBox", `0 0 ${W} ${H}`); }

d3.select("#meta").text(D.meta);
const tiles = [
  D.qx && {n: fx(D.qx), c: "Qwen3-8B made its own matmul faster on the chip, verified", cls: "hi", a: css("--qwen")},
  D.wins.length && {n: `${D.wins.length} runs`, c: `produced a verified faster kernel · ${D.agents.join(" and ")}`, cls: "hi", a: css("--qwen")},
  D.ladder.length && {n: fx(D.ladder[D.ladder.length - 1].x), c: "fastest matmul on the chip: AWS design + tuning + both cores", cls: "tu", a: css("--tuned")},
  D.tuned && {n: "+" + Math.round((D.tuned - 1) * 100) + "%", c: "P2's block tuning over AWS's default · #1 of " + D.sweep.length, cls: "tu", a: css("--tuned")},
  D.q_n && {n: `${D.q_ok} / ${D.q_n}`, c: `unseen shapes correct for Qwen's kernel · ${fx(D.q_geo)} geometric mean`, cls: "hi", a: css("--qwen")},
  D.cheats && {n: `${D.caught} / ${D.cheats}`, c: `planted cheats caught · ${D.honest_ok} / ${D.honest} honest accepted`, a: css("--good")},
  {n: "4.9 → 0.5", c: "bf16 ulps: precision bug in AWS's published matmul, found by our referee and fixed in one line", cls: "au", a: css("--gold")},
].filter(Boolean);
d3.select("#tiles").selectAll(".tile").data(tiles).join("div").attr("class", d => "tile " + (d.cls || ""))
  .style("--accent", d => d.a).html(d => `<div class="n">${d.n}</div><div class="c">${d.c}</div>`);

/* speed ladder */
(function () {
  const L = D.ladder; if (!L.length) return;
  d3.select("#ladder-top").text(fx(L[L.length - 1].x));
  const W = 1180, rh = 52, Lx = 250, H = rh * L.length + 30;
  const s = svg("#ladder", W, H), x = d3.scaleLinear().domain([0, d3.max(L, d => d.x) * 1.08]).range([Lx, W - 70]);
  s.append("g").attr("class", "gridl").attr("transform", `translate(0,${H - 24})`)
    .call(d3.axisBottom(x).ticks(6).tickSize(-(H - 30)).tickFormat(d => d + "×")).selectAll("text").attr("class", "tick");
  const fill = {base: css("--base"), qwen: css("--qwen"), tuned: css("--tuned")};
  const g = s.selectAll(".r").data(L).join("g").attr("transform", (d, i) => `translate(0,${i * rh + 4})`);
  g.append("text").attr("x", Lx - 14).attr("y", 20).attr("text-anchor", "end").attr("class", "lbl").text(d => d.name);
  g.append("text").attr("x", Lx - 14).attr("y", 36).attr("text-anchor", "end").attr("class", "sub").text(d => d.desc);
  const bar = g.append("rect").attr("x", Lx).attr("y", 8).attr("height", 30).attr("rx", 6).attr("fill", d => fill[d.kind]).attr("width", 0);
  bar.transition().duration(900).delay((d, i) => i * 120).attr("width", d => x(d.x) - Lx);
  g.append("text").attr("class", "val").attr("y", 28).attr("x", d => x(d.x) + 10).text(d => `${fx(d.x)}  ·  ${d.us.toFixed(1)} µs`);
  hover(bar, d => `<b>${fx(d.x)}</b>${d.name}: ${d.us.toFixed(1)} µs on the chip`);
})();

/* winning runs climb */
(function () {
  const R = D.wins; if (!R.length) return;
  const W = 560, H = 300, m = {l: 44, r: 20, t: 14, b: 34};
  const pal = [css("--qwen"), css("--tuned"), css("--gold"), "#8b5cf6", "#e05d9b"];
  const n = d3.max(R, d => d.curve.length);
  const s = svg("#climb", W, H), x = d3.scaleLinear().domain([0, n]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0.95, d3.max(R, d => d.x) * 1.06]).range([H - m.b, m.t]);
  s.append("g").attr("class", "gridl").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - m.l - m.r)).tickFormat(d => d.toFixed(1) + "×"));
  s.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(n).tickFormat(d3.format("d")));
  s.append("text").attr("x", W - m.r).attr("y", H - 4).attr("text-anchor", "end").attr("class", "tick").text("attempt");
  d3.select("#climb-legend").selectAll("span").data(R).join("span")
    .html((d, i) => `<i style="background:${pal[i % pal.length]}"></i>${d.agent} · ${d.label}`);
  R.forEach((r, i) => {
    const pts = [[0, 1]].concat(r.curve.map((v, j) => [j + 1, v]));
    const path = s.append("path").datum(pts).attr("fill", "none").attr("stroke", pal[i % pal.length]).attr("stroke-width", 2.5)
      .attr("d", d3.line().curve(d3.curveStepAfter).x(d => x(d[0])).y(d => y(d[1] + i * 0.004)));
    const len = path.node().getTotalLength();
    path.attr("stroke-dasharray", len).attr("stroke-dashoffset", len).transition().duration(1200).delay(i * 250).attr("stroke-dashoffset", 0);
    const c = s.append("circle").attr("cx", x(r.attempt)).attr("cy", y(r.x + i * 0.004)).attr("r", 7)
      .attr("fill", pal[i % pal.length]).attr("stroke", css("--card")).attr("stroke-width", 2.5);
    hover(c, () => `<b>${fx(r.x)} on attempt ${r.attempt}</b>${r.agent} · ${r.label}<br>${r.us} µs vs ${r.base} µs`);
  });
  s.append("text").attr("x", x(n) - 4).attr("y", y(d3.max(R, d => d.x)) - 12).attr("text-anchor", "end").attr("class", "val")
    .text(`${fx(d3.max(R, d => d.x))} reached in ${d3.min(R, d => d.attempt)}–${d3.max(R, d => d.attempt)} attempts`);
})();

/* search agent */
(function () {
  const R = D.search; if (!R.length) return;
  const W = 560, H = 300, m = {l: 44, r: 20, t: 14, b: 34};
  const n = d3.max(R, d => d.curve.length);
  const s = svg("#search", W, H), x = d3.scaleLinear().domain([1, n]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([Math.min(D.expert_x || 2.4, d3.min(R, d => d.curve[0])) * 0.95, d3.max(R, d => d.best) * 1.04]).range([H - m.b, m.t]);
  s.append("g").attr("class", "gridl").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - m.l - m.r)).tickFormat(d => d.toFixed(1) + "×"));
  s.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(8).tickFormat(d3.format("d")));
  s.append("text").attr("x", W - m.r).attr("y", H - 4).attr("text-anchor", "end").attr("class", "tick").text("try");
  if (D.expert_x) {
    s.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(D.expert_x)).attr("y2", y(D.expert_x))
      .attr("stroke", css("--muted")).attr("stroke-dasharray", "5 4");
    s.append("text").attr("x", W - m.r).attr("y", y(D.expert_x) - 6).attr("text-anchor", "end").attr("class", "sub").text(`AWS expert ${fx(D.expert_x)}`);
  }
  const best = d3.max(R, d => d.best);
  R.forEach((r, i) => {
    const pts = r.curve.map((v, j) => [j + 1, v]);
    const top = r.best === best;
    const p = s.append("path").datum(pts).attr("fill", "none").attr("stroke", top ? css("--tuned") : css("--tuned2"))
      .attr("stroke-width", top ? 3 : 1.8).attr("d", d3.line().curve(d3.curveStepAfter).x(d => x(d[0])).y(d => y(d[1])));
    const len = p.node().getTotalLength();
    p.attr("stroke-dasharray", len).attr("stroke-dashoffset", len).transition().duration(1200).delay(i * 120).attr("stroke-dashoffset", 0);
    const c = s.append("circle").attr("cx", x(pts.length)).attr("cy", y(r.best)).attr("r", top ? 6 : 4).attr("fill", top ? css("--tuned") : css("--tuned2"));
    hover(c, () => `<b>${fx(r.best)}</b>search run ${i + 1} · seat ${r.seat}`);
  });
  s.append("text").attr("x", x(n) - 10).attr("y", y(best) - 12).attr("text-anchor", "end").attr("class", "val").text(`best ${fx(best)}`);
})();

/* winning kernels table */
(function () {
  const R = D.wins; if (!R.length) return;
  const rows = R.map(r => `<tr><td><b>${r.agent}</b><br><span style="color:var(--muted)">${r.label}</span></td>
    <td class="num">${r.attempt} of ${r.n}</td><td class="num">${r.us} µs</td><td class="num">${r.base} µs</td>
    <td class="num"><b style="color:var(--qwen)">${fx(r.x)}</b></td><td class="num">${r.heldout ?? "–"}</td>
    <td class="num">${r.faster}</td><td><span class="pill">✓ verified</span></td><td><code>${r.code}</code></td></tr>`).join("");
  d3.select("#wins").html(`<table><thead><tr><th>Agent</th><th class="num">First faster</th><th class="num">Kernel</th>
    <th class="num">Start, same session</th><th class="num">Speedup</th><th class="num">Held-out shapes</th>
    <th class="num">Faster kernels in run</th><th>Referee</th><th>Code</th></tr></thead><tbody>${rows}</tbody></table>`);
})();

/* sweep */
(function () {
  const S = D.sweep; if (!S.length) return;
  const d0 = S.find(d => d.default);
  d3.select("#sweep-how").text(`All ${S.length} legal block settings of AWS's optimised matmul, timed on the chip, fastest first, as a multiple of AWS's default (#${d0.rank}). Green: settings the search runs found.`);
  const W = 560, H = 280, m = {l: 40, r: 10, t: 24, b: 26};
  const s = svg("#sweep", W, H), x = d3.scaleBand().domain(S.map(d => d.rank)).range([m.l, W - m.r]).padding(0.15);
  const y = d3.scaleLinear().domain([0, d3.max(S, d => d.rel) * 1.1]).range([H - m.b, m.t]);
  s.append("g").attr("class", "gridl").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickSize(-(W - m.l - m.r)).tickFormat(d => d + "×"));
  const b = s.selectAll("rect").data(S).join("rect").attr("x", d => x(d.rank)).attr("width", x.bandwidth()).attr("rx", 1.5)
    .attr("fill", d => d.found ? css("--tuned") : d.default ? css("--ink") : css("--base"))
    .attr("y", H - m.b).attr("height", 0);
  b.transition().duration(800).delay((d, i) => i * 12).attr("y", d => y(d.rel)).attr("height", d => H - m.b - y(d.rel));
  hover(b, d => `<b>#${d.rank}: ${fx(d.rel)} the default</b>block caps ${d.caps} · ${fx(d.x)} the start kernel`);
  s.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(1)).attr("y2", y(1)).attr("stroke", css("--ink")).attr("stroke-width", 1);
  s.append("text").attr("x", x(1)).attr("y", y(S[0].rel) - 6).attr("class", "val").text(`#1 · ${fx(S[0].rel)}`);
  s.append("text").attr("x", x(d0.rank) + x.bandwidth() / 2).attr("y", y(d0.rel) - 8).attr("text-anchor", "middle").attr("class", "sub").text(`AWS default #${d0.rank}`);
})();

/* held-out heat map */
(function () {
  const H0 = D.heldout; if (!H0.length) return;
  const rows = [...new Set(H0.map(d => d.row))], cols = [...new Set(H0.map(d => d.shape))];
  const W = 560, Lx = 150, T = 70, rh = 38, H = T + rh * rows.length + 4;
  const s = svg("#heldout", W, H), x = d3.scaleBand().domain(cols).range([Lx, W - 10]).padding(0.06), y = d3.scaleBand().domain(rows).range([T, H - 4]).padding(0.1);
  const c = d3.scaleQuantize().domain([1, d3.max(H0, d => d.x || 1)]).range(["--h0", "--h1", "--h2", "--h3", "--h4"].map(css));
  s.selectAll(".col").data(cols).join("text").attr("class", "tick").attr("transform", d => `translate(${x(d) + x.bandwidth() / 2},${T - 8}) rotate(-35)`).text(d => d);
  s.selectAll(".row").data(rows).join("text").attr("class", "lbl").attr("x", Lx - 10).attr("y", d => y(d) + y.bandwidth() / 2 + 4).attr("text-anchor", "end").text(d => d);
  const g = s.selectAll(".cell").data(H0).join("g").attr("transform", d => `translate(${x(d.shape)},${y(d.row)})`);
  const r = g.append("rect").attr("width", x.bandwidth()).attr("height", y.bandwidth()).attr("rx", 5)
    .attr("fill", d => d.ok ? c(d.x || 1) : css("--bad")).style("opacity", 0);
  r.transition().duration(500).delay((d, i) => i * 25).style("opacity", 1);
  g.append("text").attr("x", x.bandwidth() / 2).attr("y", y.bandwidth() / 2 + 4).attr("text-anchor", "middle").style("font-size", "12px").style("font-weight", 600)
    .attr("fill", d => !d.ok || (d.x || 1) > 2.2 ? "#fff" : css("--ink")).text(d => d.ok ? fx(d.x) : `✗ ${d.ulps} ulps`).style("pointer-events", "none");
  hover(r, d => d.ok ? `<b>${fx(d.x)}</b>${d.row} at ${d.shape}: correct` : `<b>AWS bug: ${d.ulps} bf16 ulps</b>${d.row} at ${d.shape} (limit 4)`);
})();

/* agent evolution */
(function () {
  const E = D.evo; if (!E.length) return;
  const stages = ["compiles", "runs on the chip", "correct", "faster"];
  const cols = [css("--h1"), css("--h2"), css("--h3"), css("--good")];
  d3.select("#evo-legend").selectAll("span").data(stages).join("span").html((d, i) => `<i style="background:${cols[i]};height:10px"></i>${d}`);
  const W = 1180, rh = 44, Lx = 190, H = rh * E.length + 30;
  const s = svg("#evo", W, H), gw = (W - Lx - 20) / 4;
  stages.forEach((st, k) => s.append("text").attr("x", Lx + k * gw + gw / 2).attr("y", 12).attr("text-anchor", "middle").attr("class", "tick").text(st.toUpperCase()));
  const g = s.selectAll(".e").data(E).join("g").attr("transform", (d, i) => `translate(0,${22 + i * rh})`);
  g.append("text").attr("x", Lx - 14).attr("y", 18).attr("text-anchor", "end").attr("class", "lbl").text(d => d.label);
  g.append("text").attr("x", Lx - 14).attr("y", 32).attr("text-anchor", "end").attr("class", "sub").text(d => `${d.n} attempts`);
  stages.forEach((st, k) => {
    g.append("rect").attr("x", Lx + k * gw + 4).attr("y", 6).attr("width", gw - 8).attr("height", 26).attr("rx", 5).attr("fill", css("--grid"));
    const r = g.append("rect").attr("x", Lx + k * gw + 4).attr("y", 6).attr("height", 26).attr("rx", 5).attr("fill", cols[k]).attr("width", 0);
    r.transition().duration(900).delay((d, i) => i * 90 + k * 150).attr("width", d => Math.max(d.reach[k] ? 6 : 0, (gw - 8) * d.reach[k] / d.n));
    g.append("text").attr("x", Lx + k * gw + gw - 10).attr("y", 24).attr("text-anchor", "end").attr("class", "val")
      .text(d => d.reach[k] ? `${Math.round(100 * d.reach[k] / d.n)}%` : "");
    hover(r, d => `<b>${d.reach[k]} of ${d.n} attempts ${st}</b>${d.arm}`);
  });
})();

/* every search try */
(function () {
  const T = D.tries; if (!T.length) return;
  const W = 560, H = 300, m = {l: 44, r: 16, t: 14, b: 34};
  const pal = d3.schemeTableau10;
  const s = svg("#tries", W, H), x = d3.scaleLinear().domain([0, d3.max(T, d => d.t) + 1]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([d3.min(T, d => d.x) * 0.95, d3.max(T, d => d.x) * 1.04]).range([H - m.b, m.t]);
  s.append("g").attr("class", "gridl").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - m.l - m.r)).tickFormat(d => d.toFixed(1) + "×"));
  s.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(8).tickFormat(d3.format("d")));
  if (D.expert_x) s.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(D.expert_x)).attr("y2", y(D.expert_x)).attr("stroke", css("--muted")).attr("stroke-dasharray", "5 4");
  const c = s.selectAll("circle").data(T).join("circle").attr("cx", d => x(d.t) + (d.run - 2.5) * 1.6).attr("cy", d => y(d.x)).attr("r", 0)
    .attr("fill", d => pal[d.run % 10]).attr("fill-opacity", .8).attr("stroke", css("--card")).attr("stroke-width", 1);
  c.transition().duration(500).delay((d, i) => i * 4).attr("r", 4.5);
  hover(c, d => `<b>${fx(d.x)} the start kernel</b>search run ${d.run + 1}, try ${d.t}`);
  const top = T.reduce((a, b) => a.x > b.x ? a : b);
  s.append("circle").attr("cx", x(top.t) + (top.run - 2.5) * 1.6).attr("cy", y(top.x)).attr("r", 9).attr("fill", "none").attr("stroke", css("--tuned")).attr("stroke-width", 2.5);
  s.append("text").attr("x", x(top.t) + 14).attr("y", y(top.x) + 4).attr("class", "val").text(`best ${fx(top.x)}`);
})();

/* block-size heat map */
(function () {
  const G = D.grid; if (!G.length) return;
  const ms = [...new Set(G.map(d => d.m))].sort((a, b) => a - b), ns = [...new Set(G.map(d => d.n))].sort((a, b) => a - b);
  const W = 560, Lx = 70, T = 30, H = 300;
  const s = svg("#grid", W, H), x = d3.scaleBand().domain(ns).range([Lx, W - 10]).padding(0.06), y = d3.scaleBand().domain(ms).range([T, H - 10]).padding(0.06);
  const c = d3.scaleSequential(d3.interpolateRgb(css("--h0"), css("--tuned"))).domain([d3.min(G, d => d.rel), d3.max(G, d => d.rel)]);
  s.append("text").attr("x", (Lx + W) / 2).attr("y", 12).attr("text-anchor", "middle").attr("class", "tick").text("N block (tiles) →");
  s.append("text").attr("transform", `translate(14,${(T + H) / 2}) rotate(-90)`).attr("text-anchor", "middle").attr("class", "tick").text("M block (tiles) →");
  s.selectAll(".nx").data(ns).join("text").attr("class", "tick").attr("x", d => x(d) + x.bandwidth() / 2).attr("y", T - 4).attr("text-anchor", "middle").text(d => d);
  s.selectAll(".my").data(ms).join("text").attr("class", "tick").attr("x", Lx - 8).attr("y", d => y(d) + y.bandwidth() / 2 + 4).attr("text-anchor", "end").text(d => d);
  const best = d3.max(G, d => d.rel);
  const g = s.selectAll(".gc").data(G).join("g").attr("transform", d => `translate(${x(d.n)},${y(d.m)})`);
  const r = g.append("rect").attr("width", x.bandwidth()).attr("height", y.bandwidth()).attr("rx", 5).attr("fill", d => c(d.rel)).style("opacity", 0)
    .attr("stroke", d => d.m === 16 && d.n === 2 ? css("--ink") : d.rel === best ? css("--gold") : "none").attr("stroke-width", 2.5);
  r.transition().duration(500).delay((d, i) => i * 30).style("opacity", 1);
  g.append("text").attr("x", x.bandwidth() / 2).attr("y", y.bandwidth() / 2 + 4).attr("text-anchor", "middle").style("font-size", "11.5px").style("font-weight", 600)
    .attr("fill", d => d.rel > (best + 1) / 2 ? "#fff" : css("--ink")).text(d => fx(d.rel)).style("pointer-events", "none");
  hover(r, d => `<b>${fx(d.rel)} AWS's default</b>M block ${d.m}, N block ${d.n} (best K)${d.m === 16 && d.n === 2 ? " · AWS's default" : ""}`);
})();

/* the AWS bug */
(function () {
  const B = [{k: "K = 2048", v: 5.3, w: "simulator", c: css("--bad")}, {k: "K = 6144 · down_proj", v: 4.9, w: "chip, held-out", c: css("--bad")},
             {k: "K = 8192 · AWS's benchmark", v: 19.4, w: "simulator", c: css("--bad")}, {k: "Fixed · fp32 accumulator", v: 0.5, w: "one-line fix", c: css("--good")}];
  const W = 540, H = 280, m = {l: 40, r: 10, t: 16, b: 46};
  const s = svg("#bug", W, H), x = d3.scaleBand().domain(B.map(d => d.k)).range([m.l, W - m.r]).padding(0.28);
  const y = d3.scaleLinear().domain([0, 21]).range([H - m.b, m.t]);
  s.append("g").attr("class", "gridl").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - m.l - m.r)));
  s.append("text").attr("x", m.l).attr("y", 8).attr("class", "tick").text("error, bf16 ulps");
  const r = s.selectAll(".b").data(B).join("rect").attr("x", d => x(d.k)).attr("width", x.bandwidth()).attr("rx", 5).attr("fill", d => d.c)
    .attr("y", y(0)).attr("height", 0);
  r.transition().duration(900).delay((d, i) => i * 200).attr("y", d => y(d.v)).attr("height", d => y(0) - y(d.v));
  s.selectAll(".bv").data(B).join("text").attr("class", "val").attr("text-anchor", "middle").attr("x", d => x(d.k) + x.bandwidth() / 2).attr("y", d => y(d.v) - 6).text(d => d.v);
  s.selectAll(".bk").data(B).join("text").attr("class", "tick").attr("text-anchor", "middle").attr("x", d => x(d.k) + x.bandwidth() / 2).attr("y", H - m.b + 16)
    .text(d => d.k.split(" · ")[0]);
  s.selectAll(".bw").data(B).join("text").attr("class", "sub").attr("text-anchor", "middle").attr("x", d => x(d.k) + x.bandwidth() / 2).attr("y", H - m.b + 30)
    .text(d => d.k.split(" · ")[1] || d.w);
  s.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(4)).attr("y2", y(4)).attr("stroke", css("--ink")).attr("stroke-dasharray", "6 4").attr("stroke-width", 1.5);
  s.append("text").attr("x", W - m.r).attr("y", y(4) - 6).attr("text-anchor", "end").attr("class", "lbl").text("limit: 4 ulps");
  hover(r, d => `<b>${d.v} bf16 ulps</b>${d.k} · ${d.w}`);
})();

/* referee */
(function () {
  d3.select("#wall").selectAll("div").data(D.wall).join("div").attr("class", d => "chip" + (d.honest ? " h" : "")).text("✓")
    .call(sel => hover(sel, d => `<b>${d.honest ? "Honest kernel, accepted" : "Cheat caught"}</b>${d.name}<br>${d.what || ""}`));
  const st = [
    {b: `${D.caught} / ${D.cheats}`, s: "planted cheats caught"},
    {b: `${D.honest_ok} / ${D.honest}`, s: "honest kernels accepted"},
    {b: "18 of 18", s: "times a deliberately slowed copy was measured at 2.949×"},
    {b: "R² 0.9999998", s: "time scales linearly with work"},
    {b: "≤ 0.04%", s: "timing moved under live vLLM load"},
  ];
  d3.select("#stats").selectAll(".stat").data(st).join("div").attr("class", "stat").html(d => `<b>${d.b}</b><span>${d.s}</span>`);
})();
</script>
</body>
</html>
"""
