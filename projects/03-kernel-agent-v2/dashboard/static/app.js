/* app.js — the dashboard. Vanilla JS, no build step.
   Color rules (from the dataviz method): the stacked token chart uses the six
   categorical slots in FIXED order (color follows the component, never the rank);
   the taxonomy chart is a single-hue magnitude bar (sequential default); status
   colors appear only on solved/confidently-wrong chips, always with a text label. */

const SLOT = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"];
const PARTS = [                      // fixed slot order for the token stack
  ["reference", "reference + header"],
  ["kernel", "previous kernel"],
  ["feedback", "feedback"],
  ["ledger", "ledger"],
  ["docs", "docs"],
  ["scratch", "scratch"],
];
const SURFACE = "#1a1a19";
const INK = "#c3c2b7";
const MUTED = "#898781";
const GRID = "#2c2c2a";

let selectedRun = null;
let anyLive = false;

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g,
    c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

/* ---------- KPI row ---------- */

function levelName(n) {
  const names = {1:"relu(ax+b)", 2:"row sum", 3:"row max", 4:"RMSNorm", 5:"softmax",
                 6:"transpose", 7:"matmul", 8:"layernorm", 9:"banded attention",
                 10:"conv1d", 11:"cumsum", 12:"variance", 13:"argmax"};
  return names[n] || `level ${n}`;
}

function renderKpis(summary) {
  const el = document.getElementById("kpis");
  const levels = Object.entries(summary.levels || {}).sort((a, b) => a[0] - b[0]);
  if (!levels.length) {
    el.innerHTML = `<div class="empty">no runs yet — start one with
      <code class="inline">python agent.py --all</code></div>`;
    return;
  }
  const solvedTotal = levels.reduce((n, [_lv, A]) => n + A.solved_runs, 0);
  const runsTotal = levels.reduce((n, [_lv, A]) => n + A.runs, 0);
  let html = `<div class="tile hero"><div class="label">levels solved</div>
    <div class="value">${solvedTotal}<span class="of">/${levels.length}</span></div>
    <div class="sub">across ${runsTotal} level-runs</div></div>`;
  for (const [lv, A] of levels) {
    const isHoldout = +lv > 10;
    html += `<div class="tile ${A.solve_rate > 0 ? "solved" : ""} ${isHoldout ? "holdout" : ""}">
      <div class="label">L${lv} · ${esc(levelName(lv))}</div>
      <div class="value">${A.solved_runs}<span class="of">/${A.runs}</span></div>
      <div class="sub">best ${A.best.toFixed(2)} · mean ${A.mean.toFixed(2)}
       · ${Math.round(A.attempts / Math.max(1, A.runs))} att/run</div>
    </div>`;
  }
  el.innerHTML = html;
}

/* ---------- token stacked bars ---------- */

function renderTokens(runs) {
  const host = document.getElementById("tokchart");
  const legend = document.getElementById("toklegend");
  const run = selectedRun ? runs.find(r => r.run_id === selectedRun) : runs[0];
  const attempts = (run && run.summary.tokens_by_attempt) || [];
  if (!attempts.length) {
    host.innerHTML = `<div class="empty">no attempts recorded yet</div>`;
    legend.innerHTML = "";
    return;
  }
  // newest left; each attempt one column of stacked segments
  const data = attempts.slice().reverse();
  const W = Math.max(560, host.clientWidth || 600), H = 190;
  const pad = {l: 42, r: 8, t: 10, b: 22};
  const maxTotal = Math.max(1, ...data.map(d =>
    PARTS.reduce((s, [k]) => s + (d.parts[k] || 0), 0)));
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const n = data.length;
  const slot = Math.min(24, Math.max(6, (iw / n) * 0.55));
  const gap = Math.max(2, (iw / n) * 0.45);
  const y = v => pad.t + ih * (1 - v / maxTotal);

  let bars = "";
  let grid = "";
  for (let g = 0; g <= 4; g++) {
    const v = maxTotal * g / 4;
    grid += `<line x1="${pad.l}" y1="${y(v)}" x2="${W - pad.r}" y2="${y(v)}"
      stroke="${GRID}" stroke-width="1"/>
      <text x="${pad.l - 6}" y="${y(v) + 3.5}" fill="${MUTED}" font-size="10"
        text-anchor="end">${Math.round(v)}</text>`;
  }
  data.forEach((d, i) => {
    const x = pad.l + i * (iw / n) + gap / 2;
    let acc = 0;
    PARTS.forEach(([k, _label], si) => {
      const v = d.parts[k] || 0;
      if (!v) return;
      const y0 = y(acc + v), h = Math.max(1, y(acc) - y(acc + v) - 2); // 2px surface gap
      bars += `<rect x="${x}" y="${y0}" width="${slot}" height="${h}" rx="2"
        fill="${SLOT[si]}"><title>L${d.level} r${d.round} ${esc(k)}: ${v} tok</title></rect>`;
      acc += v;
    });
    if (d.solved) {  // solved marker: a ring above the bar
      bars += `<circle cx="${x + slot / 2}" cy="${y(acc) - 8}" r="3.5" fill="none"
        stroke="#0ca30c" stroke-width="2"><title>solved</title></circle>`;
    }
  });
  host.innerHTML = `<svg width="${W}" height="${H}" role="img"
    aria-label="tokens per attempt stacked by component">${grid}${bars}</svg>`;
  legend.innerHTML = PARTS.map(([k, label], si) =>
    `<span class="key"><span class="sw" style="background:${SLOT[si]}"></span>${esc(label)}</span>`
  ).join("") + `<span class="key" style="color:${MUTED}">○ solved attempt</span>`;
}

/* ---------- taxonomy bars ---------- */

function renderTaxonomy(summary) {
  const host = document.getElementById("taxchart");
  const entries = Object.entries(summary.taxonomy || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    host.innerHTML = `<div class="empty">no failures recorded</div>`;
    return;
  }
  const W = Math.max(300, host.clientWidth || 380);
  const rowH = 22, pad = {l: 118, r: 34, t: 4};
  const H = pad.t + entries.length * rowH + 6;
  const max = Math.max(1, ...entries.map(e => e[1]));
  const iw = W - pad.l - pad.r;
  let svg = "";
  entries.forEach(([tax, n], i) => {
    const yy = pad.t + i * rowH;
    const w = Math.max(2, iw * n / max);
    svg += `<text x="${pad.l - 8}" y="${yy + 14}" fill="${INK}" font-size="11"
      text-anchor="end">${esc(tax)}</text>
      <rect x="${pad.l}" y="${yy + 3}" width="${w}" height="14" rx="3"
        fill="${SLOT[0]}"><title>${esc(tax)}: ${n}</title></rect>
      <text x="${pad.l + w + 6}" y="${yy + 14}" fill="${MUTED}" font-size="11">${n}</text>`;
  });
  host.innerHTML = `<svg width="${W}" height="${H}" role="img"
    aria-label="failure taxonomy counts">${svg}</svg>`;
}

/* ---------- runs table ---------- */

function renderRuns(runs) {
  const tb = document.querySelector("#runstable tbody");
  anyLive = runs.some(r => !r.done);
  document.getElementById("live").className = "live " + (anyLive ? "" : "idle");
  document.getElementById("live").innerHTML =
    `<span class="dot"></span>${anyLive ? "run live" : "idle"}`;
  if (!runs.length) {
    tb.innerHTML = `<tr><td colspan="7" class="empty">no runs yet</td></tr>`;
    return;
  }
  tb.innerHTML = runs.map(r => {
    const s = r.summary;
    const when = new Date((r.started || 0) * 1000).toLocaleTimeString();
    const cfg = `r${r.rounds}×s${r.samples}${r.repeat > 1 ? `×${r.repeat}` : ""}${r.greedy ? " greedy" : ""}`;
    return `<tr class="click ${selectedRun === r.run_id ? "selected" : ""}" data-run="${esc(r.run_id)}">
      <td class="strong">${esc(r.run_id)}</td>
      <td>${esc(r.model || "")}${r.tag ? ` · ${esc(r.tag)}` : ""}</td>
      <td>${(r.levels || []).length === 0 ? "—" : esc((r.levels || []).join(","))}</td>
      <td>${esc(cfg)} · ${esc(when)}</td>
      <td class="strong">${s.solved_n}/${(r.levels || []).length || (s.levels ? Object.keys(s.levels).length : 0)}</td>
      <td>${s.attempts}${s.tool_calls ? ` <span style="color:${MUTED}">+${s.tool_calls}tools</span>` : ""}</td>
      <td>${r.done ? `<span class="dot-done"></span>done` : `<span class="dot-run"></span>running`}</td>
    </tr>`;
  }).join("");
  tb.querySelectorAll("tr.click").forEach(tr =>
    tr.addEventListener("click", () => selectRun(tr.dataset.run)));
}

/* ---------- calibration ---------- */

function renderCalibration(runs) {
  const el = document.getElementById("calibration");
  const agg = {high: [0, 0], medium: [0, 0], low: [0, 0]};
  let cw = 0;
  for (const r of runs) {
    for (const rec of []) {} // per-run trace is fetched only on select; aggregate from summary
    const levels = r.summary.levels || {};
    for (const L of Object.values(levels)) { /* taxonomy-only here */ }
  }
  // full aggregation needs the traces; do it from the selected run + a sweep endpoint
  // -- simplest honest path: fetch every run's trace once (small files, local)
  Promise.all(runs.map(r => jget(`/api/run/${encodeURIComponent(r.run_id)}`)))
    .then(all => {
      for (const {trace} of all) {
        for (const rec of trace) {
          if (rec.type !== "attempt") continue;
          const c = rec.confidence;
          if (c && c in agg && rec.confidence !== "none") {
            agg[c][0] += 1;
            if (!rec.solved) agg[c][1] += 1;
          }
          if (rec.taxonomy === "confidently-wrong") cw += 1;
        }
      }
      const rows = ["high", "medium", "low"].filter(k => agg[k][0]).map(k => {
        const [claims, wrong] = agg[k];
        return `<tr><td class="strong">${k}</td><td>${claims}</td><td>${wrong}</td>
          <td>${claims ? Math.round(100 * (claims - wrong) / claims) : 0}%</td></tr>`;
      });
      el.innerHTML = `<table><thead><tr><th>confidence</th><th>claims</th>
        <th>wrong</th><th>right</th></tr></thead><tbody>${rows.join("") ||
        `<tr><td colspan="4" class="empty">no confidence claims yet</td></tr>`}</tbody></table>`
        + (cw ? `<p class="hint" style="color:var(--critical);margin-top:8px;">
              confidently-wrong events: ${cw}</p>` : "");
    })
    .catch(() => { el.innerHTML = `<div class="empty">calibration unavailable</div>`; });
}

/* ---------- trace viewer ---------- */

const TAX_CHIP = t => `<span class="chip tax">${esc(t || "")}</span>`;

async function selectRun(runId) {
  selectedRun = runId;
  document.getElementById("trace-section").style.display = "";
  document.getElementById("trace-run-name").textContent = runId;
  document.querySelectorAll("#runstable tr").forEach(tr =>
    tr.classList.toggle("selected", tr.dataset.run === runId));
  const {meta, trace} = await jget(`/api/run/${encodeURIComponent(runId)}`);
  const host = document.getElementById("trace");
  const levels = [...new Set(trace.filter(r => r.type === "attempt" || r.type === "tool")
                               .map(r => r.level))].sort((a, b) => a - b);
  if (!levels.length) { host.innerHTML = `<div class="empty">no records yet</div>`; return; }

  let html = "";
  for (const lv of levels) {
    const name = levelName(lv);
    const end = trace.find(r => r.type === "level_end" && r.level === lv);
    html += `<div class="lvl"><h4>level ${lv} · ${esc(name)}
      ${end ? (end.solved
        ? `<span class="chip solved">SOLVED round ${(end.solved_round || 0) + 1}</span>`
        : `<span class="chip">not solved · best ${(end.best || 0).toFixed(2)}</span>`) : ""}
    </h4>`;
    for (const rec of trace) {
      if (rec.level !== lv) continue;
      if (rec.type === "tool") {
        html += `<div class="tool-line"><b>${esc(rec.tool)}:</b>
          <code class="inline">${esc(rec.ask)}</code>
          ${rec.result && rec.result !== "(suggested to the model)"
            ? ` → <span style="color:${MUTED}">${esc(String(rec.result).slice(0, 160))}</span>` : ""}
        </div>`;
      } else if (rec.type === "attempt") {
        const tok = rec.tokens || {parts: {}};
        const parts = tok.parts || {};
        const maxPart = Math.max(1, ...PARTS.map(([k]) => parts[k] || 0));
        const tokrows = PARTS.filter(([k]) => parts[k]).map(([k, label], si) =>
          `<div class="tokbars" style="grid-template-columns:92px 1fr 44px">
            <span>${esc(label)}</span>
            <span><span class="bar" style="background:${SLOT[si]};width:${Math.round(100 * (parts[k] || 0) / maxPart)}%"></span></span>
            <span class="n">${parts[k]}</span>
          </div>`).join("");
        const total = tok.total || Object.values(parts).reduce((a, b) => a + b, 0);
        const cw = rec.taxonomy === "confidently-wrong" && !rec.solved;
        html += `<div class="attempt">
          <div class="head">
            <span class="r">r${rec.round}·s${rec.sample ?? "—"}</span>
            <span class="reward">${(rec.reward ?? 0).toFixed(2)}</span>
            ${rec.solved ? `<span class="chip solved">solved</span>` : TAX_CHIP(rec.taxonomy)}
            ${rec.confidence && rec.confidence !== "none"
              ? `<span class="chip conf-${rec.confidence}">conf: ${esc(rec.confidence)}</span>` : ""}
            ${cw ? `<span class="chip cw">confidently wrong</span>` : ""}
            ${rec.finish === "length" ? `<span class="chip cw">truncated</span>` : ""}
            <span style="flex:1"></span>
            <span style="color:${MUTED};font-size:11px;">${total} tok in
              ${tok.api_prompt_tokens ? `· api ${tok.api_prompt_tokens}` : ""}</span>
          </div>
          <div class="body">
            ${rec.verdict ? `<div class="block"><div class="bl">verdict</div>
              <pre>${esc(rec.verdict)}</pre></div>` : ""}
            ${rec.instruction ? `<div class="block"><div class="bl">instruction sent back</div>
              <div class="instr">${esc(rec.instruction)}</div></div>` : ""}
            ${tokrows ? `<div class="block"><div class="bl">prompt composition (est. tokens)</div>
              ${tokrows}</div>` : ""}
            ${rec.code ? `<div class="block"><div class="bl">code</div>
              <pre class="code">${esc(rec.code)}</pre></div>` : ""}
          </div>
        </div>`;
      }
    }
    html += `</div>`;
  }
  host.innerHTML = html;
  host.querySelectorAll(".attempt .head").forEach(h =>
    h.addEventListener("click", () => h.parentElement.classList.toggle("open")));
  // auto-open the last attempt of each unsolved level (the interesting failure)
  const opens = host.querySelectorAll(".attempt");
  if (opens.length) opens[opens.length - 1].classList.add("open");
}

/* ---------- refresh loop ---------- */

async function refresh() {
  try {
    const runs = await jget("/api/runs");
    renderRuns(runs);
    const summary = await jget("/api/summary");
    renderKpis(summary);
    renderTokens(runs);
    renderTaxonomy(summary);
    renderCalibration(runs);
    if (selectedRun && runs.some(r => r.run_id === selectedRun)) {
      await selectRun(selectedRun);      // cheap; local files
    } else if (runs.length && !selectedRun) {
      await selectRun(runs[0].run_id);
    }
  } catch (e) {
    console.warn("refresh failed:", e);
  }
}

document.getElementById("auto").addEventListener("change", e => {
  if (e.target.checked) tick();
});

let timer = null;
async function tick() {
  await refresh();
  clearTimeout(timer);
  if (document.getElementById("auto").checked) {
    timer = setTimeout(tick, anyLive ? 3000 : 10000);
  }
}
tick();
