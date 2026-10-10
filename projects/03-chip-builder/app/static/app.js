'use strict';

const MAX_ROUNDS = 5;
const POLL_FAST = 600;   // while requests are streaming
const POLL_SLOW = 1200;

const BLOCK_STYLE = {
  compute_cluster: ['var(--compute-fill)', 'var(--compute-stroke)', 'Compute'],
  hbm: ['var(--hbm-fill)', 'var(--hbm-stroke)', 'HBM'],
  sbuf: ['var(--sbuf-fill)', 'var(--sbuf-stroke)', 'SRAM'],
  dma: ['var(--dma-fill)', 'var(--dma-stroke)', 'DMA'],
  router: ['#f4f4f5', '#a1a1aa', 'Router'],
  io: ['#f4f4f5', '#a1a1aa', 'I/O'],
};
const SHORT = {
  'HBM bandwidth': 'HBM BW', 'HBM capacity': 'HBM', 'Compute cores': 'Cores', 'Compute': 'TFLOPS',
  'On-chip SRAM': 'SRAM', 'DMA engines': 'DMA', 'Utilization': 'Util', 'Power limit': 'Power cap',
  'Cooling': 'Cooling', 'Compute clusters': 'Clusters', 'HBM stacks': 'HBM stacks', 'Links': 'Links',
};
const DEFAULT_OBJECTIVE = { metric: 'ttft_p99_ms', direction: 'min', short: 'p99 TTFT', label: 'p99 time-to-first-token', unit: 'ms' };
const LOWER_IS_BETTER = { ttft_p99_ms: true, area_mm2: true, power_w: true, max_temperature_c: true, throughput_tokens_s: false };

const state = {
  scenarios: [], scenario: null, campaignId: null, timer: null, model: 'the model',
  rendered: new Set(), entries: new Map(), selected: null, baseline: null, objective: DEFAULT_OBJECTIVE,
  checkerShown: false,
};
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ helpers

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'style') node.style.cssText = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

const SVG_NS = 'http://www.w3.org/2000/svg';
function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const fmt = (v, digits = 0) => (num(v) === null ? '—' : v.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }));
const short = (v) => (num(v) === null ? '—' : Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : +v.toPrecision(3));
const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

function fmtMs(v) {
  if (num(v) === null) return '—';
  if (v >= 1000) return `${(v / 1000).toFixed(v >= 10000 ? 1 : 2)} s`;
  return v < 10 ? `${v.toFixed(1)} ms` : `${Math.round(v)} ms`;
}
function fmtMetric(metric, v) {
  if (metric === 'ttft_p99_ms') return fmtMs(v);
  if (metric === 'throughput_tokens_s') return `${fmt(v, v < 100 ? 1 : 0)} tok/s`;
  if (metric === 'area_mm2') return `${fmt(v)} mm²`;
  if (metric === 'power_w') return `${fmt(v)} W`;
  return short(v);
}
const withUnit = (v, unit) => (unit === 'ms' ? fmtMs(v) : `${short(v)} ${unit}`);
const pct = (value, base) => (num(value) === null || !num(base) ? null : ((value - base) / base) * 100);
const signed = (p, digits = 1) => `${p > 0 ? '+' : '−'}${Math.abs(p).toFixed(digits)}%`;
const better = (metric, change) => (change === null || Math.abs(change) < 0.05 ? null : (LOWER_IS_BETTER[metric] ? change < 0 : change > 0));

// Tween a number in place so live values glide instead of jumping.
function tween(node, to, render, ms = 500) {
  const from = num(node._value);
  node._value = to;
  if (num(to) === null || from === null || reducedMotion()) { node.textContent = render(to); return; }
  const start = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - start) / ms);
    node.textContent = render(from + (to - from) * (1 - Math.pow(1 - t, 3)));
    if (t < 1 && node._value === to) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function transition(update) {
  if (document.startViewTransition && !reducedMotion()) document.startViewTransition(update);
  else update();
}

function sparkline(values, color) {
  const pts = values.filter((v) => num(v) !== null).slice(-40);
  const root = svg('svg', { viewBox: '0 0 100 28', preserveAspectRatio: 'none' });
  if (pts.length < 2) return root;
  const hi = Math.max(...pts, 1e-6), lo = Math.min(...pts, 0);
  const xy = pts.map((v, i) => [(i / (pts.length - 1)) * 100, 26 - ((v - lo) / Math.max(hi - lo, 1e-6)) * 24]);
  const line = xy.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ');
  root.append(svg('path', { d: `${line} L100 28 L0 28 Z`, fill: color, 'fill-opacity': 0.08 }));
  root.append(svg('path', { d: line, fill: 'none', stroke: color, 'stroke-width': 1.5, 'vector-effect': 'non-scaling-stroke' }));
  return root;
}

// ------------------------------------------------------------------ diagram

function blockLabel(c) {
  const index = Number(String(c.id).split('-').pop());
  if (c.type === 'sbuf') return 'SRAM';
  const name = (BLOCK_STYLE[c.type] || [])[2] || c.type;
  return Number.isFinite(index) ? `${name} ${index + 1}` : c.id;
}
function blockValue(c) {
  if (c.type === 'compute_cluster' && c.compute_tflops) return `${Math.round(c.compute_tflops)} TF`;
  if (c.type === 'hbm' && c.capacity_gb) return `${+c.capacity_gb.toFixed(1)} GB · ${+c.bandwidth_tb_s.toFixed(2)} TB/s`;
  if (c.type === 'sbuf' && c.capacity_gb) return `${+c.capacity_gb.toFixed(1)} GB`;
  if (c.type === 'dma' && c.dma_engines > 1) return `×${c.dma_engines}`;
  return '';
}

function drawDiagram(design, changedIds = [], previousBoxes = {}) {
  const S = 300, P = 10;
  const root = svg('svg', { viewBox: `0 0 ${S + 2 * P} ${S + 2 * P}`, role: 'img', 'aria-label': 'Chip floorplan' });
  const components = (design && design.components) || [];
  const connections = (design && design.connections) || [];
  const changed = new Set(changedIds || []);
  const box = (c) => ({ x: P + c.x * S, y: P + c.y * S, w: c.width * S, h: c.height * S });

  root.append(svg('rect', { x: P, y: P, width: S, height: S, rx: 12, fill: '#fff', stroke: '#e4e4e7' }));
  for (const c of components.filter((c) => c.type === 'noc')) {
    const b = box(c);
    root.append(svg('rect', { x: b.x, y: b.y, width: b.w, height: b.h, rx: 8, fill: '#f4f4f5', stroke: '#d4d4d8', 'stroke-dasharray': '4 3' }));
  }
  const byId = Object.fromEntries(components.map((c) => [c.id, c]));
  const maxBw = Math.max(0.01, ...connections.map((e) => e.bandwidth_tb_s || 0));
  for (const edge of connections) {
    const a = byId[edge.source], b = byId[edge.target];
    if (!a || !b) continue;
    const pa = box(a), pb = box(b);
    root.append(svg('line', { class: 'link-line', x1: pa.x + pa.w / 2, y1: pa.y + pa.h / 2, x2: pb.x + pb.w / 2, y2: pb.y + pb.h / 2,
      'stroke-width': (0.8 + 2.6 * (edge.bandwidth_tb_s || 0) / maxBw).toFixed(2) }));
  }
  for (const [id, [x, y, w, h]] of Object.entries(previousBoxes || {})) {
    const now = byId[id];
    if (!now || (Math.abs(now.x - x) < 0.006 && Math.abs(now.y - y) < 0.006 && Math.abs(now.width - w) < 0.006)) continue;
    root.append(svg('rect', { class: 'ghost', x: P + x * S, y: P + y * S, width: w * S, height: h * S, rx: 5 }));
  }
  components.filter((c) => c.type !== 'noc').forEach((c, i) => {
    const b = box(c);
    const [fill, stroke] = BLOCK_STYLE[c.type] || BLOCK_STYLE.io;
    const isChanged = changed.has(c.id);
    const g = svg('g', { class: isChanged ? 'blk blk-changed' : 'blk', style: `animation-delay:${i * 25}ms` });
    g.append(svg('rect', { x: b.x, y: b.y, width: b.w, height: b.h, rx: 5, fill,
      stroke: isChanged ? 'var(--changed)' : stroke, 'stroke-width': isChanged ? 2.4 : 1.2 }));
    const size = Math.max(6, Math.min(11, b.w / 6.5, b.h / 2.4));
    const value = blockValue(c);
    const twoLines = value && b.h > size * 3.2 && b.w - 6 > value.length * size * 0.85 * 0.62;
    const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
    g.append(svg('text', { x: cx, y: twoLines ? cy - size * 0.25 : cy + size * 0.35, 'text-anchor': 'middle',
      'font-size': size, 'font-weight': 600, 'font-family': 'Geist, sans-serif', fill: '#18181b' }, blockLabel(c)));
    if (twoLines) g.append(svg('text', { x: cx, y: cy + size * 1.05, 'text-anchor': 'middle', 'font-size': size * 0.85,
      'font-family': 'Geist Mono, monospace', fill: '#71717a' }, value));
    root.append(g);
  });
  return root;
}

// IDs of blocks that are new, moved or resized relative to another design.
function changedVs(before, after) {
  const old = Object.fromEntries(((before && before.components) || []).map((c) => [c.id, c]));
  return ((after && after.components) || []).filter((c) => c.type !== 'noc' && (!old[c.id] ||
    ['x', 'y', 'width', 'height'].some((k) => Math.abs((c[k] || 0) - (old[c.id][k] || 0)) > 0.005))).map((c) => c.id);
}

// ------------------------------------------------------------- 01 problem

function objectiveChip(o) {
  return el('span', { class: 'chip accent' }, `${o.direction === 'min' ? '↓' : '↑'} ${o.short}`);
}

function renderScenarios() {
  $('scenarios').replaceChildren(...state.scenarios.map((s, i) => {
    const c = s.constraints;
    const card = el('button', { type: 'button', class: 'scenario', role: 'radio', 'aria-checked': String(state.scenario === s.id),
      'data-id': s.id, style: `animation-delay:${i * 50}ms` },
      el('h3', {}, s.title), el('p', {}, s.problem),
      el('div', { class: 'obj' }, objectiveChip(s.objective), el('span', { class: 'chip' }, `≤ ${c.max_power_w} W`),
        el('span', { class: 'chip' }, `≤ ${c.max_area_mm2} mm²`)));
    card.addEventListener('click', () => pickScenario(s.id));
    return card;
  }));
}

function pickScenario(id) {
  state.scenario = id;
  const s = state.scenarios.find((x) => x.id === id);
  if (s) $('goal').value = s.goal;
  for (const card of document.querySelectorAll('.scenario')) card.setAttribute('aria-checked', String(card.dataset.id === id));
}

function lockProblem(locked) {
  $('start').disabled = locked;
  $('goal').disabled = locked;
  for (const card of document.querySelectorAll('.scenario')) card.disabled = locked;
}

// ------------------------------------------------------------- 02 measure

function renderLanes(live) {
  const requests = live.requests || [];
  const list = $('lanes');
  if (list.children.length !== requests.length) {
    list.replaceChildren(...requests.map((r, i) => el('li', { class: 'lane queued', style: `animation-delay:${i * 40}ms` },
      el('span', {}, `#${r.id}`), el('span', { class: 'track' }, el('span', { class: 'fill' })), el('span', { class: 'info' }, 'queued'))));
  }
  requests.forEach((r, i) => {
    const lane = list.children[i];
    lane.className = `lane ${r.state}`;
    const fill = lane.querySelector('.fill');
    if (r.state === 'decode' || r.state === 'done') fill.style.width = `${Math.min(100, (r.tokens / Math.max(r.max_tokens, 1)) * 100)}%`;
    lane.querySelector('.info').textContent = {
      queued: 'queued', prefill: 'prefill…', error: 'error',
      decode: `${r.tokens} tok · ttft ${fmtMs(r.ttft_ms)}`,
      done: `${r.tokens} tok · ttft ${fmtMs(r.ttft_ms)}`,
    }[r.state] || r.state;
  });
}

function metricTile(key, label, src, spark) {
  return el('div', { class: 'card metric-tile', 'data-key': key },
    el('div', { class: 'k' }, el('span', {}, label), el('span', { class: 'src' }, src)),
    el('div', { class: 'v' }), spark ? el('div', { class: 'spark' }) : null);
}

function renderLiveMetrics(campaign) {
  const box = $('live-metrics');
  if (!box.children.length) {
    box.append(metricTile('done', 'Requests done', 'live'), metricTile('elapsed', 'Elapsed', 'live'),
      metricTile('tps', 'Tokens / s', 'live', true), metricTile('util', 'NeuronCores busy', 'neuron-monitor', true),
      metricTile('ttft', 'First token (median)', 'live'), metricTile('kv', 'KV cache in use', 'vLLM'));
  }
  const live = campaign.live || {};
  const reqs = live.requests || [];
  const samples = live.samples || [];
  const last = samples[samples.length - 1] || {};
  const ttfts = reqs.map((r) => r.ttft_ms).filter((v) => num(v) !== null).sort((a, b) => a - b);
  const t = campaign.telemetry;   // final, measured numbers replace the live ones
  const v = (t && t.vllm) || {}, n = (t && t.neuron_monitor) || {};
  const set = (key, value, render, sub) => {
    const tile = box.querySelector(`[data-key="${key}"]`);
    const valueEl = tile.querySelector('.v');
    if (!valueEl.firstChild || valueEl.firstChild.nodeType !== 3) valueEl.replaceChildren(document.createTextNode(''));
    tween(valueEl.firstChild, value, render);
    let small = valueEl.querySelector('small');
    if (sub) { if (!small) { small = el('small'); valueEl.append(small); } small.textContent = sub; } else if (small) small.remove();
  };
  set('done', t ? v.requests : reqs.filter((r) => r.state === 'done').length, (x) => fmt(x), `/ ${t ? v.replayed || reqs.length : reqs.length}`);
  set('elapsed', t ? v.wall_s : live.elapsed_s, (x) => `${fmt(x, 1)} s`);
  set('tps', t ? v.throughput_tokens_s : last.tps, (x) => fmt(x, 0));
  set('util', t && num(n.core_utilization_pct) !== null ? n.core_utilization_pct : last.util, (x) => `${fmt(x, 0)}%`,
    t && num(n.core_utilization_peak_pct) !== null ? `peak ${fmt(n.core_utilization_peak_pct, 0)}%` : null);
  set('ttft', t ? v.ttft_p50_ms : ttfts[Math.floor(ttfts.length / 2)], fmtMs, t && num(v.ttft_p99_ms) !== null ? `p99 ${fmtMs(v.ttft_p99_ms)}` : null);
  set('kv', t ? v.kv_cache_peak_pct : last.kv, (x) => `${fmt(x, 1)}%`, t ? 'peak' : null);
  box.querySelector('[data-key="tps"] .spark').replaceChildren(sparkline(samples.map((s) => s.tps), 'var(--accent)'));
  box.querySelector('[data-key="util"] .spark').replaceChildren(sparkline(samples.map((s) => s.util), 'var(--good)'));
  if (t) for (const tile of box.children) tile.querySelector('.src').textContent = tile.dataset.key === 'util' ? 'neuron-monitor' : 'measured';
}

function renderFinding(campaign) {
  const base = campaign.baseline_metrics;
  if (!base || state.findingShown) return;
  state.findingShown = true;
  const o = state.objective;
  const cal = campaign.calibration || {};
  const failed = (campaign.baseline_checks || []).filter((r) => r.passed === false);
  const scale = cal.measured_prompt_tokens && cal.prompt_tokens !== cal.measured_prompt_tokens
    ? ` with ${fmt(cal.prompt_tokens)}-token prompts (measured ${fmt(cal.measured_prompt_tokens)})` : '';
  let headline;
  if (failed.length) headline = `Today's chip does not fit this budget: ${failed.map(checkText).join('; ')}.`;
  else if (o.metric === 'ttft_p99_ms') headline = `On today's chip, the slowest 1% of requests wait ${fmtMs(base.ttft_p99_ms)} for their first token under this scenario's bursts${scale}.`;
  else if (o.metric === 'throughput_tokens_s') headline = `Today's chip sustains ${fmtMetric('throughput_tokens_s', base.throughput_tokens_s)} when saturated${scale}.`;
  else headline = `Today's die is ${fmtMetric('area_mm2', base.area_mm2)}, serving at ${fmtMs(base.ttft_p99_ms)} p99 TTFT and ${fmtMetric('throughput_tokens_s', base.throughput_tokens_s)}.`;
  const why = base.decode_bound
    ? `Decode is ${base.decode_bound}-bound (${fmtMs(base.decode_memory_ms)} of HBM traffic vs ${fmtMs(base.decode_compute_ms)} of compute per token) and prefill is ${base.prefill_bound}-bound.`
    : '';
  const s = (campaign.telemetry || {}).system_trace || {};
  const trace = s.available ? ` The Neuron trace shows the NeuronCores ${fmt(s.device_busy_pct, 0)}% busy during traffic, with collectives at ${fmt(s.collective_pct, 1)}% of execution.` : '';
  $('finding').className = `finding${failed.length ? ' infeasible' : ''}`;
  $('finding').replaceChildren(el('h3', {}, failed.length ? 'The problem: over budget' : 'The problem, measured'),
    el('p', {}, headline), why || trace ? el('p', {}, why + trace) : null);
  $('finding').hidden = false;
}

function kv(rows) {
  return el('dl', { class: 'kv' }, rows.filter(Boolean).map(([k, v]) => [el('dt', {}, k), el('dd', {}, v)]));
}
function source(title, data, rows) {
  const ok = data && data.available;
  return el('div', { class: 'source' },
    el('div', { class: 'source-head' }, el('span', {}, title), el('span', { class: `tag ${ok ? 'measured' : 'missing'}` }, ok ? 'measured' : 'not available')),
    ok ? kv(rows) : el('p', { class: 'missing-note' }, (data && data.reason) || 'Not collected.',
      data && data.hint ? ['. To capture one: ', el('code', {}, data.hint), '.'] : null));
}

function renderChecker(campaign) {
  if (state.checkerShown || !campaign.baseline_metrics) return;
  state.checkerShown = true;
  const t = campaign.telemetry || {};
  const v = t.vllm || {}, n = t.neuron_monitor || {}, s = t.system_trace || {};
  const cal = campaign.calibration || {};
  const base = campaign.baseline_metrics || {};
  const measured = el('div', { class: 'card checker-col' }, el('h3', {}, 'Measured on your Trainium'),
    source('vLLM · Prometheus', v, [
      ['TTFT p50 / p99', `${fmtMs(v.ttft_p50_ms)} / ${fmtMs(v.ttft_p99_ms)}`],
      ['Queue · prefill', `${fmtMs(v.queue_ms)} · ${fmtMs(v.prefill_ms)}`],
      ['Per output token', fmtMs(v.inter_token_ms)],
      ['Tokens in / out', `${fmt(v.prompt_tokens_mean)} / ${fmt(v.generation_tokens_mean)}`],
    ]),
    source('neuron-monitor', n, [
      ['NeuronCore busy', `${fmt(n.core_utilization_pct)}% avg · ${fmt(n.core_utilization_peak_pct)}% peak`],
      ['Device memory', `${fmt(n.device_memory_gb, 1)} / ${fmt(n.device_memory_total_gb)} GB`],
    ]),
    source('Neuron Explorer system trace', s, [
      ['NeuronCores busy (traffic)', `${fmt(s.device_busy_pct, 1)}%`],
      ['Prefill · decode step', `${fmtMs(s.prefill_step_ms)} · ${fmtMs(s.decode_step_ms)}`],
      ['Collectives / host copies', `${fmt(s.collective_pct, 1)}% / ${fmt(s.host_copy_pct, 1)}%`],
      ['HBM in use', `${fmt(s.hbm_used_gb, 1)} GB`],
    ]));
  const calibrated = el('div', { class: 'card checker-col' }, el('h3', {}, 'Simulator, fitted to it'),
    el('div', { class: 'source' }, el('div', { class: 'source-head' }, el('span', {}, 'Calibration'),
      el('span', { class: `tag ${cal.live ? 'modeled' : 'missing'}` }, cal.live ? 'calibrated' : 'defaults')),
    kv([['Prefill', `${fmtMs(cal.prefill_overhead_ms)} host + ×${fmt(cal.prefill_scale, 1)} roofline`],
      ['Decode step', `${fmtMs(cal.decode_overhead_ms)} host + ×${fmt(cal.decode_scale, 2)} roofline`],
      ['Collectives', `${fmt((cal.collective_share || 0) * 100, 1)}% of device time`],
      ['Memory footprint', num(cal.memory_footprint_gb) === null ? 'estimated' : `${fmt(cal.memory_footprint_gb, 1)} GB measured`],
      ['Host / device split', cal.split_source === 'system trace' ? 'from system trace' : 'not available']])),
    el('div', { class: 'source' }, el('div', { class: 'source-head' }, el('span', {}, 'Scenario workload'), el('span', { class: 'tag modeled' }, 'modeled')),
      kv([['Tokens in / out', `${fmt(cal.prompt_tokens)} / ${fmt(cal.generation_tokens)}`],
        ['Bursts of 4, load', `${fmt((cal.load || 0) * 100)}% of today`],
        ['Today: p99 TTFT', fmtMs(base.ttft_p99_ms)], ['Today: throughput', fmtMetric('throughput_tokens_s', base.throughput_tokens_s)]])));
  const o = state.objective;
  const checker = el('div', { class: 'card checker-col' }, el('h3', {}, 'Checker'),
    el('p', { class: 'missing-note' }, `Score: ${o.direction === 'min' ? 'lower' : 'higher'} ${o.label}. Every design must pass:`),
    checkList(campaign.baseline_checks, true),
    el('p', { class: 'missing-note' }, 'A failed or slower design goes back to the model with the limiting resource and an absolute target that would fix it.'));
  $('checker-grid').replaceChildren(measured, calibrated, checker);
  $('judging').hidden = false;
}

// ------------------------------------------------------------- 03 explore

function renderJudge(campaign) {
  if ($('judge').children.length || !campaign.baseline_checks) return;
  const o = state.objective;
  $('judge').replaceChildren(el('span', { class: 'chip score' }, `Score: ${o.direction === 'min' ? 'lower' : 'higher'} ${o.short}`),
    ...(campaign.baseline_checks || []).map((r) => el('span', { class: 'chip' }, `${r.label} ${r.op} ${withUnit(r.limit, r.unit)}`)),
    el('span', { class: 'chip' }, 'valid layout & wiring'));
}

function renderCalls(campaign) {
  const calls = campaign.calls || [];
  const box = $('calls');
  const done = (campaign.attempts || []).length;
  const running = campaign.status === 'running';
  const round = running ? Math.max(done + 1, ...calls.map((c) => c.round || 0)) : done;
  const tokens = calls.reduce((sum, c) => sum + (c.tokens || 0), 0);
  if (!calls.length && !running) { box.hidden = true; return; }
  box.hidden = false;
  const inRound = calls.filter((c) => c.round === round);
  const simulating = running && /simulating/i.test(campaign.activity || '');
  const stage = (key, label) => {
    const mine = inRound.filter((c) => c.stage === key);
    const lastCall = mine[mine.length - 1];
    let cls = '', note = '';
    if (key === 'simulate') {
      cls = simulating ? 'running' : (campaign.attempts || []).some((a) => a.round === round) ? 'done' : '';
      note = simulating ? 'checking…' : '';
    } else if (lastCall) {
      cls = lastCall.status === 'running' ? 'running' : lastCall.status === 'error' ? 'error' : 'done';
      note = lastCall.status === 'running' ? `${lastCall.tokens} tok` : `${(lastCall.ms / 1000).toFixed(1)} s · ${lastCall.tokens} tok`;
      if (mine.length > 1) note += ` · ×${mine.length}`;
    }
    return el('span', { class: `stage ${cls}` }, el('span', { class: 'dot' }), label, note ? el('span', { class: 'n' }, note) : null);
  };
  const arrow = () => el('span', { class: 'stage-arrow' }, '→');
  box.className = 'card calls';
  box.replaceChildren(
    el('div', { class: 'calls-head' },
      el('span', { class: 'round' }, running ? 'Round ' : 'Last round ', el('b', {}, String(round || 1)), ` · ${state.model}`),
      el('span', { class: 'total' }, `${calls.length} model calls · ${fmt(tokens)} tokens`)),
    el('div', { class: 'stages' }, stage('plan', 'Plan'), arrow(), stage('floorplan', 'Floorplan'), arrow(),
      stage('wiring', 'Wiring'), arrow(), stage('simulate', 'Simulate & check')));
  $('model-badge').classList.toggle('busy', calls.some((c) => c.status === 'running'));
}

const failedChecks = (attempt) => (attempt.checks || []).filter((row) => row.passed === false);
function checkText(row) {
  if (row.key === 'topology_valid') return 'invalid layout or wiring';
  return `${row.label} ${withUnit(row.value, row.unit)} ${row.op === '≥' ? '<' : '>'} ${withUnit(row.limit, row.unit)}`;
}

function verdict(entry) {
  const a = entry.attempt;
  if (entry.isBaseline) return { text: 'Today', cls: '' };
  if (a.duplicate) return { text: 'Duplicate', cls: 'fail' };
  if (!a.metrics) return { text: 'No design', cls: 'fail' };
  if (!a.accepted) return { text: 'Rejected', cls: 'fail' };
  return a.improved ? { text: 'New best', cls: 'pass' } : { text: 'Not better', cls: '' };
}

function deltaChip(d) {
  if (d.kind === 'layout' || num(d.pct) === null) return el('span', { class: 'chip layout' }, d.text.replace('Moved or resized', 'Moved'));
  return el('span', { class: 'chip' }, `${SHORT[d.label] || d.label} ${short(d.before)}→${short(d.after)}${d.unit ? ' ' + d.unit : ''} `,
    d.pct > 0 ? el('span', { class: 'up' }, '▲') : el('span', { class: 'down' }, '▼'));
}

function outcome(entry) {
  const a = entry.attempt, m = a.metrics, o = state.objective;
  if (entry.isBaseline) return el('div', { class: 'outcome' }, el('span', {}, o.short), el('b', {}, fmtMetric(o.metric, m && m[o.metric])));
  if (!m) return el('div', { class: 'outcome' }, el('span', {}, a.duplicate ? 'Repeats an earlier design' : 'No design produced'),
    el('b', { class: 'bad' }, '✕'));
  const failed = failedChecks(a);
  if (failed.length) return el('div', { class: 'outcome' }, el('b', { class: 'bad' }, '✕'), el('span', { class: 'why' }, checkText(failed[0])));
  const change = pct(m[o.metric], state.baseline && state.baseline[o.metric]);
  const good = better(o.metric, change);
  return el('div', { class: 'outcome' }, el('span', {}, `${o.short} vs today`),
    el('b', { class: good === null ? '' : good ? 'good' : 'bad' }, change === null ? '—' : signed(change)));
}

function addTile(entry, index) {
  state.entries.set(entry.id, entry);
  const a = entry.attempt;
  const v = verdict(entry);
  const item = el('li', { class: `tile${v.cls === 'fail' ? ' rejected' : ''}`, 'data-id': entry.id, style: `animation-delay:${Math.min(index, 4) * 60}ms` },
    el('button', { type: 'button', 'aria-label': `${entry.label}: show details` },
      el('div', { class: 'tile-top' }, el('span', { class: 'tile-num' }, entry.label), el('span', { class: `pill ${v.cls}` }, v.text)),
      el('div', { class: 'tile-title', title: a.name || '' }, entry.isBaseline ? 'Trainium2-like chip' : (a.name || 'Untitled design')),
      el('div', { class: 'diagram-wrap' }, a.design ? drawDiagram(a.design, a.changed_ids, a.previous_boxes) : el('div', { class: 'skeleton', style: 'animation:none' })),
      el('div', { class: 'chips' }, entry.isBaseline ? el('span', { class: 'chip' }, 'Starting point') : (a.deltas || []).slice(0, 2).map(deltaChip)),
      outcome(entry)));
  item.querySelector('button').addEventListener('click', () => select(entry.id));
  $('iterations').insertBefore(item, $('iterations').querySelector('.tile.working'));
}

function setWorkingTile(campaign) {
  let tile = $('iterations').querySelector('.tile.working');
  const next = (campaign.attempts || []).length + 1;
  const running = campaign.status === 'running' && campaign.baseline_metrics && next <= campaign.max_rounds;
  if (!running) { if (tile) tile.remove(); return; }
  if (!tile) {
    tile = el('li', { class: 'tile working' }, el('button', { type: 'button', disabled: 'true', tabindex: '-1' },
      el('div', { class: 'tile-top' }, el('span', { class: 'tile-num' }), el('span', { class: 'pill' }, 'Designing')),
      el('div', { class: 'tile-title' }, 'The model is working…'), el('div', { class: 'skeleton' }), el('div', { class: 'working-step' })));
    $('iterations').append(tile);
  }
  tile.querySelector('.tile-num').textContent = `Iteration ${next}`;
  tile.querySelector('.working-step').textContent = (campaign.activity || '').replace(/^Round \d+: /, '');
}

function select(id) {
  const same = state.selected === id;
  transition(() => {
    state.selected = same ? null : id;
    for (const tile of document.querySelectorAll('.tile[data-id]')) tile.setAttribute('aria-current', String(tile.dataset.id === state.selected));
    if (same) { $('detail').hidden = true; return; }
    $('detail').replaceChildren(detailCard(state.entries.get(id)));
    $('detail').hidden = false;
  });
  if (!same) setTimeout(() => $('detail').scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 60);
}

function deltaRows(deltas) {
  if (!deltas || !deltas.length) return el('p', { class: 'why-text' }, 'No parameter changes.');
  return el('div', { class: 'delta-rows' }, deltas.map((d) => el('div', { class: 'delta-row' },
    el('span', { class: 'name' }, d.kind === 'layout' && num(d.pct) === null ? d.text : d.label),
    el('span', { class: 'vals' }, d.kind === 'layout' && num(d.pct) === null ? ''
      : `${short(d.before)} → ${short(d.after)} ${d.unit || ''}${num(d.pct) === null ? '' : '  ' + signed(d.pct, 0)}`),
    num(d.pct) === null ? null : el('span', { class: 'bar' }, el('i', { style: `width:${Math.min(100, Math.max(4, Math.abs(d.pct)))}%` })))));
}

function checkList(checks, limitsOnly = false) {
  return el('ul', { class: 'checks' }, (checks || []).map((row) => el('li', { class: row.passed || limitsOnly ? '' : 'fail' },
    el('span', { class: 'icon', style: limitsOnly ? 'background:var(--faint)' : '' }, limitsOnly ? '·' : row.passed ? '✓' : '✕'),
    el('span', {}, row.key === 'topology_valid' ? 'Valid layout & wiring' : `${row.label} ${row.op} ${withUnit(row.limit, row.unit)}`),
    el('span', { class: 'val' }, row.key === 'topology_valid' ? 'invalid' : limitsOnly ? `today ${withUnit(row.value, row.unit)}` : withUnit(row.value, row.unit)))));
}

function metricsBlock(m, isBaseline) {
  if (m.topology_valid === false) return el('p', { class: 'why-text' }, 'Not simulated: the layout or wiring was invalid.');
  const metrics = [...new Set([state.objective.metric, 'ttft_p99_ms', 'throughput_tokens_s'])];
  const names = { ttft_p99_ms: 'p99 TTFT', throughput_tokens_s: 'Throughput', area_mm2: 'Die area' };
  return el('dl', { class: 'metric-row' }, metrics.map((key) => {
    const change = isBaseline ? null : pct(m[key], state.baseline && state.baseline[key]);
    const good = better(key, change);
    return el('div', { class: 'metric' }, el('dt', {}, names[key]),
      el('dd', {}, fmtMetric(key, m[key]), change === null || good === null ? null : el('small', { class: good ? 'good' : 'bad' }, signed(change))));
  }));
}

function detailCard(entry) {
  const a = entry.attempt;
  const v = verdict(entry);
  const why = entry.isBaseline
    ? 'The reference chip: 4 compute clusters, 2 HBM stacks with 96 GB at 2.9 TB/s, 24 GB SRAM. Each iteration edits the best chip found so far.'
    : (a.rationale || (a.changes || [])[0] || a.feedback || '');
  const story = el('div', { class: 'detail-col' },
    el('div', { class: 'detail-head' }, el('span', { class: 'tile-num' }, entry.label), el('span', { class: `pill ${v.cls}` }, v.text)),
    el('h3', {}, entry.isBaseline ? "Today's chip" : (a.name || 'Untitled design')),
    why ? el('section', {}, el('h4', {}, 'Why'), el('p', { class: 'why-text' }, why)) : null,
    entry.isBaseline ? null : el('section', {}, el('h4', {}, 'What changed'), deltaRows(a.deltas)),
    a.steps && a.steps.length ? el('section', {}, el('h4', {}, 'Model calls this round'), el('div', { class: 'steps' }, a.steps.map((s) =>
      el('span', { class: `chip ${s.includes('fallback') ? 'fallback' : s.includes('repaired') || s.includes('widened') ? 'repaired' : ''}` },
        s.replace(/ \(automatic fallback.*/, ' · fallback'))))) : null);
  const result = el('div', { class: 'detail-col' },
    a.metrics ? el('section', {}, el('h4', {}, 'Simulated result'), metricsBlock(a.metrics, entry.isBaseline)) : null,
    a.checks && a.checks.length ? el('section', {}, el('h4', {}, 'Checker'), checkList(a.checks)) : null,
    a.feedback && !entry.isBaseline ? el('section', {}, el('h4', {}, 'Feedback sent to the model'), el('pre', { class: 'feedback' }, a.feedback)) : null);
  return el('article', { class: 'card detail-card' }, story, result);
}

function renderChart(campaign) {
  const base = campaign.baseline_metrics;
  const o = state.objective, key = o.metric, min = o.direction === 'min';
  const points = [{ x: 0, y: base[key], kind: campaign.baseline_feasible === false ? 'infeasible' : 'base' }];
  for (const a of campaign.attempts || []) {
    if (!a.metrics || a.metrics.topology_valid === false || num(a.metrics[key]) === null) continue;
    points.push({ x: a.round, y: a.metrics[key], kind: a.accepted ? (a.improved ? 'best' : 'ok') : 'bad', id: a.id });
  }
  const card = $('chart');
  if (points.length < 2) { card.hidden = true; return; }
  const cacheKey = JSON.stringify([points, campaign.status]);
  if (card.dataset.key === cacheKey) return;
  card.dataset.key = cacheKey;
  card.hidden = false;
  const W = Math.max(320, card.clientWidth - 32), H = 140, L = 64, R = 16, T = 10, B = 22;
  const maxX = Math.max(campaign.max_rounds, ...points.map((p) => p.x));
  const ys = points.map((p) => p.y);
  const lo = Math.min(...ys) * 0.94, hi = Math.max(...ys) * 1.04;
  const sx = (x) => L + (x / maxX) * (W - L - R);
  const sy = (y) => T + (1 - (y - lo) / Math.max(hi - lo, 1e-6)) * (H - T - B);
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}` });
  const label = { 'font-size': 10, fill: '#a1a1aa', 'font-family': 'Geist Mono, monospace' };
  for (const tick of [lo, (lo + hi) / 2, hi]) {
    root.append(svg('line', { x1: L, x2: W - R, y1: sy(tick), y2: sy(tick), stroke: '#f4f4f5' }));
    root.append(svg('text', { ...label, x: L - 8, y: sy(tick) + 4, 'text-anchor': 'end' }, fmtMetric(key, tick)));
  }
  for (let i = 0; i <= maxX; i++) root.append(svg('text', { ...label, x: sx(i), y: H - 4, 'text-anchor': 'middle' }, i === 0 ? 'today' : `#${i}`));
  root.append(svg('line', { x1: L, x2: W - R, y1: sy(base[key]), y2: sy(base[key]), stroke: '#d4d4d8', 'stroke-dasharray': '3 4' }));
  let best = points[0].kind === 'base' ? base[key] : null;
  let d = best === null ? '' : `M ${sx(0)} ${sy(best)}`;
  for (const p of points.slice(1)) {
    if (p.kind === 'best' && (best === null || (min ? p.y < best : p.y > best))) {
      d += best === null ? `M ${sx(p.x)} ${sy(p.y)}` : ` H ${sx(p.x)} V ${sy(p.y)}`;
      best = p.y;
    } else if (best !== null) d += ` H ${sx(p.x)}`;
  }
  if (d) root.append(svg('path', { d, class: 'draw', pathLength: 1, fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2, 'stroke-linejoin': 'round' }));
  const winnerId = campaign.status === 'completed' && campaign.best_attempt ? campaign.best_attempt.id : null;
  points.forEach((p, i) => {
    const color = p.kind === 'bad' ? 'var(--bad)' : p.kind === 'best' ? 'var(--accent)' : p.kind === 'base' ? '#71717a' : '#a1a1aa';
    root.append(svg('circle', { class: 'pt', cx: sx(p.x), cy: sy(p.y), r: p.id && p.id === winnerId ? 6 : 4,
      fill: p.kind === 'bad' || p.kind === 'infeasible' ? '#fff' : color, stroke: color, 'stroke-width': 1.6, style: `animation-delay:${200 + i * 80}ms` }));
  });
  const change = best === null ? null : pct(best, base[key]);
  card.replaceChildren(el('div', { class: 'chart-title' },
    el('span', {}, `${o.short} by iteration · ${min ? 'lower' : 'higher'} is better`),
    el('span', {}, 'Best so far ', el('b', {}, best === null ? 'none yet' : fmtMetric(key, best)), change !== null && Math.abs(change) > 0.05 ? ` (${signed(change)})` : '')), root);
}

// ------------------------------------------------------------- 04 solution

function renderSolution(campaign) {
  const best = campaign.best_attempt;
  const base = campaign.baseline_metrics;
  const o = state.objective;
  const body = $('solution-body');
  if (!best || !best.metrics) {
    $('solution-title').textContent = "No design beat today's chip";
    $('solution-sub').textContent = campaign.baseline_feasible === false
      ? 'None of the proposals passed every check for this budget. Open the rejected designs above to see which limit each one hit.'
      : `Every proposal either failed a check or did not improve ${o.label} by at least 1%.`;
    body.replaceChildren(el('div', { class: 'card compare none' }, el('p', { class: 'why-text' }, `Stopped: ${campaign.stop_reason || 'done'}.`)));
    return;
  }
  const m = best.metrics;
  const change = pct(m[o.metric], base[o.metric]);
  $('solution-title').textContent = campaign.baseline_feasible === false
    ? `Iteration ${best.round} fits the budget today's chip breaks`
    : `Iteration ${best.round}: ${o.short} ${change === null ? 'improved' : signed(change)}`;
  $('solution-sub').textContent = best.rationale || '';
  const metric = (key, label, primary) => {
    const ch = pct(m[key], base[key]);
    const good = better(key, ch);
    const value = el('span', {}, fmtMetric(key, base[key]));
    setTimeout(() => { value._value = base[key]; tween(value, m[key], (x) => fmtMetric(key, x), 1100); }, 300);
    return el('div', { class: `delta-big${primary ? ' primary' : ''}` }, el('div', { class: 'k' }, label),
      el('div', { class: 'v' }, value, ch === null || good === null ? null : el('span', { class: `pct ${good ? 'good' : 'bad'}` }, signed(ch))),
      el('div', { class: 'from' }, `was ${fmtMetric(key, base[key])}`));
  };
  const keys = [[o.metric, o.short, true], ['ttft_p99_ms', 'p99 TTFT'], ['throughput_tokens_s', 'Throughput'], ['power_w', 'Power'], ['area_mm2', 'Die area']]
    .filter(([key], i, all) => all.findIndex(([k]) => k === key) === i);
  const passed = (best.checks || []).filter((r) => r.passed).length;
  body.replaceChildren(el('div', { class: 'card compare' },
    el('div', { class: 'versus' },
      el('figure', {}, el('figcaption', {}, el('b', {}, "Today's chip"), el('span', {}, fmtMetric(o.metric, base[o.metric]))),
        el('div', { class: 'diagram-wrap' }, drawDiagram(campaign.baseline_design))),
      el('div', { class: 'arrow', 'aria-hidden': 'true' }, '→'),
      el('figure', {}, el('figcaption', {}, el('b', {}, `Proposed · iteration ${best.round}`), el('span', {}, fmtMetric(o.metric, m[o.metric]))),
        el('div', { class: 'diagram-wrap' }, drawDiagram(best.design, changedVs(campaign.baseline_design, best.design))))),
    el('div', { class: 'deltas-big' }, keys.map(([key, label, primary]) => metric(key, label, primary))),
    el('div', { class: 'solution-text' },
      el('section', {}, el('h4', {}, 'What changed vs today'), deltaRows(best.baseline_deltas)),
      el('section', {}, el('h4', {}, 'Checker verdict'), checkList(best.checks))),
    el('div', { class: 'solution-foot' }, el('span', {}, `Passed ${passed} of ${(best.checks || []).length} checks · ${(campaign.attempts || []).length} iterations · ${(campaign.calls || []).length} model calls`),
      el('a', { class: 'link', href: `/api/reports/${encodeURIComponent(campaign.id)}`, download: 'chip-design-report.md' }, 'Download report'))));
}

// ------------------------------------------------------------------ flow

function setRail(step) {
  const order = ['problem', 'measure', 'explore', 'solution'];
  const index = order.indexOf(step);
  for (const li of $('rail').children) {
    const i = order.indexOf(li.dataset.step);
    li.classList.toggle('done', i < index || (step === 'solution' && i === index));
    li.classList.toggle('active', i === index && step !== 'solution');
  }
}

function setStatus(text, kind = '') {
  $('status').hidden = !text;
  $('status').textContent = text || '';
  $('status').className = `status ${kind}`;
}

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { 'Content-Type': 'application/json', ...(options.headers || {}) } });
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (_) { /* keep statusText */ }
    throw new Error(detail);
  }
  return response.json();
}

function reset() {
  clearTimeout(state.timer);
  Object.assign(state, { rendered: new Set(), entries: new Map(), selected: null, baseline: null, checkerShown: false, findingShown: false });
  for (const id of ['lanes', 'live-metrics', 'iterations', 'judge', 'checker-grid', 'solution-body']) $(id).replaceChildren();
  delete $('chart').dataset.key;
  for (const id of ['measure', 'explore', 'solution', 'finding', 'judging', 'detail', 'chart']) $(id).hidden = true;
}

function watch(id) {
  state.campaignId = id;
  history.replaceState(null, '', `?run=${encodeURIComponent(id)}`);
  poll();
}

async function start(event) {
  event.preventDefault();
  reset();
  lockProblem(true);
  setStatus('Checking that the model is being served…');
  try {
    const campaign = await api('/api/campaigns', { method: 'POST',
      body: JSON.stringify({ scenario: state.scenario, goal: $('goal').value.trim(), max_rounds: MAX_ROUNDS }) });
    setStatus('');
    watch(campaign.id);
  } catch (error) {
    setStatus(`Could not start: ${error.message}`, 'error');
    lockProblem(false);
  }
}

function baselineEntry(campaign) {
  return { id: 'baseline', isBaseline: true, label: 'Today', attempt: {
    name: "Today's chip", design: campaign.baseline_design, metrics: campaign.baseline_metrics, checks: campaign.baseline_checks } };
}

async function poll() {
  let campaign;
  try {
    campaign = await api(`/api/campaigns/${encodeURIComponent(state.campaignId)}?compact=1`);
  } catch (error) {
    if (/not found/i.test(error.message)) {
      history.replaceState(null, '', location.pathname);
      setStatus('That run is no longer on the server. Pick a problem and start a new one.', 'error');
      lockProblem(false);
      return;
    }
    setStatus(`Lost connection, retrying (${error.message})`, 'error');
    state.timer = setTimeout(poll, POLL_SLOW * 2);
    return;
  }
  state.objective = campaign.objective || DEFAULT_OBJECTIVE;
  if (campaign.scenario && state.scenario !== campaign.scenario.id) pickScenario(campaign.scenario.id);
  if (campaign.goal) $('goal').value = campaign.goal;

  // 02 Measure
  const firstShow = $('measure').hidden;
  $('measure').hidden = false;
  if (firstShow) setTimeout(() => $('measure').scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
  renderLanes(campaign.live || { requests: [] });
  renderLiveMetrics(campaign);
  const phase = (campaign.live || {}).phase;
  $('measure-sub').textContent = campaign.baseline_metrics ? 'Measured. These numbers calibrate the simulator every design is checked in.'
    : phase === 'trace' ? 'Requests done. Reading the Neuron system trace and calibrating the simulator…'
    : 'Streaming requests to the live model while vLLM, neuron-monitor and the Neuron trace record what the hardware does.';

  if (campaign.baseline_metrics) {
    if (!state.baseline) {
      state.baseline = campaign.baseline_metrics;
      renderFinding(campaign);
      renderChecker(campaign);
      $('explore').hidden = false;
      renderJudge(campaign);
      addTile(baselineEntry(campaign), 0);
    }
    let index = state.rendered.size + 1;
    for (const attempt of campaign.attempts || []) {
      if (state.rendered.has(attempt.id)) continue;
      state.rendered.add(attempt.id);
      addTile({ id: attempt.id, label: `Iteration ${attempt.round}`, attempt }, index++);
    }
    setWorkingTile(campaign);
    renderCalls(campaign);
    renderChart(campaign);
  }

  if (campaign.status === 'completed') return finish(campaign);
  if (campaign.status === 'failed') {
    setStatus(campaign.error || 'The design run failed.', 'error');
    $('measure-sub').textContent = campaign.error || 'The run failed.';
    setRail(campaign.baseline_metrics ? 'explore' : 'measure');
    lockProblem(false);
    return;
  }
  setRail(campaign.baseline_metrics ? 'explore' : 'measure');
  $('model-badge').classList.toggle('busy', !campaign.baseline_metrics || (campaign.calls || []).some((c) => c.status === 'running'));
  state.timer = setTimeout(poll, campaign.baseline_metrics ? POLL_SLOW : POLL_FAST);
}

function finish(campaign) {
  lockProblem(false);
  setRail('solution');
  $('model-badge').classList.remove('busy');
  const best = campaign.best_attempt;
  const tile = document.querySelector(`.tile[data-id="${CSS.escape(best ? best.id : 'baseline')}"]`);
  if (tile && best) {
    tile.classList.add('winner');
    const pill = tile.querySelector('.pill');
    pill.className = 'pill best';
    pill.textContent = 'Winner';
  }
  renderSolution(campaign);
  transition(() => { $('solution').hidden = false; });
  setTimeout(() => $('solution').scrollIntoView({ behavior: 'smooth', block: 'start' }), 120);
}

async function loadConfig() {
  const badge = $('model-badge');
  try {
    const config = await api('/api/config');
    state.model = String(config.model || 'model').replace(/^openai\//, '');
    $('lanes-model').textContent = state.model;
    if (config.mode === 'mock') $('model-label').textContent = `${state.model} · offline mode`;
    else if (config.ready) { badge.classList.add('live'); $('model-label').textContent = state.model; }
    else {
      badge.classList.add('down');
      $('model-label').textContent = `${state.model} · not reachable`;
      setStatus(config.error, 'error');
    }
  } catch (_) { /* badge keeps its default */ }
}

async function init() {
  $('brief-form').addEventListener('submit', start);
  setRail('problem');
  loadConfig();
  try {
    state.scenarios = await api('/api/scenarios');
  } catch (_) {
    state.scenarios = [];
  }
  renderScenarios();
  pickScenario((state.scenarios[0] || {}).id || 'chat');
  const resumeId = new URLSearchParams(location.search).get('run');
  if (resumeId) {
    reset();
    lockProblem(true);
    watch(resumeId);
  }
}

init();
