"""Staged chip designer that fits a small-context local model.

One design round is split into short, focused model calls instead of asking
for the whole component graph at once:

1. plan        - pick architecture totals and explain the intended changes
2. floorplan   - assign each block to a die region (code computes exact boxes)
3. interconnect - wire the placed blocks

Code, not the model, derives per-block resources from the plan totals, so the
graph always matches its scalar fields. Each stage is validated on its own and
gets one targeted repair call with the exact errors; if that still fails, a
deterministic layout or wiring keeps the round usable and the step log says so.
"""

from __future__ import annotations

import inspect
import json
import math
import time
from typing import Any, Callable

from .llm_client import ProposalError
from .models import Component, Connection, HardwareDesign

# (system prompt, user prompt, max output tokens) -> parsed JSON object
ChatFn = Callable[[str, str, int], dict]

SYSTEM = (
    "You are a senior AI-accelerator architect working with a chip simulator. "
    "Reply with exactly one JSON object and nothing else: no prose, no markdown."
)

# Bounded search space. The checker still owns feasibility; these bounds only
# keep a single proposal from wandering into meaningless territory.
BOUNDS: dict[str, tuple[float, float, type]] = {
    "compute_clusters": (1, 8, int),
    "compute_cores": (2, 16, int),
    "peak_compute_tflops": (1000.0, 5000.0, float),
    "hbm_stacks": (1, 6, int),
    "hbm_gb": (32.0, 192.0, float),
    "hbm_bandwidth_tb_s": (1.0, 8.0, float),
    "sbuf_gb": (8.0, 96.0, float),
    "dma_engines": (1, 16, int),
    "utilization": (0.4, 0.95, float),
    "power_limit_w": (300.0, 1000.0, float),
    "cooling_capacity_w": (300.0, 1000.0, float),
}
SCALARS = [key for key in BOUNDS if key not in {"compute_clusters", "hbm_stacks"}]
MAX_DMA_BLOCKS = 4
REGIONS = ("left", "right", "top", "bottom", "center")
NOC_AREA_MM2 = 120.0
MIN_GAP = 0.02


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: float, digits: int = 4) -> float:
    return float(round(value, digits))


class ChipDesignAgent:
    def __init__(self, chat: ChatFn, report: Callable[[str], None] | None = None,
                 max_tokens: int = 3000, log_call: Callable[[dict], None] | None = None) -> None:
        self.chat = chat
        self.report = report or (lambda _text: None)
        self.log_call = log_call or (lambda _entry: None)
        self.max_tokens = max_tokens
        self.steps: list[str] = []
        self.round = 0
        try:
            self._streams = "progress" in inspect.signature(chat).parameters
        except (TypeError, ValueError):
            self._streams = False

    def _ask(self, stage: str, prompt: str) -> dict:
        """One model call, logged (and updated live while it streams) for the progress view."""
        entry = {"round": self.round, "stage": stage, "status": "running", "tokens": 0,
                 "started": time.time(), "ms": None}
        self.log_call(entry)
        started = time.monotonic()
        try:
            if self._streams:
                result = self.chat(SYSTEM, prompt, self.max_tokens, progress=lambda n: entry.__setitem__("tokens", n))
            else:
                result = self.chat(SYSTEM, prompt, self.max_tokens)
            entry["status"] = "done"
            return result
        except Exception:
            entry["status"] = "error"
            raise
        finally:
            entry["ms"] = round((time.monotonic() - started) * 1000.0)

    # ------------------------------------------------------------------ public

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        start = dict(context.get("baseline_config") or HardwareDesign.baseline().model_dump(mode="json"))
        round_no = int(context.get("round", 0)) + 1
        self.round = round_no

        self.report(f"Round {round_no}: planning the architecture")
        plan = self._plan(context, start)
        totals = plan["targets"]

        self.report(f"Round {round_no}: placing blocks on the die")
        blocks = self._blocks(totals)
        placements = self._floorplan(blocks, start, plan["rationale"])

        self.report(f"Round {round_no}: wiring the interconnect")
        links = self._interconnect(blocks, placements, totals, start)

        design = self._assemble(start, totals, blocks, placements, links)
        return {
            "name": plan["name"],
            "rationale": plan["rationale"],
            "changes": plan["changes"],
            "design": design,
            "steps": self.steps,
            "source": "model",
        }

    # -------------------------------------------------------------- stage 1

    def _plan(self, context: dict[str, Any], start: dict[str, Any]) -> dict[str, Any]:
        current = self._current_totals(start)
        prompt = self._plan_prompt(context, start, current)
        errors: list[str] = []
        for attempt in range(2):
            text = prompt if not errors else (
                prompt + "\n\nYour previous reply was rejected: " + "; ".join(errors)
                + ". Return the corrected JSON object.")
            try:
                raw = self._ask("plan", text)
                plan = self._parse_plan(raw, current)
                self.steps.append("plan" if attempt == 0 else "plan (repaired)")
                return plan
            except ProposalError as exc:
                errors = [str(exc)]
        raise ProposalError(f"model could not produce a valid plan: {errors[0]}")

    def _plan_prompt(self, context: dict[str, Any], start: dict[str, Any], current: dict[str, Any]) -> str:
        constraints = context.get("constraints") or {}
        baseline = context.get("baseline_metrics") or {}
        best = context.get("best_metrics") or baseline
        caps = []
        if "max_power_w" in constraints:
            caps.append(f"power <= {constraints['max_power_w']:g} W")
        if "max_temperature_c" in constraints:
            caps.append(f"peak temperature <= {constraints['max_temperature_c']:g} C")
        if "max_area_mm2" in constraints:
            caps.append(f"area <= {constraints['max_area_mm2']:g} mm2")
        if "max_p99_ttft_ms" in constraints:
            caps.append(f"p99 TTFT <= {constraints['max_p99_ttft_ms']:.4g} ms (no worse than today)")
        objective = context.get("objective") or {"label": "p99 time-to-first-token (TTFT)", "direction": "min"}
        floor = float(constraints.get("throughput_floor", 1.0))
        throughput_floor = _num(baseline.get("throughput_tokens_s"))
        workload = context.get("workload") or {}
        shape = (f" ({workload.get('prompt_tokens')} prompt and {workload.get('generation_tokens')} output tokens per request, "
                 f"bursts at {float(workload.get('load', 0.8)):.0%} of today's capacity)") if workload.get("prompt_tokens") else ""
        history = context.get("history") or []
        history_lines = "\n".join(
            f"- round {item.get('round')}: {item.get('name')} | changes: {'; '.join(item.get('changes') or []) or 'n/a'}"
            f" | {item.get('outcome')}" for item in history[-4:]) or "- none yet"
        bounds = ", ".join(f"{key} {lo:g}-{hi:g}" for key, (lo, hi, _kind) in BOUNDS.items())
        signals = "\n".join(f"- {line}" for line in context.get("signals") or []) or "- no live measurements available"
        strategy = self._strategy(objective, current, best, constraints)
        return f"""Design brief: {context.get('goal') or 'Minimize p99 time-to-first-token without reducing throughput.'}

Workload: a {start.get('model_size_b', 13):g}B-parameter LLM with {start.get('weight_bytes_per_parameter', 2):g}-byte weights, served under a bursty request trace{shape}.
Score: {'lower' if objective['direction'] == 'min' else 'higher'} simulated {objective['label']} is better.
Hard checks: throughput >= {self._fmt(None if throughput_floor is None else throughput_floor * floor)} tokens/s{'' if floor == 1 else f' ({floor:.0%} of today)'}, weights + KV cache fit in HBM{', ' + ', '.join(caps) if caps else ''}.

How the simulator behaves:
- prefill and decode time = max(compute time, HBM traffic time); compute time scales with 1 / (TFLOPS x utilization)
- usable HBM bandwidth is capped by the max flow from HBM stacks to compute clusters through the links
- power ~= 85 + 420 x utilization x min(TFLOPS / 2500, 1.8) W; above min(power_limit_w, cooling_capacity_w) the chip throttles
- area ~= 42 mm2 per core + 1.45 mm2 per GB HBM + 0.65 mm2 per GB SRAM + 3.5 mm2 per DMA engine + 120 mm2 NoC
- compute clusters placed close together or at the die centre run hotter

Measured on the live Trainium chip and from the simulator:
{signals}

What moves the score:
{strategy}

Current best design totals: {json.dumps(current)}
Its simulated result: {self._metrics_line(best)}
Checker feedback on the last proposal (it may have been rejected; you are editing the current best design above, so read its numbers as absolute targets): {context.get('feedback') or 'none'}
Earlier rounds:
{history_lines}

Allowed ranges: {bounds}. compute_clusters must not exceed compute_cores.
Change one to three things that address the biggest measured bottleneck, and say which measurement motivated it. Do not repeat a rejected idea.

Return:
{{"name": "<=6 word statement of the change, e.g. Widen HBM path", "rationale": "one sentence: the bottleneck you saw and why this fixes it", "changes": ["up to 4 short concrete changes"], "targets": {{{', '.join(f'"{key}": <number>' for key in BOUNDS)}}}}}"""

    def _parse_plan(self, raw: Any, current: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ProposalError("reply must be a JSON object")
        targets = raw.get("targets")
        if not isinstance(targets, dict):
            raise ProposalError('missing "targets" object')
        resolved: dict[str, Any] = {}
        for key, (lo, hi, kind) in BOUNDS.items():
            value = _num(targets.get(key))
            if value is None:
                value = current[key]
            value = min(hi, max(lo, value))
            resolved[key] = int(round(value)) if kind is int else _round(value, 4)
        resolved["compute_clusters"] = min(resolved["compute_clusters"], resolved["compute_cores"])
        changes = raw.get("changes")
        if isinstance(changes, str):
            changes = [changes]
        changes = [str(item).strip() for item in changes or [] if str(item).strip()][:4]
        rationale = str(raw.get("rationale") or "").strip()
        name = str(raw.get("name") or "").strip()[:60] or "Untitled design"
        return {"name": name, "rationale": rationale, "changes": changes, "targets": resolved}

    @staticmethod
    def _strategy(objective: dict[str, Any], current: dict[str, Any], best: dict[str, Any],
                  constraints: dict[str, Any]) -> str:
        """Objective-specific guidance, so latency advice does not pull an area or throughput search off course."""
        metric = objective.get("metric", "ttft_p99_ms")
        decode = best.get("decode_bound")
        prefill = best.get("prefill_bound")
        if metric == "area_mm2":
            parts = {"compute cores": current["compute_cores"] * 42.0, "HBM": current["hbm_gb"] * 1.45,
                     "SRAM": current["sbuf_gb"] * 0.65, "DMA": current["dma_engines"] * 3.5, "NoC": 120.0}
            need = _num(best.get("peak_memory_gb"))
            lines = ["- area by part: " + ", ".join(f"{k} {v:.0f} mm2" for k, v in sorted(parts.items(), key=lambda kv: -kv[1])),
                     "- every mm2 removed is progress; adding anything makes the score worse"]
            if need:
                lines.append(f"- HBM must still hold weights + KV cache: hbm_gb >= {math.ceil(need)}")
            if decode == "memory":
                lines.append("- decode is memory-bound, so compute cores and TFLOPS are slack: cut them first, keep HBM bandwidth")
            if "max_p99_ttft_ms" in constraints:
                lines.append(f"- p99 TTFT may rise up to {constraints['max_p99_ttft_ms']:.4g} ms; spend that slack on removing area")
            return "\n".join(lines)
        if metric == "throughput_tokens_s":
            return ("- the chip is saturated, so throughput rises only if each request is served faster\n"
                    f"- decode is {decode or 'unknown'}-bound and prefill is {prefill or 'unknown'}-bound: grow the resource of the bound phase\n"
                    "- stay inside the power and area caps; extra resources on the non-bound side are wasted")
        return (f"- p99 TTFT is queueing plus prefill: decode is {decode or 'unknown'}-bound and prefill is {prefill or 'unknown'}-bound\n"
                "- speed up the bound phase first; resources on the non-bound side add area and power for nothing")

    # -------------------------------------------------------------- stage 2

    @staticmethod
    def _blocks(totals: dict[str, Any]) -> list[dict[str, Any]]:
        """List every block the plan implies, with its resources and area."""
        blocks: list[dict[str, Any]] = []
        clusters, cores = totals["compute_clusters"], totals["compute_cores"]
        for i in range(clusters):
            share = cores // clusters + (1 if i < cores % clusters else 0)
            blocks.append({"id": f"compute-{i}", "type": "compute_cluster", "label": f"Compute {i + 1}",
                           "core_count": share, "compute_tflops": totals["peak_compute_tflops"] / clusters,
                           "area_mm2": share * 42.0})
        stacks = totals["hbm_stacks"]
        for i in range(stacks):
            gb = totals["hbm_gb"] / stacks
            blocks.append({"id": f"hbm-{i}", "type": "hbm", "label": f"HBM {i + 1}", "capacity_gb": gb,
                           "bandwidth_tb_s": totals["hbm_bandwidth_tb_s"] / stacks, "area_mm2": gb * 1.45})
        blocks.append({"id": "sbuf-0", "type": "sbuf", "label": "SRAM", "capacity_gb": totals["sbuf_gb"],
                       "area_mm2": totals["sbuf_gb"] * 0.65})
        dma_blocks = min(totals["dma_engines"], MAX_DMA_BLOCKS)
        for i in range(dma_blocks):
            engines = totals["dma_engines"] // dma_blocks + (1 if i < totals["dma_engines"] % dma_blocks else 0)
            blocks.append({"id": f"dma-{i}", "type": "dma", "label": f"DMA {i + 1}", "dma_engines": engines,
                           "area_mm2": engines * 3.5})
        blocks.append({"id": "noc-0", "type": "noc", "label": "Interconnect", "area_mm2": NOC_AREA_MM2})
        return blocks

    def _floorplan(self, blocks: list[dict[str, Any]], start: dict[str, Any], rationale: str) -> dict[str, list[float]]:
        """The model assigns each block to a die region; code turns regions into exact boxes.

        Small models are unreliable at raw coordinates, but good at "HBM on the
        left and top edges, compute in the centre". Regions whose contents did not
        change keep their previous boxes, so only real layout changes show up.
        """
        placeable = [block for block in blocks if block["type"] != "noc"]
        previous = self._infer_regions(start)
        rows = [f"- {block['id']} ({block['type']}), currently {previous.get(block['id'], 'new')}" for block in placeable]
        prompt = f"""Assign each block of an accelerator die to a region: "left", "right", "top", "bottom" (the four die edges) or "center".
Code converts regions into exact, non-overlapping coordinates; blocks in one region share it evenly.

Physical rules:
- HBM stacks must sit on a die edge (their PHYs are at the edge), never in the center
- compute clusters go in the center; at least one block must be in the center
- spreading HBM stacks over more edges shortens the wires to compute; crowding one edge makes each stack thinner
- SRAM should border the compute it feeds; DMA engines sit on an edge near I/O

Blocks:
{chr(10).join(rows)}

Design intent: {rationale or 'n/a'}
Keep a block where it is unless moving it serves the intent.

Return: {{"regions": {{"<block id>": "left|right|top|bottom|center"}}}} listing every block above."""
        ids = {block["id"]: block["type"] for block in placeable}
        errors: list[str] = []
        for attempt in range(2):
            text = prompt if not errors else (
                prompt + "\n\nYour previous answer was rejected:\n- " + "\n- ".join(errors[:8])
                + "\nFix these problems and return every block again.")
            try:
                raw = self._ask("floorplan", text)
            except ProposalError as exc:
                errors = [str(exc)]
                continue
            regions, errors = self._check_regions(raw, ids)
            if not errors:
                self.steps.append("floorplan" if attempt == 0 else "floorplan (repaired)")
                return self._pack(placeable, regions, start, previous)
        self.steps.append("floorplan (automatic fallback: " + (errors[0] if errors else "no reply") + ")")
        regions = {block_id: previous.get(block_id) or self._default_region(kind) for block_id, kind in ids.items()}
        return self._pack(placeable, regions, start, previous)

    @staticmethod
    def _default_region(kind: str) -> str:
        return {"hbm": "left", "sbuf": "top", "dma": "bottom"}.get(kind, "center")

    @staticmethod
    def _infer_regions(design: dict[str, Any]) -> dict[str, str]:
        """Read the region of every block in an existing design from its centre point."""
        regions = {}
        for c in design.get("components") or []:
            if c.get("type") == "noc":
                continue
            cx = float(c.get("x", 0)) + float(c.get("width", 0)) / 2
            cy = float(c.get("y", 0)) + float(c.get("height", 0)) / 2
            regions[c.get("id")] = ("left" if cx < 0.2 else "right" if cx > 0.8 else
                                    "top" if cy < 0.17 else "bottom" if cy > 0.83 else "center")
        return regions

    @staticmethod
    def _check_regions(raw: Any, ids: dict[str, str]) -> tuple[dict[str, str], list[str]]:
        regions = raw.get("regions") if isinstance(raw, dict) else None
        if not isinstance(regions, dict):
            return {}, ['missing "regions" object']
        out: dict[str, str] = {}
        errors: list[str] = []
        for block_id, kind in ids.items():
            region = str(regions.get(block_id, "")).strip().lower()
            if region not in REGIONS:
                errors.append(f"{block_id} needs a region (one of {', '.join(REGIONS)})")
            elif kind == "hbm" and region == "center":
                errors.append(f"{block_id} is an HBM stack and must sit on a die edge")
            else:
                out[block_id] = region
        if out and "center" not in out.values():
            errors.append("at least one block (normally the compute clusters) must be in the center")
        return out, errors

    @classmethod
    def _pack(cls, placeable: list[dict[str, Any]], regions: dict[str, str], start: dict[str, Any],
              previous: dict[str, str]) -> dict[str, list[float]]:
        """Deterministic, overlap-free coordinates for a region assignment."""
        margin, side, edge, gap = 0.02, 0.16, 0.13, MIN_GAP
        used = set(regions.values())
        x0 = margin + side + gap if "left" in used else margin
        x1 = 1 - margin - side - gap if "right" in used else 1 - margin
        y0 = margin + edge + gap if "top" in used else margin
        y1 = 1 - margin - edge - gap if "bottom" in used else 1 - margin
        frames = {"left": (margin, y0, side, y1 - y0), "right": (1 - margin - side, y0, side, y1 - y0),
                  "top": (x0, margin, x1 - x0, edge), "bottom": (x0, 1 - margin - edge, x1 - x0, edge),
                  "center": (x0, y0, x1 - x0, y1 - y0)}
        order = {"compute_cluster": 0, "hbm": 1, "sbuf": 2, "dma": 3}
        old_boxes = {c.get("id"): [float(c.get(k, 0)) for k in ("x", "y", "width", "height")]
                     for c in start.get("components") or []}
        previous_used = set(previous.values())
        boxes: dict[str, list[float]] = {}
        for region, (fx, fy, fw, fh) in frames.items():
            members = sorted((b["id"] for b in placeable if regions[b["id"]] == region),
                             key=lambda i: (order.get(next(b["type"] for b in placeable if b["id"] == i), 9), i))
            if not members:
                continue
            same = sorted(members) == sorted(k for k, v in previous.items() if v == region)
            if same and previous_used == used and all(m in old_boxes for m in members):
                for m in members:
                    boxes[m] = [_round(v) for v in old_boxes[m]]
                continue
            n = len(members)
            if region == "center":
                cols = math.ceil(math.sqrt(n))
                rows = math.ceil(n / cols)
                cw, ch = (fw - gap * (cols - 1)) / cols, (fh - gap * (rows - 1)) / rows
                for i, m in enumerate(members):
                    r, c = divmod(i, cols)
                    boxes[m] = [_round(fx + c * (cw + gap)), _round(fy + r * (ch + gap)), _round(cw), _round(ch)]
            elif region in {"left", "right"}:
                h = (fh - gap * (n - 1)) / n
                for i, m in enumerate(members):
                    boxes[m] = [_round(fx), _round(fy + i * (h + gap)), _round(fw), _round(h)]
            else:
                w = (fw - gap * (n - 1)) / n
                for i, m in enumerate(members):
                    boxes[m] = [_round(fx + i * (w + gap)), _round(fy), _round(w), _round(fh)]
        boxes["noc-0"] = cls._noc_box(boxes)
        return boxes

    @staticmethod
    def _noc_box(placements: dict[str, list[float]]) -> list[float]:
        """Span the interconnect overlay across the compute region."""
        compute = [box for key, box in placements.items() if key.startswith("compute-")] or list(placements.values())
        x0 = max(0.0, min(b[0] for b in compute) - 0.02)
        y0 = max(0.0, min(b[1] for b in compute) - 0.02)
        x1 = min(1.0, max(b[0] + b[2] for b in compute) + 0.02)
        y1 = min(1.0, max(b[1] + b[3] for b in compute) + 0.02)
        return [_round(x0), _round(y0), _round(x1 - x0), _round(y1 - y0)]

    @classmethod
    def _fallback_layout(cls, blocks: list[dict[str, Any]]) -> dict[str, list[float]]:
        """Deterministic default layout: HBM split over the sides, compute grid in the centre."""
        placeable = [b for b in blocks if b["type"] != "noc"]
        hbm = [b["id"] for b in placeable if b["type"] == "hbm"]
        regions = {b["id"]: cls._default_region(b["type"]) for b in placeable}
        for i, block_id in enumerate(hbm):
            regions[block_id] = "left" if i < math.ceil(len(hbm) / 2) else "right"
        return cls._pack(placeable, regions, {}, {})

    # -------------------------------------------------------------- stage 3

    def _interconnect(self, blocks: list[dict[str, Any]], placements: dict[str, list[float]],
                      totals: dict[str, Any], start: dict[str, Any]) -> list[Connection]:
        bandwidth = totals["hbm_bandwidth_tb_s"]
        rows = []
        for block in blocks:
            x, y, w, h = placements[block["id"]]
            rows.append(f"- {block['id']} ({block['type']}) centre ({x + w / 2:.2f}, {y + h / 2:.2f})")
        previous = [[c.get("source"), c.get("target"), c.get("bandwidth_tb_s")] for c in start.get("connections") or []]
        prompt = f"""Wire an accelerator die. The blocks are already placed:
{chr(10).join(rows)}

HBM provides {bandwidth:g} TB/s in total across {totals['hbm_stacks']} stacks ({bandwidth / totals['hbm_stacks']:.3g} TB/s each); there are {totals['compute_clusters']} compute clusters.

Rules:
- every compute cluster must reach at least one HBM stack, directly or through noc-0
- connect sbuf-0 and every DMA block to noc-0 or to a compute cluster
- each link is [source id, target id, bandwidth in TB/s] with bandwidth between 0.05 and {2 * bandwidth:g}
- the links must be able to carry all {bandwidth:g} TB/s from the HBM stacks to the compute clusters (max flow), or the extra HBM bandwidth is wasted: give each HBM link its stack's bandwidth, and make the links into compute add up to at least {bandwidth:g} TB/s
- the simulator also scores average hop count and wire length: short direct HBM-to-compute links help

Previous links, for reference: {json.dumps(previous)}

Return: {{"links": [["hbm-0", "noc-0", 1.0], ...]}}"""
        ids = {block["id"]: block["type"] for block in blocks}
        errors: list[str] = []
        for attempt in range(2):
            text = prompt if not errors else (
                prompt + "\n\nYour previous wiring was rejected:\n- " + "\n- ".join(errors[:8])
                + "\nFix these problems and return all links again.")
            try:
                raw = self._ask("wiring", text)
            except ProposalError as exc:
                errors = [str(exc)]
                continue
            links, errors = self._check_links(raw, ids, bandwidth)
            if not errors:
                flow = self._flow(start, totals, blocks, placements, links)
                if flow < 0.9 * bandwidth:
                    errors = [f"the links carry only {flow:.2f} of the {bandwidth:g} TB/s the HBM stacks provide to compute; "
                              "widen the links on every HBM-to-compute path so each carries its share"]
                    if attempt == 1:
                        # Keep the model's topology but scale the links up to the HBM rate.
                        factor = min(4.0, bandwidth / max(flow, 1e-6))
                        links = [link.model_copy(update={"bandwidth_tb_s": _round(min(link.bandwidth_tb_s * factor, 2 * bandwidth))})
                                 for link in links]
                        self.steps.append("interconnect (links widened to the HBM rate)")
                        return links
                    continue
                self.steps.append("interconnect" if attempt == 0 else "interconnect (repaired)")
                return links
        self.steps.append("interconnect (automatic fallback: " + (errors[0] if errors else "no reply") + ")")
        return self._fallback_links(blocks, totals)

    @staticmethod
    def _check_links(raw: Any, ids: dict[str, str], bandwidth: float) -> tuple[list[Connection], list[str]]:
        items = raw.get("links") if isinstance(raw, dict) else None
        if not isinstance(items, list) or not items:
            return [], ['missing "links" list']
        merged: dict[tuple[str, str], float] = {}
        errors: list[str] = []
        for item in items:
            if isinstance(item, dict):
                item = [item.get("source"), item.get("target"), item.get("bandwidth_tb_s", item.get("bandwidth"))]
            if not isinstance(item, list) or len(item) != 3:
                errors.append(f"link {item!r} must be [source, target, bandwidth]")
                continue
            source, target, value = str(item[0]), str(item[1]), _num(item[2])
            if source not in ids or target not in ids:
                errors.append(f"link {source} -> {target} references an unknown block")
                continue
            if source == target:
                continue
            if value is None or value <= 0:
                errors.append(f"link {source} -> {target} needs a positive bandwidth")
                continue
            key = (source, target)
            merged[key] = merged.get(key, 0.0) + min(max(value, 0.05), 2 * bandwidth)
        adjacency: dict[str, set[str]] = {block_id: set() for block_id in ids}
        for source, target in merged:
            adjacency[source].add(target)
            adjacency[target].add(source)
        hbm = {block_id for block_id, kind in ids.items() if kind == "hbm"}
        for block_id, kind in ids.items():
            if kind == "compute_cluster":
                seen, todo = {block_id}, [block_id]
                while todo:
                    for nxt in adjacency[todo.pop()]:
                        if nxt not in seen:
                            seen.add(nxt)
                            todo.append(nxt)
                if not seen & hbm:
                    errors.append(f"{block_id} cannot reach any HBM stack")
            elif kind in {"sbuf", "dma", "hbm"} and not adjacency[block_id]:
                errors.append(f"{block_id} is not connected")
        links = [Connection(id=f"{source}--{target}", source=source, target=target,
                            bandwidth_tb_s=_round(value, 4), latency_ns=8 if "hbm" in source + target else 10)
                 for (source, target), value in merged.items()]
        return links, errors

    @staticmethod
    def _fallback_links(blocks: list[dict[str, Any]], totals: dict[str, Any]) -> list[Connection]:
        bandwidth = totals["hbm_bandwidth_tb_s"]
        links = []
        for block in blocks:
            kind, block_id = block["type"], block["id"]
            if kind == "noc":
                continue
            if kind == "hbm":
                links.append(Connection(id=f"{block_id}--noc-0", source=block_id, target="noc-0",
                                        bandwidth_tb_s=_round(block["bandwidth_tb_s"]), latency_ns=8))
            elif kind == "compute_cluster":
                links.append(Connection(id=f"noc-0--{block_id}", source="noc-0", target=block_id,
                                        bandwidth_tb_s=_round(bandwidth / totals["compute_clusters"]), latency_ns=10))
            elif kind == "sbuf":
                links.append(Connection(id="sbuf-0--noc-0", source="sbuf-0", target="noc-0",
                                        bandwidth_tb_s=_round(bandwidth), latency_ns=6))
            else:
                links.append(Connection(id=f"{block_id}--noc-0", source=block_id, target="noc-0",
                                        bandwidth_tb_s=_round(max(bandwidth / MAX_DMA_BLOCKS, 0.05)), latency_ns=8))
        return links

    # ------------------------------------------------------------- assembly

    @staticmethod
    def _build(start: dict[str, Any], totals: dict[str, Any], blocks: list[dict[str, Any]],
               boxes: dict[str, list[float]], edges: list[Connection]) -> HardwareDesign:
        components = []
        for block in blocks:
            x, y, w, h = boxes[block["id"]]
            components.append(Component(
                id=block["id"], type=block["type"], label=block["label"], x=x, y=y, width=w, height=h,
                layer=0 if block["type"] == "noc" else 1,
                core_count=block.get("core_count", 0), compute_tflops=_round(block.get("compute_tflops", 0)),
                capacity_gb=_round(block.get("capacity_gb", 0)), bandwidth_tb_s=_round(block.get("bandwidth_tb_s", 0)),
                dma_engines=block.get("dma_engines", 0), area_mm2=_round(block["area_mm2"], 2)))
        fields = {key: value for key, value in start.items() if key not in {"components", "connections", "name"}}
        fields.update({key: totals[key] for key in SCALARS})
        return HardwareDesign(**fields, components=components, connections=edges)

    def _flow(self, start: dict[str, Any], totals: dict[str, Any], blocks: list[dict[str, Any]],
              boxes: dict[str, list[float]], links: list[Connection]) -> float:
        """Max flow from HBM to compute through the proposed links, in TB/s."""
        from .simulator import DigitalTwinSimulator

        return DigitalTwinSimulator(self._build(start, totals, blocks, boxes, links))._topology_profile()["flow"]

    def _assemble(self, start: dict[str, Any], totals: dict[str, Any], blocks: list[dict[str, Any]],
                  placements: dict[str, list[float]], links: list[Connection]) -> dict[str, Any]:
        design = self._build(start, totals, blocks, placements, links)
        errors = design.validate_topology()
        if errors:
            # The stage checks should make this unreachable; stay safe anyway.
            self.steps.append("assembly fallback: " + errors[0])
            design = self._build(start, totals, blocks, self._fallback_layout(blocks), self._fallback_links(blocks, totals))
        return design.model_dump(mode="json")

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _current_totals(start: dict[str, Any]) -> dict[str, Any]:
        defaults = HardwareDesign.baseline().model_dump(mode="json")
        components = start.get("components") or []
        totals = {key: start.get(key, defaults.get(key)) for key in SCALARS}
        totals["compute_clusters"] = sum(1 for c in components if c.get("type") == "compute_cluster") or 4
        totals["hbm_stacks"] = sum(1 for c in components if c.get("type") == "hbm") or 2
        return {key: totals[key] for key in BOUNDS}

    @staticmethod
    def _fmt(value: Any) -> str:
        number = _num(value)
        return "n/a" if number is None else f"{number:.4g}"

    @classmethod
    def _metrics_line(cls, metrics: dict[str, Any]) -> str:
        return (f"p99 TTFT {cls._fmt(metrics.get('ttft_p99_ms'))} ms, throughput {cls._fmt(metrics.get('throughput_tokens_s'))} tokens/s, "
                f"power {cls._fmt(metrics.get('power_w'))} W, peak temperature {cls._fmt(metrics.get('max_temperature_c'))} C, "
                f"area {cls._fmt(metrics.get('area_mm2'))} mm2, throttling {bool(metrics.get('throttling'))}")
