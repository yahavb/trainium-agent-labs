"""Deterministic phase-aware roofline and queue replay simulator."""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Iterable, List, Sequence, Union

from .models import HardwareDesign, RequestOutcome, RequestSpec, RequestTrace, SimulationResult


def _make_trace(name: str, counts: Sequence[tuple[int, int, float]], description: str) -> RequestTrace:
    return RequestTrace(
        name=name,
        description=description,
        requests=[
            RequestSpec(request_id=f"{name.lower()}-{i + 1:03d}", prompt_tokens=p, output_tokens=o, arrival_ms=a)
            for i, (p, o, a) in enumerate(counts)
        ],
    )


# Fixed and reproducible request streams. Inter-arrival times are milliseconds.
TRACE_A = _make_trace("A", [(512, 128, i * 90.0) for i in range(12)], "Interactive chat, short context")
TRACE_B = _make_trace("B", [(4096, 256, i * 250.0) for i in range(8)], "Long-context generation")
TRACE_C = _make_trace("C", [(1024, 64, i * 8.0) for i in range(48)], "High-concurrency short answers")
TRACES: Dict[str, RequestTrace] = {"A": TRACE_A, "B": TRACE_B, "C": TRACE_C}


def fixed_trace(name: str) -> RequestTrace:
    """Return a fresh copy of fixed trace A, B, or C (case insensitive)."""
    key = str(name).strip().upper().removeprefix("TRACE_")
    if key not in TRACES:
        raise ValueError(f"Unknown request trace {name!r}; choose A, B, or C")
    return TRACES[key].model_copy(deep=True)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * percentile
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


class DigitalTwinSimulator:
    """Replay a fixed arrival stream through a deterministic FCFS device model."""

    def __init__(self, design: HardwareDesign | None = None):
        self.design = design or HardwareDesign.baseline()
        # Set from live measurements (see CampaignOrchestrator._calibrate): the scales map
        # roofline time onto measured Trainium prefill/decode time; spill_fraction is the share of
        # HBM traffic the system trace attributes to SRAM spills.
        self.calibration: Dict[str, float] = {"prefill_scale": 1.0, "decode_scale": 1.0, "spill_fraction": 0.05}

    def _memory_multiplier(self) -> float:
        """Spill traffic grows as on-chip SRAM shrinks relative to the 24 GB reference."""
        spill = float(self.calibration.get("spill_fraction", 0.05))
        return 1.0 + spill * min(4.0, max(0.25, 24.0 / max(self.design.sbuf_gb, 1e-3)))

    def _roofline(self, prompt_tokens: int | None) -> tuple[float, float]:
        """Return (compute_ms, memory_ms) for a prefill, or a decode step when prompt_tokens is None."""
        d = self.design
        topology = getattr(self, "_active_topology", None) or self._topology_profile()
        params = d.model_size_b * 1e9
        if prompt_tokens is None:
            flops = 2.0 * params
            bytes_moved = params * d.weight_bytes_per_parameter + 4096.0
        else:
            # Attention makes prefill work grow faster than a linear token pass.
            flops = 2.0 * params * prompt_tokens * (1.0 + min(prompt_tokens, 8192) / 4096.0)
            bytes_moved = params * d.weight_bytes_per_parameter + prompt_tokens * 4096.0
        compute_ms = flops / (topology["compute_tflops"] * 1e12 * d.utilization) * 1000.0
        memory_ms = bytes_moved * self._memory_multiplier() / (topology["hbm_bandwidth"] * 1e12) * 1000.0
        return compute_ms, memory_ms

    def _topology_profile(self):
        """Summarize resources and the HBM-to-compute max-flow in the graph."""
        d = self.design
        components = d.components or []
        edges = d.connections or []
        compute = [c for c in components if c.type == "compute_cluster"]
        hbm = [c for c in components if c.type == "hbm"]
        ids = [c.id for c in components]
        by_id = {c.id: c for c in components}
        source, sink = "__hbm_source__", "__compute_sink__"
        residual = {key: {} for key in ids + [source, sink]}
        wire_lengths = []
        for edge in edges:
            if edge.source in residual and edge.target in residual:
                # Connections model bidirectional physical links. Residual
                # capacities retain that while allowing flow rerouting.
                residual[edge.source][edge.target] = residual[edge.source].get(edge.target, 0) + edge.bandwidth_tb_s
                residual[edge.target][edge.source] = residual[edge.target].get(edge.source, 0) + edge.bandwidth_tb_s
                a, b = by_id[edge.source], by_id[edge.target]
                ax, ay = a.x + a.width / 2.0, a.y + a.height / 2.0
                bx, by = b.x + b.width / 2.0, b.y + b.height / 2.0
                wire_lengths.append(abs(ax - bx) + abs(ay - by))
        for comp in hbm:
            residual[source][comp.id] = comp.bandwidth_tb_s
        for comp in compute:
            residual[comp.id][sink] = max(d.hbm_bandwidth_tb_s, 0.001)
        flow = 0.0
        while True:
            parent = {source: None}
            queue = deque([source])
            while queue and sink not in parent:
                node = queue.popleft()
                for nxt, cap in residual[node].items():
                    if cap > 1e-9 and nxt not in parent:
                        parent[nxt] = node
                        queue.append(nxt)
            if sink not in parent:
                break
            amount, node = float("inf"), sink
            while parent[node] is not None:
                prev = parent[node]
                amount = min(amount, residual[prev][node]); node = prev
            node = sink
            while parent[node] is not None:
                prev = parent[node]
                residual[prev][node] -= amount
                residual[node][prev] = residual[node].get(prev, 0) + amount
                node = prev
            flow += amount
        # Mean shortest graph distance approximates interconnect hop cost.
        distances = []
        adjacency = {key: set() for key in ids}
        for edge in edges:
            if edge.source in adjacency and edge.target in adjacency:
                adjacency[edge.source].add(edge.target); adjacency[edge.target].add(edge.source)
        for src in hbm:
            dist = {src.id: 0}; queue = deque([src.id])
            while queue:
                node = queue.popleft()
                for nxt in adjacency[node]:
                    if nxt not in dist:
                        dist[nxt] = dist[node] + 1; queue.append(nxt)
            distances.extend(dist[c.id] for c in compute if c.id in dist)
        hops = sum(distances) / len(distances) if distances else 0.0
        return {
            "compute_tflops": sum(c.compute_tflops for c in compute),
            "hbm_gb": sum(c.capacity_gb for c in hbm),
            "hbm_bandwidth": min(sum(c.bandwidth_tb_s for c in hbm), flow),
            "sbuf_gb": sum(c.capacity_gb for c in components if c.type == "sbuf"),
            "dma_engines": sum(c.dma_engines for c in components if c.type == "dma"),
            "area_mm2": sum(c.area_mm2 for c in components),
            "hops": hops,
            "wire_length": sum(wire_lengths) / len(wire_lengths) if wire_lengths else 0.0,
            "flow": flow,
            "compute_link_bw": sum(e.bandwidth_tb_s for e in edges
                                   if by_id.get(e.source) and by_id.get(e.target)
                                   and "compute_cluster" in (by_id[e.source].type, by_id[e.target].type)),
        }

    def _collective_factor(self, topology: Dict[str, float]) -> float:
        """Collectives (measured share of device time) speed up with wider compute links."""
        share = float(self.calibration.get("collective_share", 0.0))
        base = float(self.calibration.get("baseline_compute_link_bw", 0.0))
        if share <= 0 or base <= 0:
            return 1.0
        return 1.0 - share + share * base / max(topology.get("compute_link_bw", base), 1e-6)

    def _prefill_ms(self, prompt_tokens: int) -> float:
        topology = getattr(self, "_active_topology", None) or self._topology_profile()
        # Bytes and bandwidth use decimal units so the stated 2.9 TB/s baseline is literal.
        compute_ms, memory_ms = self._roofline(prompt_tokens)
        scale = float(self.calibration.get("prefill_scale", 1.0))
        # Measured host-side overhead does not shrink on faster silicon; only device time scales.
        hardware = (max(compute_ms, memory_ms, 0.05) + topology["hops"] * 0.01 + topology["wire_length"] * 0.04) * scale
        return float(self.calibration.get("prefill_overhead_ms", 0.0)) + hardware * self._collective_factor(topology)

    def _decode_token_ms(self, concurrent: int = 1) -> float:
        topology = getattr(self, "_active_topology", None) or self._topology_profile()
        # Each step streams weights once and reads per-sequence KV state.
        compute_ms, memory_ms = self._roofline(None)
        scale = float(self.calibration.get("decode_scale", 1.0))
        hardware = (max(compute_ms, memory_ms, 0.01) + topology["hops"] * 0.01 + topology["wire_length"] * 0.04) * scale
        return float(self.calibration.get("decode_overhead_ms", 0.0)) + hardware * self._collective_factor(topology)

    def evaluate(
        self,
        design: HardwareDesign | None = None,
        trace: Union[RequestTrace, str, None] = None,
    ) -> SimulationResult:
        # Support both evaluate(design, trace) and evaluate(trace) on a simulator
        # that already owns its design.
        if trace is None and isinstance(design, (str, RequestTrace)):
            trace, design = design, None
        if trace is None:
            trace = "A"
        if design is not None:
            self.design = design if isinstance(design, HardwareDesign) else HardwareDesign.parse_obj(design)
        workload = fixed_trace(trace) if isinstance(trace, str) else trace
        if not workload.requests:
            return self._empty_result(workload.name)

        d = self.design
        topology_errors = d.validate_topology()
        topology = self._topology_profile()
        if topology_errors:
            return self._invalid_result(workload.name, topology_errors)
        # Use the topology-derived totals throughout the replay. A topology
        # edit therefore changes simulated performance and resource fit.
        self._active_topology = topology
        raw_power = 85.0 + d.utilization * min(d.peak_compute_tflops / 2500.0, 1.8) * 420.0
        power_cap = min(d.power_limit_w, d.cooling_capacity_w)
        throttling = raw_power > power_cap
        factor = min(1.0, power_cap / raw_power) if throttling else 1.0
        ordered = sorted(enumerate(workload.requests), key=lambda item: (item[1].arrival_ms, item[0]))
        # Estimate peak resident memory from weights, the largest active KV
        # allocation, and a fixed work-buffer allowance.
        weight_gb = d.model_size_b * 1e9 * d.weight_bytes_per_parameter / 1e9
        max_kv_gb = max((r.prompt_tokens + r.output_tokens) * d.kv_bytes_per_token for r in workload.requests)
        measured = self.calibration.get("memory_footprint_gb")
        # Prefer the footprint measured on the device: vLLM preallocates its KV pool, so
        # weights + KV + runtime buffers are already in it.
        peak_memory = (max(float(measured), weight_gb + max_kv_gb) if measured
                       else weight_gb + max_kv_gb + min(topology["sbuf_gb"], 8.0))
        memory_fit = peak_memory <= topology["hbm_gb"]

        # FCFS queue replay: requests are admitted by arrival time, then each
        # receives a prefill phase followed by serial autoregressive decode.
        # Overload is represented through queue delay and the memory-fit flag.
        now_ms = 0.0
        outcomes: List[RequestOutcome] = []
        total_tokens = 0
        for index, request in ordered:
            start = max(now_ms, request.arrival_ms)
            prefill = self._prefill_ms(request.prompt_tokens) / factor
            first_token = start + prefill
            decode = request.output_tokens * self._decode_token_ms() / factor
            complete = first_token + decode
            request_id = request.request_id or f"{workload.name.lower()}-{index + 1:03d}"
            outcomes.append(RequestOutcome(
                request_id=request_id,
                arrival_ms=request.arrival_ms,
                first_token_ms=first_token,
                completion_ms=complete,
                ttft_ms=max(0.0, first_token - request.arrival_ms),
                output_tokens=request.output_tokens,
            ))
            now_ms = complete
            total_tokens += request.prompt_tokens + request.output_tokens

        duration = max(now_ms - min(r.arrival_ms for r in workload.requests), 0.001)
        power = min(raw_power, power_cap)

        # Coarse package-area proxy used only as a transparent architectural
        # constraint. It is not a physical-design estimate.
        area_mm2 = topology["area_mm2"] or (
            120.0
            + d.compute_cores * 42.0
            + d.hbm_gb * 1.45
            + d.sbuf_gb * 0.65
            + d.dma_engines * 3.5
        )

        # A stable spatial gradient provides an 8x8 thermal proxy. Temperatures
        # rise with power and are hottest near the center of the package.
        base_c = 28.0 + power / 32.0
        grid = []
        for row in range(8):
            cells = []
            for col in range(8):
                radial = abs(row - 3.5) + abs(col - 3.5)
                cells.append(round(base_c + max(0.0, 4.0 - radial) * 1.7, 2))
            grid.append(cells)

        component_metrics = {}
        component_temperatures = {}
        bottlenecks = []
        if edges := (d.connections or []):
            min_bw = min(edge.bandwidth_tb_s for edge in edges)
            bottlenecks = sorted({edge.source for edge in edges if edge.bandwidth_tb_s <= min_bw * 1.001} |
                                 {edge.target for edge in edges if edge.bandwidth_tb_s <= min_bw * 1.001})
        for component in d.components or []:
            local_power = power * (component.compute_tflops / max(topology["compute_tflops"], 1.0)
                                   if component.type == "compute_cluster" else
                                   component.capacity_gb / max(topology["hbm_gb"], 1.0)
                                   if component.type == "hbm" else .04)
            cx, cy = component.x + component.width / 2.0, component.y + component.height / 2.0
            center_coupling = max(0.0, 1.0 - (abs(cx - .5) + abs(cy - .5))) * 3.0
            neighbor_coupling = 0.0
            for other in d.components or []:
                if other.id == component.id or other.type != "compute_cluster":
                    continue
                ox, oy = other.x + other.width / 2.0, other.y + other.height / 2.0
                neighbor_coupling += max(0.0, .35 - (abs(cx - ox) + abs(cy - oy))) * 4.0
            temperature = round(base_c + center_coupling + neighbor_coupling
                                + min(18.0, local_power / max(component.area_mm2, 1.0) * 4), 2)
            component_temperatures[component.id] = temperature
            component_metrics[component.id] = {
                "type": component.type,
                "core_count": component.core_count,
                "compute_tflops": component.compute_tflops,
                "capacity_gb": component.capacity_gb,
                "bandwidth_tb_s": component.bandwidth_tb_s,
                "dma_engines": component.dma_engines,
                "area_mm2": component.area_mm2,
                "temperature_c": temperature,
                "wire_length_factor": round(topology["wire_length"], 4),
            }

        typical_prompt = sorted(r.prompt_tokens for r in workload.requests)[len(workload.requests) // 2]
        prefill_c, prefill_m = self._roofline(typical_prompt)
        decode_c, decode_m = self._roofline(None)
        p_scale = float(self.calibration.get("prefill_scale", 1.0))
        d_scale = float(self.calibration.get("decode_scale", 1.0))

        return SimulationResult(
            trace_name=workload.name,
            prefill_compute_ms=round(prefill_c * p_scale, 4), prefill_memory_ms=round(prefill_m * p_scale, 4),
            decode_compute_ms=round(decode_c * d_scale, 4), decode_memory_ms=round(decode_m * d_scale, 4),
            prefill_bound="compute" if prefill_c >= prefill_m else "memory",
            decode_bound="compute" if decode_c >= decode_m else "memory",
            raw_power_w=round(raw_power, 2),
            usable_hbm_bandwidth_tb_s=round(topology["hbm_bandwidth"], 4),
            ttft_p50_ms=round(_percentile([o.ttft_ms for o in outcomes], 0.50), 3),
            ttft_p95_ms=round(_percentile([o.ttft_ms for o in outcomes], 0.95), 3),
            ttft_p99_ms=round(_percentile([o.ttft_ms for o in outcomes], 0.99), 3),
            throughput_tokens_s=round(total_tokens / (duration / 1000.0), 3),
            memory_fit=memory_fit,
            peak_memory_gb=round(peak_memory, 3),
            power_w=round(power, 2),
            area_mm2=round(area_mm2, 2),
            max_temperature_c=max(max(row) for row in grid),
            thermal_grid=grid,
            throttling=throttling,
            throttle_factor=round(factor, 4),
            duration_ms=round(duration, 3),
            completed_requests=len(outcomes),
            request_outcomes=outcomes,
            topology_valid=True,
            topology_errors=[],
            bottleneck_component_ids=bottlenecks,
            component_metrics=component_metrics,
            component_temperatures=component_temperatures,
        )

    def _invalid_result(self, trace_name: str, errors: List[str]) -> SimulationResult:
        """Invalid architectures are explicit failed runs, never scored as wins."""
        return SimulationResult(
            trace_name=trace_name, ttft_p50_ms=0, ttft_p95_ms=0, ttft_p99_ms=0,
            throughput_tokens_s=0, memory_fit=False, peak_memory_gb=0, power_w=0,
            area_mm2=0, max_temperature_c=0,
            thermal_grid=[[0.0 for _ in range(8)] for _ in range(8)], throttling=False,
            topology_valid=False, topology_errors=errors,
        )

    def _empty_result(self, trace_name: str) -> SimulationResult:
        return SimulationResult(
            trace_name=trace_name, ttft_p50_ms=0, ttft_p95_ms=0, ttft_p99_ms=0,
            throughput_tokens_s=0, memory_fit=True, peak_memory_gb=0, power_w=0,
            area_mm2=0, max_temperature_c=0,
            thermal_grid=[[0.0 for _ in range(8)] for _ in range(8)], throttling=False,
        )

