"""Shared data models for the AccelTwin simulator."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Component(BaseModel):
    """One renderable block in a normalized chip floorplan."""

    id: str
    type: Literal["compute_cluster", "hbm", "sbuf", "dma", "noc", "router", "io"]
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    layer: int = Field(default=0, ge=0)
    label: str = ""
    core_count: int = Field(default=0, ge=0)
    compute_tflops: float = Field(default=0, ge=0)
    capacity_gb: float = Field(default=0, ge=0)
    bandwidth_tb_s: float = Field(default=0, ge=0)
    dma_engines: int = Field(default=0, ge=0)
    area_mm2: float = Field(default=0, ge=0)

    model_config = ConfigDict(extra="ignore")


class Connection(BaseModel):
    """A directed data path between two component IDs."""

    id: str
    source: str
    target: str
    bandwidth_tb_s: float = Field(gt=0)
    latency_ns: float = Field(default=10, ge=0)
    label: str = ""

    model_config = ConfigDict(extra="ignore")


def _baseline_topology(compute_cores: int, peak_compute_tflops: float, hbm_gb: float,
                       hbm_bandwidth_tb_s: float, sbuf_gb: float, dma_engines: int):
    """Build a deterministic illustrative topology consistent with scalar fields."""
    components = []
    # Four logical compute tiles, arranged as a 2x2 mesh. For scalar-only
    # legacy inputs, distribute aggregate compute evenly across these tiles.
    for i in range(4):
        components.append(Component(id=f"compute-{i}", type="compute_cluster",
            x=0.30 + (i % 2) * 0.25, y=0.22 + (i // 2) * 0.34,
            width=0.20, height=0.25, layer=1, label=f"Compute cluster {i + 1}",
            core_count=compute_cores // 4 + (1 if i < compute_cores % 4 else 0),
            compute_tflops=peak_compute_tflops / 4.0,
            area_mm2=(compute_cores * 42.0) / 4.0))
    for i in range(2):
        components.append(Component(id=f"hbm-{i}", type="hbm",
            x=0.03 + i * 0.78, y=0.25, width=0.16, height=0.42,
            layer=1, label=f"HBM stack {i + 1}", capacity_gb=hbm_gb / 2.0,
            bandwidth_tb_s=hbm_bandwidth_tb_s / 2.0,
            area_mm2=(hbm_gb * 1.45) / 2.0))
    components.extend([
        Component(id="sbuf-0", type="sbuf", x=.30, y=.02, width=.40, height=.13, layer=1,
            label="On-chip SRAM", capacity_gb=sbuf_gb, area_mm2=sbuf_gb * .65),
        Component(id="noc-0", type="noc", x=.25, y=.18, width=.50, height=.64,
            label="2D mesh interconnect", area_mm2=120),
    ])
    for i in range(dma_engines):
        components.append(Component(id=f"dma-{i}", type="dma", x=.28 + (i % 4) * .12,
            y=.86, width=.08, height=.08, layer=1, label=f"DMA {i + 1}", dma_engines=1,
            area_mm2=3.5))
    connections = []
    for h in range(2):
        connections.append(Connection(id=f"hbm{h}-noc", source=f"hbm-{h}",
            target="noc-0", bandwidth_tb_s=hbm_bandwidth_tb_s / 2.0,
            latency_ns=8, label="HBM to mesh"))
    for c in range(4):
        connections.append(Connection(id=f"noc-compute{c}", source="noc-0",
            target=f"compute-{c}", bandwidth_tb_s=hbm_bandwidth_tb_s / 4.0,
            latency_ns=10, label="Mesh to compute"))
    connections.append(Connection(id="sbuf-noc", source="sbuf-0", target="noc-0",
        bandwidth_tb_s=max(hbm_bandwidth_tb_s, .01), latency_ns=6, label="SRAM to mesh"))
    for i in range(dma_engines):
        connections.append(Connection(id=f"dma{i}-noc", source=f"dma-{i}", target="noc-0",
            bandwidth_tb_s=max(hbm_bandwidth_tb_s / max(dma_engines, 1), .01), latency_ns=8))
    return components, connections


class HardwareDesign(BaseModel):
    """A compact, editable description of an accelerator instance.

    Values use decimal TB/s and GB. ``peak_compute_tflops`` is aggregate BF16
    throughput for the logical device described by ``compute_cores``.
    """

    name: str = "Trainium2-like baseline"
    compute_cores: int = Field(default=4, gt=0)
    peak_compute_tflops: float = Field(default=2500.0, gt=0)
    hbm_gb: float = Field(default=96.0, gt=0)
    hbm_bandwidth_tb_s: float = Field(default=2.9, gt=0)
    sbuf_gb: float = Field(default=24.0, ge=0)
    dma_engines: int = Field(default=4, ge=0)
    power_limit_w: float = Field(default=600.0, gt=0)
    cooling_capacity_w: float = Field(default=650.0, gt=0)
    model_size_b: float = Field(default=13.0, gt=0)
    weight_bytes_per_parameter: float = Field(default=2.0, gt=0)
    kv_bytes_per_token: float = Field(default=0.0005, gt=0)
    utilization: float = Field(default=0.72, gt=0, le=1)
    components: Optional[List[Component]] = None
    connections: Optional[List[Connection]] = None

    model_config = ConfigDict(extra="ignore")

    def model_post_init(self, __context: Any) -> None:
        # Scalar-only callers remain supported, but every instance now has a
        # direct renderable topology. Explicit topology is preserved verbatim.
        if self.components is None or self.connections is None:
            components, connections = _baseline_topology(self.compute_cores,
                self.peak_compute_tflops, self.hbm_gb, self.hbm_bandwidth_tb_s,
                self.sbuf_gb, self.dma_engines)
            if self.components is None:
                self.components = components
            if self.connections is None:
                self.connections = connections

    def validate_topology(self) -> List[str]:
        """Return deterministic topology and scalar-consistency errors."""
        components = self.components or []
        connections = self.connections or []
        errors: List[str] = []
        ids = [c.id for c in components]
        if len(ids) != len(set(ids)):
            errors.append("component IDs must be unique")
        cids = set(ids)
        supported_types = {"compute_cluster", "hbm", "sbuf", "dma", "noc", "router", "io"}
        for c in components:
            if c.type not in supported_types:
                errors.append(f"component {c.id} has unsupported type {c.type}")
            if c.x + c.width > 1.000001 or c.y + c.height > 1.000001:
                errors.append(f"component {c.id} is outside normalized floorplan bounds")
        for i, a in enumerate(components):
            for b in components[i + 1:]:
                overlap_x = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
                overlap_y = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
                if a.layer == b.layer and overlap_x > .015 and overlap_y > .015 and a.type not in {"noc", "router"} and b.type not in {"noc", "router"}:
                    errors.append(f"components {a.id} and {b.id} overlap on layer {a.layer}")
        types = {c.type for c in components}
        for required in ("compute_cluster", "hbm", "sbuf", "noc"):
            if required not in types:
                errors.append(f"required component type {required} is missing")
        if self.dma_engines > 0 and "dma" not in types:
            errors.append("required component type dma is missing")
        seen_edges = set()
        adjacency = {cid: set() for cid in cids}
        for edge in connections:
            if edge.source not in cids or edge.target not in cids:
                errors.append(f"connection {edge.id} references an unknown component")
                continue
            if edge.id in seen_edges:
                errors.append(f"connection IDs must be unique: {edge.id}")
            seen_edges.add(edge.id)
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
        compute_ids = [c.id for c in components if c.type == "compute_cluster"]
        hbm_ids = [c.id for c in components if c.type == "hbm"]
        for compute_id in compute_ids:
            reached = {compute_id}
            todo = [compute_id]
            while todo:
                for nxt in adjacency.get(todo.pop(), ()):
                    if nxt not in reached:
                        reached.add(nxt); todo.append(nxt)
            if not any(h in reached for h in hbm_ids):
                errors.append(f"compute component {compute_id} has no HBM path")
        total_compute = sum(c.compute_tflops for c in components if c.type == "compute_cluster")
        total_cores = sum(c.core_count for c in components if c.type == "compute_cluster")
        total_hbm = sum(c.capacity_gb for c in components if c.type == "hbm")
        total_hbm_bw = sum(c.bandwidth_tb_s for c in components if c.type == "hbm")
        total_sbuf = sum(c.capacity_gb for c in components if c.type == "sbuf")
        total_dma = sum(c.dma_engines for c in components if c.type == "dma")
        for label, actual, expected in (
            ("compute throughput", total_compute, self.peak_compute_tflops),
            ("logical core count", total_cores, self.compute_cores),
            ("HBM capacity", total_hbm, self.hbm_gb),
            ("HBM bandwidth", total_hbm_bw, self.hbm_bandwidth_tb_s),
            ("SBUF capacity", total_sbuf, self.sbuf_gb),
            ("DMA engines", total_dma, self.dma_engines),
        ):
            if abs(actual - expected) > max(.01, abs(expected) * .01):
                errors.append(f"topology {label} ({actual:g}) does not match scalar value ({expected:g})")
        if not any(c.bandwidth_tb_s > 0 for c in components if c.type == "hbm"):
            errors.append("HBM components must declare positive bandwidth")
        return errors

    @classmethod
    def baseline(cls) -> "HardwareDesign":
        """Return the required four-core, 96 GB, 2.9 TB/s reference design."""
        return cls()

    @classmethod
    def trainium2_baseline(cls) -> "HardwareDesign":
        return cls.baseline()


class RequestSpec(BaseModel):
    request_id: str = ""
    prompt_tokens: int = Field(default=512, ge=1)
    output_tokens: int = Field(default=128, ge=1)
    arrival_ms: float = Field(default=0.0, ge=0)

    model_config = ConfigDict(extra="ignore")


class RequestTrace(BaseModel):
    name: str
    description: str = ""
    requests: List[RequestSpec]

    model_config = ConfigDict(extra="ignore")


class RequestOutcome(BaseModel):
    request_id: str
    arrival_ms: float
    first_token_ms: float
    completion_ms: float
    ttft_ms: float
    output_tokens: int


class SimulationResult(BaseModel):
    trace_name: str
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    throughput_tokens_s: float
    memory_fit: bool
    peak_memory_gb: float
    power_w: float
    area_mm2: float = 0.0
    max_temperature_c: float = 0.0
    thermal_grid: List[List[float]]
    throttling: bool
    throttle_factor: float = 1.0
    duration_ms: float = 0.0
    completed_requests: int = 0
    request_outcomes: List[RequestOutcome] = Field(default_factory=list)
    topology_valid: bool = True
    topology_errors: List[str] = Field(default_factory=list)
    bottleneck_component_ids: List[str] = Field(default_factory=list)
    component_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    component_temperatures: Dict[str, float] = Field(default_factory=dict)
    # Roofline split for a typical request, so the checker can name the limit.
    prefill_compute_ms: float = 0.0
    prefill_memory_ms: float = 0.0
    decode_compute_ms: float = 0.0
    decode_memory_ms: float = 0.0
    prefill_bound: str = ""
    decode_bound: str = ""
    raw_power_w: float = 0.0
    usable_hbm_bandwidth_tb_s: float = 0.0

    model_config = ConfigDict(extra="ignore")

