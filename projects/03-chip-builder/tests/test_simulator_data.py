import json

from app.models import HardwareDesign, RequestSpec, RequestTrace
from app.simulator import DigitalTwinSimulator, fixed_trace
from app.storage import Storage
from app.telemetry import NeuronMonitorAdapter, parse_vllm_metrics


def test_baseline_and_replay_are_deterministic():
    design = HardwareDesign.baseline()
    assert (design.compute_cores, design.hbm_gb, design.hbm_bandwidth_tb_s) == (4, 96, 2.9)
    sim = DigitalTwinSimulator()
    first = sim.evaluate(design, fixed_trace("A"))
    second = sim.evaluate(design, "A")
    assert first.model_dump() == second.model_dump()
    assert len(first.thermal_grid) == 8
    assert all(len(row) == 8 for row in first.thermal_grid)
    assert first.ttft_p99_ms >= first.ttft_p50_ms
    assert first.completed_requests == 12
    assert first.area_mm2 > 0
    assert first.max_temperature_c == max(max(row) for row in first.thermal_grid)
    assert first.topology_valid
    assert sum(c.capacity_gb for c in design.components if c.type == "hbm") == 96
    assert len([c for c in design.components if c.type == "compute_cluster"]) == 4
    assert len(first.component_metrics) == len(design.components)
    assert first.component_temperatures


def test_connection_bandwidth_changes_simulated_score():
    design = HardwareDesign.baseline()
    slower_connections = [
        edge.model_copy(update={"bandwidth_tb_s": 0.1})
        if edge.target.startswith("compute-") else edge
        for edge in design.connections
    ]
    constrained = design.model_copy(update={"connections": slower_connections})
    fast_result = DigitalTwinSimulator().evaluate(design, "A")
    constrained_result = DigitalTwinSimulator().evaluate(constrained, "A")
    assert constrained_result.topology_valid
    assert constrained_result.ttft_p50_ms > fast_result.ttft_p50_ms
    assert constrained_result.throughput_tokens_s < fast_result.throughput_tokens_s


def test_floorplan_placement_changes_wire_latency_and_temperature():
    design = HardwareDesign.baseline()
    moved_components = [
        component.model_copy(update={"x": 0.35}) if component.id == "sbuf-0" else component
        for component in design.components
    ]
    moved = design.model_copy(update={"components": moved_components})
    baseline_result = DigitalTwinSimulator().evaluate(design, "A")
    moved_result = DigitalTwinSimulator().evaluate(moved, "A")
    assert moved_result.topology_valid
    assert moved_result.ttft_p50_ms != baseline_result.ttft_p50_ms
    assert moved_result.component_temperatures != baseline_result.component_temperatures


def test_disconnected_compute_topology_fails_simulation():
    design = HardwareDesign.baseline()
    disconnected = design.model_copy(update={
        "connections": [edge for edge in design.connections if edge.target != "compute-3"]
    })
    result = DigitalTwinSimulator().evaluate(disconnected, "A")
    assert not result.topology_valid
    assert any("compute-3 has no HBM path" in error for error in result.topology_errors)
    assert not result.memory_fit
    assert result.throughput_tokens_s == 0


def test_memory_overflow_and_power_throttle_are_reported():
    design = HardwareDesign(model_size_b=100, hbm_gb=16, power_limit_w=100, cooling_capacity_w=120)
    trace = RequestTrace(name="tiny", requests=[RequestSpec(prompt_tokens=16, output_tokens=4)])
    result = DigitalTwinSimulator().evaluate(design, trace)
    assert not result.memory_fit
    assert result.throttling
    assert result.throttle_factor < 1


def test_sqlite_and_jsonl_round_trip(tmp_path):
    storage = Storage(tmp_path / "runs.db", tmp_path / "runs.jsonl")
    design = HardwareDesign.baseline()
    result = DigitalTwinSimulator().evaluate(design, "C")
    run_id = storage.create(result, design)
    saved = storage.get(run_id)
    assert saved["result"]["trace_name"] == "C"
    assert storage.list(limit=1)[0]["id"] == run_id
    lines = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == run_id


def test_prometheus_parser_and_neuron_mock_fallback():
    metrics = parse_vllm_metrics(
        "# HELP vllm:num_requests_running requests\n"
        "vllm:num_requests_running{engine=\"0\"} 2\n"
        "vllm:num_requests_running{engine=\"1\"} 3\n"
        "vllm:time_to_first_token_seconds_sum 1.25\n"
    )
    assert metrics["vllm:num_requests_running"] == 5
    assert metrics["vllm:time_to_first_token_seconds_sum"] == 1.25
    adapter = NeuronMonitorAdapter()
    assert adapter.collect()["source"] == "mock"
    assert adapter.is_mock

