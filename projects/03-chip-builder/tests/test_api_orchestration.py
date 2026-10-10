from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app import api
from app.llm_client import ProposalError
from app.llm_client import LLMClient
from app.orchestrator import CampaignOrchestrator


class FakeSimulator:
    def evaluate(self, design, trace):
        candidate = bool(design)
        return {
            "trace_name": getattr(trace, "name", trace),
            "ttft_p50_ms": 40.0 if candidate else 50.0,
            "ttft_p95_ms": 65.0 if candidate else 80.0,
            "ttft_p99_ms": 80.0 if candidate else 100.0,
            "throughput_tokens_s": 120.0 if candidate else 100.0,
            "memory_fit": True,
            "peak_memory_gb": 20.0,
            "power_w": 300.0,
            "thermal_grid": [[60.0] * 8 for _ in range(8)],
            "area_mm2": 500.0,
        }


class MockProposer:
    mode = "mock"

    def propose(self, prompt, schema, context=None):
        round_no = int((context or {}).get("round", 0))
        return {"name": f"mock design {round_no + 1}", "rationale": "test proposal", "design": {"candidate": round_no + 1}}

    def repair(self, prompt, invalid, error, schema):
        return {"name": "repaired design", "design": {"candidate": 9}}


def setup_function():
    api.orchestrator = CampaignOrchestrator(FakeSimulator(), proposer=MockProposer())
    api.orchestrator._models = lambda: (None, None)


def test_baseline_and_get_by_id():
    with TestClient(api.app) as client:
        response = client.post("/api/baselines/run", json={"trace": "A"})
        assert response.status_code == 200
        baseline = response.json()
        assert baseline["metrics"]["ttft_p99_ms"] == 100.0
        fetched = client.get(f"/api/baselines/{baseline['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["id"] == baseline["id"]


def test_campaign_hard_gates_report_design_and_sse():
    with TestClient(api.app) as client:
        baseline = client.post("/api/baselines/run", json={"trace": "A"}).json()
        calibration = client.post("/api/calibrations", json={
            "baseline_id": baseline["id"], "observed_metrics": {"ttft_p99_ms": 102.0}
        })
        assert calibration.status_code == 200

        created = client.post("/api/campaigns", json={
            "baseline_id": baseline["id"], "max_rounds": 2,
            "constraints": {"max_power_w": 350, "max_temperature_c": 80, "max_area_mm2": 600},
        })
        assert created.status_code == 202
        campaign_id = created.json()["id"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            campaign = client.get(f"/api/campaigns/{campaign_id}").json()
            if campaign["status"] in {"completed", "failed"}:
                break
            time.sleep(0.02)
        assert campaign["status"] == "completed"
        assert len(campaign["attempts"]) >= 1
        best = campaign["best_attempt"]
        assert best["accepted"] is True
        assert all(best["gate_results"].values())

        design_response = client.get(f"/api/designs/{best['id']}")
        assert design_response.status_code == 200
        assert design_response.json()["design"] == best["design"]
        report = client.get(f"/api/reports/{campaign_id}")
        assert report.status_code == 200
        assert "predicted p99 TTFT" in report.text
        events = client.get(f"/api/events/{campaign_id}")
        assert events.status_code == 200
        assert "campaign.completed" in events.text


def test_duplicate_candidate_early_stops_without_second_simulation():
    class DuplicateProposer(MockProposer):
        def propose(self, prompt, schema, context=None):
            return {"name": "duplicate", "design": {"same": 1}}

    simulator = FakeSimulator()
    api.orchestrator = CampaignOrchestrator(simulator, proposer=DuplicateProposer())
    api.orchestrator._models = lambda: (None, None)
    with TestClient(api.app) as client:
        baseline = client.post("/api/baselines/run", json={}).json()
        response = client.post("/api/campaigns", json={"baseline_id": baseline["id"], "max_rounds": 5})
        campaign_id = response.json()["id"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            campaign = client.get(f"/api/campaigns/{campaign_id}").json()
            if campaign["status"] == "completed":
                break
            time.sleep(0.02)
        assert campaign["status"] == "completed"
        assert campaign["stop_reason"] == "duplicate design"
        assert campaign["attempts"][-1]["duplicate"] is True


def test_throughput_gate_rejects_candidate():
    class SlowSimulator(FakeSimulator):
        def evaluate(self, design, trace):
            result = super().evaluate(design, trace)
            if design:
                result["throughput_tokens_s"] = 90.0
            return result

    api.orchestrator = CampaignOrchestrator(SlowSimulator(), proposer=MockProposer())
    api.orchestrator._models = lambda: (None, None)
    with TestClient(api.app) as client:
        baseline = client.post("/api/baselines/run", json={}).json()
        campaign_id = client.post("/api/campaigns", json={
            "baseline_id": baseline["id"], "max_rounds": 1,
        }).json()["id"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            campaign = client.get(f"/api/campaigns/{campaign_id}").json()
            if campaign["status"] == "completed":
                break
            time.sleep(0.02)
        attempt = campaign["attempts"][0]
        assert attempt["accepted"] is False
        assert attempt["gate_results"]["throughput_at_least_baseline"] is False


def test_invalid_proposal_gets_exactly_one_repair_attempt():
    class InvalidProposer:
        mode = "strict"

        def __init__(self):
            self.propose_calls = 0
            self.repair_calls = 0

        def propose(self, prompt, schema, context=None):
            self.propose_calls += 1
            return {"unexpected": True}

        def repair(self, prompt, invalid, error, schema):
            self.repair_calls += 1
            raise ProposalError("still invalid")

    proposer = InvalidProposer()
    api.orchestrator = CampaignOrchestrator(FakeSimulator(), proposer=proposer)
    api.orchestrator._models = lambda: (None, None)
    with TestClient(api.app) as client:
        baseline = client.post("/api/baselines/run", json={}).json()
        campaign_id = client.post("/api/campaigns", json={
            "baseline_id": baseline["id"], "max_rounds": 3,
        }).json()["id"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            campaign = client.get(f"/api/campaigns/{campaign_id}").json()
            if campaign["status"] in {"completed", "failed"}:
                break
            time.sleep(0.02)
        assert campaign["status"] == "failed"
        assert proposer.propose_calls == 1
        assert proposer.repair_calls == 1


def test_topology_is_a_hard_gate_and_checker_feedback_is_actionable():
    gates = CampaignOrchestrator._gate_results(
        {"topology_valid": False, "memory_fit": True, "throughput_tokens_s": 120,
         "power_w": 200, "max_temperature_c": 70, "area_mm2": 500},
        baseline_throughput=100, constraints={}, topology_required=True,
        topology_errors=["connection c0 targets missing component ghost"],
    )
    assert gates["topology_valid"] is False
    feedback = CampaignOrchestrator._feedback(
        gates, {"topology_errors": ["connection c0 targets missing component ghost"],
                "disconnected_component_ids": ["compute-1"],
                "thermal_bottleneck_component_ids": ["compute-1"]},
        baseline_throughput=100, best_p99=90,
        topology_errors=["connection c0 targets missing component ghost"],
    )
    assert "missing component ghost" in feedback
    assert "compute-1" in feedback
    assert "reconnect all component IDs" in feedback


def test_offline_proposer_returns_complete_changed_chip_graph():
    source = {"compute_cores": 4, "components": [
        {"id": "compute-0", "type": "compute_cluster", "x": 0.1, "y": 0.1,
         "width": 0.2, "height": 0.2, "layer": "default", "core_count": 4},
    ], "connections": []}
    proposal = LLMClient._rule_proposal({"round": 0, "baseline_config": source})
    assert proposal["design"]["components"]
    assert proposal["design"]["connections"] == []
    moved = proposal["design"]["components"][0]
    assert (moved["x"], moved["y"]) != (source["components"][0]["x"], source["components"][0]["y"])
    assert proposal["rationale"]


def test_design_diff_exposes_physical_layout_and_link_changes():
    before = {"components": [{"id": "hbm-0", "type": "hbm", "x": 0.1, "y": 0.2}],
              "connections": [{"id": "link-0", "source": "hbm-0", "target": "compute-0"}]}
    after = {"components": [{"id": "hbm-0", "type": "hbm", "x": 0.2, "y": 0.2}],
             "connections": [{"id": "link-1", "source": "hbm-0", "target": "compute-1"}]}
    diff = CampaignOrchestrator._design_diff(before, after)
    assert diff["components"]["modified"][0]["id"] == "hbm-0"
    assert diff["connections"]["removed"][0]["id"] == "link-0"
    assert diff["connections"]["added"][0]["id"] == "link-1"
