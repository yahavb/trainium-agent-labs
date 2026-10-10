from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app import api
from app.agent import ChipDesignAgent
from app.models import HardwareDesign
from app.orchestrator import CampaignOrchestrator
from tests.test_agent import GOOD_LINKS, ScriptedChat

REGIONS = {"regions": {"hbm-0": "left", "hbm-1": "right", "sbuf-0": "top",
                       **{f"compute-{i}": "center" for i in range(4)}, **{f"dma-{i}": "bottom" for i in range(4)}}}
LINKS = {"links": [["hbm-0", "noc-0", 1.5], ["hbm-1", "noc-0", 1.5], ["sbuf-0", "noc-0", 1.0]]
         + [["noc-0", f"compute-{i}", 0.8] for i in range(4)] + [[f"dma-{i}", "noc-0", 0.2] for i in range(4)]}


def run(proposer, body):
    api.orchestrator = CampaignOrchestrator(proposer=proposer)
    with TestClient(api.app) as client:
        campaign_id = client.post("/api/campaigns", json=body).json()["id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            campaign = client.get(f"/api/campaigns/{campaign_id}?compact=1").json()
            if campaign["status"] in {"completed", "failed"}:
                return campaign
            time.sleep(0.02)
    raise AssertionError("campaign did not finish")


def test_scenarios_are_listed_with_objectives():
    with TestClient(api.app) as client:
        scenarios = client.get("/api/scenarios").json()
    assert {s["id"] for s in scenarios} == {"chat", "rag", "throughput", "edge", "cost"}
    assert all({"problem", "goal", "objective", "constraints"} <= set(s) for s in scenarios)
    assert next(s for s in scenarios if s["id"] == "cost")["objective"]["metric"] == "area_mm2"


def test_infeasible_baseline_means_first_passing_design_wins():
    # Today's chip draws ~387 W and is ~457 mm², over the edge scenario's 350 W / 450 mm².
    frugal = {"name": "Trim compute and SRAM", "rationale": "Fit the power and area budget.", "changes": [],
              "targets": {"peak_compute_tflops": 1500, "utilization": 0.6, "sbuf_gb": 8}}

    class Proposer:
        mode, model = "openai", "openai/gpt-oss-20b"

        def design_round(self, context):
            return ChipDesignAgent(ScriptedChat([REGIONS], links=(LINKS,), plan=frugal)).run(context)

    campaign = run(Proposer(), {"scenario": "edge", "max_rounds": 1})
    assert campaign["status"] == "completed"
    assert campaign["baseline_feasible"] is False
    assert any(row["passed"] is False for row in campaign["baseline_checks"])
    assert campaign["attempts"][0]["accepted"] and campaign["best_attempt"]["round"] == 1


def test_cost_down_scores_area_and_guards_latency():
    gates = CampaignOrchestrator._gate_results(
        {"throughput_tokens_s": 100, "memory_fit": True, "ttft_p99_ms": 120, "area_mm2": 400},
        100, {"max_p99_ttft_ms": 100.0})
    assert gates["latency_guard"] is False
    rows = CampaignOrchestrator._checks({"ttft_p99_ms": 120, "throughput_tokens_s": 100}, {"hbm_gb": 96}, 100,
                                        {"max_p99_ttft_ms": 100.0, "throughput_floor": 0.7})
    assert next(r for r in rows if r["key"] == "throughput_at_least_baseline")["limit"] == 70
    assert next(r for r in rows if r["key"] == "latency_guard")["passed"] is False


def test_every_model_call_is_logged_and_streams_token_counts():
    calls = []

    class StreamingChat(ScriptedChat):
        def __call__(self, system, user, max_tokens, progress=None):
            for n in range(1, 4):
                progress(n * 10)
            return super().__call__(system, user, max_tokens)

    plan = {"name": "x", "rationale": "", "changes": [], "targets": {}}
    ChipDesignAgent(StreamingChat([REGIONS], links=(LINKS,), plan=plan), log_call=calls.append).run(
        {"round": 2, "baseline_config": HardwareDesign.baseline().model_dump()})
    assert [c["stage"] for c in calls] == ["plan", "floorplan", "wiring"]
    assert all(c["status"] == "done" and c["tokens"] == 30 and c["round"] == 3 and c["ms"] is not None for c in calls)


def test_area_objective_prompt_steers_toward_removing_area():
    chat = ScriptedChat([REGIONS], links=(LINKS,), plan={"name": "x", "rationale": "", "changes": [], "targets": {}})
    ChipDesignAgent(chat).run({"round": 0, "baseline_config": HardwareDesign.baseline().model_dump(),
                               "objective": {"metric": "area_mm2", "direction": "min", "label": "die area"},
                               "best_metrics": {"decode_bound": "memory", "peak_memory_gb": 55.2},
                               "constraints": {"max_p99_ttft_ms": 17000.0}})
    plan_prompt = chat.prompts[0]
    assert "Score: lower simulated die area is better." in plan_prompt
    assert "compute cores 168 mm2" in plan_prompt and "hbm_gb >= 56" in plan_prompt
    assert "compute cores and TFLOPS are slack" in plan_prompt
