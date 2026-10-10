from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app import api
from app.agent import ChipDesignAgent
from app.llm_client import LLMClient
from app.models import HardwareDesign
from app.orchestrator import CampaignOrchestrator
from app.simulator import DigitalTwinSimulator

PLAN = {
    "name": "Wider memory, six clusters",
    "rationale": "Decode is bandwidth bound, so add HBM bandwidth and spread compute.",
    "changes": ["Raise HBM bandwidth to 3.6 TB/s", "Split compute into six clusters"],
    "targets": {"compute_clusters": 6, "compute_cores": 6, "hbm_stacks": 3, "hbm_bandwidth_tb_s": 3.6},
}
GOOD_LINKS = {"links": [["hbm-0", "noc-0", 1.2], ["hbm-1", "noc-0", 1.2], ["hbm-2", "noc-0", 1.2],
                        ["sbuf-0", "noc-0", 2], ["dma-0", "noc-0", 0.5], ["dma-1", "noc-0", 0.5],
                        ["dma-2", "noc-0", 0.5], ["dma-3", "noc-0", 0.5]]
              + [["noc-0", f"compute-{i}", 0.6] for i in range(6)]}


class ScriptedChat:
    """Answers each stage from a queue and records the prompts it saw."""

    def __init__(self, floorplans, links=(GOOD_LINKS,), plan=PLAN):
        self.plan, self.floorplans, self.links = plan, list(floorplans), list(links)
        self.prompts: list[str] = []

    def __call__(self, system, user, max_tokens):
        self.prompts.append(user)
        if "Return:\n{\"name\"" in user:
            return self.plan
        if '"regions"' in user:
            return self.floorplans.pop(0) if self.floorplans else {}
        return self.links.pop(0) if self.links else {}


def good_floorplan():
    return {"regions": {"hbm-0": "left", "hbm-1": "right", "hbm-2": "left", "sbuf-0": "top",
                        **{f"compute-{i}": "center" for i in range(6)},
                        **{f"dma-{i}": "bottom" for i in range(4)}}}


def test_staged_rounds_produce_a_valid_design_from_small_calls():
    chat = ScriptedChat([good_floorplan()])
    proposal = ChipDesignAgent(chat).run({"round": 0, "baseline_config": HardwareDesign.baseline().model_dump()})
    design = HardwareDesign.model_validate(proposal["design"])
    assert design.validate_topology() == []
    assert len(chat.prompts) == 3
    assert proposal["steps"] == ["plan", "floorplan", "interconnect"]
    assert sum(c.type == "compute_cluster" for c in design.components) == 6
    assert design.hbm_bandwidth_tb_s == 3.6
    assert proposal["changes"] == PLAN["changes"]
    # No single prompt carries the whole graph as JSON.
    assert all(len(prompt) < 6000 for prompt in chat.prompts)
    assert DigitalTwinSimulator().evaluate(design, "C").topology_valid


def test_bad_floorplan_gets_one_targeted_repair_then_falls_back():
    overlapping = good_floorplan()
    overlapping["regions"]["hbm-1"] = "center"
    chat = ScriptedChat([overlapping, good_floorplan()])
    proposal = ChipDesignAgent(chat).run({"round": 0, "baseline_config": HardwareDesign.baseline().model_dump()})
    assert proposal["steps"][1] == "floorplan (repaired)"
    assert "hbm-1 is an HBM stack and must sit on a die edge" in chat.prompts[2]

    chat = ScriptedChat([overlapping, overlapping], links=({"links": []}, {"links": []}))
    proposal = ChipDesignAgent(chat).run({"round": 0, "baseline_config": HardwareDesign.baseline().model_dump()})
    assert proposal["steps"][1].startswith("floorplan (automatic fallback")
    assert proposal["steps"][2].startswith("interconnect (automatic fallback")
    assert HardwareDesign.model_validate(proposal["design"]).validate_topology() == []


def test_disconnected_wiring_is_reported_back_to_the_model():
    broken = {"links": [link for link in GOOD_LINKS["links"] if link[1] != "compute-5"]}
    chat = ScriptedChat([good_floorplan()], links=(broken, GOOD_LINKS))
    proposal = ChipDesignAgent(chat).run({"round": 0, "baseline_config": HardwareDesign.baseline().model_dump()})
    assert "compute-5 cannot reach any HBM stack" in chat.prompts[-1]
    assert proposal["steps"][-1] == "interconnect (repaired)"


def test_json_is_extracted_from_reasoning_prose():
    raw = 'Let me think. The best plan is {"name": "x", "targets": {"hbm_gb": 128}} and that is final.'
    assert LLMClient._parse_json(raw)["targets"]["hbm_gb"] == 128


def test_campaign_uses_staged_designer_and_records_changes():
    class StagedProposer:
        mode = "openai"
        model = "openai/gpt-oss-20b"

        def design_round(self, context):
            context["report"]("Round 1: planning the architecture")
            chat = ScriptedChat([good_floorplan()])
            return ChipDesignAgent(chat, report=context["report"]).run(context)

    api.orchestrator = CampaignOrchestrator(proposer=StagedProposer())
    with TestClient(api.app) as client:
        assert client.get("/api/config").json()["model"] == "openai/gpt-oss-20b"
        campaign_id = client.post("/api/campaigns", json={"max_rounds": 1}).json()["id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            campaign = client.get(f"/api/campaigns/{campaign_id}?compact=1").json()
            if campaign["status"] in {"completed", "failed"}:
                break
            time.sleep(0.02)
    assert campaign["status"] == "completed"
    attempt = campaign["attempts"][0]
    assert attempt["changes"] == PLAN["changes"]
    texts = [delta["text"] for delta in attempt["deltas"]]
    assert "HBM bandwidth 2.9 → 3.6 TB/s" in texts
    assert "Compute clusters 4 → 6" in texts
    assert attempt["deltas"][0]["pct"] == 50.0          # largest change first (4 → 6 cores)
    assert {row["key"] for row in attempt["checks"]} >= {"throughput_at_least_baseline", "memory_fit"}
    assert "bound" in attempt["feedback"]
    assert "compute-5" in attempt["changed_ids"]
    assert "request_outcomes" not in attempt["metrics"]


def test_unserved_model_blocks_the_run_instead_of_faking_iterations():
    class WrongModel(LLMClient):
        def check(self):
            return {"ready": False, "served": ["Qwen/Qwen3-8B"],
                    "error": "openai/gpt-oss-20b is not being served; the server has Qwen/Qwen3-8B."}

    api.orchestrator = CampaignOrchestrator(proposer=WrongModel(mode="auto"))
    with TestClient(api.app) as client:
        config = client.get("/api/config").json()
        assert config["ready"] is False
        response = client.post("/api/campaigns", json={"max_rounds": 2})
    assert response.status_code == 503
    assert "Qwen/Qwen3-8B" in response.json()["detail"]


def test_model_dropping_mid_run_fails_the_campaign_loudly():
    from app.llm_client import ModelUnavailableError

    class DropsOut:
        mode = "auto"
        model = "openai/gpt-oss-20b"

        def design_round(self, context):
            raise ModelUnavailableError("openai/gpt-oss-20b did not answer within 240 s")

    api.orchestrator = CampaignOrchestrator(proposer=DropsOut())
    with TestClient(api.app) as client:
        campaign_id = client.post("/api/campaigns", json={"max_rounds": 3}).json()["id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            campaign = client.get(f"/api/campaigns/{campaign_id}?compact=1").json()
            if campaign["status"] in {"completed", "failed"}:
                break
            time.sleep(0.02)
    assert campaign["status"] == "failed"
    assert "did not answer" in campaign["error"]
    assert campaign["attempts"] == []


def test_narrow_links_are_sent_back_then_widened_to_the_hbm_rate():
    narrow = {"links": [[a, b, bw / 4] for a, b, bw in GOOD_LINKS["links"]]}
    chat = ScriptedChat([good_floorplan()], links=(narrow, narrow))
    proposal = ChipDesignAgent(chat).run({"round": 0, "baseline_config": HardwareDesign.baseline().model_dump()})
    assert "the links carry only 0.90 of the 3.6 TB/s" in chat.prompts[-1]
    assert proposal["steps"][-1] == "interconnect (links widened to the HBM rate)"
    design = HardwareDesign.model_validate(proposal["design"])
    assert DigitalTwinSimulator(design)._topology_profile()["flow"] >= 3.5


def test_unchanged_regions_keep_their_exact_boxes():
    baseline = HardwareDesign.baseline().model_dump()
    same_shape = {"name": "Raise utilization", "rationale": "", "changes": [], "targets": {"utilization": 0.8}}
    regions = {"regions": {"hbm-0": "left", "hbm-1": "right", "sbuf-0": "top",
                           **{f"compute-{i}": "center" for i in range(4)}, **{f"dma-{i}": "bottom" for i in range(4)}}}
    links = {"links": [l for l in GOOD_LINKS["links"] if l[0] != "hbm-2" and l[1] not in {"compute-4", "compute-5"}]}
    proposal = ChipDesignAgent(ScriptedChat([regions], links=(links,), plan=same_shape)).run(
        {"round": 0, "baseline_config": baseline})
    before = {c["id"]: (c["x"], c["y"], c["width"], c["height"]) for c in baseline["components"] if c["type"] != "noc"}
    after = {c["id"]: (c["x"], c["y"], c["width"], c["height"]) for c in proposal["design"]["components"] if c["type"] != "noc"}
    assert before == after
