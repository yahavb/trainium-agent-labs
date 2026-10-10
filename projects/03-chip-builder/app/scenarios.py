"""Design problems the agent can be pointed at.

Each scenario frames a problem on today's chip, what "better" means (the
objective), the hard budget, and how the measured workload is reshaped. The
live replay always measures the model as it is served; the workload knobs
scale that measurement (e.g. 4x longer prompts) before simulation.
"""

from __future__ import annotations

from typing import Any

OBJECTIVES: dict[str, dict[str, str]] = {
    "ttft": {"key": "ttft", "metric": "ttft_p99_ms", "direction": "min",
             "label": "p99 time-to-first-token", "short": "p99 TTFT", "unit": "ms"},
    "throughput": {"key": "throughput", "metric": "throughput_tokens_s", "direction": "max",
                   "label": "throughput", "short": "Throughput", "unit": "tok/s"},
    "area": {"key": "area", "metric": "area_mm2", "direction": "min",
             "label": "die area", "short": "Die area", "unit": "mm²"},
}

_BUDGET = {"max_power_w": 600, "max_temperature_c": 85, "max_area_mm2": 650}

SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "chat",
        "title": "Interactive chat",
        "problem": "Bursts of chat requests queue up, and users wait too long for the first token.",
        "goal": "Minimize p99 time-to-first-token for bursty chat traffic without reducing throughput.",
        "objective": "ttft",
        "constraints": {**_BUDGET, "throughput_floor": 1.0},
        "workload": {"prompt_scale": 1.0, "output_scale": 1.0, "load": 0.8},
    },
    {
        "id": "rag",
        "title": "Long-context RAG",
        "problem": "Retrieval packs 4× longer prompts into each request, so prefill dominates the wait.",
        "goal": "Minimize p99 time-to-first-token when prompts are 4× longer than today's traffic.",
        "objective": "ttft",
        "constraints": {**_BUDGET, "throughput_floor": 1.0},
        "workload": {"prompt_scale": 4.0, "output_scale": 0.5, "load": 0.8},
    },
    {
        "id": "throughput",
        "title": "Batch throughput",
        "problem": "Offline jobs saturate the chip; what matters is tokens per second, as long as latency does not regress.",
        "goal": "Maximize throughput on a saturated chip while keeping p99 time-to-first-token no worse than today.",
        "objective": "throughput",
        "constraints": {**_BUDGET, "throughput_floor": 1.0, "latency_ceiling": 1.0},
        "workload": {"prompt_scale": 1.0, "output_scale": 2.0, "load": 1.5},
    },
    {
        "id": "edge",
        "title": "Power-capped part",
        "problem": "A smaller part with a 350 W and 450 mm² budget must serve the same model; today's chip does not fit.",
        "goal": "Minimize p99 time-to-first-token within 350 W, 75 °C and 450 mm², keeping at least 70% of today's throughput.",
        "objective": "ttft",
        "constraints": {"max_power_w": 350, "max_temperature_c": 75, "max_area_mm2": 450, "throughput_floor": 0.7},
        "workload": {"prompt_scale": 1.0, "output_scale": 1.0, "load": 0.6},
    },
    {
        "id": "cost",
        "title": "Cost-down",
        "problem": "Die area drives cost. How small can the chip get before chat latency or throughput suffers?",
        "goal": "Minimize die area while keeping p99 time-to-first-token and throughput at least as good as today.",
        "objective": "area",
        "constraints": {**_BUDGET, "throughput_floor": 1.0, "latency_ceiling": 1.0},
        "workload": {"prompt_scale": 1.0, "output_scale": 1.0, "load": 0.8},
    },
]

DEFAULT_SCENARIO = "chat"


def get_scenario(scenario_id: str | None) -> dict[str, Any]:
    for scenario in SCENARIOS:
        if scenario["id"] == (scenario_id or DEFAULT_SCENARIO):
            return scenario
    raise ValueError(f"unknown scenario {scenario_id!r}; choose one of {', '.join(s['id'] for s in SCENARIOS)}")


def public_scenarios() -> list[dict[str, Any]]:
    return [{**s, "objective": OBJECTIVES[s["objective"]]} for s in SCENARIOS]
