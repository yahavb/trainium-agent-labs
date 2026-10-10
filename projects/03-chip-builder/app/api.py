"""FastAPI endpoints for baselines, calibration, and bounded design campaigns."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .orchestrator import CampaignOrchestrator

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


orchestrator = CampaignOrchestrator()
app = FastAPI(title="AccelTwin", version="0.1.0", description="Checker-driven accelerator design exploration")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/how", include_in_schema=False)
async def how_it_works() -> FileResponse:
    return FileResponse(STATIC_DIR / "how.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "acceltwin"}


@app.get("/api/scenarios")
async def scenarios() -> list[dict[str, Any]]:
    from .scenarios import public_scenarios
    return public_scenarios()


@app.get("/api/config")
async def config() -> dict[str, Any]:
    proposer = orchestrator.proposer
    check = getattr(proposer, "check", None)
    status = await asyncio.to_thread(check) if callable(check) else {"ready": True, "error": None}
    return {"model": getattr(proposer, "model", None), "mode": getattr(proposer, "mode", "custom"),
            "ready": status.get("ready", True), "error": status.get("error")}


_HEAVY_METRICS = {"request_outcomes", "thermal_grid", "component_metrics", "component_temperatures"}


def _compact(campaign: dict[str, Any]) -> dict[str, Any]:
    """Drop per-request and per-cell detail the timeline UI never draws."""
    def slim_metrics(metrics: Any) -> Any:
        return {k: v for k, v in metrics.items() if k not in _HEAVY_METRICS} if isinstance(metrics, dict) else metrics

    def slim_attempt(attempt: Any) -> Any:
        if not isinstance(attempt, dict):
            return attempt
        slim = {k: v for k, v in attempt.items() if k not in {"design_diff", "baseline_design_diff"}}
        slim["metrics"] = slim_metrics(attempt.get("metrics"))
        return slim

    out = {k: v for k, v in campaign.items() if k not in {"events", "report", "latest_design"}}
    out["baseline_metrics"] = slim_metrics(campaign.get("baseline_metrics"))
    out["source_baseline_metrics"] = slim_metrics(campaign.get("source_baseline_metrics"))
    out["attempts"] = [slim_attempt(a) for a in campaign.get("attempts", [])]
    out["best_attempt"] = slim_attempt(campaign.get("best_attempt"))
    return out


@app.get("/api/state")
async def state() -> dict[str, Any]:
    baseline = next(reversed(orchestrator.baselines.values()), None) if orchestrator.baselines else None
    if baseline is None:
        try:
            from .models import HardwareDesign
            initial_design = HardwareDesign.baseline().model_dump(mode="json")
        except Exception:
            initial_design = None
    else:
        initial_design = baseline.get("design")
    metrics = (baseline or {}).get("metrics", {})
    grid = metrics.get("thermal_grid", []) if isinstance(metrics, dict) else []
    flat_grid = [value for row in grid for value in row] if grid else []
    low = min(flat_grid, default=0)
    high = max(flat_grid, default=1)
    span = max(high - low, 0.001)
    campaigns = list(orchestrator.campaigns.values())
    latest = campaigns[-1] if campaigns else None
    return {
        "connected": bool((baseline or {}).get("measured_telemetry")),
        "measurement_source": (baseline or {}).get("measurement_source", "modeled"),
        "baseline_id": (baseline or {}).get("id"),
        "design": initial_design,
        "metrics": {
            **metrics,
            "p99_ttft_ms": metrics.get("ttft_p99_ms"),
            "throughput_tokens_per_second": metrics.get("throughput_tokens_s"),
            "power_watts": metrics.get("power_w"),
            "temperature_c": max((max(row) for row in grid if row), default=None),
            "memory_bandwidth_tb_s": (baseline or {}).get("design", {}).get("hbm_bandwidth_tb_s"),
            "thermal_grid": grid,
        },
        "campaign": ({"status": latest.get("status"), "step": len(latest.get("attempts", [])),
                      "total_steps": latest.get("max_rounds", 5)} if latest else None),
        "updated_at": baseline.get("created_at") if baseline else None,
        "tile_utilization": [(value - low) / span for value in flat_grid],
    }


@app.post("/api/baselines/run")
async def run_baseline(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return await orchestrator.run_baseline(payload)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/baselines/{baseline_id}")
async def get_baseline(baseline_id: str) -> dict[str, Any]:
    baseline = orchestrator.get_baseline(baseline_id)
    if baseline is None:
        raise HTTPException(status_code=404, detail="baseline not found")
    return baseline


@app.post("/api/calibrations")
async def create_calibration(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return orchestrator.create_calibration(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/campaigns", status_code=202)
async def create_campaign(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        campaign = await orchestrator.create_campaign(payload)
        return {key: value for key, value in campaign.items() if key not in {"events", "report"}}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/campaigns/{campaign_id}")
async def get_campaign(campaign_id: str, compact: bool = False) -> dict[str, Any]:
    campaign = orchestrator.get_campaign(campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    if compact:
        return _compact(campaign)
    return {key: value for key, value in campaign.items() if key not in {"events", "report"}}


@app.get("/api/designs/{attempt_id}")
async def get_design(attempt_id: str) -> dict[str, Any]:
    design = orchestrator.get_design(attempt_id)
    if design is None:
        raise HTTPException(status_code=404, detail="design attempt not found")
    return design


@app.get("/api/reports/{campaign_id}", response_class=PlainTextResponse)
async def get_report(campaign_id: str) -> PlainTextResponse:
    report = orchestrator.get_report(campaign_id)
    if report is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    return PlainTextResponse(report, media_type="text/markdown; charset=utf-8")


@app.get("/api/events/{campaign_id}")
async def campaign_events(campaign_id: str, request: Request) -> StreamingResponse:
    try:
        orchestrator.event_snapshot(campaign_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="campaign not found") from exc
    try:
        after_id = int(request.headers.get("last-event-id", "0"))
    except ValueError:
        after_id = 0

    async def stream():
        cursor = max(0, after_id)
        while True:
            if await request.is_disconnected():
                break
            try:
                events, finished = orchestrator.event_snapshot(campaign_id, cursor)
            except KeyError:
                break
            for item in events:
                cursor = item["id"]
                yield f"id: {cursor}\nevent: {item['event']}\ndata: {json.dumps(item['data'], default=str)}\n\n"
            if finished and not events:
                yield "event: stream.complete\ndata: {}\n\n"
                break
            if not events:
                yield ": keep-alive\n\n"
                await asyncio.sleep(0.35)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"
    })

