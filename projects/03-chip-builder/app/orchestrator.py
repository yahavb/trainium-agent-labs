"""Bounded, checker-driven design campaign orchestration."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import re
import time
import uuid
from typing import Any

from .llm_client import LLMClient, ModelUnavailableError, ProposalError
from .report import render_campaign_report
from .scenarios import OBJECTIVES, get_scenario


MIN_IMPROVEMENT = 0.01


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return value


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


class CampaignOrchestrator:
    """Owns in-process campaign state and delegates simulation/persistence."""

    def __init__(self, simulator: Any = None, storage: Any = None, proposer: Any = None) -> None:
        self.simulator = simulator or self._make_simulator()
        self.storage = storage or self._make_storage()
        self.proposer = proposer or LLMClient()
        self.baselines: dict[str, dict[str, Any]] = {}
        self.calibrations: dict[str, dict[str, Any]] = {}
        self.campaigns: dict[str, dict[str, Any]] = {}
        self.designs: dict[str, dict[str, Any]] = {}
        self._live_traces: dict[str, Any] = {}

    @staticmethod
    def _make_simulator() -> Any:
        try:
            from .simulator import DigitalTwinSimulator
            return DigitalTwinSimulator()
        except (ImportError, AttributeError):
            return None

    @staticmethod
    def _make_storage() -> Any:
        try:
            from .storage import Storage
            return Storage()
        except (ImportError, AttributeError):
            return None

    def _models(self) -> tuple[Any, Any]:
        try:
            from .models import HardwareDesign
            from .simulator import fixed_trace
            return HardwareDesign, fixed_trace
        except (ImportError, AttributeError):
            return None, None

    def _build_design(self, data: dict[str, Any]) -> Any:
        model, _ = self._models()
        if model is None:
            return dict(data)
        allowed = set(getattr(model, "model_fields", getattr(model, "__fields__", {})))
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"design contains unsupported fields: {', '.join(sorted(unknown))}")
        if hasattr(model, "model_validate"):
            return model.model_validate(data)
        return model.parse_obj(data)

    @staticmethod
    def _topology_errors(design: Any, metrics: Any = None) -> list[str]:
        """Return structural errors from both the design model and checker."""
        errors: list[str] = []
        validator = getattr(design, "validate_topology", None)
        if callable(validator):
            try:
                errors.extend(str(error) for error in (validator() or []))
            except Exception as exc:
                errors.append(f"topology validation failed: {exc}")
        metric_errors = _get(metrics, "topology_errors", "topology_violations", default=[])
        if isinstance(metric_errors, str):
            metric_errors = [metric_errors]
        if isinstance(metric_errors, (list, tuple)):
            errors.extend(str(error) for error in metric_errors)
        return list(dict.fromkeys(errors))

    @staticmethod
    def _has_topology(design: Any, metrics: Any = None) -> bool:
        return (_get(design, "components") is not None or _get(design, "connections") is not None
                or _get(metrics, "topology_valid") is not None)

    @staticmethod
    def _design_diff(previous: Any, current: Any) -> dict[str, Any]:
        """Explain what physically changed between two complete design graphs."""
        before, after = _plain(previous) or {}, _plain(current) or {}
        keys = set(before) | set(after)
        scalar_changes = {key: {"from": before.get(key), "to": after.get(key)}
                          for key in sorted(keys - {"components", "connections"})
                          if before.get(key) != after.get(key)}
        def indexed(items: Any) -> dict[str, Any]:
            return {str(item.get("id")): item for item in items or [] if isinstance(item, dict) and item.get("id") is not None}
        old_components, new_components = indexed(before.get("components")), indexed(after.get("components"))
        old_connections, new_connections = indexed(before.get("connections")), indexed(after.get("connections"))
        def changes(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
            return {"added": [new[key] for key in sorted(new.keys() - old.keys())],
                    "removed": [old[key] for key in sorted(old.keys() - new.keys())],
                    "modified": [{"id": key, "from": old[key], "to": new[key]}
                                 for key in sorted(old.keys() & new.keys()) if old[key] != new[key]]}
        return {"scalars": scalar_changes, "components": changes(old_components, new_components),
                "connections": changes(old_connections, new_connections)}

    _DELTA_LABELS = {
        "compute_cores": ("Compute cores", ""),
        "peak_compute_tflops": ("Compute", " TFLOPS"),
        "hbm_gb": ("HBM capacity", " GB"),
        "hbm_bandwidth_tb_s": ("HBM bandwidth", " TB/s"),
        "sbuf_gb": ("On-chip SRAM", " GB"),
        "dma_engines": ("DMA engines", ""),
        "utilization": ("Utilization", ""),
        "power_limit_w": ("Power limit", " W"),
        "cooling_capacity_w": ("Cooling", " W"),
    }

    @classmethod
    def _change_summary(cls, previous: Any, current: Any) -> tuple[list[dict[str, Any]], list[str]]:
        """Parameter deltas (largest first) plus the IDs of blocks that were added, moved, or resized."""
        before, after = _plain(previous) or {}, _plain(current) or {}
        fmt = lambda value: f"{value:.3g}" if isinstance(value, float) else str(value)
        deltas: list[dict[str, Any]] = []

        def add(label: str, a: Any, b: Any, unit: str = "", kind: str = "resource") -> None:
            pct = (float(b) - float(a)) / float(a) * 100.0 if isinstance(a, (int, float)) and a else None
            deltas.append({"label": label, "before": a, "after": b, "unit": unit.strip(), "pct": pct, "kind": kind,
                           "text": f"{label} {fmt(a)} → {fmt(b)}{unit}"})

        for key, (label, unit) in cls._DELTA_LABELS.items():
            a, b = before.get(key), after.get(key)
            if a is not None and b is not None and a != b:
                add(label, a, b, unit)
        old = {c.get("id"): c for c in before.get("components") or [] if isinstance(c, dict)}
        new = {c.get("id"): c for c in after.get("components") or [] if isinstance(c, dict)}
        for kind, label in (("compute_cluster", "Compute clusters"), ("hbm", "HBM stacks")):
            a = sum(1 for c in old.values() if c.get("type") == kind)
            b = sum(1 for c in new.values() if c.get("type") == kind)
            if a != b:
                add(label, a, b, kind="structure")
        changed = [cid for cid, comp in new.items() if cid not in old or any(
            abs(float(comp.get(k) or 0) - float(old[cid].get(k) or 0)) > 0.005
            for k in ("x", "y", "width", "height"))]
        deltas.sort(key=lambda d: -abs(d["pct"] or 0))
        moved = [cid for cid in changed if cid in old]
        if moved:
            deltas.append({"label": "Layout", "before": None, "after": None, "unit": "", "pct": None, "kind": "layout",
                           "text": f"Moved or resized {len(moved)} block{'s' if len(moved) != 1 else ''}"})
        a, b = len(before.get("connections") or []), len(after.get("connections") or [])
        if a != b:
            add("Links", a, b, kind="layout")
        return deltas, changed

    @staticmethod
    def _previous_boxes(previous: Any, ids: list[str]) -> dict[str, list[float]]:
        """Where changed blocks sat before, so the diagram can draw a ghost outline."""
        old = {c.get("id"): c for c in (_plain(previous) or {}).get("components") or [] if isinstance(c, dict)}
        return {cid: [old[cid].get(k, 0) for k in ("x", "y", "width", "height")] for cid in ids if cid in old}

    def _signals(self, campaign: dict[str, Any]) -> list[str]:
        """Plain-language measurements the planner sees every round."""
        telemetry = campaign.get("telemetry") or {}
        vllm = telemetry.get("vllm") or {}
        neuron = telemetry.get("neuron_monitor") or {}
        system = telemetry.get("system_trace") or {}
        base = campaign.get("baseline_metrics") or {}
        f = lambda v, d=0: "n/a" if v is None else f"{v:.{d}f}"
        lines = []
        if vllm.get("available"):
            lines.append(f"vLLM on Trainium (measured): TTFT p50 {f(vllm.get('ttft_p50_ms'))} ms, p99 {f(vllm.get('ttft_p99_ms'))} ms; "
                         f"queue {f(vllm.get('queue_ms'))} ms, prefill {f(vllm.get('prefill_ms'))} ms, "
                         f"{f(vllm.get('inter_token_ms'), 1)} ms per output token; {f(vllm.get('prompt_tokens_mean'))} prompt and "
                         f"{f(vllm.get('generation_tokens_mean'))} output tokens per request; KV cache peak {f(vllm.get('kv_cache_peak_pct'))}%.")
        if neuron.get("available"):
            lines.append(f"neuron-monitor (measured): NeuronCore utilization {f(neuron.get('core_utilization_pct'))}% "
                         f"(peak {f(neuron.get('core_utilization_peak_pct'))}%), {f(neuron.get('effective_tflops'), 1)} effective TFLOPS, "
                         f"{f(neuron.get('device_memory_gb'), 1)} GB device memory in use.")
        if system.get("available"):
            line = (f"Neuron Explorer system trace (measured during traffic): NeuronCores busy {f(system.get('device_busy_pct'))}% "
                    f"of the time, so serving is {'device-bound and faster hardware cuts latency directly' if (system.get('device_busy_pct') or 0) > 80 else 'partly host-bound'}; "
                    f"on-device prefill {f(system.get('prefill_step_ms'))} ms and decode step {f(system.get('decode_step_ms'), 1)} ms; "
                    f"collectives take {f(system.get('collective_pct'), 1)}% of execution"
                    f"{' (the chip interconnect is not the bottleneck)' if (system.get('collective_pct') or 100) < 5 else ''}; "
                    f"host copies {f(system.get('host_copy_pct'), 1)}%; {f(system.get('hbm_used_gb'), 1)} GB of HBM in use.")
            ai, ridge = system.get("arithmetic_intensity"), system.get("ridge_point")
            if ai is not None and ridge is not None:
                line += (f" Device profile: arithmetic intensity {f(ai, 1)} FLOP/B vs ridge {f(ridge)} → "
                         f"{'memory' if ai < ridge else 'compute'}-bound; HBM bandwidth utilization {f(system.get('mbu_pct'))}%.")
            lines.append(line)
        cal = campaign.get("calibration") or {}
        if cal.get("prefill_overhead_ms"):
            lines.append(f"Calibration: {f(cal['prefill_overhead_ms'])} ms of each prefill is host-side overhead that no chip change removes; "
                         f"only the on-device part scales with hardware. Memory fit uses the measured {f(cal.get('memory_footprint_gb'), 1)} GB footprint.")
        if not campaign.get("baseline_feasible", True):
            failed = [row["label"] for row in campaign.get("baseline_checks") or [] if row.get("passed") is False]
            lines.append("Today's chip does not meet this scenario's budget (" + ", ".join(failed)
                         + "); the first design that passes every check wins, then later ones must beat it.")
        if base.get("prefill_bound"):
            lines.append(f"Simulator on the baseline: prefill is {base['prefill_bound']}-bound, decode is {base['decode_bound']}-bound.")
        return lines

    def _default_design(self) -> Any:
        model, _ = self._models()
        if model is None:
            return {}
        baseline = getattr(model, "baseline", None)
        return baseline() if callable(baseline) else model()

    def _trace(self, trace: Any) -> Any:
        if trace is None:
            return "A"
        model, fixed_trace = self._models()
        if fixed_trace and isinstance(trace, str):
            try:
                return fixed_trace(trace)
            except (ValueError, KeyError):
                return trace
        return trace

    async def _evaluate(self, design: Any, trace: Any) -> Any:
        if self.simulator is None:
            raise RuntimeError("DigitalTwinSimulator is unavailable")
        fn = getattr(self.simulator, "evaluate", None) or getattr(self.simulator, "run", None)
        if fn is None:
            raise RuntimeError("simulator must provide evaluate(design, trace)")
        if inspect.iscoroutinefunction(fn):
            return await fn(design, trace)
        return await asyncio.to_thread(fn, design, trace)

    def _save_result(self, result: Any, design: Any) -> str:
        if self.storage is not None and callable(getattr(self.storage, "create", None)):
            try:
                stored = self.storage.create(result, design=design)
                if isinstance(stored, str):
                    return stored
                return str(_get(stored, "id", "run_id", default=uuid.uuid4()))
            except TypeError:
                stored = self.storage.create(result, design)
                if isinstance(stored, str):
                    return stored
                return str(_get(stored, "id", "run_id", default=uuid.uuid4()))
        return str(uuid.uuid4())

    async def run_baseline(self, request: dict[str, Any]) -> dict[str, Any]:
        design_data = request.get("design")
        design = self._build_design(design_data) if design_data else self._default_design()
        trace = self._trace(request.get("trace", request.get("trace_name", "A")))
        result = await self._evaluate(design, trace)
        measured: dict[str, Any] = {}
        measurement_source = "modeled"
        if os.getenv("ACCELTWIN_MOCK_TELEMETRY", "false").lower() not in {"1", "true", "yes"}:
            try:
                from .telemetry import NeuronMonitorAdapter, collect_vllm_metrics
                metrics_url = os.getenv("ACCELTWIN_VLLM_METRICS_URL", "http://127.0.0.1:8000/metrics")
                vllm = collect_vllm_metrics(metrics_url)
                neuron = NeuronMonitorAdapter(mock_fallback=False).collect()
                if vllm or neuron:
                    measured = {"vllm": vllm, "neuron": neuron}
                    measurement_source = "measured+modeled"
            except Exception:
                # Disconnected demos retain the deterministic simulator baseline.
                pass
        run_id = self._save_result(result, design)
        record = {
            "id": run_id,
            "design": _plain(design),
            "trace": _plain(trace),
            "metrics": _plain(result),
            "measured_telemetry": measured,
            "measurement_source": measurement_source,
            "created_at": time.time(),
        }
        self.baselines[run_id] = record
        return record

    def get_baseline(self, baseline_id: str) -> dict[str, Any] | None:
        if baseline_id in self.baselines:
            return self.baselines[baseline_id]
        stored = self._storage_get(baseline_id)
        return stored

    def create_calibration(self, request: dict[str, Any]) -> dict[str, Any]:
        baseline_id = request.get("baseline_id")
        baseline = self.get_baseline(baseline_id) if baseline_id else None
        if baseline_id and baseline is None:
            raise KeyError("baseline not found")
        record = {
            "id": str(uuid.uuid4()),
            "baseline_id": baseline_id,
            "observed_metrics": request.get("observed_metrics", request.get("metrics", {})),
            "source": request.get("source", "manual"),
            "notes": request.get("notes"),
            "created_at": time.time(),
        }
        self.calibrations[record["id"]] = record
        return record

    async def create_campaign(self, request: dict[str, Any]) -> dict[str, Any]:
        check = getattr(self.proposer, "check", None)
        if callable(check):
            status = await asyncio.to_thread(check)
            if not status.get("ready", True):
                raise ModelUnavailableError(status.get("error") or "the model is not available")
        constraints = request.get("constraints", {})
        if not isinstance(constraints, dict):
            raise ValueError("constraints must be a JSON object")
        # A scenario supplies the budget, objective and workload shape; explicit
        # request fields still win. Legacy baseline_id runs only use one if asked.
        scenario = get_scenario(request.get("scenario")) if request.get("scenario") or not request.get("baseline_id") else None
        constraints = {**(scenario["constraints"] if scenario else {}), **constraints}
        for cap in ("max_power_w", "max_temperature_c", "max_area_mm2"):
            if cap in request and cap not in constraints:
                constraints[cap] = request[cap]
            if cap in constraints:
                try:
                    constraints[cap] = float(constraints[cap])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{cap} must be a non-negative number") from exc
                if constraints[cap] < 0:
                    raise ValueError(f"{cap} must be a non-negative number")
        rounds = min(5, max(1, int(request.get("max_rounds", 5))))
        try:
            patience = max(1, int(request.get("early_stop_patience", 2)))
        except (TypeError, ValueError) as exc:
            raise ValueError("early_stop_patience must be an integer") from exc
        baseline_id = request.get("baseline_id")
        if not baseline_id:
            # Measure the live workload in the background, then calibrate and run.
            campaign = self._new_campaign(request, rounds, patience, constraints, scenario)
            self.campaigns[campaign["id"]] = campaign
            self._emit(campaign, "campaign.started", {"max_rounds": rounds, "trace": "live"})
            asyncio.create_task(self._prepare_and_run(campaign, request))
            return campaign
        baseline = self.get_baseline(baseline_id)
        if baseline is None:
            raise KeyError("baseline not found")
        # Re-evaluate the unchanged baseline hardware on the campaign trace so
        # candidates are never compared against a different workload.
        campaign_trace = request.get("trace", "C")
        baseline_design = self._build_design(baseline.get("design", {}))
        campaign_baseline_result = await self._evaluate(baseline_design, self._trace(campaign_trace))
        campaign_baseline_metrics = _plain(campaign_baseline_result)
        campaign_id = str(uuid.uuid4())
        campaign = {
            "id": campaign_id,
            "status": "queued",
            "baseline_id": baseline_id,
            "baseline_design": baseline.get("design", {}),
            "baseline_metrics": campaign_baseline_metrics,
            "source_baseline_metrics": baseline.get("metrics", {}),
            "trace": campaign_trace,
            "goal": request.get("goal", "Minimize p99 TTFT without reducing throughput"),
            "max_rounds": rounds,
            "early_stop_patience": patience,
            "constraints": constraints,
            "attempts": [],
            "events": [],
            "best_attempt": None,
            "latest_design": baseline.get("design", {}),
            "activity": "Starting",
            "created_at": time.time(),
        }
        self.campaigns[campaign_id] = campaign
        self._emit(campaign, "campaign.started", {"max_rounds": rounds, "baseline_id": baseline_id, "trace": campaign_trace})
        asyncio.create_task(self._run_campaign(campaign, request))
        return campaign

    @staticmethod
    def _new_campaign(request: dict[str, Any], rounds: int, patience: int, constraints: dict[str, Any],
                      scenario: dict[str, Any] | None) -> dict[str, Any]:
        scenario = scenario or get_scenario(None)
        return {
            "id": str(uuid.uuid4()), "status": "measuring", "baseline_id": None,
            "baseline_design": None, "baseline_metrics": None, "trace": "live",
            "scenario": {k: scenario[k] for k in ("id", "title", "problem")},
            "objective": OBJECTIVES[request.get("objective") or scenario["objective"]],
            "workload": dict(scenario["workload"]),
            "live": {"phase": "measure", "requests": [], "samples": [], "tokens": 0, "elapsed_s": 0.0, "concurrency": 0},
            "calls": [],
            "goal": request.get("goal") or scenario["goal"],
            "max_rounds": rounds, "early_stop_patience": patience, "constraints": constraints,
            "attempts": [], "events": [], "best_attempt": None, "latest_design": None,
            "telemetry": None, "calibration": None, "activity": "Measuring the live workload",
            "created_at": time.time(),
        }

    async def _prepare_and_run(self, campaign: dict[str, Any], request: dict[str, Any]) -> None:
        try:
            self._activity(campaign, "Measuring the live workload on Trainium")
            telemetry = await asyncio.to_thread(self._measure, campaign.get("live"))
            self._activity(campaign, "Calibrating the simulator to the measurements")
            calibration, trace, design = self._calibrate(telemetry, campaign.get("workload"))
            result = await self._evaluate(design, trace)
        except Exception as exc:
            campaign["status"] = "failed"
            campaign["error"] = f"Could not measure or calibrate the baseline: {exc}"
            self._emit(campaign, "campaign.failed", {"error": campaign["error"]})
            return
        self._live_traces[campaign["id"]] = trace
        design_data, metrics = _plain(design), _plain(result)
        self._apply_guards(campaign["constraints"], metrics)
        campaign.update(telemetry=telemetry, calibration=calibration, baseline_design=design_data,
                        baseline_metrics=metrics, latest_design=design_data,
                        baseline_checks=self._checks(metrics, design_data, metrics.get("throughput_tokens_s"),
                                                     campaign["constraints"]))
        if campaign.get("live"):
            campaign["live"]["phase"] = "design"
        self._emit(campaign, "baseline.measured", {"calibration": calibration})
        await self._run_campaign(campaign, request)

    @staticmethod
    def _apply_guards(constraints: dict[str, Any], baseline_metrics: Any) -> None:
        """Turn relative guards (e.g. 'no worse than today') into absolute limits."""
        ceiling = constraints.get("latency_ceiling")
        p99 = _get(baseline_metrics, "ttft_p99_ms")
        if ceiling is not None and p99 is not None and "max_p99_ttft_ms" not in constraints:
            constraints["max_p99_ttft_ms"] = round(float(p99) * float(ceiling), 3)

    def _measure(self, live: dict[str, Any] | None = None) -> dict[str, Any]:
        """Replay a burst on the live server; every source degrades to a stated reason."""
        from .telemetry import measure_live_workload, read_system_trace

        if os.getenv("ACCELTWIN_MOCK_TELEMETRY", "false").lower() in {"1", "true", "yes"} \
                or getattr(self.proposer, "mode", "") in {"mock", "rule", "rules", "offline"}:
            reason = "offline mode (ACCELTWIN_MOCK_TELEMETRY or ACCELTWIN_MOCK_LLM)"
            return {"vllm": {"available": False, "reason": reason},
                    "neuron_monitor": {"available": False, "reason": reason},
                    "system_trace": {"available": False, "reason": reason}}
        base_url = getattr(self.proposer, "base_url", None) or os.getenv("ACCELTWIN_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
        try:
            return measure_live_workload(
                base_url, os.getenv("ACCELTWIN_VLLM_METRICS_URL", "http://127.0.0.1:8000/metrics"),
                getattr(self.proposer, "model", None) or "", requests=int(os.getenv("ACCELTWIN_REPLAY_REQUESTS", "8")),
                concurrency=int(os.getenv("ACCELTWIN_REPLAY_CONCURRENCY", "4")), live=live)
        except Exception as exc:
            return {"vllm": {"available": False, "reason": f"{exc.__class__.__name__}: {exc}"[:200]},
                    "neuron_monitor": {"available": False, "reason": "not sampled"},
                    "system_trace": read_system_trace()}

    def _calibrate(self, telemetry: dict[str, Any], workload: dict[str, Any] | None = None) -> tuple[dict[str, Any], Any, Any]:
        """Fit the simulator to the measurements and build the scenario's evaluation trace from them."""
        workload = {"prompt_scale": 1.0, "output_scale": 1.0, "load": 0.8, **(workload or {})}
        from .models import RequestSpec, RequestTrace

        vllm = telemetry.get("vllm") or {}
        system = telemetry.get("system_trace") or {}
        design = self._default_design()
        live = bool(vllm.get("available"))
        weights_gb = vllm.get("model_weights_gb")
        if weights_gb and hasattr(design, "model_copy"):
            design = design.model_copy(update={"model_size_b": round(weights_gb / design.weight_bytes_per_parameter, 3)})
        spill = system.get("spill_fraction") if system.get("available") else None
        calibration = {"live": live, "prefill_scale": 1.0, "decode_scale": 1.0,
                       "spill_fraction": spill if spill is not None else 0.05,
                       "spill_source": "system trace" if spill is not None else "default",
                       "prompt_tokens": round(vllm.get("prompt_tokens_mean") or 512),
                       "generation_tokens": round(vllm.get("generation_tokens_mean") or 128),
                       "model_weights_gb": weights_gb, "trace": "live" if live else "C"}
        sim = self.simulator
        if not callable(getattr(sim, "_roofline", None)):
            return calibration, self._trace("C"), design

        clamp = lambda value: float(min(100.0, max(0.02, value)))
        sim.calibration = {"prefill_scale": 1.0, "decode_scale": 1.0, "spill_fraction": calibration["spill_fraction"]}
        sim.design, sim._active_topology = design, None
        sim._active_topology = sim._topology_profile()
        modeled_prefill = max(sim._roofline(calibration["prompt_tokens"]))
        modeled_decode = max(sim._roofline(None))
        # With a system trace, split each measured time into on-device time (which the
        # roofline scales) and host overhead (which no chip change removes).
        trace = system if system.get("available") else {}
        neuron = telemetry.get("neuron_monitor") or {}
        calibration.update(prefill_overhead_ms=0.0, decode_overhead_ms=0.0, collective_share=0.0,
                           baseline_compute_link_bw=round(sim._active_topology.get("compute_link_bw", 0.0), 4),
                           memory_footprint_gb=None, split_source="none")
        if live and vllm.get("prefill_ms"):
            device = min(float(trace.get("prefill_step_ms") or vllm["prefill_ms"]), vllm["prefill_ms"])
            calibration["prefill_overhead_ms"] = round(vllm["prefill_ms"] - device, 3)
            calibration["prefill_scale"] = round(clamp(device / modeled_prefill), 4)
        if live and vllm.get("inter_token_ms"):
            device = min(float(trace.get("decode_step_ms") or vllm["inter_token_ms"]), vllm["inter_token_ms"])
            calibration["decode_overhead_ms"] = round(vllm["inter_token_ms"] - device, 3)
            calibration["decode_scale"] = round(clamp(device / modeled_decode), 4)
        if trace:
            calibration["split_source"] = "system trace"
            calibration["collective_share"] = round(float(trace.get("collective_pct") or 0.0) / 100.0, 4)
            calibration["device_busy_pct"] = trace.get("device_busy_pct")
        footprint = trace.get("hbm_used_gb") or (neuron.get("device_memory_gb") if neuron.get("available") else None)
        if live and footprint:
            calibration["memory_footprint_gb"] = round(float(footprint), 2)
            calibration["memory_source"] = "system trace" if trace.get("hbm_used_gb") else "neuron-monitor"
        sim.calibration = {k: calibration[k] for k in ("prefill_scale", "decode_scale", "spill_fraction", "prefill_overhead_ms",
                                                       "decode_overhead_ms", "collective_share", "baseline_compute_link_bw",
                                                       "memory_footprint_gb")}
        # Reshape the measured workload for the scenario (e.g. 4x longer prompts).
        calibration["measured_prompt_tokens"] = calibration["prompt_tokens"]
        calibration["measured_generation_tokens"] = calibration["generation_tokens"]
        calibration["prompt_tokens"] = max(1, round(calibration["prompt_tokens"] * float(workload["prompt_scale"])))
        calibration["generation_tokens"] = max(1, round(calibration["generation_tokens"] * float(workload["output_scale"])))
        modeled_prefill = max(sim._roofline(calibration["prompt_tokens"]))
        calibration["trace"] = "live" if live else "default"
        calibration["load"] = float(workload["load"])
        # Bursts the size of the measured replay concurrency, spaced so the baseline
        # runs at the scenario's load (0.8 by default): requests queue inside each
        # burst, so p99 TTFT scales smoothly with service time instead of sitting on
        # an overload cliff. Loads above 1 model a saturated, throughput-bound chip.
        service_ms = (calibration["prefill_overhead_ms"] + modeled_prefill * calibration["prefill_scale"]
                      + calibration["generation_tokens"] * (calibration["decode_overhead_ms"] + modeled_decode * calibration["decode_scale"]))
        burst = max(1, int(vllm.get("concurrency") or 4))
        interval = round(burst * service_ms / max(float(workload["load"]), 0.05), 3)
        mix = (0.5, 1.0, 1.5, 1.0, 0.75, 1.25)
        trace = RequestTrace(name=calibration["trace"], description="Bursts shaped from the measured vLLM workload", requests=[
            RequestSpec(request_id=f"live-{i + 1:03d}", prompt_tokens=max(1, round(calibration["prompt_tokens"] * mix[i % 6])),
                        output_tokens=calibration["generation_tokens"], arrival_ms=(i // burst) * interval) for i in range(32)])
        calibration.update(arrival_interval_ms=interval, burst_size=burst)
        return calibration, trace, design

    _CHECK_LABELS = {"throughput_at_least_baseline": "Throughput ≥ baseline", "memory_fit": "Model fits in HBM",
                     "latency_guard": "p99 TTFT no worse than today",
                     "topology_valid": "Valid layout & wiring", "power_cap": "Power", "temperature_cap": "Peak temperature",
                     "area_cap": "Die area"}

    @classmethod
    def _checks(cls, metrics: Any, design: Any, baseline_throughput: float | None,
                constraints: dict[str, Any]) -> list[dict[str, Any]]:
        """The checker's gates as value-vs-limit rows the UI can draw."""
        m = _plain(metrics) or {}
        d = _plain(design) or {}
        rows = []
        if baseline_throughput is not None:
            rows.append({"key": "throughput_at_least_baseline", "label": "Throughput", "value": m.get("throughput_tokens_s"),
                         "limit": round(baseline_throughput * float(constraints.get("throughput_floor", 1.0)), 3),
                         "unit": "tok/s", "op": "≥"})
        if "max_p99_ttft_ms" in constraints:
            rows.append({"key": "latency_guard", "label": "p99 TTFT", "value": m.get("ttft_p99_ms"),
                         "limit": constraints["max_p99_ttft_ms"], "unit": "ms", "op": "≤"})
        rows.append({"key": "memory_fit", "label": "Memory", "value": m.get("peak_memory_gb"), "limit": d.get("hbm_gb"),
                     "unit": "GB", "op": "≤"})
        for key, metric, cap, label, unit in (("power_cap", "power_w", "max_power_w", "Power", "W"),
                                              ("temperature_cap", "max_temperature_c", "max_temperature_c", "Temperature", "°C"),
                                              ("area_cap", "area_mm2", "max_area_mm2", "Area", "mm²")):
            if cap in constraints:
                rows.append({"key": key, "label": label, "value": m.get(metric), "limit": constraints[cap], "unit": unit, "op": "≤"})
        for row in rows:
            value, limit = row["value"], row["limit"]
            row["passed"] = (value is not None and limit is not None
                             and (value >= limit - 1e-9 if row["op"] == "≥" else value <= limit + 1e-9))
        if m.get("topology_valid") is False:
            rows.insert(0, {"key": "topology_valid", "label": "Layout", "value": None, "limit": None, "unit": "",
                            "op": "", "passed": False})
        return rows

    async def _run_campaign(self, campaign: dict[str, Any], request: dict[str, Any]) -> None:
        campaign["status"] = "running"
        baseline_metrics = campaign["baseline_metrics"]
        baseline_throughput = self._metric(baseline_metrics, "throughput_tokens_s", "throughput_tokens_per_sec")
        objective = campaign.get("objective") or OBJECTIVES["ttft"]
        campaign.setdefault("objective", objective)
        self._apply_guards(campaign["constraints"], baseline_metrics)
        # An infeasible starting chip (e.g. over a tighter budget) is not a bar to
        # beat: the first design that passes every check becomes the best.
        baseline_gates = self._gate_results(baseline_metrics, baseline_throughput, campaign["constraints"])
        campaign["baseline_feasible"] = all(baseline_gates.values())
        best_p99 = self._metric(baseline_metrics, objective["metric"]) if campaign["baseline_feasible"] else None
        best_design = campaign["baseline_design"]
        best_metrics = baseline_metrics
        history: list[dict[str, Any]] = []
        seen: set[str] = set()
        stale_rounds = 0
        design_schema = self._design_schema()
        for round_no in range(1, campaign["max_rounds"] + 1):
            prompt = self._prompt(campaign, round_no, best_design, best_p99)
            context = {
                "round": round_no - 1,
                "max_rounds": campaign["max_rounds"],
                "goal": campaign["goal"],
                "constraints": campaign["constraints"],
                "baseline_config": best_design,
                "baseline_metrics": baseline_metrics,
                "best_metrics": best_metrics,
                "history": history,
                "feedback": campaign["attempts"][-1].get("feedback") if campaign["attempts"] else None,
                "current_design": campaign.get("latest_design", best_design),
                "baseline_design": campaign.get("baseline_design", {}),
                "report": lambda text: self._activity(campaign, text),
                "signals": self._signals(campaign),
                "objective": objective,
                "workload": campaign.get("calibration"),
                "log_call": lambda entry: self._log_call(campaign, entry),
            }
            raw: dict[str, Any] = {}
            if callable(getattr(self.proposer, "design_round", None)):
                # Staged multi-call design runs in a worker thread so the API
                # keeps serving progress while the model works.
                try:
                    raw = await asyncio.to_thread(self.proposer.design_round, context)
                    design = self._build_design(self._extract_design(raw))
                except ModelUnavailableError as exc:
                    campaign["status"] = "failed"
                    campaign["error"] = f"Model unavailable: {exc}"
                    self._emit(campaign, "campaign.failed", {"error": campaign["error"]})
                    return
                except Exception as exc:
                    attempt = {"id": str(uuid.uuid4()), "round": round_no, "name": "Proposal failed",
                               "accepted": False, "feedback": f"The model could not produce a design: {exc}"}
                    campaign["attempts"].append(attempt)
                    history.append({"round": round_no, "name": attempt["name"], "outcome": "no valid design"})
                    self._emit(campaign, "proposal.invalid", {"round": round_no, "error": str(exc)})
                    stale_rounds += 1
                    if stale_rounds >= campaign["early_stop_patience"]:
                        campaign["stop_reason"] = "no accepted improvement"
                        break
                    continue
            else:
                try:
                    raw = self.proposer.propose(prompt, design_schema, context)
                    design = self._extract_design(raw)
                    design = self._build_design(design)
                except Exception as first_error:
                    self._emit(campaign, "proposal.invalid", {"round": round_no, "error": str(first_error)})
                    # Exactly one repair attempt. If the local model is unavailable in
                    # auto mode, its deterministic fallback keeps the demo runnable.
                    try:
                        raw = self.proposer.repair(prompt, json.dumps(raw), str(first_error), design_schema)
                        design = self._build_design(self._extract_design(raw))
                    except Exception as repair_error:
                        if getattr(self.proposer, "mode", "") in {"mock", "rule", "rules", "offline"}:
                            raw = self.proposer._rule_proposal(context)
                            design = self._build_design(self._extract_design(raw))
                        else:
                            campaign["status"] = "failed"
                            campaign["error"] = f"proposal failed after one repair: {repair_error}"
                            self._emit(campaign, "campaign.failed", {"error": campaign["error"]})
                            return

            design_data = _plain(design)
            # Canonical full-design serialization includes all component positions,
            # dimensions, layers, and links, so a topology edit is never collapsed
            # into a duplicate scalar configuration.
            fingerprint = json.dumps(design_data, sort_keys=True, separators=(",", ":"), default=str)
            if fingerprint in seen or fingerprint == json.dumps(best_design, sort_keys=True, separators=(",", ":"), default=str):
                attempt = {
                    "id": str(uuid.uuid4()), "round": round_no, "name": _get(raw, "name", default=f"candidate {round_no}"),
                    "design": design_data, "accepted": False, "duplicate": True,
                    "feedback": "Duplicate design rejected without simulation.",
                }
                campaign["attempts"].append(attempt)
                self.designs[attempt["id"]] = attempt
                self._emit(campaign, "design.rejected", attempt)
                campaign["stop_reason"] = "duplicate design"
                break
            seen.add(fingerprint)
            previous_design = campaign.get("latest_design", campaign.get("baseline_design", {}))
            design_diff = self._design_diff(previous_design, design_data)
            deltas, changed_ids = self._change_summary(best_design, design_data)
            rationale = _get(raw, "rationale", default="") or ""
            changes = [str(item) for item in _get(raw, "changes", default=None) or [] if str(item).strip()]
            if not changes and rationale:
                changes = [rationale]
            campaign["latest_design"] = design_data
            self._emit(campaign, "design.proposed", {
                "round": round_no,
                "name": _get(raw, "name", default=f"candidate {round_no}"),
                "rationale": _get(raw, "rationale", default=""),
                "design": design_data,
                "design_diff": design_diff,
            })
            self._activity(campaign, f"Round {round_no}: simulating the design")
            try:
                result = await self._evaluate(design, self._live_traces.get(campaign["id"]) or self._trace(campaign["trace"]))
                metrics = _plain(result)
            except Exception as exc:
                attempt = {"id": str(uuid.uuid4()), "round": round_no, "design": design_data,
                           "accepted": False, "design_diff": design_diff,
                           "feedback": f"Simulation failed: {exc}"}
                campaign["attempts"].append(attempt)
                self.designs[attempt["id"]] = attempt
                self._emit(campaign, "design.evaluated", attempt)
                continue

            topology_errors = self._topology_errors(design, metrics)
            gates = self._gate_results(metrics, baseline_throughput, campaign["constraints"],
                                       topology_required=self._has_topology(design, metrics),
                                       topology_errors=topology_errors)
            p99 = self._metric(metrics, "ttft_p99_ms", "predicted_p99_ttft_ms")
            score = self._metric(metrics, objective["metric"])
            accepted = all(gates.values())
            # Require a real gain so simulator noise never counts as progress.
            improved = accepted and score is not None and (best_p99 is None or (
                score < best_p99 * (1 - MIN_IMPROVEMENT) if objective["direction"] == "min"
                else score > best_p99 * (1 + MIN_IMPROVEMENT)))
            attempt_id = self._save_result(result, design)
            attempt = {
                "id": attempt_id, "round": round_no,
                "name": _get(raw, "name", default=f"candidate {round_no}"),
                "rationale": _get(raw, "rationale", default=""),
                "design": design_data, "metrics": metrics, "gate_results": gates,
                "accepted": accepted, "improved": improved,
                "changes": changes, "deltas": deltas, "changed_ids": changed_ids,
                "source": _get(raw, "source", default="model"), "steps": _get(raw, "steps", default=[]),
                "topology_errors": topology_errors,
                "design_diff": design_diff,
                "baseline_design_diff": self._design_diff(campaign.get("baseline_design", {}), design_data),
                "feedback": self._feedback(gates, metrics, baseline_throughput, best_p99, topology_errors,
                                           design=design_data, constraints=campaign["constraints"], objective=objective),
                "checks": self._checks(metrics, design_data, baseline_throughput, campaign["constraints"]),
                "previous_boxes": self._previous_boxes(best_design, changed_ids),
                "baseline_deltas": self._change_summary(campaign.get("baseline_design") or {}, design_data)[0],
            }
            campaign["attempts"].append(attempt)
            self.designs[attempt_id] = attempt
            self._emit(campaign, "design.evaluated", attempt)
            failed = [gate for gate, passed in gates.items() if not passed]
            history.append({"round": round_no, "name": attempt["name"], "changes": changes,
                            "outcome": (f"{objective['short']} {score:g} {objective['unit']}, " if score is not None else "")
                            + ("rejected: " + ", ".join(failed) if failed
                               else "new best" if improved else "passed but not better")})
            if improved:
                best_p99 = score
                best_design = design_data
                best_metrics = metrics
                campaign["best_attempt"] = attempt
                stale_rounds = 0
                self._emit(campaign, "campaign.best_updated", {"attempt_id": attempt_id, objective["metric"]: score,
                            "design": design_data, "design_diff": attempt["design_diff"],
                            "baseline_design_diff": attempt["baseline_design_diff"]})
            else:
                stale_rounds += 1
            target = campaign["constraints"].get("target_p99_ttft_ms")
            if target is not None and best_p99 is not None and best_p99 <= float(target):
                campaign["stop_reason"] = "target p99 TTFT reached"
                break
            if stale_rounds >= campaign["early_stop_patience"]:
                campaign["stop_reason"] = "no accepted improvement"
                break
        campaign["status"] = "completed"
        campaign["activity"] = "Done"
        campaign["completed_at"] = time.time()
        campaign["report"] = render_campaign_report(campaign)
        self._emit(campaign, "campaign.completed", {"stop_reason": campaign.get("stop_reason"), "rounds": len(campaign["attempts"])})

    @staticmethod
    def _extract_design(proposal: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(proposal, dict):
            raise ProposalError("proposal must be an object")
        extra = set(proposal) - {"name", "rationale", "design", "changes", "steps", "source"}
        if extra:
            raise ProposalError(f"proposal contains unexpected fields: {', '.join(sorted(extra))}")
        if "name" in proposal and not isinstance(proposal["name"], str):
            raise ProposalError("proposal name must be a string")
        if "rationale" in proposal and not isinstance(proposal["rationale"], str):
            raise ProposalError("proposal rationale must be a string")
        design = proposal.get("design")
        if not isinstance(design, dict) or not design:
            raise ProposalError("proposal must include a non-empty design object")
        return design

    def _design_schema(self) -> dict[str, Any]:
        model, _ = self._models()
        if model is not None:
            schema = model.model_json_schema() if hasattr(model, "model_json_schema") else model.schema()
            schema["additionalProperties"] = False
            return {"type": "object", "properties": {
                "name": {"type": "string"}, "rationale": {"type": "string"}, "design": schema
            }, "required": ["design"], "additionalProperties": False}
        return {"type": "object", "properties": {"name": {"type": "string"}, "rationale": {"type": "string"},
                "design": {"type": "object", "properties": {
                    "components": {"type": "array", "items": {"type": "object"}},
                    "connections": {"type": "array", "items": {"type": "object"}},
                }, "required": ["components", "connections"]}}, "required": ["design"], "additionalProperties": False}

    @staticmethod
    def _metric(metrics: Any, *names: str) -> float | None:
        value = _get(metrics, *names)
        try:
            return None if value is None else float(value)
        except (ValueError, TypeError):
            return None

    @classmethod
    def _gate_results(cls, metrics: Any, baseline_throughput: float | None, constraints: dict[str, Any],
                      topology_required: bool = False, topology_errors: list[str] | None = None) -> dict[str, bool]:
        throughput = cls._metric(metrics, "throughput_tokens_s", "throughput_tokens_per_sec")
        memory_fit = _get(metrics, "memory_fit", "memory_fits", default=False)
        power = cls._metric(metrics, "power_w", "power_watts")
        temp = cls._metric(metrics, "max_temperature_c", "temperature_c", "peak_temperature_c")
        if temp is None:
            grid = _get(metrics, "thermal_grid")
            if isinstance(grid, list):
                vals = [v for row in grid if isinstance(row, list) for v in row if isinstance(v, (int, float))]
                temp = max(vals) if vals else None
        area = cls._metric(metrics, "area_mm2", "area")
        floor = float(constraints.get("throughput_floor", 1.0))
        gates = {
            "throughput_at_least_baseline": throughput is not None and (baseline_throughput is None or throughput >= baseline_throughput * floor - 1e-9),
            "memory_fit": bool(memory_fit),
        }
        if "max_p99_ttft_ms" in constraints:
            p99 = cls._metric(metrics, "ttft_p99_ms", "predicted_p99_ttft_ms")
            gates["latency_guard"] = p99 is not None and p99 <= float(constraints["max_p99_ttft_ms"]) * (1 + 1e-9)
        if topology_required:
            reported_topology_valid = _get(metrics, "topology_valid", default=not topology_errors)
            gates["topology_valid"] = bool(reported_topology_valid) and not bool(topology_errors)
        for gate, metric, cap in (("power_cap", power, "max_power_w"), ("temperature_cap", temp, "max_temperature_c"), ("area_cap", area, "max_area_mm2")):
            if cap in constraints:
                gates[gate] = metric is not None and metric <= float(constraints[cap])
        return gates

    @classmethod
    def _diagnosis(cls, gates: dict[str, bool], metrics: Any, design: dict[str, Any],
                   constraints: dict[str, Any]) -> list[str]:
        """Name the limiting resource and the size of change that would fix each failure."""
        m = _plain(metrics) or {}
        out: list[str] = []
        u = float(design.get("utilization") or 0.72)
        tflops = float(design.get("peak_compute_tflops") or 0)
        if gates.get("power_cap") is False:
            cap = float(constraints["max_power_w"])
            budget = (cap - 85.0) / 420.0
            out.append(f"Power {m.get('power_w', 0):.0f} W (unthrottled {m.get('raw_power_w', 0):.0f} W) exceeds the {cap:.0f} W cap: "
                       f"at utilization {u:.2f} peak compute must be at most {2500.0 * budget / u:.0f} TFLOPS "
                       f"(now {tflops:.0f}), or utilization at most {budget / min(tflops / 2500.0, 1.8):.2f}.")
        if gates.get("area_cap") is False:
            cap = float(constraints["max_area_mm2"])
            over = float(m.get("area_mm2") or 0) - cap
            out.append(f"Area {m.get('area_mm2', 0):.0f} mm² is {over:.0f} mm² over the {cap:.0f} mm² cap; "
                       f"compute cores use {float(design.get('compute_cores') or 0) * 42:.0f} mm², HBM {float(design.get('hbm_gb') or 0) * 1.45:.0f} mm², "
                       f"SRAM {float(design.get('sbuf_gb') or 0) * 0.65:.0f} mm²: with everything else unchanged it fits at "
                       f"compute_cores <= {int(float(design.get('compute_cores') or 0) - math.ceil(over / 42))} "
                       f"or hbm_gb <= {math.floor(float(design.get('hbm_gb') or 0) - over / 1.45)}.")
        if gates.get("memory_fit") is False:
            need = float(m.get("peak_memory_gb") or 0)
            out.append(f"Weights + KV cache need {need:.1f} GB but HBM has {float(design.get('hbm_gb') or 0):g} GB: "
                       f"set hbm_gb >= {math.ceil(need)}.")
        usable, provisioned = m.get("usable_hbm_bandwidth_tb_s"), design.get("hbm_bandwidth_tb_s")
        if usable and provisioned and usable < 0.9 * float(provisioned):
            out.append(f"The links deliver only {float(usable):.2f} of the {float(provisioned):g} TB/s the HBM stacks provide: "
                       f"widen the HBM-to-compute links (directly or through the NoC) so the extra bandwidth reaches compute.")
        if m.get("decode_bound"):
            dc, dm = float(m.get("decode_compute_ms") or 0), float(m.get("decode_memory_ms") or 0)
            pc, pm = float(m.get("prefill_compute_ms") or 0), float(m.get("prefill_memory_ms") or 0)
            if m["decode_bound"] == "memory":
                out.append(f"Decode is memory-bound ({dm:.2f} ms of HBM traffic vs {dc:.2f} ms of compute per token): extra TFLOPS will not "
                           f"speed it up; raise HBM bandwidth, HBM-to-compute link capacity, or SRAM to cut spills.")
            else:
                out.append(f"Decode is compute-bound ({dc:.2f} ms compute vs {dm:.2f} ms memory per token): raise TFLOPS or utilization.")
            out.append(f"Prefill is {m.get('prefill_bound')}-bound ({pc:.1f} ms compute vs {pm:.1f} ms memory for a typical prompt).")
        return out

    @classmethod
    def _feedback(cls, gates: dict[str, bool], metrics: Any, baseline_throughput: float | None,
                  best_p99: float | None, topology_errors: list[str] | None = None,
                  design: dict[str, Any] | None = None, constraints: dict[str, Any] | None = None,
                  objective: dict[str, Any] | None = None) -> str:
        issues = [key for key, passed in gates.items() if not passed]
        details = list(topology_errors or [])
        disconnected = _get(metrics, "disconnected_component_ids", "disconnected_components", default=[])
        bottlenecks = _get(metrics, "bottleneck_component_ids", "bottlenecks", default=[])
        if not disconnected:
            disconnected = sorted({component_id for error in details
                                   for component_id in re.findall(r"compute component ([^ ]+) has no HBM path", error)})
        if isinstance(disconnected, (list, tuple)) and disconnected:
            details.append("Disconnected component IDs: " + ", ".join(map(str, disconnected)))
        if isinstance(bottlenecks, (list, tuple)) and bottlenecks:
            ids_text = ", ".join(map(str, bottlenecks))
            details.append("Resource/bandwidth bottleneck component IDs: " + ids_text)
            details.append("Inspect or widen the constrained links incident to: " + ids_text)
        if _get(metrics, "thermal_bottleneck_component_ids"):
            details.append("Thermal bottleneck component IDs: " + ", ".join(map(str, _get(metrics, "thermal_bottleneck_component_ids"))))
        if _get(metrics, "bandwidth_bottleneck_component_ids"):
            details.append("Bandwidth bottleneck component IDs: " + ", ".join(map(str, _get(metrics, "bandwidth_bottleneck_component_ids"))))
        component_temperatures = _get(metrics, "component_temperatures", default={}) or {}
        if isinstance(component_temperatures, dict) and component_temperatures:
            hottest = max(float(value) for value in component_temperatures.values())
            hot_ids = [str(component_id) for component_id, value in component_temperatures.items()
                       if float(value) >= hottest - 0.5]
            details.append(f"Hottest component IDs ({hottest:g} °C): " + ", ".join(hot_ids))
        diagnosis = cls._diagnosis(gates, metrics, design or {}, constraints or {}) if design else []
        if objective and objective.get("metric") == "area_mm2" and design:
            # Latency advice ("raise HBM bandwidth") would push an area search the wrong way.
            diagnosis = [d.replace("raise HBM bandwidth, HBM-to-compute link capacity, or SRAM to cut spills",
                                   "compute cores and TFLOPS are slack and can be cut to save area") for d in diagnosis]
        if issues:
            prefix = "Hard gate failure: " + ", ".join(issues)
            details = diagnosis[:3] + details
            if details:
                prefix += ". Checker findings: " + "; ".join(details)
            actions = []
            if "topology_valid" in issues:
                actions.append("repair listed topology violations, reconnect all component IDs, and preserve scalar/topology consistency")
            if "memory_fit" in issues:
                actions.append("increase HBM capacity or reduce model/KV allocation")
            if "throughput_at_least_baseline" in issues:
                actions.append("increase the active compute or memory-bandwidth path")
            if "latency_guard" in issues:
                actions.append("keep p99 TTFT at or below today's: restore HBM bandwidth or compute on the bound phase")
            if "temperature_cap" in issues or "power_cap" in issues:
                actions.append("reduce hotspot activity or improve cooling without violating other caps")
            if "area_cap" in issues:
                actions.append("reduce component count or footprint")
            return prefix + (". Next round: " + "; ".join(actions) + "." if actions else "")
        p99 = cls._metric(metrics, "ttft_p99_ms", "predicted_p99_ttft_ms")
        tail = (" " + " ".join(diagnosis[-2:])) if diagnosis else ""
        if objective and objective.get("metric") != "ttft_p99_ms":
            value = cls._metric(metrics, objective["metric"])
            worse = value is not None and best_p99 is not None and (
                value >= best_p99 if objective["direction"] == "min" else value <= best_p99)
            if worse:
                return (f"All hard gates pass; {objective['short']} ({value:g} {objective['unit']}) did not beat the best "
                        f"({best_p99:g} {objective['unit']}).{tail}")
            return f"All hard gates pass; candidate improves {objective['label']}.{tail}"
        if p99 is not None and best_p99 is not None and p99 >= best_p99:
            return f"All hard gates pass; p99 TTFT ({p99:g} ms) did not improve on best ({best_p99:g} ms).{tail}"
        return f"All hard gates pass; candidate improves predicted p99 TTFT.{tail}"

    @staticmethod
    def _prompt(campaign: dict[str, Any], round_no: int, best_design: dict[str, Any], best_p99: float | None) -> str:
        return (
            "Act as a chip-builder AI. Propose the complete physical accelerator design as strict JSON fields name, rationale, design. "
            "Every round must provide the full components array (IDs, type, x/y, width/height, layer and relevant capacities) and "
            "connections array (source/target IDs and bandwidth); return the entire graph, not a scalar-only patch. "
            "Change the actual layout/topology in response to checker findings. The design must satisfy the model schema and topology validator. "
            f"User design brief: {campaign.get('goal', 'Minimize p99 TTFT without reducing throughput')}. "
            "Primary scored objective: lower predicted p99 TTFT. Hard gates: topology validity, throughput at least baseline, memory fit, and all specified power, temperature, and area caps. "
            f"Round {round_no} of at most {campaign['max_rounds']}. Baseline metrics: "
            f"{json.dumps(campaign['baseline_metrics'], default=str)}. Current best p99 TTFT: {best_p99}. "
            f"Current best design: {json.dumps(best_design, default=str)}. Latest proposed design: {json.dumps(campaign.get('latest_design', best_design), default=str)}. "
            f"Checker feedback: {json.dumps(campaign['attempts'][-1].get('feedback') if campaign['attempts'] else None)}. "
            "Return a complete corrected graph with every connection endpoint referencing an existing component ID."
        )

    @staticmethod
    def _log_call(campaign: dict[str, Any], entry: dict[str, Any]) -> None:
        """Keep the most recent model calls (mutated in place as they stream) for the live view."""
        calls = campaign.setdefault("calls", [])
        calls.append(entry)
        del calls[:-40]

    def _activity(self, campaign: dict[str, Any], text: str) -> None:
        campaign["activity"] = text
        self._emit(campaign, "agent.step", {"text": text})

    @staticmethod
    def _emit(campaign: dict[str, Any], event_type: str, data: dict[str, Any]) -> None:
        campaign["events"].append({"id": len(campaign["events"]) + 1, "event": event_type, "data": data, "time": time.time()})

    def _storage_get(self, item_id: str) -> dict[str, Any] | None:
        if self.storage is None or not callable(getattr(self.storage, "get", None)):
            return None
        try:
            value = self.storage.get(item_id)
            if not value:
                return None
            record = _plain(value)
            if isinstance(record, dict) and "result" in record and "metrics" not in record:
                record["metrics"] = record["result"]
            return record
        except (KeyError, LookupError):
            return None

    def get_design(self, attempt_id: str) -> dict[str, Any] | None:
        return self.designs.get(attempt_id) or self._storage_get(attempt_id)

    def get_campaign(self, campaign_id: str) -> dict[str, Any] | None:
        return self.campaigns.get(campaign_id)

    def get_report(self, campaign_id: str) -> str | None:
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return None
        return campaign.get("report") or render_campaign_report(campaign)

    def event_snapshot(self, campaign_id: str, after_id: int = 0) -> tuple[list[dict[str, Any]], bool]:
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            raise KeyError("campaign not found")
        events = [event for event in campaign["events"] if event["id"] > after_id]
        return events, campaign["status"] in {"completed", "failed"}

