"""Prometheus vLLM telemetry parsing and optional Neuron monitoring."""

from __future__ import annotations

import logging
import re
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|[+-]?Inf)(?:\s+\d+)?$")


def parse_vllm_metrics(text: str) -> Dict[str, float]:
    """Parse Prometheus exposition samples into a flat metric-name mapping.

    Label sets are ignored and duplicate samples are summed, which is useful
    for common per-engine counters. Histogram buckets remain individual names.
    """
    parsed: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        name, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        parsed[name] = parsed.get(name, 0.0) + value
    return parsed


def collect_vllm_metrics(url: str = "http://localhost:8000/metrics", timeout: float = 1.5) -> Dict[str, float]:
    """Fetch and parse a vLLM Prometheus endpoint; return an empty map offline."""
    try:
        request = urllib.request.Request(url, headers={"Accept": "text/plain"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return parse_vllm_metrics(response.read().decode("utf-8", errors="replace"))
    except Exception as exc:  # graceful fallback is expected when server is absent
        logger.debug("Could not collect vLLM metrics from %s: %s", url, exc)
        return {}


class NeuronMonitorAdapter:
    """Optional adapter for AWS Neuron monitoring, with deterministic mock mode.

    The adapter accepts an injected monitor to keep the core app independent
    of the optional ``neuron-monitor`` package and to make deployments testable.
    """

    def __init__(self, monitor: Any = None, mock_fallback: bool = True):
        self.monitor = monitor
        self.mock_fallback = mock_fallback
        self.is_mock = monitor is None

    def collect(self) -> Dict[str, Any]:
        if self.monitor is not None:
            try:
                raw = self.monitor.get_metrics() if hasattr(self.monitor, "get_metrics") else self.monitor()
                if isinstance(raw, dict):
                    self.is_mock = False
                    return raw
            except Exception as exc:
                logger.debug("Neuron monitor unavailable: %s", exc)
        if not self.mock_fallback:
            return {}
        self.is_mock = True
        return {"source": "mock", "neuron_utilization": 0.0, "neuron_memory_used_gb": 0.0, "temperature_c": 0.0}


NeuronMonitor = NeuronMonitorAdapter



# ---------------------------------------------------------------------------
# Live workload measurement: vLLM Prometheus deltas, neuron-monitor sampling,
# and the Neuron Explorer system/device trace summary.
# ---------------------------------------------------------------------------

import json
import math
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

NEURON_BIN = "/opt/aws/neuron/bin"
_LABELLED = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+(\S+)')


def scrape_prometheus(text: str) -> Dict[str, Any]:
    """Parse exposition text into summed scalars plus histogram buckets/sum/count."""
    scalars: Dict[str, float] = {}
    hists: Dict[str, Dict[str, Any]] = {}
    histogram_names = {m.group(1) for m in re.finditer(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)_bucket\{", text, re.M)}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _LABELLED.match(line.strip())
        if not match:
            continue
        name, labels, raw = match.groups()
        try:
            value = float(raw)
        except ValueError:
            continue
        if not math.isfinite(value) and raw not in {"+Inf", "Inf"}:
            continue
        if name.endswith("_bucket"):
            le = re.search(r'le="([^"]+)"', labels or "")
            if le:
                bound = float("inf") if le.group(1) in {"+Inf", "Inf"} else float(le.group(1))
                buckets = hists.setdefault(name[:-7], {"buckets": {}, "sum": 0.0, "count": 0.0})["buckets"]
                buckets[bound] = buckets.get(bound, 0.0) + value
            continue
        for suffix in ("_sum", "_count"):
            base = name[: -len(suffix)]
            if name.endswith(suffix) and base in histogram_names:
                hists.setdefault(base, {"buckets": {}, "sum": 0.0, "count": 0.0})[suffix[1:]] += value
                break
        else:
            scalars[name] = scalars.get(name, 0.0) + value
    return {"scalars": scalars, "hists": hists}


def _hist_delta(before: Dict[str, Any], after: Dict[str, Any], name: str) -> Dict[str, Any]:
    a, b = after["hists"].get(name), before["hists"].get(name, {"buckets": {}, "sum": 0.0, "count": 0.0})
    if not a:
        return {"buckets": {}, "sum": 0.0, "count": 0.0}
    return {"buckets": {le: a["buckets"][le] - b["buckets"].get(le, 0.0) for le in a["buckets"]},
            "sum": a["sum"] - b["sum"], "count": a["count"] - b["count"]}


def hist_quantile(hist: Dict[str, Any], q: float) -> Optional[float]:
    """Prometheus-style histogram_quantile with linear interpolation inside a bucket."""
    count = hist.get("count") or 0
    if count <= 0:
        return None
    rank, previous_bound, previous_count = q * count, 0.0, 0.0
    for bound in sorted(hist["buckets"]):
        cumulative = hist["buckets"][bound]
        if cumulative >= rank:
            if math.isinf(bound):
                return previous_bound
            span = cumulative - previous_count
            return previous_bound + (bound - previous_bound) * ((rank - previous_count) / span if span else 1.0)
        previous_bound, previous_count = bound, cumulative
    return previous_bound


def summarize_vllm(before: Dict[str, Any], after: Dict[str, Any], wall_s: float,
                   kv_peak: Optional[float] = None) -> Dict[str, Any]:
    """Turn two scrapes taken around a replay into the metrics the checker uses."""
    def mean_ms(name: str) -> Optional[float]:
        h = _hist_delta(before, after, name)
        return h["sum"] / h["count"] * 1000.0 if h["count"] else None

    def mean(name: str) -> Optional[float]:
        h = _hist_delta(before, after, name)
        return h["sum"] / h["count"] if h["count"] else None

    def counter(*names: str) -> float:
        for name in names:
            if name in after["scalars"]:
                return after["scalars"][name] - before["scalars"].get(name, 0.0)
        return 0.0

    ttft = _hist_delta(before, after, "vllm:time_to_first_token_seconds")
    prompt_tokens = counter("vllm:prompt_tokens_total", "vllm:prompt_tokens")
    generation_tokens = counter("vllm:generation_tokens_total", "vllm:generation_tokens")
    flops = counter("vllm:estimated_flops_per_gpu_total", "vllm:estimated_flops_per_gpu")
    read_bytes = counter("vllm:estimated_read_bytes_per_gpu_total", "vllm:estimated_read_bytes_per_gpu")
    write_bytes = counter("vllm:estimated_write_bytes_per_gpu_total", "vllm:estimated_write_bytes_per_gpu")
    p50, p99 = hist_quantile(ttft, 0.50), hist_quantile(ttft, 0.99)
    return {
        "requests": int(counter("vllm:request_success_total", "vllm:request_success") or ttft["count"]),
        "ttft_p50_ms": None if p50 is None else p50 * 1000.0,
        "ttft_p99_ms": None if p99 is None else p99 * 1000.0,
        "queue_ms": mean_ms("vllm:request_queue_time_seconds"),
        "prefill_ms": mean_ms("vllm:request_prefill_time_seconds"),
        "decode_ms": mean_ms("vllm:request_decode_time_seconds"),
        "inter_token_ms": mean_ms("vllm:inter_token_latency_seconds"),
        "prompt_tokens_mean": mean("vllm:request_prompt_tokens"),
        "generation_tokens_mean": mean("vllm:request_generation_tokens"),
        "throughput_tokens_s": (prompt_tokens + generation_tokens) / wall_s if wall_s > 0 else None,
        "kv_cache_peak_pct": None if kv_peak is None else kv_peak * 100.0,
        "preemptions": counter("vllm:num_preemptions_total", "vllm:num_preemptions"),
        "arithmetic_intensity": flops / (read_bytes + write_bytes) if read_bytes + write_bytes > 0 else None,
        "model_weights_gb": (after["scalars"].get("vllm_neuron:model_load_size_bytes") or 0) / 1e9 or None,
    }


def _neuron_tool(name: str) -> Optional[str]:
    return shutil.which(name) or shutil.which(name, path=NEURON_BIN)


class NeuronMonitorSampler:
    """Run neuron-monitor in the background and aggregate its JSON reports."""

    def __init__(self) -> None:
        self.samples: List[Dict[str, float]] = []
        self.hardware: Dict[str, Any] = {}
        self.error: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "NeuronMonitorSampler":
        binary = _neuron_tool("neuron-monitor")
        if not binary:
            self.error = "neuron-monitor not installed"
            return self
        # One report per second for runtime counters only; the default period is too coarse for a short replay.
        config = Path(os.getenv("TMPDIR", "/tmp")) / "acceltwin-neuron-monitor.json"
        config.write_text(json.dumps({"period": "1s", "system_metrics": [], "neuron_runtimes": [
            {"tag_filter": ".*", "metrics": [{"type": "neuroncore_counters"}, {"type": "memory_used"}]}]}))
        try:
            self._proc = subprocess.Popen([binary, "-c", str(config)], stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, text=True)
        except OSError as exc:
            self.error = str(exc)
            return self
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            try:
                self._ingest(json.loads(line))
            except (ValueError, KeyError, TypeError):
                continue

    def _ingest(self, report: Dict[str, Any]) -> None:
        self.hardware = report.get("neuron_hardware_info") or self.hardware
        utilizations, flops, device_bytes = [], 0.0, 0.0
        for runtime in report.get("neuron_runtime_data") or []:
            data = runtime.get("report") or {}
            cores = ((data.get("neuroncore_counters") or {}).get("neuroncores_in_use") or {})
            for core in cores.values():
                utilizations.append(float(core.get("neuroncore_utilization") or 0.0))
                flops += float(core.get("effective_flops") or 0.0)
            device_bytes += float(((data.get("memory_used") or {}).get("neuron_runtime_used_bytes") or {})
                                  .get("neuron_device") or 0.0)
        if utilizations:
            self.samples.append({"utilization": sum(utilizations) / len(utilizations),
                                 "peak_core": max(utilizations), "flops": flops, "device_gb": device_bytes / 1e9})

    def stop(self) -> Dict[str, Any]:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=2)
        busy = [s for s in self.samples if s["utilization"] > 0] or self.samples
        if not busy:
            return {"available": False, "reason": self.error or "no neuron-monitor samples"}
        return {
            "available": True,
            "samples": len(self.samples),
            "core_utilization_pct": sum(s["utilization"] for s in busy) / len(busy),
            "core_utilization_peak_pct": max(s["peak_core"] for s in busy),
            "effective_tflops": (sum(s["flops"] for s in busy) / len(busy) / 1e12) or None,
            "device_memory_gb": max(s["device_gb"] for s in self.samples),
            "device_type": self.hardware.get("neuron_device_type"),
            "device_memory_total_gb": (self.hardware.get("neuron_device_memory_size") or 0) / 1e9 or None,
            "cores": self.hardware.get("neuroncore_per_device_count"),
        }


_TRACE_KEYS = ("mm_arithmetic_intensity", "peak_flops_bandwidth_ratio", "mbu_estimated_percent",
               "dma_active_time_percent", "hbm_read_bytes", "hbm_write_bytes", "spill_save_bytes",
               "spill_reload_bytes", "total_time", "model_flops")


def _collect_summaries(node: Any, found: List[Dict[str, Any]]) -> None:
    if isinstance(node, dict):
        if any(key in node for key in _TRACE_KEYS):
            found.append(node)
        for value in node.values():
            _collect_summaries(value, found)
    elif isinstance(node, list):
        for value in node:
            _collect_summaries(value, found)


def summarize_trace_json(payload: Any) -> Dict[str, Any]:
    """Reduce one or more Neuron Explorer Summary rows to the bottleneck signals."""
    rows: List[Dict[str, Any]] = []
    _collect_summaries(payload, rows)
    if not rows:
        return {"available": False, "reason": "trace summary had no Summary rows"}

    def avg(key: str) -> Optional[float]:
        values = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        return sum(values) / len(values) if values else None

    def total(key: str) -> float:
        return sum(float(r[key]) for r in rows if isinstance(r.get(key), (int, float)))

    hbm = total("hbm_read_bytes") + total("hbm_write_bytes")
    spill = total("spill_save_bytes") + total("spill_reload_bytes")
    return {
        "available": True,
        "profiles": len(rows),
        "arithmetic_intensity": avg("mm_arithmetic_intensity"),
        "ridge_point": avg("peak_flops_bandwidth_ratio"),
        "mbu_pct": avg("mbu_estimated_percent"),
        "dma_active_pct": avg("dma_active_time_percent"),
        "spill_fraction": spill / hbm if hbm else None,
        "hbm_gb_moved": hbm / 1e9 if hbm else None,
    }


def _loads_any(text: str) -> Any:
    """Parse a whole JSON document, or the first JSON value embedded in log text."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char in "{[":
            try:
                return decoder.raw_decode(text, index)[0]
            except ValueError:
                continue
    return None


def summarize_system_profile(data: Dict[str, Any], gap_s: float = 5.0) -> Dict[str, Any]:
    """Where device time went during the traffic in a Neuron runtime system profile.

    Only the last burst of executions is used (startup warm-up is separated from
    the traffic by an idle gap), so the numbers describe serving, not loading.
    """
    events = data.get("trace_event") or []
    execs = [e for e in events if e.get("name") == "nc_exec_running" and e.get("duration")]
    if not execs:
        return {"available": False, "reason": "the system profile has no NeuronCore executions"}
    spans = sorted((int(e["timestamp"]), int(e["timestamp"]) + int(e["duration"])) for e in execs)
    window_start, window_end, last_end = spans[0][0], spans[0][1], spans[0][1]
    for start, end in spans[1:]:
        if start - last_end > gap_s * 1e9:
            window_start = start
        last_end = max(last_end, end)
        window_end = last_end
    inside = lambda e: window_start <= int(e.get("timestamp", 0)) <= window_end
    window = [e for e in execs if inside(e)]

    def total(name: str) -> float:
        return sum(float(e.get("duration") or 0) for e in events if e.get("name") == name and inside(e))

    busy: Dict[Any, List[tuple]] = {}
    for e in window:
        busy.setdefault((e.get("process_id"), e.get("device_core_idx", e.get("nc_idx"))), []).append(
            (int(e["timestamp"]), int(e["timestamp"]) + int(e["duration"])))
    length = max(window_end - window_start, 1)
    fractions = []
    for intervals in busy.values():
        merged, cursor = 0, None
        for start, end in sorted(intervals):
            if cursor is None or start > cursor:
                merged += end - start
                cursor = end
            elif end > cursor:
                merged += end - cursor
                cursor = end
        fractions.append(merged / length)
    # Decode steps dominate the count, so the median execution is a decode step;
    # executions far longer than that are context encoding (prefill).
    durations = sorted(float(e["duration"]) / 1e6 for e in window)
    decode_ms = durations[len(durations) // 2]
    long_runs = [d for d in durations if d > 3 * decode_ms]
    prefill_ms = sum(long_runs) / len(long_runs) if long_runs else None
    exec_ns = sum(float(e["duration"]) for e in window)
    per_hbm: Dict[str, Dict[str, float]] = {}
    for m in data.get("device_mem_usage") or []:
        row = per_hbm.setdefault(str(m.get("hbm_idx")), {"total": 0.0, "spill": 0.0})
        row["total"] = max(row["total"], float(m.get("total_bytes") or 0))
        row["spill"] = max(row["spill"], float(m.get("dram_spill_bytes") or 0) + float(m.get("dma_rings_spill_bytes") or 0))
    return {
        "available": True,
        "kind": "system profile",
        "window_s": length / 1e9,
        "executions_per_core": round(len(window) / max(len(busy), 1)),
        "device_busy_pct": 100.0 * sum(fractions) / len(fractions),
        "decode_step_ms": decode_ms,
        "prefill_step_ms": prefill_ms,
        "collective_pct": 100.0 * total("cc_running") / exec_ns if exec_ns else None,
        "host_copy_pct": 100.0 * (total("dmem_buf_copyin") + total("dmem_buf_copyout")) / exec_ns if exec_ns else None,
        "host_overhead_us": (total("nrt_model_submit") + total("nrta_execute_schedule")) / max(len(window), 1) / 1000.0,
        "hbm_used_gb": sum(r["total"] for r in per_hbm.values()) / 1e9 or None,
        "spill_mb": sum(r["spill"] for r in per_hbm.values()) / 1e6,
    }


def read_system_trace(trace_dir: Optional[str] = None, timeout: float = 900.0) -> Dict[str, Any]:
    """Summarize a Neuron Explorer capture (NEURON_RT_INSPECT_OUTPUT_DIR), caching the result.

    vllm-neuron writes the profile when its workers shut down, so a capture is:
    start vLLM with inspection on, send traffic, stop vLLM, then read it here.
    """
    directory = Path(trace_dir or os.getenv("ACCELTWIN_NEURON_TRACE_DIR", "/workspace/neuron-trace"))
    hint = (f"start vLLM with VLLM_NEURON_WORKER_TERMINATION_TIMEOUT=60 NEURON_RT_INSPECT_ENABLE=1 "
            f"NEURON_RT_INSPECT_OUTPUT_DIR={directory}, send traffic, then stop it so the profile is written")
    files = [p for p in directory.rglob("*") if p.is_file() and not p.name.startswith(".acceltwin")] if directory.is_dir() else []
    if not files:
        return {"available": False, "reason": f"no capture in {directory}", "hint": hint}
    cache = directory / ".acceltwin-summary.json"
    if cache.exists() and cache.stat().st_mtime >= max(p.stat().st_mtime for p in files):
        try:
            return json.loads(cache.read_text())
        except ValueError:
            pass
    binary = _neuron_tool("neuron-explorer")
    if not binary:
        return {"available": False, "reason": "neuron-explorer not installed", "hint": hint}
    import tempfile

    with tempfile.TemporaryDirectory(prefix="acceltwin-trace-") as work:
        try:
            # The json export writes system_profile.json (and device profiles, if captured) into the cwd.
            subprocess.run([binary, "view", "-d", str(directory), "--output-format", "json", "--disable-ui"],
                           cwd=work, capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"available": False, "reason": f"neuron-explorer view failed: {exc}", "hint": hint}
        system_file = Path(work) / "system_profile.json"
        if not system_file.exists():
            return {"available": False, "reason": "neuron-explorer produced no system profile", "hint": hint}
        with system_file.open() as handle:
            summary = summarize_system_profile(json.load(handle))
        device = [summarize_trace_json(_loads_any(p.read_text())) for p in Path(work).glob("*.json")
                  if p.name != "system_profile.json" and p.stat().st_size < 200_000_000]
        device = [d for d in device if d.get("available")]
        if device:
            summary.update({k: v for k, v in device[0].items() if k not in {"available", "profiles"}})
            summary["kind"] = "system + device profile"
    summary["source_dir"] = str(directory)
    try:
        cache.write_text(json.dumps(summary))
    except OSError:
        pass
    return summary


# A shared document in front of each question makes prompts the length of a real
# assistant turn (system prompt + context) rather than a bare one-liner.
REPLAY_CONTEXT = (
    "You are a support assistant for a cloud ML platform. Use the reference notes when relevant.\n\n"
    "Reference notes: Inference requests pass through an admission queue, then a prefill phase that "
    "processes the whole prompt in parallel and writes the key/value cache, then a decode phase that "
    "produces one token per step and re-reads the model weights and the cache on every step. Prefill "
    "is usually limited by matrix-multiply throughput; decode at small batch sizes is usually limited "
    "by memory bandwidth, because each step moves every weight once for very little arithmetic. "
    "Time to first token is queue time plus prefill time. Tail latency rises sharply once arrivals "
    "approach the service rate, so small speedups can remove most queueing. The accelerator has high-"
    "bandwidth memory stacks beside the compute dies, on-chip SRAM used as a software-managed buffer, "
    "DMA engines that move tensors between them, and an on-chip network linking everything. When "
    "intermediate tensors do not fit in SRAM they spill to HBM and are reloaded, which costs bandwidth. "
    "Power rises with active compute and is capped by the package and cooling; above the cap the chip "
    "throttles its clock. Larger dies cost more and yield worse.\n\nQuestion: "
)

REPLAY_PROMPTS = [
    "Explain how a hash table handles collisions, with a short example.",
    "Summarize the trade-offs between tensor parallelism and pipeline parallelism for LLM inference.",
    "Write a Python function that merges two sorted lists, then explain its complexity.",
    "Why does KV-cache size limit batch size during LLM serving? Answer in a paragraph.",
    "Describe what an HBM stack is and why accelerators use it.",
    "Give three ways to reduce time-to-first-token for a chat service.",
    "Explain the roofline model in plain language.",
    "What is the difference between prefill and decode in transformer inference?",
]


def _stream_tokens(response: Any) -> Any:
    """Yield one item per streamed token from an OpenAI-compatible SSE response (content or reasoning)."""
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
            if text:
                yield delta


def measure_live_workload(base_url: str, metrics_url: str, model: str, requests: int = 8,
                          concurrency: int = 4, max_tokens: int = 160, timeout: float = 300.0,
                          live: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Replay a short burst against the live server and measure it three ways.

    Requests stream, so ``live`` (if given) is updated in place as tokens arrive:
    per-request state and token counts, plus utilization / KV / tokens-per-second
    samples. The API serves that dict to the page while the replay runs.
    """
    import httpx  # local import keeps parse helpers usable without httpx
    from concurrent.futures import ThreadPoolExecutor

    live = live if live is not None else {}
    lanes = [{"id": i + 1, "state": "queued", "tokens": 0, "max_tokens": max_tokens, "ttft_ms": None, "total_ms": None}
             for i in range(requests)]
    live.update(phase="measure", requests=lanes, samples=[], tokens=0, elapsed_s=0.0, concurrency=concurrency)

    def scrape() -> Dict[str, Any]:
        return scrape_prometheus(httpx.get(metrics_url, timeout=5.0).text)

    before = scrape()
    monitor = NeuronMonitorSampler().start()
    kv_peak = [0.0]
    stop = threading.Event()
    started = time.monotonic()

    def watch() -> None:
        last_tokens, last_t = 0, time.monotonic()
        while not stop.wait(0.5):
            kv = None
            try:
                kv = scrape()["scalars"].get("vllm:kv_cache_usage_perc", 0.0)
                kv_peak[0] = max(kv_peak[0], kv)
            except Exception:  # sampling is best effort
                pass
            now, tokens = time.monotonic(), sum(lane["tokens"] for lane in lanes)
            latest = monitor.samples[-1] if monitor.samples else {}
            live["samples"].append({"t": round(now - started, 2), "tps": round((tokens - last_tokens) / max(now - last_t, 1e-3), 1),
                                    "util": latest.get("utilization"), "kv": None if kv is None else round(kv * 100, 2)})
            live["tokens"], live["elapsed_s"] = tokens, round(now - started, 2)
            last_tokens, last_t = tokens, now

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    errors: List[str] = []

    def one(index: int) -> None:
        lane = lanes[index]
        lane["state"] = "prefill"
        sent = time.monotonic()
        body = {"model": model, "max_tokens": max_tokens, "temperature": 0.7, "stream": True,
                "messages": [{"role": "user", "content": REPLAY_CONTEXT + REPLAY_PROMPTS[index % len(REPLAY_PROMPTS)]}]}
        try:
            with httpx.stream("POST", f"{base_url.rstrip('/')}/chat/completions", json=body, timeout=timeout) as response:
                response.raise_for_status()
                for _ in _stream_tokens(response):
                    if lane["ttft_ms"] is None:
                        lane["ttft_ms"] = round((time.monotonic() - sent) * 1000.0, 1)
                        lane["state"] = "decode"
                    lane["tokens"] += 1
            lane["state"] = "done"
        except Exception as exc:
            lane["state"] = "error"
            errors.append(str(exc))
        lane["total_ms"] = round((time.monotonic() - sent) * 1000.0, 1)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(one, range(requests)))
    wall = time.monotonic() - started
    stop.set()
    watcher.join(timeout=2)
    neuron = monitor.stop()
    vllm = summarize_vllm(before, scrape(), wall, kv_peak[0])
    vllm.update({"available": vllm["requests"] > 0, "wall_s": wall, "errors": len(errors),
                 "replayed": requests, "concurrency": concurrency})
    if not vllm["available"]:
        vllm["reason"] = errors[0][:160] if errors else "no requests completed"
    live["tokens"], live["elapsed_s"] = sum(lane["tokens"] for lane in lanes), round(wall, 2)
    live["phase"] = "trace"
    trace = read_system_trace()
    return {"vllm": vllm, "neuron_monitor": neuron, "system_trace": trace}
