"""OpenAI-compatible proposer with strict JSON parsing and a local fallback."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx


DEFAULT_MODEL = "openai/gpt-oss-20b"


class ProposalError(ValueError):
    """The proposer did not return a valid structured design."""


class ModelUnavailableError(RuntimeError):
    """The configured model server cannot be reached or does not serve the model."""


class ModelDiscoveryError(ProposalError):
    """The OpenAI-compatible endpoint has no discoverable served model."""


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        mode: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("ACCELTWIN_LLM_BASE_URL") or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model = model or os.getenv("ACCELTWIN_LLM_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_MODEL
        self.api_key = api_key or os.getenv("ACCELTWIN_LLM_API_KEY") or os.getenv("LLM_API_KEY", "local")
        explicit_mode = os.getenv("ACCELTWIN_LLM_MODE")
        if os.getenv("ACCELTWIN_MOCK_LLM", "false").lower() in {"1", "true", "yes"}:
            explicit_mode = "mock"
        self.mode = (mode or explicit_mode or "auto").lower()
        # gpt-oss on Neuron is slow per token and spends some of its budget on
        # reasoning, so calls get a generous timeout and token budget.
        self.timeout = timeout or float(os.getenv("ACCELTWIN_LLM_TIMEOUT", "240"))
        self.max_tokens = int(os.getenv("ACCELTWIN_LLM_MAX_TOKENS", "3000"))
        self.reasoning_effort = os.getenv("ACCELTWIN_LLM_REASONING_EFFORT", "low").strip()
        self.json_mode = os.getenv("ACCELTWIN_LLM_JSON_MODE", "false").lower() in {"1", "true", "yes"}

    def design_round(self, context: dict[str, Any]) -> dict[str, Any]:
        """Design one complete chip through several small, validated model calls."""
        if self.mode in {"mock", "rule", "rules", "offline"}:
            return self._rule_proposal(context)
        from .agent import ChipDesignAgent

        # No silent offline fallback here: a design round either comes from the
        # model or fails loudly, so the UI never passes scripted edits off as AI.
        try:
            return ChipDesignAgent(self.chat_json, report=context.get("report"), max_tokens=self.max_tokens,
                                   log_call=context.get("log_call")).run(context)
        except httpx.TimeoutException as exc:
            raise ModelUnavailableError(f"{self.model} did not answer within {self.timeout:g} s") from exc
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(f"{self.model} at {self.base_url} failed: {exc}") from exc

    def check(self) -> dict[str, Any]:
        """Report whether the configured model is actually being served."""
        if self.mode in {"mock", "rule", "rules", "offline"}:
            return {"ready": True, "served": [], "error": None}
        try:
            response = httpx.get(f"{self.base_url}/models", headers={"Authorization": f"Bearer {self.api_key}"},
                                 timeout=5.0)
            response.raise_for_status()
            served = [item.get("id") for item in response.json().get("data", []) if isinstance(item, dict)]
        except (httpx.HTTPError, ValueError) as exc:
            return {"ready": False, "served": [],
                    "error": f"No model server at {self.base_url} ({exc.__class__.__name__}). Is serve.sh running?"}
        if self.model not in served:
            return {"ready": False, "served": served,
                    "error": f"{self.model} is not being served; the server has {', '.join(map(str, served)) or 'no models'}. "
                             f"Restart serve.sh with MODEL={self.model}, or set ACCELTWIN_LLM_MODEL."}
        return {"ready": True, "served": served, "error": None}

    def chat_json(self, system: str, user: str, max_tokens: int | None = None,
                  progress: Any = None) -> dict[str, Any]:
        """Send one chat turn and parse the single JSON object it returns.

        The reply streams so ``progress(tokens)`` can report live token counts
        (reasoning tokens included); set ACCELTWIN_LLM_STREAM=false to disable.
        """
        body: dict[str, Any] = {
            "model": self.model or self._discover_model(),
            "temperature": 0.3,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        url, headers = f"{self.base_url}/chat/completions", {"Authorization": f"Bearer {self.api_key}"}
        if os.getenv("ACCELTWIN_LLM_STREAM", "true").lower() in {"0", "false", "no"}:
            response = httpx.post(url, headers=headers, json=body, timeout=self.timeout)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            raw, finish = (choice.get("message") or {}).get("content") or "", choice.get("finish_reason")
        else:
            parts: list[str] = []
            tokens, finish = 0, None
            with httpx.stream("POST", url, headers=headers, json={**body, "stream": True}, timeout=self.timeout) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(line[5:])
                    except ValueError:
                        continue
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        finish = choice.get("finish_reason") or finish
                        if delta.get("content"):
                            parts.append(delta["content"])
                        if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                            tokens += 1
                            if progress:
                                progress(tokens)
            raw = "".join(parts)
        if not raw.strip():
            raise ProposalError(f"model returned no answer (finish_reason={finish}); "
                                "raise ACCELTWIN_LLM_MAX_TOKENS or lower the reasoning effort")
        return self._parse_json(raw)

    def propose(self, prompt: str, schema: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return exactly one JSON design object. `auto` falls back when vLLM is unavailable."""
        if self.mode in {"mock", "rule", "rules", "offline"}:
            return self._rule_proposal(context or {})
        try:
            return self._request(prompt, schema)
        except (httpx.HTTPError, ModelDiscoveryError):
            if self.mode in {"openai", "remote", "strict"}:
                raise
            return self._rule_proposal(context or {})

    def repair(self, prompt: str, invalid: str, error: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Ask the model once to repair invalid JSON; the caller enforces the one-repair limit."""
        repair_prompt = (
            f"{prompt}\n\nYour previous response failed validation: {error}. "
            f"Return only one JSON object matching this schema: {json.dumps(schema)}. "
            f"Previous response:\n{invalid}"
        )
        return self._request(repair_prompt, schema)

    def _request(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        model = self.model or self._discover_model()
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "Return exactly one JSON object. No markdown or commentary."},
                    {"role": "user", "content": f"{prompt}\n\nRequired JSON schema:\n{json.dumps(schema)}"},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        raw = payload["choices"][0]["message"]["content"]
        return self._parse_json(raw)

    def _discover_model(self) -> str:
        response = httpx.get(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=min(self.timeout, 10.0),
        )
        response.raise_for_status()
        models = response.json().get("data", [])
        if not models or not isinstance(models[0].get("id"), str):
            raise ModelDiscoveryError("the local endpoint did not report a served model ID; set LLM_MODEL")
        self.model = models[0]["id"]
        return self.model

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        candidate = raw.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            candidate = fenced.group(1)
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            # Reasoning models sometimes wrap the object in prose; take the
            # first complete JSON object in the reply.
            value = LLMClient._first_object(candidate)
            if value is None:
                raise ProposalError(f"response is not valid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ProposalError("response must be a JSON object")
        return value

    @staticmethod
    def _first_object(text: str) -> dict[str, Any] | None:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                value, _end = decoder.raw_decode(text, match.start())
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    @staticmethod
    def _rule_proposal(context: dict[str, Any]) -> dict[str, Any]:
        """Deterministic offline chip builder with explicit, round-varying graphs."""
        round_index = int(context.get("round", 0))
        baseline_config = dict(context.get("current_design") or context.get("baseline_config") or {})
        config = dict(baseline_config)
        components = config.get("components")
        connections = config.get("connections")
        if isinstance(components, list) and isinstance(connections, list) and components:
            components = [dict(component) for component in components]
            connections = [dict(connection) for connection in connections]
            by_type = lambda *names: [c for c in components if str(c.get("type", "")).lower() in names]
            compute = by_type("compute", "compute_cluster", "core", "cores")
            hbm = by_type("hbm", "memory", "hbm_stack")
            sram = by_type("sram", "sbuf", "buffer")
            noc = by_type("noc", "interconnect", "network")
            def shift_one(candidates: list[dict[str, Any]], axis: str, direction: float) -> bool:
                """Apply a small normalized move only when bounds and spacing hold."""
                for component in candidates:
                    old = float(component.get(axis, 0.0))
                    extent = float(component.get("width" if axis == "x" else "height", 0.0))
                    for delta in (direction * 0.035, -direction * 0.035, direction * 0.02, -direction * 0.02):
                        new = old + delta
                        if new < 0 or new + extent > 1:
                            continue
                        trial = dict(component)
                        trial[axis] = new
                        overlaps = False
                        candidate_is_overlay = str(component.get("type", "")).lower() in {"noc", "router"}
                        for other in components:
                            if candidate_is_overlay or other is component or str(other.get("type", "")).lower() in {"noc", "router"}:
                                continue
                            if str(other.get("layer", "default")) != str(component.get("layer", "default")):
                                continue
                            ax = new if axis == "x" else float(component.get("x", 0.0))
                            ay = new if axis == "y" else float(component.get("y", 0.0))
                            aw = extent if axis == "x" else float(component.get("width", 0.0))
                            ah = extent if axis == "y" else float(component.get("height", 0.0))
                            ox, oy = float(other.get("x", 0.0)), float(other.get("y", 0.0))
                            ow, oh = float(other.get("width", 0.0)), float(other.get("height", 0.0))
                            if min(ax + aw, ox + ow) - max(ax, ox) > 0.015 and min(ay + ah, oy + oh) - max(ay, oy) > 0.015:
                                overlaps = True
                                break
                        if not overlaps:
                            component[axis] = round(new, 4)
                            return True
                return False
            # Use bounded, graph-preserving physical edits. Layout nudges remain
            # within common 0..100 floorplan coordinates and do not alter totals.
            action = round_index % 5
            if action == 0 and hbm:
                shift_one(hbm, "x", 1 if round_index % 2 == 0 else -1)
                bandwidth_scale = 1.20
                if "hbm_bandwidth_tb_s" in config:
                    config["hbm_bandwidth_tb_s"] = round(float(config["hbm_bandwidth_tb_s"]) * bandwidth_scale, 4)
                for component in hbm:
                    component["bandwidth_tb_s"] = round(float(component.get("bandwidth_tb_s", 0)) * bandwidth_scale, 4)
                for connection in connections:
                    connection["bandwidth_tb_s"] = round(float(connection.get("bandwidth_tb_s", .01)) * bandwidth_scale, 4)
                if "utilization" in config:
                    config["utilization"] = min(.95, round(float(config.get("utilization", .72)) + .05, 3))
                rationale = "Widen the memory fabric and HBM interfaces by 20%, reposition the HBM stack, and raise useful compute utilization to relieve the decode bandwidth bottleneck."
            elif action == 1 and compute:
                # Separate existing clusters to spread heat and expose their
                # distinct placement to the live chip renderer.
                shift_one(compute, "y", -1 if round_index % 2 == 1 else 1)
                rationale = "Shift a compute cluster to explore a cooler physical placement while preserving the graph."
            elif action == 2 and (noc or connections):
                if noc:
                    shift_one(noc, "x", 1 if round_index % 2 == 0 else -1)
                else:
                    shift_one(compute or hbm or components, "x", 1 if round_index % 2 == 0 else -1)
                rationale = "Shift the interconnect region to improve physical locality while preserving link capacity."
            elif action == 3 and sram:
                shift_one(sram, "x", -1 if round_index % 2 == 1 else 1)
                rationale = "Move the on-chip buffer closer to compute to shorten local data movement."
            elif hbm or compute:
                shift_one(hbm + compute, "y", 1 if round_index % 2 == 0 else -1)
                rationale = "Rebalance memory and compute placement to improve locality without changing resources."
            else:
                # Even minimal schemas produce a visible graph change.
                first = components[0]
                shift_one([first], "x", 1 if round_index % 2 == 0 else -1)
                rationale = "Reposition the first functional block to explore a distinct physical floorplan."
            config["components"] = components
            config["connections"] = connections
            return {"name": f"offline chip layout {round_index + 1}", "rationale": rationale, "design": config,
                    "source": "offline"}

        # Compatibility for scalar-only models/callers during migration.
        # These knobs are illustrative defaults. The simulator owns allowed ranges and
        # the orchestration layer subjects every candidate to its hard gates.
        knobs = ("utilization", "hbm_bandwidth_tb_s", "peak_compute_tflops", "sbuf_gb", "dma_engines")
        knob = knobs[round_index % len(knobs)]
        current = config.get(knob)
        if knob == "utilization":
            config[knob] = min(0.95, round(float(current or 0.72) + 0.05, 2))
        elif knob == "hbm_bandwidth_tb_s":
            config[knob] = min(8.0, round(float(current or 2.9) + 0.4, 2))
        elif knob == "peak_compute_tflops":
            config[knob] = min(5000.0, round(float(current or 2500.0) * 1.15, 2))
        elif knob == "sbuf_gb":
            config[knob] = min(96.0, float(current or 24.0) + 4.0)
        else:
            config[knob] = min(16, int(current or 4) + 1)
        return {
            "name": f"offline architecture candidate {round_index + 1}",
            "rationale": f"Explore a bounded increase to {knob}; hard gates remain authoritative.",
            "design": config,
            "source": "offline",
        }

