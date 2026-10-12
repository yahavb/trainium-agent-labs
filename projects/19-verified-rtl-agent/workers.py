"""The only file that talks to a model. Every guard here was paid for on the seats.

DESIGN.md 6.4 and constraints C3-C8:
  * thinking OFF on Qwen3, or rounds take minutes and return nothing
  * NEVER send `seed` -- it kills the vLLM engine (EngineDeadError, HTTP 500) for everyone on that seat
  * at most 4 requests in flight per seat (max_num_seqs=4)
  * prompt + max_tokens under 8192, and finish_reason logged on every reply
  * HTTP 500 is never retried: a dead engine is checked with /health and the seat is dropped

    python workers.py --ping http://localhost:8000/v1      # one tiny request, prints the Reply
"""
from __future__ import annotations

import argparse
import os
import socket
import threading
import time
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

import httpx

QWEN = os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-8B")
CONTEXT = 8192
MARGIN = 64
PER_SEAT = 4
HTTP_TIMEOUT = 900


class SeatDown(Exception):
    """The seat's engine is dead. Stop sending it work."""


class ModelError(Exception):
    """One request failed (4xx, a 500 on a healthy server, a second connection error)."""


@dataclass
class Reply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    latency_s: float
    seat: str
    model: str

    def to_dict(self):
        return asdict(self)


_lock = threading.Lock()
_sems: dict = {}
_clients: dict = {}


def seat_name(base_url: str) -> str:
    """seat-93 for http://seat-93.seat:8000/v1; this pod's own name for localhost."""
    host = urlparse(base_url).hostname or base_url
    if host in ("localhost", "127.0.0.1", "::1"):
        return os.environ.get("SEAT_NAME") or socket.gethostname()
    if host.startswith("seat-"):
        return host.split(".")[0]
    return host.split(".")[0]


def _sem(base_url):
    with _lock:
        return _sems.setdefault(base_url, threading.BoundedSemaphore(PER_SEAT))


def _client(base_url):
    with _lock:
        if base_url not in _clients:
            # The shared gpt-oss endpoint's certificate does not match its hostname (gptoss/README).
            _clients[base_url] = httpx.Client(timeout=HTTP_TIMEOUT, verify=not base_url.startswith("https"))
        return _clients[base_url]


def is_gptoss(model: str) -> bool:
    return "gpt-oss" in model


def build_body(prompt: str, model: str, max_tokens: int, temperature: float, top_p: float) -> dict:
    est = len(prompt) // 4
    if is_gptoss(model):
        max_tokens = max(max_tokens, 2500)   # it reasons first; below ~2500 the answer comes back empty
    max_tokens = max(64, min(max_tokens, CONTEXT - MARGIN - est))
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
    if not is_gptoss(model):
        body["chat_template_kwargs"] = {"enable_thinking": False}
    guard(body)
    return body


def guard(body: dict):
    if "seed" in body:
        raise ValueError("refusing to send `seed`: it crashes the engine on the seats (DESIGN C6)")


def healthy(base_url: str) -> bool:
    root = base_url.rstrip("/")
    root = root[: -len("/v1")] if root.endswith("/v1") else root
    try:
        return _client(base_url).get(root + "/health", timeout=5).status_code == 200
    except httpx.HTTPError:
        return False


def ask(base_url: str, prompt: str, model: str = QWEN, max_tokens: int = 1500,
        temperature: float = 0.6, top_p: float = 0.95) -> Reply:
    body = build_body(prompt, model, max_tokens, temperature, top_p)
    url = base_url.rstrip("/") + "/chat/completions"
    seat = seat_name(base_url)
    with _sem(base_url):
        t0 = time.time()
        for attempt in (1, 2):
            try:
                r = _client(base_url).post(url, json=body)
                break
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as e:
                if attempt == 2:
                    if not healthy(base_url):
                        raise SeatDown(seat) from e
                    raise ModelError(f"connection failed twice: {e}") from e
                time.sleep(10)
            except httpx.TimeoutException as e:
                raise ModelError(f"no answer within {HTTP_TIMEOUT}s") from e
        latency = time.time() - t0
    if r.status_code >= 500:
        if not healthy(base_url):
            raise SeatDown(seat)
        raise ModelError(f"HTTP {r.status_code}: {r.text[:200]}")
    if r.status_code != 200:
        raise ModelError(f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    usage = data.get("usage") or {}
    return Reply(text=msg.get("content") or "",
                 prompt_tokens=usage.get("prompt_tokens", len(prompt) // 4),
                 completion_tokens=usage.get("completion_tokens", 0),
                 finish_reason=choice.get("finish_reason") or "unknown",
                 latency_s=round(latency, 2), seat=seat, model=model)


def selftest() -> int:
    fails = 0
    try:
        guard({"model": "m", "seed": 1})
        fails += 1
        print("  FAIL  a body with `seed` was allowed")
    except ValueError:
        print("  ok    a body with `seed` raises before sending")
    b = build_body("x" * 4000, QWEN, 1500, 0.6, 0.95)
    ok = b["chat_template_kwargs"] == {"enable_thinking": False} and "seed" not in b
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  Qwen body has thinking off and no seed")
    b = build_body("x" * 30000, QWEN, 1500, 0.6, 0.95)
    ok = 7500 + b["max_tokens"] <= CONTEXT - MARGIN
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  a long prompt lowers max_tokens to fit 8192 ({b['max_tokens']})")
    b = build_body("hi", "gpt-oss-20b", 1500, 0.6, 0.95)
    ok = "chat_template_kwargs" not in b and b["max_tokens"] >= 2500
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  gpt-oss body: no chat_template_kwargs, max_tokens >= 2500")
    ok = seat_name("http://seat-94.seat:8000/v1") == "seat-94"
    fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  seat name from a teammate URL")
    print("ALL OK" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ping", metavar="BASE_URL", help="send one tiny request and print the Reply")
    ap.add_argument("--model", default=QWEN)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if a.ping:
        print("healthy:", healthy(a.ping))
        print(ask(a.ping, "Reply with the single word: ready", model=a.model, max_tokens=16).to_dict())
    else:
        ap.print_help()
