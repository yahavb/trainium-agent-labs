"""Model client: exact token counts, context-budget enforcement, finish_reason checks, disk cache.

The server's context (8192) is input + output. A request that leaves too little room for the answer
truncates silently (finish_reason=length), so the answer budget is computed from the exact prompt
token count, and every truncation or empty answer is reported, never parsed.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

BASE_URL = os.environ.get("KERNEL_AGENT_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-8B")
CONTEXT = int(os.environ.get("KERNEL_AGENT_CONTEXT", "8192"))
MIN_ANSWER = 1500       # below this a kernel answer is likely to truncate; refuse rather than send
CACHE_DIR = Path(os.environ.get("KERNEL_AGENT_CACHE", Path(__file__).parent.parent / ".cache"))


@dataclass
class Reply:
    content: str
    reasoning: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    max_tokens: int
    latency_s: float
    cached: bool = False

    @property
    def ok(self):
        return self.finish_reason == "stop" and bool(self.content.strip())

    def problem(self):
        if self.finish_reason == "length":
            return (f"truncated (finish_reason=length) after {self.completion_tokens} tokens "
                    f"with max_tokens={self.max_tokens}")
        if not self.content.strip():
            return f"empty answer ({len(self.reasoning)} chars of reasoning, finish={self.finish_reason})"
        return None


def _post(path, body, timeout):
    root = BASE_URL.rstrip("/")
    url = (root[: -len("/v1")] if path.startswith("/tokenize") and root.endswith("/v1") else root) + path
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{e.code} from {path}: {e.read().decode(errors='replace')[:500]}") from None


def count_tokens(messages, think=False):
    """Exact prompt tokens after the chat template, from the server's own tokenizer."""
    body = {"model": MODEL, "messages": messages, "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": think}}
    return _post("/tokenize", body, 30)["count"]


def count_text(text):
    """Tokens in raw text (no chat template): used to split a prompt's cost by section."""
    return _post("/tokenize", {"model": MODEL, "prompt": text}, 30)["count"]


def chat(messages, max_tokens=4000, temperature=0.6, sample=0, think=False, use_cache=True):
    """`sample` only distinguishes cache entries. A per-request `seed` would make this reproducible,
    but on Neuron vLLM it kills the engine (no device RNG generator), so it is never sent."""
    prompt_tokens = count_tokens(messages, think)
    budget = min(max_tokens, CONTEXT - prompt_tokens - 16)
    if budget < MIN_ANSWER:
        raise ValueError(f"prompt is {prompt_tokens} tokens; only {budget} left for the answer")
    body = {"model": MODEL, "messages": messages, "max_tokens": budget,
            "temperature": temperature, "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": think}}
    key = hashlib.sha256(json.dumps([body, sample], sort_keys=True).encode()).hexdigest()[:24]
    path = CACHE_DIR / f"{key}.json"
    if use_cache and path.exists():
        return Reply(**{**json.loads(path.read_text()), "cached": True})

    t0 = time.time()
    r = _post("/chat/completions", body, 600)
    ch = r["choices"][0]
    msg = ch["message"]
    reply = Reply(content=msg.get("content") or "",
                  reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
                  finish_reason=ch.get("finish_reason") or "",
                  prompt_tokens=r["usage"]["prompt_tokens"],
                  completion_tokens=r["usage"]["completion_tokens"],
                  max_tokens=budget, latency_s=round(time.time() - t0, 2))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(reply)))
    return reply
