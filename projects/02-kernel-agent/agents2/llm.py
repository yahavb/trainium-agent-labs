"""The model client, token counting, and the prompt packer that keeps every call inside the context.

Every call is one user message, with no history: the ledger carries what happened, and each role's
prompt is rebuilt from it. So the 8K context is a limit per call, and the packer enforces it.
"""

import dataclasses
import re
import sys
import threading
import time

REASONING_KEYS = ("reasoning", "reasoning_content")


class PromptTooLong(Exception):
    pass


# ---------------------------------------------------------------- token counting

class Tokens:
    """Counts with the served model's own tokenizer when the pod's model cache has it, otherwise
    estimates at 3 characters a token, which over-counts code slightly. Over-counting is the safe side."""

    def __init__(self, model):
        self.tok = None
        self.how = "estimate (3 chars/token)"
        try:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
            self.how = f"{model} tokenizer"
        except Exception:
            pass
        self.lock = threading.Lock()

    def count(self, text):
        if not text:
            return 0
        if self.tok is None:
            return len(text) // 3 + 1
        with self.lock:
            return len(self.tok.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------- the packer

@dataclasses.dataclass
class Section:
    name: str
    text: str
    priority: int = 0          # dropped lowest first; required sections never are
    required: bool = False


def pack(sections, cap, count):
    """Join the sections in order, dropping optional ones lowest-priority first until the prompt fits
    `cap` tokens. Code is always a required section, so it is never cut mid-line."""
    keep = [s for s in sections if s.text and s.text.strip()]
    sizes = {s.name: count(s.text) for s in keep}
    total = sum(sizes.values())
    dropped = []
    for s in sorted((s for s in keep if not s.required), key=lambda s: s.priority):
        if total <= cap:
            break
        total -= sizes[s.name]
        dropped.append(s.name)
    if total > cap:
        raise PromptTooLong(f"required sections need {total} tokens, cap is {cap}: "
                            + ", ".join(f"{k}={v}" for k, v in sizes.items()))
    text = "\n\n".join(s.text.strip("\n") for s in keep if s.name not in dropped)
    return text, dict(prompt_tokens_est=total, sections=sizes, dropped=dropped)


# ---------------------------------------------------------------- the client

class LLM:
    def __init__(self, cfg, tokens):
        self.cfg = cfg
        self.tokens = tokens
        self.context = {}          # (base, model) -> max_model_len
        self.lock = threading.Lock()
        self.events = None         # set by agent2.py: every call, from every role, goes to the log

    def connect(self):
        """Ask every endpoint in use what it serves, before the first call. A wrong URL or model name
        then fails with one line instead of a failed thread."""
        import httpx
        for role in self.cfg.roles.values():
            key = (role.base, role.model)
            if key in self.context:
                continue
            try:
                r = httpx.get(f"{role.base}/models", timeout=30, verify=False)
                r.raise_for_status()
                served = {m["id"]: m for m in r.json().get("data", [])}
            except Exception as e:
                sys.exit(f"cannot list the models at {role.base}/models: {type(e).__name__}: {e}")
            if role.model not in served:
                sys.exit(f"{role.base} does not serve {role.model!r}; it serves {sorted(served)}")
            self.context[key] = served[role.model].get("max_model_len") or 4096
        return self.context

    def chat(self, role_name, sections, tags=None, temperature=None, max_tokens=None):
        """Pack the sections for this role, send one request, return (text, meta).

        meta carries the packer's report, the sampler settings and the server's token counts, and
        becomes one `call` line in the log."""
        from agents2.config import CONTEXT_MARGIN
        role = self.cfg.roles[role_name]
        ctx = self.context.get((role.base, role.model), 8192)
        want = max_tokens or role.max_tokens
        cap = min(role.prompt_cap, ctx - want - CONTEXT_MARGIN)
        prompt, report = pack(sections, cap, self.tokens.count)
        budget = min(want, ctx - report["prompt_tokens_est"] - CONTEXT_MARGIN)
        temperature = role.temperature if temperature is None else temperature
        meta = dict(role=role_name, model=role.model, base=role.base, temperature=temperature,
                    top_p=role.top_p, max_tokens=budget, error=None, **report, **(tags or {}))
        # Never send "seed" (it killed a seat server) or "logprobs" (HTTP 500 on this plugin).
        body = dict(model=role.model, messages=[{"role": "user", "content": prompt}],
                    max_tokens=budget, temperature=temperature, top_p=role.top_p,
                    chat_template_kwargs={"enable_thinking": False})
        text, meta = self._send(role, body, meta, prompt)
        if self.events:
            self.events.write("call", **meta)
        return text, meta

    def _send(self, role, body, meta, prompt):
        import httpx
        t0 = time.perf_counter()
        r = None
        for attempt in range(self.cfg.retries + 1):
            try:
                r = httpx.post(f"{role.base}/chat/completions", json=body, timeout=900, verify=False)
            except httpx.HTTPError as e:
                meta["error"] = f"{type(e).__name__}: {e}"
            else:
                if r.status_code == 200:
                    meta["error"] = None
                    break
                meta["error"] = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code < 500:
                    break
            if attempt < self.cfg.retries:
                time.sleep(5 * 2 ** attempt)
        meta["gen_seconds"] = round(time.perf_counter() - t0, 2)
        meta["prompt"] = prompt
        if meta["error"]:
            return "", meta
        payload = r.json()
        usage = payload.get("usage") or {}
        ch = payload["choices"][0]
        msg = ch.get("message", {})
        content = msg.get("content") or ""
        meta.update(reply=content, finish_reason=ch.get("finish_reason"),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    reasoning_chars=len(next((msg[k] for k in REASONING_KEYS if msg.get(k)), "")))
        return content, meta


# ---------------------------------------------------------------- no model

class OfflineLLM:
    """No model. The planner gets a canned plan, the coder first a broken copy of the reference kernel
    (no @nki.jit) and then the reference itself, the debugger and reviewer a canned change. It
    exercises every role and the manager end to end with no endpoint. Never report a number."""

    def __init__(self, cfg, tokens):
        self.cfg = cfg
        self.tokens = tokens
        self.context = {}
        self.writes = {}
        self.lock = threading.Lock()
        self.events = None

    def connect(self):
        return {}

    def chat(self, role_name, sections, tags=None, temperature=None, max_tokens=None):
        role = self.cfg.roles[role_name]
        prompt, report = pack(sections, role.prompt_cap, self.tokens.count)
        tags = tags or {}
        meta = dict(role=role_name, model="offline", error=None, gen_seconds=0.0, prompt=prompt,
                    temperature=temperature, **report, **tags)
        level = tags.get("level", 1)
        if role_name == "planner":
            text = ("APPROACH: replay the reference kernel (offline)\nCALLS: nisa.dma_copy, nl.ndarray\n"
                    "LAYOUT: as the reference\nSTEPS: 1 replay")
        elif role_name == "coder":
            ref = open(f"reference_level{level}.py").read()
            text = (f"```python\n{ref.replace('@nki.jit', '', 1)}\n```" if tags.get("mode") == "write"
                    else f"```python\n{ref}\n```")
        else:
            text = "CAUSE: offline\nLINE: 1\nCHANGE: put @nki.jit back on the entry point"
        meta["reply"] = text
        if self.events:
            self.events.write("call", **meta)
        return text, meta


def numbered(code):
    """Code with line numbers, for the debugger, which has to name a line."""
    lines = code.splitlines()
    w = len(str(len(lines)))
    return "\n".join(f"{i + 1:>{w}}| {l}" for i, l in enumerate(lines))


def head(text, n=100):
    """A failure's fingerprint: its first n characters with every number replaced, so the same mistake
    at a different size still counts as the same mistake."""
    return re.sub(r"\d+", "N", (text or "")[:n])
