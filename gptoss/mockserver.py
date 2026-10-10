#!/usr/bin/env python3
"""
mockserver.py — a fake gpt-oss endpoint, for building while offline.

The real endpoint serves 4 concurrent sequences for the whole room, and it is
network-restricted. So: develop against this, then point at the real thing.

    python mockserver.py                    # listens on :8000
    GPTOSS_BASE_URL=http://localhost:8000 python chat.py
    GPTOSS_BASE_URL=http://localhost:8000 python web.py --port 8080
    GPTOSS_BASE_URL=http://localhost:8000 python probe.py --path /agg/v1 --quick

It imitates the parts that matter: the /agg and /disagg path prefixes, both API
surfaces, SSE streaming, a separate reasoning channel, `usage`, `system_fingerprint`,
greedy determinism, an 8192-token limit that rejects overlong prompts, and a
concurrency cap of 4 that queues rather than errors. It does NOT imitate the model —
replies are canned. Do not benchmark this and report the numbers.
"""

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 4
CHARS_PER_TOKEN = 4
sem = threading.Semaphore(MAX_NUM_SEQS)


def toks(s):
    return max(1, len(s) // CHARS_PER_TOKEN)


def reply_for(prompt):
    """Canned but deterministic — same input, same output, like the real greedy server."""
    p = prompt.lower()
    if "plum-4417" in p or "calibration key" in p:
        return "PLUM-4417"
    if "json array" in p:
        rows = [{"id": i, "name": f"row-{i}", "score": (i * 37) % 101,
                 "tag": ["alpha", "beta", "gamma"][i % 3]} for i in range(1, 41)]
        return json.dumps(rows)
    if "single word: ok" in p or "word ok" in p:
        return "ok"
    if "single word: ready" in p:
        return "ready"
    if "capital of france" in p:
        return " Paris."
    return ("This is the mock server. It returns canned text so you can build the client "
            "without the real endpoint. Point GPTOSS_BASE_URL at the real one when you are "
            "ready to measure anything.")


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = urlparse(self.path).path
        if path.endswith("/models"):
            self._json(200, {"object": "list",
                             "data": [{"id": "gpt-oss-20b", "object": "model"}]})
        elif path.endswith("/health"):
            self._json(200, {"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"] or 0)) or b"{}")
        is_chat = path.endswith("/chat/completions")
        if not (is_chat or path.endswith("/completions")):
            self.send_error(404)
            return

        tp = "tp16" if "/disagg/" in path else "tp32"
        fingerprint = f"vllm-0.21.0-{tp}-mock"

        if is_chat:
            prompt = "\n".join(m.get("content", "") for m in body.get("messages", []))
        else:
            prompt = body.get("prompt", "")
        max_tokens = int(body.get("max_tokens", 256))
        in_tok = toks(prompt)

        if in_tok + max_tokens > MAX_MODEL_LEN:
            self._json(400, {"error": {
                "message": f"This model's maximum context length is {MAX_MODEL_LEN} tokens. "
                           f"However, you requested {in_tok + max_tokens} tokens "
                           f"({in_tok} in the messages, {max_tokens} in the completion).",
                "type": "BadRequestError", "code": 400}})
            return

        text = reply_for(prompt)
        think = ("Considering the request and the constraints, then answering directly."
                 if is_chat else "")
        out_tok = toks(text)

        with sem:  # queues past 4, exactly like max_num_seqs
            if not body.get("stream"):
                time.sleep(0.15 + out_tok * 0.004)
                ch = ({"index": 0, "message": {"role": "assistant", "content": text,
                                               "reasoning": think},
                       "finish_reason": "stop"}
                      if is_chat else
                      {"index": 0, "text": text, "finish_reason": "stop"})
                self._json(200, {
                    "id": "mock", "object": "chat.completion" if is_chat else "text_completion",
                    "model": "gpt-oss-20b", "system_fingerprint": fingerprint, "choices": [ch],
                    "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok,
                              "total_tokens": in_tok + out_tok}})
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def sse(obj):
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()

            def frame(delta=None, text_piece=None, finish=None, usage=None):
                ch = ({"index": 0, "delta": delta or {}, "finish_reason": finish}
                      if is_chat else
                      {"index": 0, "text": text_piece or "", "finish_reason": finish})
                o = {"id": "mock", "model": "gpt-oss-20b", "system_fingerprint": fingerprint,
                     "object": "chat.completion.chunk" if is_chat else "text_completion",
                     "choices": [ch]}
                if usage:
                    o["usage"] = usage
                return o

            time.sleep(0.2)  # stand-in for prefill / time-to-first-token
            if is_chat and think:
                for w in re.findall(r"\S+\s*", think):
                    sse(frame(delta={"reasoning": w}))
                    time.sleep(0.01)
            for w in re.findall(r"\S+\s*|\s+", text):
                sse(frame(delta={"content": w}, text_piece=w))
                time.sleep(0.008)
            sse(frame(finish="stop",
                      usage={"prompt_tokens": in_tok, "completion_tokens": out_tok,
                             "total_tokens": in_tok + out_tok}))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"mock gpt-oss on http://localhost:{a.port}  "
          f"(paths /agg/v1/... and /disagg/v1/... both work)")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
