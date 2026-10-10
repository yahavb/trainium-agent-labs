#!/usr/bin/env python3
"""
dashboard/server.py — serves the traces. Python stdlib only, so it runs on the laptop
next to the synced runs/ directory with nothing installed.

    python dashboard/server.py                 # http://localhost:8765
    python dashboard/server.py --runs ../runs --port 8765

The agent appends to runs/<run_id>/trace.jsonl while it works; the dashboard reads from
disk on every API request, so a run in progress simply appears and grows. A run is
"running" until meta.json carries "done".
"""

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

_cache = {}   # path -> (mtime, parsed)


def _read_cached(path, parse):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    hit = _cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        value = parse(path)
    except Exception:
        return None
    _cache[path] = (mtime, value)
    return value


def read_trace(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue        # a partially flushed line -- the next poll gets it
    return records


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def summarize(run_dir):
    """Per-run rollup for the runs table and the top-of-page KPIs."""
    meta = _read_cached(os.path.join(run_dir, "meta.json"), _load_json) or {}
    trace = _read_cached(os.path.join(run_dir, "trace.jsonl"), read_trace) or []
    levels = {}
    taxonomy = {}
    ended = set()          # levels whose level_end arrived -- their counts are final
    tokens_by_attempt = []
    attempts = 0
    tool_calls = 0
    for rec in trace:
        t = rec.get("type")
        if t == "attempt":
            attempts += 1
            lv = str(rec.get("level"))
            L = levels.setdefault(lv, dict(attempts=0, solved=False, best=0.0,
                                           solved_round=None, taxonomies={}))
            L["attempts"] += 1
            L["best"] = max(L["best"], rec.get("reward") or 0.0)
            if rec.get("solved"):
                L["solved"] = True
                L["solved_round"] = rec.get("round")
            tax = rec.get("taxonomy")
            # live taxonomy while the level runs; superseded by level_end below (which
            # carries the same counts -- adding both would double every failure)
            if tax and not rec.get("solved") and lv not in ended:
                L["taxonomies"][tax] = L["taxonomies"].get(tax, 0) + 1
            tok = rec.get("tokens") or {}
            if tok.get("parts"):
                tokens_by_attempt.append(dict(level=lv, round=rec.get("round"),
                                              solved=bool(rec.get("solved")),
                                              parts=tok["parts"],
                                              api=tok.get("api_prompt_tokens")))
        elif t == "tool":
            tool_calls += 1
        elif t == "level_end":
            lv = str(rec.get("level"))
            ended.add(lv)
            L = levels.setdefault(lv, dict(attempts=0, solved=False, best=0.0,
                                           solved_round=None, taxonomies={}))
            L["solved"] = bool(rec.get("solved"))
            if rec.get("solved"):
                L["solved_round"] = rec.get("solved_round")
            L["taxonomies"] = dict(rec.get("taxonomy_counts") or {})
    for lv, L in levels.items():
        for tax, n in L["taxonomies"].items():
            taxonomy[tax] = taxonomy.get(tax, 0) + n
    solved_levels = sorted([lv for lv, L in levels.items() if L["solved"]], key=int)
    return dict(attempts=attempts, tool_calls=tool_calls, levels=levels,
                taxonomy=taxonomy, tokens_by_attempt=tokens_by_attempt,
                solved_levels=solved_levels, solved_n=len(solved_levels))


class Handler(BaseHTTPRequestHandler):
    runs_dir = "."

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/runs":
            runs = []
            if os.path.isdir(self.runs_dir):
                for name in sorted(os.listdir(self.runs_dir), reverse=True):
                    d = os.path.join(self.runs_dir, name)
                    if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "meta.json")):
                        continue
                    meta = _read_cached(os.path.join(d, "meta.json"), _load_json) or {}
                    s = summarize(d)
                    runs.append(dict(run_id=name, model=meta.get("model"),
                                     tag=(meta.get("args") or {}).get("tag") or "",
                                     levels=meta.get("levels"), rounds=meta.get("rounds"),
                                     samples=meta.get("samples"), repeat=meta.get("repeat"),
                                     greedy=meta.get("greedy"), started=meta.get("started"),
                                     done=meta.get("done"), summary=s))
            runs.sort(key=lambda r: r.get("started") or 0, reverse=True)
            return self._send(200, json.dumps(runs).encode())

        m = re.match(r"^/api/run/([\w.\-]+)$", self.path)
        if m:
            d = os.path.join(self.runs_dir, m.group(1))
            meta = _read_cached(os.path.join(d, "meta.json"), _load_json) or {}
            trace = _read_cached(os.path.join(d, "trace.jsonl"), read_trace) or []
            # a trace line carries the full prompt only in the tokens' parts; the raw
            # prompt text is not stored (too big) -- the reply IS, capped
            return self._send(200, json.dumps(dict(meta=meta, trace=trace)).encode())

        if self.path == "/api/summary":
            agg = {}
            if os.path.isdir(self.runs_dir):
                for name in sorted(os.listdir(self.runs_dir)):
                    d = os.path.join(self.runs_dir, name)
                    if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "meta.json")):
                        continue
                    s = summarize(d)
                    for lv, L in s["levels"].items():
                        A = agg.setdefault(lv, dict(runs=0, solved_runs=0, best=0.0,
                                                    rewards=[], attempts=0,
                                                    taxonomies={}))
                        A["runs"] += 1
                        A["solved_runs"] += 1 if L["solved"] else 0
                        A["best"] = max(A["best"], L["best"])
                        A["rewards"].append(L["best"])
                        A["attempts"] += L["attempts"]
                        for tax, n in L["taxonomies"].items():
                            A["taxonomies"][tax] = A["taxonomies"].get(tax, 0) + n
            for lv, A in agg.items():
                A["mean"] = (sum(A["rewards"]) / len(A["rewards"])) if A["rewards"] else 0.0
                A["solve_rate"] = A["solved_runs"] / A["runs"] if A["runs"] else 0.0
                del A["rewards"]
            taxonomy = {}
            for A in agg.values():
                for tax, n in A["taxonomies"].items():
                    taxonomy[tax] = taxonomy.get(tax, 0) + n
            return self._send(200, json.dumps(dict(levels=agg, taxonomy=taxonomy)).encode())

        # static files
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            path = "/index.html"
        safe = os.path.normpath(path).lstrip("/")
        full = os.path.join(STATIC, safe)
        if not full.startswith(STATIC) or not os.path.isfile(full):
            return self._send(404, b"not found", "text/plain")
        ctype = {"html": "text/html", "js": "text/javascript",
                 "css": "text/css"}.get(full.rsplit(".", 1)[-1], "application/octet-stream")
        with open(full, "rb") as f:
            return self._send(200, f.read(), ctype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "runs"))
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    Handler.runs_dir = a.runs
    print(f"dashboard: {a.runs} -> http://localhost:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
