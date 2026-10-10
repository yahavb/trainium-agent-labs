#!/usr/bin/env python3
"""Opt-in live controller; original main/solve/ask/prompt/extraction stay in use."""

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time

from curriculum import evaluate, process_guard


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Instrumentation:
    def __init__(self, agent, policy, requests, grades):
        self.agent, self.policy = agent, policy
        self.requests, self.grades = requests, grades
        self.local = threading.local()
        self.lock = threading.Lock()
        self.context = {}
        self.runs = {}
        self.round = 0
        self.sample = 0

    def append(self, stream, row):
        with self.lock:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()

    def post(self, original, url, **kwargs):
        context = self.local.context
        body = kwargs["json"]
        prompt = body["messages"][0]["content"]
        row = dict(context, started_at_utc=utc(),
                   prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                   request=body, endpoint=url,
                   request_options={k: v for k, v in kwargs.items() if k != "json"})
        started = time.perf_counter()
        try:
            response = original(url, **kwargs)
            row["http_status"] = response.status_code
            try:
                payload = response.json()
            except ValueError:
                row["non_json_response"] = response.text
            else:
                row["response"] = payload
                row["response_metadata"] = {k: v for k, v in payload.items() if k not in ("choices", "usage")}
                row["usage"] = payload.get("usage")
                row["finish_reasons"] = [c.get("finish_reason") for c in payload.get("choices", [])]
            return response
        except BaseException as exc:
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            row.update(ended_at_utc=utc(), elapsed_seconds=time.perf_counter() - started)
            self.append(self.requests, row)

    def parallel(self, args, prompt, n):
        process_guard()
        self.sample = 0
        context = dict(self.context, round=self.round)
        self.round += 1

        def one(sample):
            self.local.context = dict(context, sample=sample)
            return self.original_ask(args, prompt)

        # Same fan-out and submission-order collection as canonical ask_parallel.
        with cf.ThreadPoolExecutor(max_workers=n) as executor:
            futures = [executor.submit(one, sample) for sample in range(1, n + 1)]
            return [future.result() for future in futures]

    def solve(self, args, level, log):
        self.runs[level] = self.runs.get(level, 0) + 1
        self.context = {"run": self.runs[level], "level": level, "policy": self.policy}
        self.round = 0
        return self.original_solve(args, level, log)

    def grade(self, source, level):
        process_guard()
        self.sample += 1
        result, details = evaluate(self.agent, source, level, self.policy)
        row = dict(self.context, round=self.round - 1, sample=self.sample,
                   code_sha256=hashlib.sha256(source.encode()).hexdigest(),
                   reward=result[0], parts=result[1], feedback=result[2], observed_at_utc=utc(),
                   **details)
        self.append(self.grades, row)
        return result

    def install(self):
        import httpx
        self.original_ask = self.agent.ask
        self.original_solve = self.agent.solve
        originals = (self.agent.grade, self.agent.ask_parallel, self.agent.solve, httpx.post)
        self.agent.grade = self.grade
        self.agent.ask_parallel = self.parallel
        self.agent.solve = self.solve
        httpx.post = lambda url, **kw: self.post(originals[3], url, **kw)
        return originals

    def restore(self, originals):
        import httpx
        self.agent.grade, self.agent.ask_parallel, self.agent.solve, httpx.post = originals
