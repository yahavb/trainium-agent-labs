"""The log. One JSON line per model call, plan, check, change, verdict and thread end.

attempts.jsonl is also written in agent.py's schema (run, level, round, sample, reward, parts, code_sha,
code, feedback), so the comparison scripts read both systems' runs.
"""

import json
import threading
import time


class Events:
    def __init__(self, path, attempts_path=None):
        self.f = open(path, "a")
        self.attempts = open(attempts_path, "a") if attempts_path else None
        self.lock = threading.Lock()

    def write(self, event, **fields):
        line = json.dumps(dict(t=round(time.time(), 2), event=event, **fields), default=str)
        with self.lock:
            self.f.write(line + "\n")
            self.f.flush()

    def attempt(self, **fields):
        if not self.attempts:
            return
        line = json.dumps(fields, default=str)
        with self.lock:
            self.attempts.write(line + "\n")
            self.attempts.flush()

    def close(self):
        self.f.close()
        if self.attempts:
            self.attempts.close()
