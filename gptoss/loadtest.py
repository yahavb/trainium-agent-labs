#!/usr/bin/env python3
"""
loadtest.py — how many people can actually share this endpoint?

`probe.py` check 9/10 fire N requests at once and measure raw throughput. That is the
wrong shape for answering "can 100 students use this." A person asks a question, waits
for the answer, then spends 20-40 seconds reading it and typing the next one — so 100
students do NOT mean 100 requests in flight. This script simulates that closed loop.

    export GPTOSS_BASE_URL="https://<the-load-balancer>"
    python loadtest.py                              # the default sweep
    python loadtest.py --users 100 --think 20       # one scenario
    python loadtest.py --users 25 --think 0         # agent loops: no human pauses
    python loadtest.py --users 100 --think 20 --max-tokens 1000   # longer answers

The number that matters is p95: what the unluckiest one in twenty actually waits.

IMPORTANT: `--max-tokens` dominates everything. Request rate scales roughly inversely
with answer length, so numbers taken at 220 tokens do not transfer to 1000-token answers.
Measure YOUR generation length.
"""

import argparse
import os
import random
import statistics
import sys
import threading
import time
import warnings

import httpx

warnings.filterwarnings("ignore")

MODEL = "gpt-oss-20b"

# Varied on purpose. Repeating one prompt measures differently (we saw identical prompts
# at high concurrency come out ~20% WORSE than varied ones), so a single-prompt benchmark
# will mislead you.
QUESTIONS = [
    "Explain how a hash table handles collisions.",
    "Write a Python function to reverse a linked list.",
    "What is the difference between a process and a thread?",
    "How does gradient descent work? Keep it short.",
    "Explain Big-O notation with two examples.",
    "What does a load balancer actually do?",
    "Write a SQL query to find duplicate rows in a table.",
    "Why is floating point arithmetic not associative?",
]


def simulate(base, paths, n_users, think, duration, max_tokens, quiet=False):
    lat, errors, lock = [], [0], threading.Lock()
    stop_at = time.time() + duration

    def student(i):
        c = httpx.Client(verify=False, timeout=httpx.Timeout(900.0, connect=20.0))
        path = paths[i % len(paths)]          # spread the room across every endpoint
        time.sleep(random.uniform(0, max(think, 1.0)))   # stagger arrivals
        while time.time() < stop_at:
            q = random.choice(QUESTIONS)
            t0 = time.perf_counter()
            try:
                with c.stream("POST", f"{base}{path}/chat/completions",
                              json={"model": MODEL,
                                    "messages": [{"role": "user", "content": q}],
                                    "max_tokens": max_tokens, "stream": True}) as r:
                    if r.status_code != 200:
                        with lock:
                            errors[0] += 1
                        continue
                    for _ in r.iter_lines():
                        pass
            except Exception:
                with lock:
                    errors[0] += 1
                continue
            with lock:
                lat.append(time.perf_counter() - t0)
            # read the answer, think, type the next message
            time.sleep(think * random.uniform(0.7, 1.3))
        c.close()

    threads = [threading.Thread(target=student, args=(i,), daemon=True)
               for i in range(n_users)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=duration + 120)
    wall = time.time() - t0

    s = sorted(lat)
    if not s:
        return dict(ok=False, errors=errors[0])
    return dict(ok=True, n=len(s), wall=wall, rps=len(s) / wall,
                p50=statistics.median(s), p95=s[int(len(s) * 0.95) - 1],
                worst=s[-1], errors=errors[0])


def show(label, r):
    if not r["ok"]:
        print(f"  {label:<42} no completions ({r['errors']} errors)")
        return
    print(f"  {label:<42} {r['n']:>4} replies  {r['rps']:>5.2f} req/s  "
          f"p50={r['p50']:>6.1f}s  p95={r['p95']:>6.1f}s  worst={r['worst']:>6.1f}s  "
          f"errs={r['errors']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("GPTOSS_BASE_URL"))
    ap.add_argument("--path", action="append",
                    help="endpoint path, repeatable (default: both /agg/v1 and /disagg/v1)")
    ap.add_argument("--users", type=int, help="simulated people")
    ap.add_argument("--think", type=float,
                    help="seconds between a reply and the next question (0 = an agent loop)")
    ap.add_argument("--duration", type=int, default=100, help="seconds per scenario")
    ap.add_argument("--max-tokens", type=int, default=220,
                    help="answer length. THIS DOMINATES THE RESULT.")
    a = ap.parse_args()

    if not a.base:
        sys.exit("Set GPTOSS_BASE_URL (or pass --base). Ask the organisers for the URL.")
    base = a.base.rstrip("/")
    paths = a.path or ["/agg/v1", "/disagg/v1"]

    print(f"base={base}  endpoints={paths}  max_tokens={a.max_tokens}  "
          f"{a.duration}s per scenario")
    print("'think' = seconds a person spends reading the answer and typing the next one.\n")

    if a.users is not None:
        think = a.think if a.think is not None else 20.0
        show(f"{a.users} users, {think:g}s think",
             simulate(base, paths, a.users, think, a.duration, a.max_tokens))
    else:
        for users, think, note in [(100, 40, "browsing"), (100, 20, "working"),
                                   (100, 8, "heads-down"), (25, 0, "agent loops")]:
            show(f"{users} users, {think}s think ({note})",
                 simulate(base, paths, users, think, a.duration, a.max_tokens))
            time.sleep(5)   # let the queue drain between scenarios

    print("\nRead p95, not p50 — it is what the unluckiest one in twenty waits.")
    print("Then re-run with YOUR --max-tokens. Answer length dominates everything here.")


if __name__ == "__main__":
    main()
