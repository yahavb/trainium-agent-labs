#!/usr/bin/env python3
"""bench_dflash2.py — decode throughput + DFlash2 acceptance against a running vLLM.

Usage: python3 /workspace/bench_dflash2.py [--url http://localhost:8000] [--max-tokens 128] [--runs 3]
Greedy (temperature 0), one request at a time, fixed prompts. Prints per-run tokens/s,
the generated text of the first prompt (sanity check), and speculative-decoding counters
from /metrics (accepted vs drafted tokens) when they exist.
"""
import argparse
import json
import time
import urllib.request

PROMPTS = [
    "Explain how a hash map works, step by step.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Summarize the causes of the French Revolution in a short paragraph.",
]


def post(url, payload, timeout=300):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def spec_metrics(base):
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception as e:
        return {"error": str(e)}
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or "spec_decode" not in line:
            continue
        name, _, value = line.rpartition(" ")
        try:
            out[name] = float(value)
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()

    model = json.loads(urllib.request.urlopen(f"{a.url}/v1/models", timeout=10).read())["data"][0]["id"]
    before = spec_metrics(a.url)

    # Warm request (excluded from timing)
    post(f"{a.url}/v1/completions", {"model": model, "prompt": "Hi", "max_tokens": 8, "temperature": 0})

    rates, total_toks, total_s = [], 0, 0.0
    for run in range(a.runs):
        for i, p in enumerate(PROMPTS):
            t0 = time.perf_counter()
            r = post(f"{a.url}/v1/completions",
                     {"model": model, "prompt": p, "max_tokens": a.max_tokens, "temperature": 0})
            dt = time.perf_counter() - t0
            n = r["usage"]["completion_tokens"]
            rates.append(n / dt)
            total_toks += n
            total_s += dt
            if run == 0 and i == 0:
                print("--- sample output (prompt 1) ---")
                print(r["choices"][0]["text"][:500])
                print("--------------------------------")
            print(f"run {run} prompt {i}: {n} tokens in {dt:.2f}s = {n/dt:.1f} tok/s")

    after = spec_metrics(a.url)
    print(f"\nmodel: {model}")
    print(f"overall: {total_toks} tokens in {total_s:.1f}s = {total_toks/total_s:.1f} tok/s "
          f"(per-request median {sorted(rates)[len(rates)//2]:.1f} tok/s)")
    deltas = {k: after[k] - before.get(k, 0.0) for k in after if isinstance(after.get(k), float)}
    if deltas:
        print("speculative decoding counters (delta over benchmark):")
        for k, v in sorted(deltas.items()):
            print(f"  {k} = {v:.0f}")
        drafted = sum(v for k, v in deltas.items() if "num_draft_tokens" in k)
        accepted = sum(v for k, v in deltas.items() if "num_accepted_tokens" in k and "per_pos" not in k)
        if drafted:
            print(f"acceptance rate: {accepted/drafted:.1%} ({accepted:.0f}/{drafted:.0f} draft tokens)")
    else:
        print("no spec_decode metrics found (baseline server, or metrics disabled)")


if __name__ == "__main__":
    main()
