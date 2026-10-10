#!/usr/bin/env python3
"""
bench_serving.py -- Qwen3-8B as served on this pod: time to first token, prefill throughput, decode
latency per token, end-to-end latency. The reference every attention change is measured against.

    python qwen3/bench_serving.py --label stock                     # against http://localhost:8000
    python qwen3/bench_serving.py --label stock --concurrency 4     # the server's 4 slots full
    python qwen3/bench_serving.py --input-lens 512 2048 --output-len 64 --repeats 10

HOW. /v1/completions with the prompt given as token ids, so every prompt is exactly N tokens with no
tokenizer in the loop; ignore_eos and max_tokens=M, so exactly M tokens come back; greedy; streamed,
so the arrival of every token is timed here on the client. Prompts are the same every run (fixed
seed). Each shape gets one warm-up request that is not counted.

  TTFT      request sent -> first token received. At concurrency 1: prefill + the first decode step
            + HTTP and the server's own Python. This is the prefill number.
  TPOT      (end-to-end - TTFT) / (M - 1): decode time per output token, per sequence.
  ITL       the gaps between consecutive streamed tokens, p50 / p90 / p99.
  prefill tok/s = N / TTFT     decode tok/s = 1 / TPOT     throughput = all output tokens / wall time

These are client-side timings: what a user of the server sees, including HTTP. Attention is one
part of them; the device profile, later, says how big a part. Every raw sample goes to
qwen3/results/serving_<label>_<time>.json, so later runs can be compared sample for sample.

Uses only the standard library, so it runs with any python in the pod.
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def get_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def prompt_tokens(n, seed):
    """n token ids from Qwen's ordinary vocabulary (special tokens start at 151643), the same every
    run. The values do not change the cost of attention; only the count does."""
    x, out = seed * 2654435761 % 2**32, []
    for _ in range(n):
        x = (1103515245 * x + 12345) % 2**31
        out.append(1000 + x % 149000)
    return out


def one_request(base, model, tokens, out_len, timeout):
    """Send one streamed completion; return its timings, or raise."""
    body = json.dumps({
        "model": model, "prompt": tokens, "max_tokens": out_len, "temperature": 0.0,
        "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{base}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    arrivals, usage = [], None
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("choices") and chunk["choices"][0].get("text") is not None:
                arrivals.append(time.perf_counter())
    t_end = time.perf_counter()
    if not arrivals:
        raise RuntimeError("the stream returned no tokens")
    n_out = (usage or {}).get("completion_tokens") or len(arrivals)
    n_in = (usage or {}).get("prompt_tokens") or len(tokens)
    ttft = arrivals[0] - t0
    e2e = t_end - t0
    return dict(prompt_tokens=n_in, output_tokens=n_out, chunks=len(arrivals), ttft=ttft, e2e=e2e,
                tpot=(e2e - ttft) / (n_out - 1) if n_out > 1 else None,
                itl=[b - a for a, b in zip(arrivals, arrivals[1:])], start=t0, end=t_end)


def run_shape(base, model, n_in, out_len, concurrency, repeats, timeout):
    """One warm-up, then `repeats` rounds of `concurrency` simultaneous requests."""
    one_request(base, model, prompt_tokens(n_in, seed=n_in), out_len, timeout)
    samples, walls = [], []
    for rep in range(repeats):
        results, errors = [None] * concurrency, []

        def worker(i):
            try:
                results[i] = one_request(base, model, prompt_tokens(n_in, seed=n_in * 1000 + rep * 10 + i),
                                         out_len, timeout)
            except Exception as e:                     # recorded, and the shape is marked failed
                errors.append(f"{type(e).__name__}: {e}")
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        walls.append(time.perf_counter() - t0)
        if errors:
            return dict(error=errors[0], samples=samples)
        samples.extend(results)
    return dict(samples=samples, walls=walls)


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(n_in, res, concurrency):
    s = res["samples"]
    ttft = [x["ttft"] for x in s]
    tpot = [x["tpot"] for x in s]
    itl = [g for x in s for g in x["itl"]]
    e2e = [x["e2e"] for x in s]
    out_tok = sum(x["output_tokens"] for x in s)
    short = [x for x in s if x["output_tokens"] != s[0]["output_tokens"] or x["prompt_tokens"] != n_in]
    return dict(input_len=n_in, n=len(s), concurrency=concurrency,
                ttft_p50_ms=pct(ttft, 50) * 1e3, ttft_p90_ms=pct(ttft, 90) * 1e3,
                ttft_spread_ms=(max(ttft) - min(ttft)) * 1e3,
                prefill_tok_s=n_in / pct(ttft, 50),
                tpot_p50_ms=pct(tpot, 50) * 1e3, tpot_p90_ms=pct(tpot, 90) * 1e3,
                itl_p50_ms=pct(itl, 50) * 1e3, itl_p90_ms=pct(itl, 90) * 1e3, itl_p99_ms=pct(itl, 99) * 1e3,
                e2e_p50_s=pct(e2e, 50), e2e_p90_s=pct(e2e, 90),
                decode_tok_s=1e3 / (pct(tpot, 50) * 1e3) if s[0]["tpot"] else float("nan"),
                throughput_tok_s=out_tok / sum(res["walls"]),
                chunks_per_token=statistics.mean(x["chunks"] / x["output_tokens"] for x in s),
                inconsistent=len(short))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--label", default="stock", help="names the results file, e.g. stock or agent_v1")
    ap.add_argument("--input-lens", type=int, nargs="+", default=[128, 512, 1024, 2048, 3584])
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-model-len", type=int, default=4096, help="serve.sh's MAX_MODEL_LEN")
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()

    try:
        models = get_json(f"{a.base}/v1/models")["data"]
    except Exception as e:
        sys.exit(f"No model server at {a.base} ({type(e).__name__}: {e}). Start it with "
                 f"/workspace/serve.sh and wait for READY.")
    model = models[0]["id"]
    print(f"server {a.base}   model {model}   output {a.output_len} tokens   concurrency "
          f"{a.concurrency}   {a.repeats} repeats per shape (+1 warm-up)")

    shapes = [n for n in a.input_lens if n + a.output_len <= a.max_model_len]
    for n in sorted(set(a.input_lens) - set(shapes)):
        print(f"  skipping input {n}: {n} + {a.output_len} exceeds max-model-len {a.max_model_len}")

    raw, rows = {}, []
    for n in shapes:
        t0 = time.perf_counter()
        res = run_shape(a.base, model, n, a.output_len, a.concurrency, a.repeats, a.timeout)
        raw[n] = res
        if "error" in res:
            print(f"  input {n:>5}: FAILED -- {res['error'][:300]}")
            continue
        row = summarize(n, res, a.concurrency)
        rows.append(row)
        print(f"  input {n:>5}: done in {time.perf_counter() - t0:.0f} s"
              + (f"  ({row['inconsistent']} samples returned an unexpected token count)"
                 if row["inconsistent"] else ""))

    print(f"\nQWEN3-8B SERVING -- {a.label}, client-side, median over {a.repeats * a.concurrency} "
          f"requests per row")
    print(f"  {'input':>6} {'TTFT p50':>9} {'p90':>8} {'spread':>7} {'prefill':>9}   {'TPOT p50':>9} {'p90':>7}"
          f"   {'ITL p50':>8} {'p99':>7}   {'e2e p50':>8}   {'decode':>8} {'thruput':>8}")
    print(f"  {'tokens':>6} {'ms':>9} {'ms':>8} {'ms':>7} {'tok/s':>9}   {'ms':>9} {'ms':>7}"
          f"   {'ms':>8} {'ms':>7}   {'s':>8}   {'tok/s':>8} {'tok/s':>8}")
    for r in rows:
        print(f"  {r['input_len']:>6} {r['ttft_p50_ms']:9.1f} {r['ttft_p90_ms']:8.1f} {r['ttft_spread_ms']:7.1f} "
              f"{r['prefill_tok_s']:9.0f}   {r['tpot_p50_ms']:9.2f} {r['tpot_p90_ms']:7.2f}   "
              f"{r['itl_p50_ms']:8.2f} {r['itl_p99_ms']:7.2f}   {r['e2e_p50_s']:8.2f}   "
              f"{r['decode_tok_s']:8.1f} {r['throughput_tok_s']:8.1f}")
    if rows and abs(rows[0]["chunks_per_token"] - 1) > 0.05:
        print(f"  note: {rows[0]['chunks_per_token']:.2f} streamed chunks per token -- the server batches "
              f"tokens into chunks, so ITL is per chunk, not per token. TPOT is unaffected.")

    os.makedirs(RESULTS, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"{time.time() % 1:.3f}"[1:]
    path = os.path.join(RESULTS, f"serving_{a.label}_c{a.concurrency}_{stamp}.json")
    with open(path, "w") as f:
        json.dump(dict(label=a.label, model=model, base=a.base, output_len=a.output_len,
                       concurrency=a.concurrency, repeats=a.repeats, summary=rows,
                       raw={str(k): v for k, v in raw.items()}), f, indent=1)
    print(f"\nraw samples: {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
