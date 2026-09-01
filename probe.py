#!/usr/bin/env python3
"""
probe.py — is this gpt-oss-20b-on-Trainium endpoint good enough for real work?

Answers, with measurements rather than vibes:
  * does /v1/chat/completions work, or only /v1/completions?
  * does streaming work?
  * what is the real usable context, and how does it fail when you exceed it?
  * can it find a fact buried deep in a long document? (long-sequence sanity)
  * can it hold a format (valid JSON) over a long generation?
  * what does concurrency do past max_num_seqs=4?
  * is a repeated long prefix any cheaper? (it should not be: prefix caching is off)
  * is sampling actually greedy/deterministic?
  * /agg vs /disagg: TTFT and tokens/sec, alone and under load

Usage:
    export GPTOSS_BASE_URL="https://<the-load-balancer>"
    python probe.py                  # both endpoints, full suite
    python probe.py --path /agg/v1   # one endpoint only
    python probe.py --quick          # skip the slow concurrency + long-context sweeps
    python probe.py --out results.json

Nothing here needs a GPU, a cluster login, or kubectl. Just network reach to the endpoint.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

MODEL = "gpt-oss-20b"
# TLS cert on the demo LB does not match its hostname, hence verify=False.
HTTP = dict(verify=False, timeout=httpx.Timeout(900.0, connect=20.0))

# The servers are launched with --max-model-len 8192 for prompt + completion TOGETHER.
DECLARED_CTX = 8192
# Rough token estimate; good enough for building test payloads. Server `usage` is truth.
CHARS_PER_TOKEN = 4

results = {}


def say(*a):
    print(*a, flush=True)


def hdr(t):
    say("\n" + "=" * 72)
    say(t)
    say("=" * 72)


def post(client, path, body, stream=False):
    """Returns (ok, payload_or_error, wall_seconds, ttft_seconds_or_None)."""
    t0 = time.perf_counter()
    if not stream:
        try:
            r = client.post(path, json=body)
        except Exception as e:
            return False, f"{type(e).__name__}: {e}", time.perf_counter() - t0, None
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}: {r.text[:400]}", dt, None
        return True, r.json(), dt, None

    ttft, chunks = None, []
    try:
        with client.stream("POST", path, json={**body, "stream": True}) as r:
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}: {r.read()[:400]!r}", time.perf_counter() - t0, None
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks.append(data)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", time.perf_counter() - t0, ttft
    return True, chunks, time.perf_counter() - t0, ttft


def text_of(payload):
    """Pull generated text out of either a completions or chat.completions response."""
    ch = payload["choices"][0]
    if "text" in ch:
        return ch["text"]
    msg = ch.get("message", {})
    # gpt-oss emits its chain of thought on a separate channel; vLLM may surface it here.
    return (msg.get("content") or "") or (msg.get("reasoning_content") or "")


def usage_of(payload):
    u = payload.get("usage") or {}
    return u.get("prompt_tokens"), u.get("completion_tokens")


def filler(n_tokens, seed_offset=0):
    """Distinct, non-repetitive prose of roughly n_tokens tokens."""
    out, i = [], seed_offset
    while sum(len(s) for s in out) < n_tokens * CHARS_PER_TOKEN:
        i += 1
        out.append(
            f"Section {i}. Field report {i} records that survey unit {i % 37} logged "
            f"{(i * 7919) % 1000} observations under protocol {chr(65 + i % 26)}, with "
            f"calibration drift of {(i % 50) / 10:.1f} percent and no anomalies noted. "
        )
    return "".join(out)


# ---------------------------------------------------------------- checks


def check_models(client, path):
    hdr(f"[1] {path} — /models reachable?")
    t0 = time.perf_counter()
    try:
        r = client.get(f"{path}/models")
    except Exception as e:
        say(f"  FAIL  {type(e).__name__}: {e}")
        say("  If this hung with no HTTP response, your IP is probably not allowlisted.")
        return False
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        say(f"  FAIL  HTTP {r.status_code}: {r.text[:200]}")
        return False
    ids = [m["id"] for m in r.json().get("data", [])]
    say(f"  ok    {dt*1000:.0f} ms   models={ids}")
    results[path]["models"] = ids
    return True


def check_apis(client, path):
    """Which of the two APIs actually work, and does streaming work on them?"""
    hdr(f"[2] {path} — which API surfaces work?")
    findings = {}

    ok, p, dt, _ = post(client, f"{path}/completions",
                        {"model": MODEL, "prompt": "The capital of France is", "max_tokens": 8})
    findings["completions"] = ok
    say(f"  /completions       {'ok  ' if ok else 'FAIL'} {dt:.2f}s  "
        f"{repr(text_of(p))[:60] if ok else p}")
    if ok:
        findings["fingerprint"] = p.get("system_fingerprint")
        say(f"  fingerprint        {findings['fingerprint']}")

    ok, p, dt, _ = post(client, f"{path}/chat/completions",
                        {"model": MODEL,
                         "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
                         "max_tokens": 16})
    findings["chat_completions"] = ok
    say(f"  /chat/completions  {'ok  ' if ok else 'FAIL'} {dt:.2f}s  "
        f"{repr(text_of(p))[:60] if ok else p}")
    if ok:
        ch = p["choices"][0].get("message", {})
        findings["has_reasoning_channel"] = bool(ch.get("reasoning_content"))
        if findings["has_reasoning_channel"]:
            say("  NOTE  response carries a separate reasoning_content channel — a chat UI must "
                "decide whether to show it.")

    api = f"{path}/chat/completions" if findings["chat_completions"] else f"{path}/completions"
    body = ({"model": MODEL, "messages": [{"role": "user", "content": "Count 1 to 20."}], "max_tokens": 80}
            if findings["chat_completions"] else
            {"model": MODEL, "prompt": "Count 1 to 20.", "max_tokens": 80})
    ok, p, dt, ttft = post(client, api, body, stream=True)
    findings["streaming"] = ok
    if ok:
        say(f"  streaming          ok   {dt:.2f}s  "
            f"ttft={f'{ttft:.2f}s' if ttft else 'n/a'}  chunks={len(p)}")
    else:
        say(f"  streaming          FAIL {dt:.2f}s  {p}")

    results[path]["apis"] = findings
    return findings


def check_context_limit(client, path, api, quick):
    """Find the real usable prompt size and see how it fails past the edge."""
    hdr(f"[3] {path} — real usable context (declared {DECLARED_CTX})")
    sizes = [1024, 4096, 7000] if quick else [1024, 2048, 4096, 6144, 7000, 7800, 8192, 9000]
    found = {}
    for want in sizes:
        prompt = filler(want) + "\n\nReply with the single word: ok"
        body = (dict(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=8)
                if "chat" in api else dict(model=MODEL, prompt=prompt, max_tokens=8))
        ok, p, dt, _ = post(client, api, body)
        if ok:
            pt, ct = usage_of(p)
            say(f"  ~{want:>5} tok  ok    {dt:6.2f}s   server counted prompt_tokens={pt}")
            found[want] = {"ok": True, "prompt_tokens": pt, "seconds": dt}
        else:
            short = re.sub(r"\s+", " ", str(p))[:220]
            say(f"  ~{want:>5} tok  REJECTED  {short}")
            found[want] = {"ok": False, "error": short}
    okd = [k for k, v in found.items() if v["ok"]]
    say(f"\n  Largest prompt accepted in this sweep: ~{max(okd) if okd else 0} tokens")
    say("  Remember this budget is prompt + completion together — asking for 2000 output "
        "tokens costs you 2000 tokens of prompt room.")
    results[path]["context"] = found


def check_needle(client, path, api, quick):
    """Long-sequence comprehension: a planted fact at several depths in a long document."""
    hdr(f"[4] {path} — can it find a fact buried in a long document?")
    secret = "The calibration key for unit 19 is PLUM-4417."
    depths = [0.1, 0.9] if quick else [0.05, 0.25, 0.5, 0.75, 0.95]
    doc_tokens = 5000
    out = {}
    for d in depths:
        body_text = filler(doc_tokens)
        cut = int(len(body_text) * d)
        doc = body_text[:cut] + "\n\n" + secret + "\n\n" + body_text[cut:]
        q = (doc + "\n\nQuestion: what is the calibration key for unit 19? "
                   "Answer with only the key.")
        b = (dict(model=MODEL, messages=[{"role": "user", "content": q}], max_tokens=32)
             if "chat" in api else dict(model=MODEL, prompt=q, max_tokens=32))
        ok, p, dt, _ = post(client, api, b)
        if not ok:
            say(f"  depth {d:>4.0%}  ERROR  {str(p)[:120]}")
            out[d] = {"ok": False}
            continue
        ans = text_of(p)
        hit = "PLUM-4417" in ans.upper()
        pt, _ = usage_of(p)
        say(f"  depth {d:>4.0%}  {'FOUND   ' if hit else 'MISSED  '} {dt:6.2f}s  "
            f"prompt_tokens={pt}  answer={ans.strip()[:60]!r}")
        out[d] = {"ok": True, "found": hit, "seconds": dt, "prompt_tokens": pt}
    hits = sum(1 for v in out.values() if v.get("found"))
    say(f"\n  Retrieved at {hits}/{len(out)} depths.")
    results[path]["needle"] = out


def check_long_structured(client, path, api):
    """Format adherence over a long generation — the thing that breaks in real agents."""
    hdr(f"[5] {path} — long structured generation (valid JSON, 40 records)")
    q = ("Emit ONLY a JSON array of exactly 40 objects, no prose, no markdown fence. "
         'Each object: {"id": <int 1-40>, "name": <string>, "score": <int 0-100>, '
         '"tag": one of "alpha"|"beta"|"gamma"}. Start with [ and end with ].')
    b = (dict(model=MODEL, messages=[{"role": "user", "content": q}], max_tokens=2400)
         if "chat" in api else dict(model=MODEL, prompt=q, max_tokens=2400))
    ok, p, dt, _ = post(client, api, b)
    if not ok:
        say(f"  FAIL  {str(p)[:300]}")
        results[path]["structured"] = {"ok": False}
        return
    raw = text_of(p)
    pt, ct = usage_of(p)
    tps = (ct / dt) if (ct and dt) else None
    m = re.search(r"\[.*\]", raw, re.S)
    parsed, n, valid = None, 0, False
    if m:
        try:
            parsed = json.loads(m.group(0))
            n = len(parsed)
            valid = isinstance(parsed, list) and all(
                isinstance(o, dict) and {"id", "name", "score", "tag"} <= set(o) for o in parsed)
        except Exception:
            pass
    say(f"  {dt:.2f}s  completion_tokens={ct}  "
        f"{f'{tps:.1f} tok/s' if tps else 'tok/s n/a'}")
    say(f"  parsed_json={parsed is not None}  records={n}/40  every_field_present={valid}")
    if parsed is None:
        say(f"  first 200 chars of raw output: {raw[:200]!r}")
    results[path]["structured"] = {"ok": True, "seconds": dt, "completion_tokens": ct,
                                   "tok_per_s": tps, "records": n, "schema_ok": valid}


def check_hard_tasks(client, path, api):
    """Tasks with a checkable answer, harder than the one-liners already tried."""
    hdr(f"[6] {path} — harder tasks with auto-checkable answers")
    tasks = [
        ("multi-constraint code",
         "Write one Python function `merge_intervals(intervals)` that merges overlapping "
         "closed intervals, handles unsorted input, empty input, and single-point intervals, "
         "and is O(n log n). Then, in the SAME code block, write 6 assert statements that "
         "would catch a wrong implementation. Output only Python.",
         lambda t: t.count("assert") >= 6 and "def merge_intervals" in t),
        ("stateful reasoning",
         "A jar has 3 red, 5 blue, 2 green marbles. I draw one, it is blue, I do NOT replace "
         "it. I draw again. Give the probability the second is blue as a reduced fraction, "
         "then on a new line the probability it is NOT blue as a reduced fraction. "
         "Output only the two fractions, one per line.",
         lambda t: "4/9" in t and "5/9" in t),
        ("instruction under pressure",
         "Summarise the water cycle. Hard constraints: exactly 3 sentences; every sentence "
         "starts with the letter W; no sentence longer than 12 words; do not use the word "
         "'water'. Output only the summary.",
         lambda t: (lambda s: len(s) == 3 and all(x.strip().upper().startswith("W") for x in s)
                    and "water" not in t.lower())(
                       [x for x in re.split(r"(?<=[.!?])\s+", t.strip()) if x.strip()])),
        ("refactor with a trap",
         "This is wrong: `def avg(xs): return sum(xs)/len(xs)`. Rewrite it to be correct for "
         "empty input and for a generator argument, keeping it under 6 lines. Explain in one "
         "sentence what specifically was wrong. Output code then the sentence.",
         lambda t: ("list(" in t or "tuple(" in t) and ("if not" in t or "== 0" in t or "len(" in t)),
        ("tool-call shape",
         'You may call tools by emitting exactly one JSON object and nothing else: '
         '{"tool":"<name>","args":{...}}. Tools: get_weather(city), get_time(tz). '
         'User asks: "what time is it in Tokyo?" Emit the call.',
         lambda t: (lambda m: bool(m) and json.loads(m.group(0)).get("tool") == "get_time")(
             re.search(r"\{.*\}", t, re.S)) if re.search(r"\{.*\}", t, re.S) else False),
    ]
    out = {}
    for name, prompt, checker in tasks:
        b = (dict(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=900)
             if "chat" in api else dict(model=MODEL, prompt=prompt, max_tokens=900))
        ok, p, dt, _ = post(client, api, b)
        if not ok:
            say(f"  {name:<26} ERROR  {str(p)[:100]}")
            out[name] = {"ok": False}
            continue
        t = text_of(p)
        try:
            passed = bool(checker(t))
        except Exception:
            passed = False
        _, ct = usage_of(p)
        say(f"  {name:<26} {'PASS' if passed else 'FAIL'}  {dt:6.2f}s  out_tok={ct}")
        if not passed:
            say(f"      got: {re.sub(chr(10), ' / ', t.strip())[:160]!r}")
        out[name] = {"ok": True, "passed": passed, "seconds": dt, "completion_tokens": ct,
                     "output": t}
    say(f"\n  {sum(1 for v in out.values() if v.get('passed'))}/{len(tasks)} passed. "
        "These are graded by crude string checks — read the failures before believing them.")
    results[path]["hard_tasks"] = out


def check_determinism(client, path, api):
    hdr(f"[7] {path} — is sampling really greedy?")
    q = "Invent a name for a coffee shop. One name only."
    b = (dict(model=MODEL, messages=[{"role": "user", "content": q}], max_tokens=24, temperature=1.5)
         if "chat" in api else dict(model=MODEL, prompt=q, max_tokens=24, temperature=1.5))
    outs = []
    for _ in range(3):
        ok, p, _, _ = post(client, api, b)
        outs.append(text_of(p).strip() if ok else f"ERR {p}")
    same = len(set(outs)) == 1
    say(f"  temperature=1.5 x3 → {'IDENTICAL (greedy confirmed)' if same else 'DIFFERENT'}")
    for o in outs:
        say(f"    {o[:70]!r}")
    if same:
        say("  So: temperature/top_p/seed are ignored server-side. No self-consistency voting, "
            "no retry-with-more-randomness, no sampling diversity. Design around it.")
    results[path]["determinism"] = {"identical": same, "samples": outs}


def check_prefix_cost(client, path, api):
    hdr(f"[8] {path} — does a repeated long prefix get cheaper? (prefix caching is off)")
    prefix = filler(3000)
    times = []
    for i in range(3):
        q = prefix + f"\n\nQuestion {i}: reply with the single word ok."
        b = (dict(model=MODEL, messages=[{"role": "user", "content": q}], max_tokens=8)
             if "chat" in api else dict(model=MODEL, prompt=q, max_tokens=8))
        ok, p, dt, _ = post(client, api, b)
        times.append(dt if ok else None)
        say(f"  call {i+1} with the same 3000-token prefix: {dt:.2f}s")
    good = [t for t in times if t]
    if len(good) >= 2:
        spread = (max(good) - min(good)) / max(good)
        say(f"  spread {spread:.0%} — if this is small, you are paying full price for the "
            "prefix every single turn.")
        say("  Consequence for a chat app: cost grows with the SQUARE of conversation length. "
            "Trim history aggressively.")
    results[path]["prefix_cost"] = {"seconds": times}


def check_concurrency(client, path, api, quick):
    """max_num_seqs=4, so 8 in flight should queue rather than scale."""
    hdr(f"[9] {path} — concurrency (server is configured for 4 sequences)")
    levels = [1, 4] if quick else [1, 2, 4, 8, 16]
    prompt = "Explain how a hash table handles collisions, in about 150 words."
    out = {}
    for n in levels:
        def one(_):
            b = (dict(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=220)
                 if "chat" in api else dict(model=MODEL, prompt=prompt, max_tokens=220))
            ok, p, dt, ttft = post(client, api, b, stream=True)
            return dt if ok else None, ttft
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            rows = list(ex.map(one, range(n)))
        wall = time.perf_counter() - t0
        lat = [r[0] for r in rows if r[0]]
        ttfts = [r[1] for r in rows if r[1]]
        if not lat:
            say(f"  n={n:<3} all failed")
            out[n] = {"ok": False}
            continue
        p50_ttft = statistics.median(ttfts) if ttfts else None
        say(f"  n={n:<3} wall={wall:6.2f}s  p50_latency={statistics.median(lat):6.2f}s  "
            f"max={max(lat):6.2f}s  p50_ttft={f'{p50_ttft:5.2f}s' if p50_ttft else '  n/a'}  "
            f"completed={len(lat)}/{n}  req/s={len(lat)/wall:.2f}")
        out[n] = {"ok": True, "wall": wall, "p50": statistics.median(lat), "max": max(lat),
                  "p50_ttft": p50_ttft, "completed": len(lat), "rps": len(lat) / wall}
    say("\n  Read this as: where does req/s stop rising? That is your real concurrency ceiling, "
        "and it caps how many people can share one endpoint.")
    results[path]["concurrency"] = out


def run(base, path, quick):
    results[path] = {}
    with httpx.Client(base_url=base, **HTTP) as client:
        if not check_models(client, path):
            return
        apis = check_apis(client, path)
        if not (apis["completions"] or apis["chat_completions"]):
            say("  Neither API works — stopping here.")
            return
        api = (f"{path}/chat/completions" if apis["chat_completions"] else f"{path}/completions")
        say(f"\n  Using {api} for the rest of the suite.")
        check_context_limit(client, path, api, quick)
        check_needle(client, path, api, quick)
        check_long_structured(client, path, api)
        check_hard_tasks(client, path, api)
        check_determinism(client, path, api)
        check_prefix_cost(client, path, api)
        check_concurrency(client, path, api, quick)


def verdict():
    hdr("VERDICT — read this, then read the numbers above")
    for path, r in results.items():
        if not r:
            continue
        say(f"\n{path}")
        a = r.get("apis", {})
        say(f"  chat API:        {'yes' if a.get('chat_completions') else 'NO — completions only'}")
        say(f"  streaming:       {'yes' if a.get('streaming') else 'no'}")
        ctx = r.get("context", {})
        okd = [k for k, v in ctx.items() if v.get("ok")]
        say(f"  usable prompt:   ~{max(okd) if okd else '?'} tokens (prompt + output combined)")
        nd = r.get("needle", {})
        if nd:
            say(f"  long-doc recall: {sum(1 for v in nd.values() if v.get('found'))}/{len(nd)} depths")
        st = r.get("structured", {})
        if st.get("ok"):
            say(f"  long JSON:       {st['records']}/40 records, schema_ok={st['schema_ok']}, "
                f"{(st['tok_per_s'] or 0):.1f} tok/s")
        ht = r.get("hard_tasks", {})
        if ht:
            say(f"  hard tasks:      {sum(1 for v in ht.values() if v.get('passed'))}/{len(ht)} passed")
        cc = {k: v for k, v in r.get("concurrency", {}).items() if v.get("ok")}
        if cc:
            rps, n = max((v["rps"], k) for k, v in cc.items())
            say(f"  peak throughput: {rps:.2f} req/s at n={n}")
    say("\nIf peak throughput is ~1 req/s, one endpoint supports a handful of simultaneous "
        "users, not a room of them. Size your event accordingly.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("GPTOSS_BASE_URL"))
    ap.add_argument("--path", action="append",
                    help="endpoint path prefix, repeatable (default: /agg/v1 and /disagg/v1)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="probe-results.json")
    args = ap.parse_args()

    if not args.base:
        sys.exit("Set GPTOSS_BASE_URL (or pass --base). Ask the organisers for the URL.")
    paths = args.path or ["/agg/v1", "/disagg/v1"]

    say(f"base = {args.base}")
    say(f"paths = {paths}   quick={args.quick}")
    for p in paths:
        try:
            run(args.base, p, args.quick)
        except KeyboardInterrupt:
            say("\ninterrupted")
            break
        except Exception as e:
            say(f"\n{p} blew up: {type(e).__name__}: {e}")
    verdict()
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    say(f"\nraw results → {args.out}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")  # unverified-TLS noise; see the cert note in the README
    main()
