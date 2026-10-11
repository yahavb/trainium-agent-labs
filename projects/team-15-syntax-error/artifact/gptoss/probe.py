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
  * does prompt LENGTH even change latency? (static shapes say maybe not)
  * how much max_tokens do you need before you get an answer and not just reasoning?
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
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

MODEL = "gpt-oss-20b"
# TLS cert on the demo LB does not match its hostname, hence verify=False.
HTTP = dict(verify=False, timeout=httpx.Timeout(900.0, connect=20.0))

# --max-model-len 8192. MEASURED: this is validated against INPUT length; output is
# then silently truncated to whatever room remains. See check 3.
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


# gpt-oss emits its chain of thought on a separate channel. This vLLM build calls the field
# `reasoning`; other builds/versions call it `reasoning_content`. Accept either.
REASONING_KEYS = ("reasoning", "reasoning_content")


def reasoning_of(d):
    for k in REASONING_KEYS:
        if d.get(k):
            return d[k]
    return ""


def text_of(payload):
    """The ANSWER only — never the reasoning.

    This distinction is the single easiest way to get a wrong result out of this
    endpoint. gpt-oss spends tokens on its `reasoning` channel BEFORE it writes any
    `content`. Ask for max_tokens=32 and you get 32 tokens of reasoning, empty
    content, and finish_reason="length" — which looks exactly like the model failing
    the task. It isn't; you just didn't give it room to answer. Hence MIN_ANSWER_TOKENS.
    """
    ch = payload["choices"][0]
    if "text" in ch:
        return ch["text"]
    return ch.get("message", {}).get("content") or ""


def think_of(payload):
    ch = payload["choices"][0]
    return reasoning_of(ch.get("message", {})) if "text" not in ch else ""


def finish_of(payload):
    return payload["choices"][0].get("finish_reason")


# Never ask for fewer than this. MEASURED on this endpoint: a trivial question needs ~64
# to produce any content at all, but a reasoning-heavy coding task burned 900 tokens of
# `reasoning` and returned EMPTY content with finish_reason="length"; at 2500 it answered
# using 1442. So the floor that matters is set by the hard tasks, not the easy ones.
MIN_ANSWER_TOKENS = 2500


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
        findings["has_reasoning_channel"] = bool(reasoning_of(ch))
        findings["reasoning_field"] = next((k for k in REASONING_KEYS if ch.get(k)), None)
        if findings["has_reasoning_channel"]:
            say(f"  reasoning field    {findings['reasoning_field']!r}")
            say(f"  NOTE  answers arrive on `content`; the chain of thought arrives on "
                f"`{findings['reasoning_field']}`. Grade the former, never the latter.")

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


def check_min_answer_tokens(client, path, api):
    """How much room does the model need before `content` appears at all?"""
    hdr(f"[2b] {path} — minimum max_tokens before you get an ANSWER and not just reasoning")
    q = "What is 17 * 23? Reply with only the number."
    out = {}
    for mt in (16, 32, 64, 128, 256, 512):
        b = (dict(model=MODEL, messages=[{"role": "user", "content": q}], max_tokens=mt)
             if "chat" in api else dict(model=MODEL, prompt=q, max_tokens=mt))
        ok, p, dt, _ = post(client, api, b)
        if not ok:
            say(f"  max_tokens={mt:<4} ERROR {str(p)[:80]}")
            continue
        c, th, fin = text_of(p), think_of(p), finish_of(p)
        say(f"  max_tokens={mt:<4} content={'YES' if c.strip() else 'EMPTY':<5} "
            f"reasoning={len(th):>4}ch  finish={fin:<8} answer={c.strip()[:30]!r}")
        out[mt] = {"content": bool(c.strip()), "reasoning_chars": len(th), "finish": fin}
    first = next((k for k, v in sorted(out.items()) if v["content"]), None)
    say(f"\n  Smallest max_tokens that produced an answer: {first}")
    say("  Below that you get finish_reason=length with EMPTY content. This looks identical "
        "to the model failing the task, and it is the #1 way to misjudge this endpoint.")
    results[path]["min_answer_tokens"] = {"table": out, "first_with_content": first}


def check_context_limit(client, path, api, quick):
    """Find the real ceiling, what it validates against, and how it fails."""
    hdr(f"[3] {path} — real usable context (declared {DECLARED_CTX})")
    sizes = [4096, 8000, 9500] if quick else [1024, 4096, 7000, 8000, 8500, 9000, 9500, 10000]
    found, ratio = {}, None
    for want in sizes:
        prompt = filler(want) + "\n\nReply with the single word: ok"
        body = (dict(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=16)
                if "chat" in api else dict(model=MODEL, prompt=prompt, max_tokens=16))
        ok, p, dt, _ = post(client, api, body)
        if ok:
            pt, _ = usage_of(p)
            ratio = pt / want if want else None
            say(f"  est~{want:>5}  ACCEPTED  {dt:6.2f}s   server prompt_tokens={pt}")
            found[want] = {"ok": True, "prompt_tokens": pt, "seconds": dt}
        else:
            short = re.sub(r"\s+", " ", str(p))[:200]
            say(f"  est~{want:>5}  REJECTED  {short}")
            found[want] = {"ok": False, "error": short}
    if ratio:
        say(f"\n  (this script's token estimate runs ~{ratio:.2f}x the server's count — "
            "the server's number is the one that matters)")

    # Is the ceiling on input alone, or input + output? Materially different for a chat app.
    say("\n  Is the limit on INPUT only, or input + output?")
    prompt = filler(8000) + "\n\nWrite a long essay about rivers."
    body = (dict(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=2000)
            if "chat" in api else dict(model=MODEL, prompt=prompt, max_tokens=2000))
    ok, p, dt, _ = post(client, api, body)
    if ok:
        pt, ct = usage_of(p)
        tot = (p.get("usage") or {}).get("total_tokens")
        say(f"  big input + max_tokens=2000 → ACCEPTED. prompt={pt} completion={ct} "
            f"total={tot} finish={finish_of(p)}")
        say(f"  So the {DECLARED_CTX} ceiling is checked against INPUT. Your output is then "
            "silently truncated to whatever room is left — no error, just a short answer "
            "and finish_reason=length. THIS is the failure mode to guard against.")
        results[path]["limit_semantics"] = {"input_only": True, "prompt": pt,
                                            "completion": ct, "total": tot,
                                            "finish": finish_of(p)}
    else:
        say(f"  big input + max_tokens=2000 → REJECTED: {str(p)[:180]}")
        say("  So the ceiling covers input + output together.")
        results[path]["limit_semantics"] = {"input_only": False, "error": str(p)[:300]}
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
        b = (dict(model=MODEL, messages=[{"role": "user", "content": q}],
                  max_tokens=MIN_ANSWER_TOKENS)
             if "chat" in api else dict(model=MODEL, prompt=q, max_tokens=MIN_ANSWER_TOKENS))
        ok, p, dt, _ = post(client, api, b)
        if not ok:
            say(f"  depth {d:>4.0%}  ERROR  {str(p)[:120]}")
            out[d] = {"ok": False}
            continue
        ans, fin = text_of(p), finish_of(p)
        hit = "PLUM-4417" in ans.upper()
        pt, _ = usage_of(p)
        say(f"  depth {d:>4.0%}  {'FOUND   ' if hit else 'MISSED  '} {dt:6.2f}s  "
            f"prompt_tokens={pt}  finish={fin}  answer={ans.strip()[:50]!r}")
        out[d] = {"ok": True, "found": hit, "seconds": dt, "prompt_tokens": pt, "finish": fin}
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
         lambda t: (("list(" in t or "tuple(" in t)          # materialises the iterable
                     or re.search(r"for\s+\w+\s+in", t))            # or accumulates in one pass
                    and ("if not" in t or "== 0" in t or "else" in t or "count" in t)),
        ("tool-call shape",
         'You may call tools by emitting exactly one JSON object and nothing else: '
         '{"tool":"<name>","args":{...}}. Tools: get_weather(city), get_time(tz). '
         'User asks: "what time is it in Tokyo?" Emit the call.',
         lambda t: (lambda m: bool(m) and json.loads(m.group(0)).get("tool") == "get_time")(
             re.search(r"\{.*\}", t, re.S)) if re.search(r"\{.*\}", t, re.S) else False),
    ]
    out = {}
    for name, prompt, checker in tasks:
        b = (dict(model=MODEL, messages=[{"role": "user", "content": prompt}],
                  max_tokens=MIN_ANSWER_TOKENS)
             if "chat" in api else dict(model=MODEL, prompt=prompt,
                                        max_tokens=MIN_ANSWER_TOKENS))
        ok, p, dt, _ = post(client, api, b)
        if not ok:
            say(f"  {name:<26} ERROR  {str(p)[:100]}")
            out[name] = {"ok": False}
            continue
        t, th, fin = text_of(p), think_of(p), finish_of(p)
        try:
            passed = bool(checker(t))
        except Exception:
            passed = False
        _, ct = usage_of(p)
        flag = "" if t.strip() else "  <NO CONTENT — only reasoning>"
        say(f"  {name:<26} {'PASS' if passed else 'FAIL'}  {dt:6.2f}s  out_tok={ct}  "
            f"finish={fin}{flag}")
        if not passed:
            say(f"      answer: {re.sub(chr(10), ' / ', t.strip())[:150]!r}")
            if not t.strip():
                say(f"      reasoning was: {re.sub(chr(10), ' / ', th.strip())[:120]!r}")
        out[name] = {"ok": True, "passed": passed, "seconds": dt, "completion_tokens": ct,
                     "finish": fin, "empty_content": not t.strip(),
                     "output": t, "reasoning": th}
    say(f"\n  {sum(1 for v in out.values() if v.get('passed'))}/{len(tasks)} passed. "
        "These are graded by crude string checks — read the failures before believing them.")
    results[path]["hard_tasks"] = out


def check_tool_calling(client, path, api):
    """Can an agent call tools? Three ways, because the obvious one does not work."""
    hdr(f"[6b] {path} — tool calling")
    if "chat" not in api:
        say("  skipped (needs the chat API)")
        return
    out = {}
    tools = [{"type": "function", "function": {
        "name": "get_time", "description": "Get the current time in a timezone",
        "parameters": {"type": "object", "properties": {"tz": {"type": "string"}},
                       "required": ["tz"]}}}]

    ok, p, dt, _ = post(client, api, dict(
        model=MODEL, messages=[{"role": "user", "content": "What time is it in Tokyo?"}],
        tools=tools, tool_choice="auto", max_tokens=MIN_ANSWER_TOKENS))
    if ok:
        msg = p["choices"][0].get("message", {})
        calls = msg.get("tool_calls") or []
        out["native_tools_param"] = bool(calls)
        say(f"  native `tools=` param     {'WORKS' if calls else 'NOT SUPPORTED'}  {dt:5.2f}s  "
            f"tool_calls={len(calls)}  content={(msg.get('content') or '')[:30]!r}")
    else:
        out["native_tools_param"] = False
        say(f"  native `tools=` param     ERROR  {str(p)[:120]}")

    # Hand-rolled JSON. The phrasing decides whether you get an answer at all: gpt-oss will
    # happily resolve the whole task inside its reasoning channel and end the turn with
    # empty `content` unless you explicitly demand a FINAL ANSWER.
    variants = [
        ("prompted, 'emit the call'",
         'You may call tools by emitting exactly one JSON object and nothing else: '
         '{"tool":"<name>","args":{...}}. Tools: get_weather(city), get_time(tz). '
         'User asks: "what time is it in Tokyo?" Emit the call.'),
        ("prompted, 'FINAL ANSWER must be'",
         'Tools available: get_weather(city), get_time(tz). The user asks: "what time is it '
         'in Tokyo?" Your FINAL ANSWER must be exactly one JSON object of the form '
         '{"tool":"...","args":{...}} and nothing else. Do not explain.'),
    ]
    for label, q in variants:
        ok, p, dt, _ = post(client, api, dict(
            model=MODEL, messages=[{"role": "user", "content": q}],
            max_tokens=MIN_ANSWER_TOKENS))
        if not ok:
            say(f"  {label:<26} ERROR  {str(p)[:100]}")
            continue
        c = text_of(p).strip()
        good = False
        m = re.search(r"\{.*\}", c, re.S)
        if m:
            try:
                good = json.loads(m.group(0)).get("tool") == "get_time"
            except Exception:
                good = False
        flag = "" if c else "  <NO CONTENT — answered only in its reasoning>"
        say(f"  {label:<26} {'OK   ' if good else 'FAIL '}  {dt:5.2f}s  "
            f"finish={finish_of(p)}{flag}")
        if c:
            say(f"      {c[:80]!r}")
        out[label] = good

    say("\n  If the native parameter is unsupported, an agent has to hand-roll tool calls in "
        "the prompt — and must demand a FINAL ANSWER explicitly, or the model resolves the "
        "task in its reasoning channel and returns nothing.")
    results[path]["tool_calling"] = out


def check_determinism(client, path, api):
    hdr(f"[7] {path} — is sampling really greedy?")
    q = "Invent a name for a coffee shop. One name only."
    b = (dict(model=MODEL, messages=[{"role": "user", "content": q}], max_tokens=MIN_ANSWER_TOKENS, temperature=1.5)
         if "chat" in api else dict(model=MODEL, prompt=q, max_tokens=MIN_ANSWER_TOKENS, temperature=1.5))
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


def check_prompt_length_cost(client, path, api):
    """Does a longer prompt cost more time? On static shapes, maybe not at all.

    The servers run --no-enable-chunked-prefill with num_batched_tokens_buckets=[8192],
    i.e. ONE compiled prefill shape. If every prefill is padded to that bucket, a
    200-token prompt costs the same as a 7500-token one, and the usual "keep your
    prompt short to go faster" advice is simply false here.
    """
    hdr(f"[8] {path} — does prompt LENGTH change time-to-first-token?")
    sizes = (200, 2000, 4000, 6000, 7500)
    reps = 2 if quick else 5
    prompts = {s: filler(s) + "\n\nReply with the single word: ok" for s in sizes}

    # Randomise the order. Measuring 200,2000,...,7500 in sequence confounds prompt length
    # with warm-up and with whatever else is hitting this shared server; you get a clean
    # rising line that is partly an artifact. Ask me how I know.
    trials = [s for s in sizes for _ in range(reps)]
    random.seed(7)
    random.shuffle(trials)

    raw = {s: [] for s in sizes}
    for s in trials:
        b = (dict(model=MODEL, messages=[{"role": "user", "content": prompts[s]}],
                  max_tokens=400)
             if "chat" in api else dict(model=MODEL, prompt=prompts[s], max_tokens=400))
        ok, p, dt, ttft = post(client, api, b, stream=True)
        if ok and ttft:
            raw[s].append(ttft)

    med = {}
    say(f"  ({reps} reps each, randomised order, max_tokens=400)")
    for s in sizes:
        v = raw[s]
        if not v:
            continue
        med[s] = statistics.median(v)
        say(f"  est~{s:>5} tok   median={med[s]:.3f}s  min={min(v):.3f}  max={max(v):.3f}  "
            f"n={len(v)}")
    if len(med) >= 3:
        ks = sorted(med)
        lo, hi = med[ks[0]], med[ks[-1]]
        say(f"\n  median ttft {lo:.3f}s at ~{ks[0]} tok vs {hi:.3f}s at ~{ks[-1]} tok "
            f"= {hi/lo:.2f}x across a {ks[-1]//ks[0]}x span of prompt length")
        # Look for plateaus: static compiled shapes make this a step function, not a line.
        steps = [(ks[i], ks[i + 1], med[ks[i + 1]] / med[ks[i]]) for i in range(len(ks) - 1)]
        jumps = [(a, b, r) for a, b, r in steps if r > 1.05]
        if jumps and hi / lo < 2:
            say("  STEPPED, not proportional — flat stretches with discrete jumps at "
                + ", ".join(f"~{a}→{b} tok ({r:.2f}x)" for a, b, r in jumps) + ".")
            say("  That is compiled shape buckets: within a bucket, extra prompt tokens are "
                "free because the prefill is padded to the bucket size either way.")
        say(f"  Practical read: a {ks[-1]//ks[0]}x longer prompt costs {(hi/lo - 1):.0%} more "
            "time to first token. Context here is CHEAP.")
        say("  So trim history to stay under the input ceiling and to leave room for the "
            "answer — not to go faster. Trimming for speed buys almost nothing.")
    results[path]["prompt_length_cost"] = {"raw": raw, "median": med}


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


def check_combined_capacity(client_unused, base, paths, quick):
    """Do the two endpoints ADD capacity, or share a bottleneck?

    Easy mistake (I made it): measure each endpoint alone, quote the bigger number, and
    call it the room's capacity. But /agg and /disagg sit on different Trainium devices,
    so if the ALB and the nginx prefix-stripping proxy in front of them are not the
    constraint, their throughput ADDS. That doubles what the room can support, and it is
    only visible if you load both AT THE SAME TIME.
    """
    hdr(f"[10] both endpoints under SIMULTANEOUS load — do they add up?")
    if len(paths) < 2:
        say("  skipped (needs two endpoint paths)")
        return
    prompt = "Explain how a hash table handles collisions, in about 150 words."
    levels = (4, 16) if quick else (4, 8, 16, 32)

    def one(path):
        c = httpx.Client(base_url=base, **HTTP)
        try:
            ok, p, dt, ttft = post(c, f"{path}/chat/completions",
                                   dict(model=MODEL,
                                        messages=[{"role": "user", "content": prompt}],
                                        max_tokens=220), stream=True)
            return (dt, ttft) if ok else (None, None)
        finally:
            c.close()

    def burst(path, n):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            rows = list(ex.map(lambda _: one(path), range(n)))
        return time.perf_counter() - t0, rows

    out = {}
    say(f"  {'n each':>7} {'total':>6} " + " ".join(f"{p.split('/')[1][:6]:>8}" for p in paths)
        + f" {'SUM r/s':>8} {'p50':>7} {'p95':>7} {'ttft':>7} {'fail':>5}")
    for n in levels:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(paths)) as ex:
            futs = {p: ex.submit(burst, p, n) for p in paths}
            per = {p: f.result() for p, f in futs.items()}
        wall = time.perf_counter() - t0
        rows = [r for w, rr in per.values() for r in rr]
        lat = sorted(r[0] for r in rows if r[0])
        tts = [r[1] for r in rows if r[1]]
        fails = sum(1 for r in rows if not r[0])
        rps = {p: sum(1 for r in rr if r[0]) / w for p, (w, rr) in per.items()}
        p95 = lat[int(len(lat) * 0.95) - 1] if lat else float("nan")
        say(f"  {n:>7} {n*len(paths):>6} "
            + " ".join(f"{rps[p]:>8.2f}" for p in paths)
            + f" {(len(lat)/wall):>8.2f} {statistics.median(lat) if lat else 0:>7.2f} "
              f"{p95:>7.2f} {statistics.median(tts) if tts else 0:>7.2f} {fails:>5}")
        out[n] = {"per_endpoint_rps": rps, "combined_rps": len(lat) / wall,
                  "p50": statistics.median(lat) if lat else None, "p95": p95,
                  "p50_ttft": statistics.median(tts) if tts else None, "failures": fails}
        time.sleep(3)

    if out:
        best = max(v["combined_rps"] for v in out.values())
        anyfail = sum(v["failures"] for v in out.values())
        say(f"\n  Combined ceiling: {best:.2f} req/s across both endpoints.")
        say(f"  Total failures across the whole sweep: {anyfail}."
            + ("  It queues rather than erroring — expect slow, not broken."
               if anyfail == 0 else ""))
        say("  Divide by your per-user request rate to size a room. And remember an agent loop "
            "is N serialised requests per iteration, not one.")
    results["combined_capacity"] = out


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
        check_min_answer_tokens(client, path, api)
        check_context_limit(client, path, api, quick)
        check_needle(client, path, api, quick)
        check_long_structured(client, path, api)
        check_hard_tasks(client, path, api)
        check_tool_calling(client, path, api)
        check_determinism(client, path, api)
        check_prompt_length_cost(client, path, api)
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
    if len(paths) > 1:
        try:
            check_combined_capacity(None, args.base, paths, args.quick)
        except Exception as e:
            say(f"\ncombined-capacity check failed: {type(e).__name__}: {e}")
    verdict()
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    say(f"\nraw results → {args.out}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")  # unverified-TLS noise; see the cert note in the README
    main()
