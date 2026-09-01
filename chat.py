#!/usr/bin/env python3
"""
chat.py — a terminal chat client for gpt-oss-20b running on AWS Trainium.

You should not have to hand-write curl to talk to a model. Run this instead.

    export GPTOSS_BASE_URL="https://<the-load-balancer>"
    pip install httpx
    python chat.py

Commands inside the chat:
    /agg /disagg      pin to one Trainium deployment
    /auto             use both, failing over automatically (the default)
    /new              clear the conversation
    /system <text>    set the system prompt
    /effort low|medium|high    gpt-oss reasoning effort (measured: has little/no
                              effect on this build -- kept so you can test it yourself)
    /think            show the model's hidden reasoning for the last turn
    /budget <n>       change the input budget (default 4500 tokens)
    /stats            per-turn timings for this session
    /save <file>      write the transcript to a file
    /quit

Deliberately one file, no framework, stdlib + httpx. Read it, change it, break it.
"""

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
import warnings

import httpx

warnings.filterwarnings("ignore")  # the demo endpoint's TLS cert does not match its hostname

MODEL = "gpt-oss-20b"
# The two deployments live under one base URL as path prefixes. They are separate Trainium
# devices with independent capacity (measured: they add, at ~104% scaling efficiency), so
# spreading across both doubles headroom AND survives one of them going away.
ENDPOINTS = ("/agg/v1", "/disagg/v1")
# gpt-oss streams its chain of thought on a separate channel. This vLLM build names the
# field `reasoning`; other builds name it `reasoning_content`. Accept either.
REASONING_KEYS = ("reasoning", "reasoning_content")
# MEASURED: max_model_len 8192 is enforced against INPUT length. Overlong input gets a
# clean 400; input that merely leaves no room silently truncates the ANSWER instead
# (finish_reason=length, no error). So keep input well under this.
CTX_TOTAL = 8192
# Leave room for the answer: the 8192 ceiling is enforced on INPUT, and anything
# left over is all the output gets before it is silently truncated.
DEFAULT_BUDGET = 4500
CHARS_PER_TOKEN = 4  # crude estimate, only used to decide what to trim

C = dict(dim="\033[2m", b="\033[1m", cyan="\033[36m", green="\033[32m",
         yellow="\033[33m", red="\033[31m", off="\033[0m")
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    C = {k: "" for k in C}


def est_tokens(s):
    return max(1, len(s) // CHARS_PER_TOKEN)


class Chat:
    def __init__(self, base, path, system, effort, budget, show_think):
        self.base = base.rstrip("/")
        self.path = path          # None = auto/failover across ENDPOINTS
        # Random start, not 0: every client process would otherwise pick the same
        # endpoint on its first turn and a whole room would land on one of the two.
        self.rr = random.randrange(len(ENDPOINTS))
        self.sick = {}            # path -> time it may be retried
        self.system = system
        self.effort = effort
        self.budget = budget
        self.show_think = show_think
        self.history = []          # [{"role","content"}]
        self.turns = []            # timing rows
        self.last_reasoning = ""
        self.use_chat_api = None   # probed on first send
        self.client = httpx.Client(verify=False, timeout=httpx.Timeout(900.0, connect=20.0))

    # ---------------------------------------------------------------- wire

    def candidates(self):
        """Endpoints to try, best first. Pinned mode returns exactly one."""
        if self.path:
            return [self.path]
        now = time.time()
        live = [p for p in ENDPOINTS if self.sick.get(p, 0) < now]
        if not live:                       # everything is cooling down; try anyway
            self.sick.clear()
            live = list(ENDPOINTS)
        self.rr += 1                       # round-robin so load spreads across both
        return live[self.rr % len(live):] + live[:self.rr % len(live)]

    def url(self, path, suffix):
        return f"{self.base}{path}{suffix}"

    def probe_api(self, path):
        """Prefer /chat/completions; fall back to /completions if it isn't there."""
        try:
            r = self.client.post(self.url(path, "/chat/completions"), json={
                "model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
            self.use_chat_api = (r.status_code == 200)
        except Exception:
            self.use_chat_api = False
        note = "chat/completions" if self.use_chat_api else "completions (chat API unavailable)"
        print(f"{C['dim']}api: {note}{C['off']}")

    def system_prompt(self):
        s = self.system or "You are a concise, accurate assistant."
        # gpt-oss reads reasoning effort out of the system prompt.
        return f"{s}\nReasoning: {self.effort}"

    def trimmed_history(self):
        """Keep the most recent turns that fit the budget.

        NOT for speed. Measured on this endpoint, a 37x longer prompt costs only ~15-20%
        more time to first token, because prefills are padded to compiled shape buckets.
        The budget exists because the 8192 ceiling is enforced on INPUT: overrun it and
        the request is rejected, approach it and the model's ANSWER is silently truncated
        to whatever room is left. Trimming protects the answer, not the clock.
        """
        budget = self.budget - est_tokens(self.system_prompt())
        if not self.history:
            return [], 0, 0

        # The newest message is the question being asked right now. It is never dropped:
        # dropping it leaves the model answering an empty conversation, which it will
        # happily do ("How can I help you?") while the user wonders where their text went.
        newest = dict(self.history[-1])
        if est_tokens(newest["content"]) > budget:
            keep_chars = max(200, budget * CHARS_PER_TOKEN)
            print(f"{C['yellow']}(your message is ~{est_tokens(newest['content'])} tokens, "
                  f"over the {budget}-token budget — truncating it to fit; raise it with "
                  f"/budget){C['off']}")
            newest["content"] = newest["content"][:keep_chars]

        kept, used = [newest], est_tokens(newest["content"])
        for m in reversed(self.history[:-1]):
            t = est_tokens(m["content"])
            if used + t > budget:
                break
            kept.append(m)
            used += t
        kept.reverse()
        dropped = len(self.history) - len(kept)
        return kept, used, dropped

    def flat_prompt(self, msgs):
        """Fallback rendering for servers without the chat API."""
        out = [f"System: {self.system_prompt()}"]
        for m in msgs:
            out.append(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}")
        out.append("Assistant:")
        return "\n\n".join(out)

    def body(self, msgs, max_tokens):
        if self.use_chat_api:
            return {"model": MODEL,
                    "messages": [{"role": "system", "content": self.system_prompt()}] + msgs,
                    "max_tokens": max_tokens, "stream": True}
        return {"model": MODEL, "prompt": self.flat_prompt(msgs),
                "max_tokens": max_tokens, "stream": True,
                "stop": ["\nUser:", "\n\nUser:"]}

    def send(self, user_text):
        """Try each candidate endpoint until one starts streaming an answer.

        Failover only happens BEFORE the first token. Once tokens are arriving we are
        committed — retrying mid-stream would duplicate half an answer, and the endpoint
        is clearly alive anyway.
        """
        self.history.append({"role": "user", "content": user_text})
        msgs, used, dropped = self.trimmed_history()
        max_tokens = max(2500, min(4096, CTX_TOTAL - used - 256))
        if dropped:
            print(f"{C['yellow']}(dropped {dropped} older message(s) to stay under "
                  f"{self.budget} tokens){C['off']}")

        tried = self.candidates()
        for attempt, path in enumerate(tried):
            if attempt:
                print(f"{C['yellow']}(retrying on {path}){C['off']}")
            ok = self._attempt(path, msgs, used, max_tokens,
                               last=(attempt == len(tried) - 1))
            if ok:
                return
            # Don't hammer a sick endpoint on every turn; give it 60s to recover.
            self.sick[path] = time.time() + 60
        self.history.pop()

    def _give_up(self):
        """Abort without failing over: the other endpoint would reject this identically.

        Returns True so send() stops trying, and drops the user turn so the conversation
        does not carry a question that was never answered.
        """
        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()
        return True

    def _attempt(self, path, msgs, used, max_tokens, last):
        """One endpoint, one try. Returns True if it produced a reply."""
        if self.use_chat_api is None:
            self.probe_api(path)

        t0 = time.perf_counter()
        ttft = None
        shown, reasoning = [], []
        finish, fingerprint, usage = None, None, {}

        try:
            with self.client.stream("POST", self.url(
                    path, "/chat/completions" if self.use_chat_api else "/completions"),
                    json=self.body(msgs, max_tokens)) as r:
                if r.status_code != 200:
                    body = r.read()[:400].decode(errors="replace")
                    print(f"{C['red']}HTTP {r.status_code} from {path}: {body}{C['off']}")
                    # A 400 is our fault (too many input tokens, bad request) and will fail
                    # identically on the other endpoint. Only fail over on server-side faults.
                    return False if r.status_code >= 500 else self._give_up()
                print(f"{C['green']}", end="", flush=True)
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        ev = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    fingerprint = ev.get("system_fingerprint") or fingerprint
                    usage = ev.get("usage") or usage
                    ch = (ev.get("choices") or [{}])[0]
                    finish = ch.get("finish_reason") or finish
                    if self.use_chat_api:
                        d = ch.get("delta", {}) or {}
                        piece = d.get("content") or ""
                        think = next((d[k] for k in REASONING_KEYS if d.get(k)), "")
                    else:
                        piece, think = ch.get("text") or "", ""
                    if think:
                        reasoning.append(think)
                        if self.show_think:
                            print(f"{C['dim']}{think}{C['off']}", end="", flush=True)
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                            if reasoning and self.show_think:
                                print(f"\n{C['dim']}{'-' * 40} answer {'-' * 40}{C['off']}\n",
                                      end="", flush=True)
                        shown.append(piece)
                        print(piece, end="", flush=True)
        except KeyboardInterrupt:
            print(f"\n{C['yellow']}(interrupted){C['off']}")
            return self._give_up()
        except Exception as e:
            print(f"{C['red']}\n{type(e).__name__}: {e}{C['off']}")
            if last and ("timed out" in str(e).lower()
                         or "Connect" in type(e).__name__):
                print(f"{C['dim']}Every endpoint failed the same way. If it hangs with no "
                      f"response at all, your network is probably not allowlisted. Ask the "
                      f"organisers.{C['off']}")
            # Nothing streamed, so it is safe to try the other endpoint.
            return False
        finally:
            print(C["off"], end="")

        dt = time.perf_counter() - t0
        text = "".join(shown).strip()
        self.last_reasoning = "".join(reasoning).strip()

        if not text and self.last_reasoning:
            # The model spent its whole budget thinking and never emitted an answer.
            print(f"\n{C['yellow']}(model produced only hidden reasoning, no answer — "
                  f"try /effort low, or ask a narrower question){C['off']}")

        self.history.append({"role": "assistant", "content": text})

        # The server does not always send `usage` on streamed responses; fall back to a
        # char/4 estimate, but mark it so nobody quotes a guess as a measurement.
        measured = bool(usage.get("completion_tokens"))
        out_tok = usage.get("completion_tokens") or est_tokens(text)
        in_tok = usage.get("prompt_tokens") or used
        tps = out_tok / dt if dt else 0
        self.turns.append(dict(seconds=dt, ttft=ttft, in_tok=in_tok, out_tok=out_tok,
                              tps=tps, path=path, measured=measured))

        bits = [f"{dt:.1f}s"]
        if ttft:
            bits.append(f"ttft {ttft:.1f}s")
        tilde = "" if measured else "~"
        bits += [f"{tilde}{tps:.0f} tok/s", f"in {tilde}{in_tok}", f"out {tilde}{out_tok}"]
        if self.last_reasoning and not self.show_think:
            bits.append(f"thought {est_tokens(self.last_reasoning)} tok (/think to see)")
        if finish == "length":
            bits.append(f"{C['yellow']}TRUNCATED{C['off']}")
        if fingerprint:
            bits.append(f"{C['dim']}{fingerprint}{C['off']}")
        print(f"\n{C['dim']}[{path}] {'  '.join(bits)}{C['off']}\n")
        return True

    # ---------------------------------------------------------------- repl

    def stats(self):
        if not self.turns:
            print("no turns yet")
            return
        for k in ("seconds", "tps", "in_tok", "out_tok"):
            vals = [t[k] for t in self.turns if t[k]]
            if vals:
                print(f"  {k:<8} p50={statistics.median(vals):8.1f}  "
                      f"min={min(vals):8.1f}  max={max(vals):8.1f}")
        print(f"  turns    {len(self.turns)}   total in={sum(t['in_tok'] for t in self.turns)} "
              f"out={sum(t['out_tok'] for t in self.turns)}")
        by = {}
        for t in self.turns:
            by.setdefault(t["path"], []).append(t)
        if len(by) > 1:
            for p, rows in by.items():
                print(f"  {p:<12} {len(rows):>3} turns  "
                      f"p50 {statistics.median([r['seconds'] for r in rows]):.1f}s  "
                      f"p50 {statistics.median([r['tps'] for r in rows]):.0f} tok/s")
        print(f"  {C['dim']}input tokens climb every turn because history is re-sent in full — "
              f"there is no prefix cache.{C['off']}")

    def repl(self):
        where = self.path or f"auto ({' + '.join(ENDPOINTS)}, with failover)"
        print(f"{C['b']}gpt-oss-20b on AWS Trainium{C['off']}  {C['dim']}{self.base} → {where}{C['off']}")
        print(f"{C['dim']}/agg /disagg /auto /new /system /effort /think /budget /stats /save "
              f"/quit   ctx {CTX_TOTAL}, budget {self.budget}{C['off']}\n")
        while True:
            try:
                line = input(f"{C['cyan']}you ›{C['off']} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not line:
                continue
            if line.startswith("/"):
                cmd, _, arg = line[1:].partition(" ")
                arg = arg.strip()
                if cmd in ("quit", "q", "exit"):
                    return
                elif cmd in ("agg", "disagg"):
                    self.path = f"/{cmd}/v1"
                    self.use_chat_api = None
                    print(f"{C['dim']}→ pinned to {self.path}{C['off']}")
                elif cmd == "auto":
                    self.path = None
                    self.sick.clear()
                    self.use_chat_api = None
                    print(f"{C['dim']}→ auto: round-robin across "
                          f"{', '.join(ENDPOINTS)} with failover{C['off']}")
                elif cmd == "new":
                    self.history.clear()
                    print(f"{C['dim']}conversation cleared{C['off']}")
                elif cmd == "system":
                    self.system = arg
                    print(f"{C['dim']}system prompt set ({est_tokens(arg)} tok){C['off']}")
                elif cmd == "effort":
                    if arg in ("low", "medium", "high"):
                        self.effort = arg
                        print(f"{C['dim']}reasoning effort = {arg}{C['off']}")
                    else:
                        print("usage: /effort low|medium|high")
                elif cmd == "think":
                    print(f"{C['dim']}{self.last_reasoning or '(nothing recorded)'}{C['off']}")
                elif cmd == "budget":
                    if arg.isdigit():
                        self.budget = int(arg)
                        print(f"{C['dim']}budget = {self.budget} tokens{C['off']}")
                    else:
                        print(f"budget is {self.budget}; usage: /budget 6000")
                elif cmd == "stats":
                    self.stats()
                elif cmd == "save":
                    fn = arg or "transcript.md"
                    with open(fn, "w") as f:
                        for m in self.history:
                            f.write(f"**{m['role']}**\n\n{m['content']}\n\n")
                    print(f"{C['dim']}→ {fn}{C['off']}")
                else:
                    print(f"unknown command /{cmd}")
                continue
            self.send(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("GPTOSS_BASE_URL"))
    ap.add_argument("--endpoint", default="auto", choices=["auto", "agg", "disagg"],
                    help="auto (default) round-robins across both and fails over")
    ap.add_argument("--system", default="")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--think", action="store_true", help="stream the hidden reasoning too")
    ap.add_argument("--ask", help="one-shot: ask this and exit")
    a = ap.parse_args()

    if not a.base:
        sys.exit("Set GPTOSS_BASE_URL (or pass --base). Ask the organisers for the URL.")

    path = None if a.endpoint == "auto" else f"/{a.endpoint}/v1"
    c = Chat(a.base, path, a.system, a.effort, a.budget, a.think)
    if a.ask:
        c.send(a.ask)
    else:
        c.repl()


if __name__ == "__main__":
    main()
