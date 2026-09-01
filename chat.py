#!/usr/bin/env python3
"""
chat.py — a terminal chat client for gpt-oss-20b running on AWS Trainium.

You should not have to hand-write curl to talk to a model. Run this instead.

    export GPTOSS_BASE_URL="https://<the-load-balancer>"
    pip install httpx
    python chat.py

Commands inside the chat:
    /agg /disagg      switch which Trainium deployment answers you
    /new              clear the conversation
    /system <text>    set the system prompt
    /effort low|medium|high    gpt-oss reasoning effort
    /think            show the model's hidden reasoning for the last turn
    /budget <n>       change the context budget (default 6000 tokens)
    /stats            per-turn timings for this session
    /save <file>      write the transcript to a file
    /quit

Deliberately one file, no framework, stdlib + httpx. Read it, change it, break it.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
import warnings

import httpx

warnings.filterwarnings("ignore")  # the demo endpoint's TLS cert does not match its hostname

MODEL = "gpt-oss-20b"
# max_model_len is 8192 for prompt + completion TOGETHER. Leave room to generate.
CTX_TOTAL = 8192
DEFAULT_BUDGET = 6000
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
        self.path = path
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

    def url(self, suffix):
        return f"{self.base}{self.path}{suffix}"

    def probe_api(self):
        """Prefer /chat/completions; fall back to /completions if it isn't there."""
        try:
            r = self.client.post(self.url("/chat/completions"), json={
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

        Prefix caching is DISABLED on these servers, so every token of history is
        re-processed from scratch on every turn. History is not free — it is the
        dominant cost of a long conversation. Hence the budget.
        """
        budget = self.budget - est_tokens(self.system_prompt())
        kept, used = [], 0
        for m in reversed(self.history):
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
        if self.use_chat_api is None:
            self.probe_api()

        self.history.append({"role": "user", "content": user_text})
        msgs, used, dropped = self.trimmed_history()
        max_tokens = max(256, min(2048, CTX_TOTAL - used - 256))
        if dropped:
            print(f"{C['yellow']}(dropped {dropped} older message(s) to stay under "
                  f"{self.budget} tokens){C['off']}")

        t0 = time.perf_counter()
        ttft = None
        shown, reasoning = [], []
        finish, fingerprint, usage = None, None, {}

        try:
            with self.client.stream("POST", self.url(
                    "/chat/completions" if self.use_chat_api else "/completions"),
                    json=self.body(msgs, max_tokens)) as r:
                if r.status_code != 200:
                    print(f"{C['red']}HTTP {r.status_code}: {r.read()[:500].decode(errors='replace')}{C['off']}")
                    self.history.pop()
                    return
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
                        think = d.get("reasoning_content") or ""
                    else:
                        piece, think = ch.get("text") or "", ""
                    if think:
                        reasoning.append(think)
                        if self.show_think:
                            print(f"{C['dim']}{think}{C['off']}", end="", flush=True)
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        shown.append(piece)
                        print(piece, end="", flush=True)
        except KeyboardInterrupt:
            print(f"\n{C['yellow']}(interrupted){C['off']}")
        except Exception as e:
            print(f"{C['red']}\n{type(e).__name__}: {e}{C['off']}")
            if "timed out" in str(e).lower() or "ConnectError" in type(e).__name__:
                print(f"{C['dim']}If this hangs with no response at all, your network is probably "
                      f"not allowlisted for the endpoint. Ask the organisers.{C['off']}")
            self.history.pop()
            return
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

        out_tok = usage.get("completion_tokens") or est_tokens(text)
        in_tok = usage.get("prompt_tokens") or used
        tps = out_tok / dt if dt else 0
        self.turns.append(dict(seconds=dt, ttft=ttft, in_tok=in_tok, out_tok=out_tok,
                              tps=tps, path=self.path))

        bits = [f"{dt:.1f}s"]
        if ttft:
            bits.append(f"ttft {ttft:.1f}s")
        bits += [f"{tps:.0f} tok/s", f"in {in_tok}", f"out {out_tok}"]
        if self.last_reasoning and not self.show_think:
            bits.append(f"thought {est_tokens(self.last_reasoning)} tok (/think to see)")
        if finish == "length":
            bits.append(f"{C['yellow']}TRUNCATED{C['off']}")
        if fingerprint:
            bits.append(f"{C['dim']}{fingerprint}{C['off']}")
        print(f"\n{C['dim']}[{self.path}] {'  '.join(bits)}{C['off']}\n")

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
        print(f"  {C['dim']}input tokens climb every turn because history is re-sent in full — "
              f"there is no prefix cache.{C['off']}")

    def repl(self):
        print(f"{C['b']}gpt-oss-20b on AWS Trainium{C['off']}  {C['dim']}{self.base}{self.path}{C['off']}")
        print(f"{C['dim']}/agg /disagg /new /system /effort /think /budget /stats /save /quit"
              f"   ctx {CTX_TOTAL}, budget {self.budget}{C['off']}\n")
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
                    print(f"{C['dim']}→ {self.path}{C['off']}")
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
    ap.add_argument("--endpoint", default="agg", choices=["agg", "disagg"])
    ap.add_argument("--system", default="")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--think", action="store_true", help="stream the hidden reasoning too")
    ap.add_argument("--ask", help="one-shot: ask this and exit")
    a = ap.parse_args()

    if not a.base:
        sys.exit("Set GPTOSS_BASE_URL (or pass --base). Ask the organisers for the URL.")

    c = Chat(a.base, f"/{a.endpoint}/v1", a.system, a.effort, a.budget, a.think)
    if a.ask:
        c.send(a.ask)
    else:
        c.repl()


if __name__ == "__main__":
    main()
