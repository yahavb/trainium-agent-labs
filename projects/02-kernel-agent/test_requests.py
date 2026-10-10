"""Check feedback_v5's request code against a stub server: no model, no NKI.

    PYTHONPATH=<their 02-kernel-agent dir>:<this dir> python test_requests.py

Checks:
  1. SAMPLING=theirs, thinking off: the request body equals their ask()'s, key for key.
  2. SAMPLING=qwen: temperature 0.7, top_p 0.8, top_k 20; thinking stays off.
  3. THINK=round0 on a first prompt, three ways the thinking can end:
       stop        -> one request, the answer after </think> is the reply;
       cut while thinking -> a second request that continues an assistant message ending in
                      Qwen's "Considering the limited time ... </think>", reply = its answer;
       cut while answering -> a second request that continues the partial answer, reply = both.
     A repair prompt is never sent with thinking on.
  4. The usage log gets one line per ask, with the level and the code's sha1.
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

CODE = "```python\nimport nki\n@nki.jit\ndef k(x):\n    return x\n```"
SEEN = []
SCRIPT = []   # canned (content, finish_reason) replies, consumed in order


class Stub(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        content, finish = SCRIPT.pop(0)
        out = json.dumps(dict(choices=[dict(message=dict(role="assistant", content=content),
                                            finish_reason=finish)],
                              usage=dict(prompt_tokens=10, completion_tokens=20))).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def run_case(env):
    """Run this file in a child with the given switches, so each import sees its own env."""
    child = os.environ.copy()
    child.update(env)
    r = subprocess.run([sys.executable, __file__, "--child"], env=child, capture_output=True,
                       text=True)
    if r.returncode:
        print(r.stdout, r.stderr)
        raise SystemExit("child failed")
    return r.stdout


def child():
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    import agent
    import feedback_v5 as v5

    class A:
        model, base, max_tokens, context, think = "Qwen/Qwen3-8B", f"http://127.0.0.1:{srv.server_port}/v1", 2500, 8192, False

    first = agent.first_prompt(1)
    repair = agent.repair_prompt(1, "x = 1", "some failure")
    case = os.environ["CASE"]
    if case == "theirs":
        SCRIPT[:] = [(CODE, "stop"), (CODE, "stop")]
        v5.ask5(A, first)
        v5._their_ask(A, first)
        print(json.dumps(dict(mine=SEEN[0], theirs=SEEN[1])))
    elif case == "qwen":
        SCRIPT[:] = [(CODE, "stop")]
        v5.ask5(A, repair)
        print(json.dumps(dict(body=SEEN[0])))
    elif case.startswith("think"):
        how = case.split("-")[1]
        if how == "stop":
            SCRIPT[:] = [("<think>\nplan\n</think>\n\n" + CODE, "stop")]
        elif how == "cut":
            SCRIPT[:] = [("<think>\nplan plan plan", "length"), (CODE, "stop")]
        else:   # cut while answering
            SCRIPT[:] = [("<think>\nplan\n</think>\n\n```python\nimport nki\n", "length"),
                         ("@nki.jit\ndef k(x):\n    return x\n```", "stop")]
        reply = v5.ask5(A, first)
        SCRIPT[:] = [(CODE, "stop")]
        v5.ask5(A, repair)
        print(json.dumps(dict(reply=reply, bodies=SEEN)))


def main():
    their_dir = os.environ["PYTHONPATH"].split(":")[0]
    os.chdir(their_dir)
    fails = 0

    def check(cond, what):
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + what)
        fails += not cond

    log = tempfile.mktemp(suffix=".jsonl")
    base = dict(MESSAGES="v4", CARD="reduce", LOOP="theirs", THINK="off", USAGE_LOG=log)

    out = json.loads(run_case(dict(base, SAMPLING="theirs", CASE="theirs")).splitlines()[-1])
    check(out["mine"] == out["theirs"], "1. SAMPLING=theirs sends exactly their request body")
    check(list(out["mine"]) == list(out["theirs"]), "   ... with the keys in the same order")

    out = json.loads(run_case(dict(base, SAMPLING="qwen", CASE="qwen")).splitlines()[-1])["body"]
    check((out["temperature"], out["top_p"], out["top_k"]) == (0.7, 0.8, 20),
          "2. SAMPLING=qwen sends 0.7 / 0.8 / top_k 20")
    check(out["chat_template_kwargs"] == {"enable_thinking": False}, "   ... with thinking off")

    for how in ("stop", "cut", "answer"):
        out = json.loads(run_case(dict(base, SAMPLING="qwen", THINK="round0",
                                       CASE=f"think-{how}")).splitlines()[-1])
        b = out["bodies"]
        check(b[0]["chat_template_kwargs"] == {"enable_thinking": True}
              and (b[0]["temperature"], b[0]["top_p"], b[0]["top_k"]) == (0.6, 0.95, 20),
              f"3. [{how}] first prompt: thinking on, Qwen's thinking sampling")
        check(b[-1]["chat_template_kwargs"] == {"enable_thinking": False},
              f"   [{how}] repair prompt: thinking off")
        check("```python" in out["reply"] and out["reply"].rstrip().endswith("```")
              and "<think>" not in out["reply"], f"   [{how}] reply is the whole code block only")
        if how == "stop":
            check(len(b) == 2, "   [stop] one request for the first prompt")
        else:
            c = b[1]
            last = c["messages"][-1]
            check(len(b) == 3 and c.get("continue_final_message") is True
                  and c.get("add_generation_prompt") is False and last["role"] == "assistant",
                  f"   [{how}] a second request continues an assistant message")
            if how == "cut":
                check(last["content"].endswith("directly now.\n</think>\n\n")
                      and last["content"].startswith("<think>\nplan plan plan"),
                      "   [cut] it carries the thinking and closes it with Qwen's sentence")
            else:
                check(last["content"] == "<think>\nplan\n</think>\n\n```python\nimport nki\n",
                      "   [answer] it carries the thinking and the partial answer")
    rows = [json.loads(l) for l in open(log)]
    check(len(rows) == 1 + 1 + 3 * 2 and all(r["level"] == 1 for r in rows)
          and all(len(r["code_sha1"]) == 40 for r in rows),
          f"4. usage log: one line per ask, level and code sha1 filled ({len(rows)} lines)")
    check(sorted(r.get("path") for r in rows if r["think"]) == ["answer-continued", "finished", "forced"],
          "   thinking paths recorded: finished, forced, answer-continued")
    print(f"\n{fails} failed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    child() if "--child" in sys.argv else main()
