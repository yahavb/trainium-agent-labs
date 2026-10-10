"""Mock model for the prompt-leak check. It never returns code taken from a reference, tutorial or
answer kernel: every reply is one of four broken kernels written here, chosen by sha256(prompt), so
the agent's repair prompts (which quote the previous kernel) cannot contain a forbidden line that the
mock put there. Logs every request body. argv: port log_path."""
import hashlib, http.server, inspect, json, os, sys, threading
sys.path.insert(0, "/h/tal-deliv/projects/02-kernel-agent")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
import nkibench, ops07, ops08  # noqa: F401  (ops07/08 register levels 9-14)

port, LOG = int(sys.argv[1]), sys.argv[2]
lock = threading.Lock()
OPS = sorted(((s["op"], n) for n, s in nkibench.LEVELS.items()), key=lambda x: -len(x[0]))


def level_of(p):
    return next((n for op, n in OPS if op in p), 1)


def reply(level, p):
    s = nkibench.LEVELS[level]
    args = list(inspect.signature(s["ref"]).parameters)
    a0 = args[0]
    head = "import nki\nimport nki.isa as nisa\nimport nki.language as nl\n\n"
    sig = f"def {s['entry']}({', '.join(args)}):\n"
    k = int(hashlib.sha256(p.encode()).hexdigest(), 16) % 4
    if k == 0:
        return "I could not write this one."
    if k == 1:   # whole-tensor tile: partition dimension error
        body = (f"    t = nl.ndarray((256, 64), dtype={a0}.dtype, buffer=nl.sbuf)\n"
                f"    nisa.dma_copy(dst=t, src={a0})\n    return {a0}\n")
        return f"```python\n{head}@nki.jit\n{sig}{body}```"
    if k == 2:   # invented API name
        body = f"    return nisa.multiply({a0}, 2.0)\n"
        return f"```python\n{head}@nki.jit\n{sig}{body}```"
    body = f"    return {a0}\n"   # no decorator: rule violation
    return f"```python\n{head}{sig}{body}```"


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        p = body["messages"][0]["content"]
        lv = level_of(p)
        ans = reply(lv, p)
        with lock, open(LOG, "a") as f:
            f.write(json.dumps(dict(level=lv, body=body), sort_keys=True) + "\n")
        out = dict(choices=[dict(message=dict(content=ans), finish_reason="stop")],
                   usage=dict(prompt_tokens=len(p) // 3, completion_tokens=len(ans) // 3))
        b = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
