import os, re, sys, json, time, argparse, pathlib, urllib.request
from checker import grade

BASE = os.environ.get("KERNEL_AGENT_BASE_URL", "http://localhost:8000/v1").rstrip("/")
BASE = BASE if BASE.endswith("/v1") else BASE + "/v1"

SPEC = """Write a Verilog module named `lfsr` for an 8-bit linear feedback shift register.
Ports: input clk, input reset_n (active low), output reg [7:0] data.
On reset_n low, data is set to 8'b10001010.{extra}
On each rising clock edge with reset_n high, shift data left by one and set the new
bit 0 to the XOR of the old bits 0, 3, 5 and 6.
Return only the Verilog module in one code block."""
EXTRA_HINT = " This must take effect immediately, without waiting for a clock edge."

def http(path, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))

def ask(messages, model):
    out = http("/chat/completions", {
        "model": model, "messages": messages, "temperature": 0.7, "max_tokens": 1500,
        "chat_template_kwargs": {"enable_thinking": False}})
    text = out["choices"][0]["message"]["content"] or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S)

def extract(text):
    m = re.search(r"```(?:verilog|systemverilog|v)?\s*\n(.*?)```", text, re.S)
    code = m.group(1) if m else text
    return code if "module" in code else None

def build_prompt(spec, history):
    msg = spec
    if history:
        code, score, fb = history[-1]
        msg += f"\n\nYour previous attempt:\n```verilog\n{code}\n```\nResult (score {score:.2f}):\n{fb}\n"
        if len(history) > 1:
            msg += "\nEarlier feedback you already received:\n" + "\n".join(
                f"- {h[2].splitlines()[0]}" for h in history[:-1])
        msg += "\nFix the problem and return the complete corrected module."
        if len(history) > 1 and history[-1][0].strip() == history[-2][0].strip():
            msg += ("\nYour last two attempts were identical and both failed. "
                    "Change the logic, not just the formatting.\n")
    return [{"role": "user", "content": msg}]

def run_once(model, spec, rounds, tag):
    history, log = [], []
    for rnd in range(rounds):
        reply = ask(build_prompt(spec, history), model)
        code = extract(reply)
        if code is None:
            score, fb, code = 0.0, "No Verilog module found. Return only Verilog in one code block.", ""
        else:
            p = pathlib.Path("candidates/gen"); p.mkdir(parents=True, exist_ok=True)
            f = p / f"lfsr_{tag}_{rnd}.v"; f.write_text(code)
            score, fb = grade("TestBench/lfsr_tb.v", f)
        history.append((code, score, fb))
        log.append({"run": tag, "round": rnd, "score": score, "feedback": fb, "code": code})
        print(f"run {tag} round {rnd}: {score:.2f}  {fb.splitlines()[0][:90]}", flush=True)
        if score == 1.0:
            break
    return log

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--hint", action="store_true", help="tell the model the reset is immediate")
    a = ap.parse_args()
    model = http("/models")["data"][0]["id"]
    spec = SPEC.format(extra=EXTRA_HINT if a.hint else "")
    all_logs, solved = [], 0
    for r in range(a.repeat):
        log = run_once(model, spec, a.rounds, r)
        all_logs += log
        if log[-1]["score"] == 1.0:
            solved += 1
    pathlib.Path("logs").mkdir(exist_ok=True)
    out = f"logs/lfsr_{'hint' if a.hint else 'nohint'}_{int(time.time())}.jsonl"
    pathlib.Path(out).write_text("\n".join(json.dumps(x) for x in all_logs))
    print(f"\nSOLVED {solved}/{a.repeat}   log: {out}")
