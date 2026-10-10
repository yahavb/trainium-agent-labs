import os, re, json, time, argparse, pathlib, urllib.request
import checker
from checker import grade

BASE = os.environ.get("KERNEL_AGENT_BASE_URL", "http://localhost:8000/v1").rstrip("/")
BASE = BASE if BASE.endswith("/v1") else BASE + "/v1"

SPECS = {
    "mine": """Write a Verilog module named `lfsr` for an 8-bit linear feedback shift register.
Ports: input clk, input reset_n (active low), output reg [7:0] data.
On reset_n low, data is set to 8'b10001010.{extra}
On each rising clock edge with reset_n high, shift data left by one and set the new
bit 0 to the XOR of the old bits 0, 3, 5 and 6.""",
    "bench": """I am trying to create a Verilog model for an LFSR. It must meet the following specifications:
- Inputs: Clock, Active-low reset
- Outputs: Data (8-bits)
The initial state should be 10001010, and the taps should be at locations 1, 4, 6, and 7.{extra}
Use module name `lfsr` with ports: input clk, input reset_n, output reg [7:0] data.""",
}
EXTRA_HINT = " The reset is asynchronous: it takes effect immediately, without waiting for a clock edge."
CODE_ONLY = "\nReturn only the Verilog module in one code block."


def http(path, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def ask(messages, model, temperature=0.7):
    out = http("/chat/completions", {
        "model": model, "messages": messages, "temperature": temperature, "max_tokens": 1500,
        "chat_template_kwargs": {"enable_thinking": False}})
    text = out["choices"][0]["message"]["content"] or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S)


def extract(text):
    """Last fenced block that contains a module; the prose may include fragments."""
    blocks = re.findall(r"```(?:verilog|systemverilog|v)?\s*\n(.*?)```", text, re.S)
    blocks = [b for b in blocks if "module" in b]
    if blocks:
        return blocks[-1]
    return text if "module" in text and "endmodule" in text else None


def build_prompt(spec, history, show_code=True, explain=False):
    if not history:                      # round 0 is identical in every condition
        return [{"role": "user", "content": spec + CODE_ONLY}]
    code, score, fb = history[-1]
    msg = spec
    if show_code:
        msg += f"\n\nYour previous attempt:\n```verilog\n{code}\n```"
    msg += f"\n\nA checker tested your previous attempt (score {score:.2f}):\n{fb}\n"
    if len(history) > 1:
        msg += "\nEarlier feedback:\n" + "\n".join(
            f"- {h[2].splitlines()[0]}" for h in history[:-1])
    if len(history) > 1 and history[-1][0].strip() == history[-2][0].strip():
        msg += ("\nYour last two attempts were identical and both failed. "
                "Change the logic, not just the formatting.")
    if explain:
        msg += ("\nFirst, for each failure above, say in one sentence which part of the design "
                "causes it. Then end with the complete corrected Verilog module in one code block.")
    else:
        msg += "\nFix the problems and return only the complete corrected Verilog module in one code block."
    return [{"role": "user", "content": msg}]


def failing_checks(fb):
    names = [l.split(":", 1)[0] for l in fb.splitlines() if re.match(r"^\w+: requirement", l)]
    return ", ".join(names) if names else fb.splitlines()[0][:70]


def run_once(model, spec, rounds, tag, exp, show_code, explain, temperature):
    history, log = [], []
    outdir = pathlib.Path("candidates/gen") / exp
    outdir.mkdir(parents=True, exist_ok=True)
    for rnd in range(rounds):
        reply = ask(build_prompt(spec, history, show_code, explain), model, temperature)
        code = extract(reply)
        if code is None:
            score, fb, code = 0.0, "format: requirement: end with the Verilog module in one code block. Observed: no module found.", ""
        else:
            f = outdir / f"lfsr_{tag}_{rnd}.v"
            f.write_text(code)
            score, fb = grade("TestBench/lfsr_tb.v", f)
        same = bool(history) and code.strip() == history[-1][0].strip()
        history.append((code, score, fb))
        log.append({"exp": exp, "run": tag, "round": rnd, "score": score,
                    "feedback": fb, "code": code, "reply": reply, "same_as_previous": same})
        status = "correct" if score == 1.0 else "failed: " + failing_checks(fb)
        print(f"run {tag} round {rnd}: {score:.2f}  {status}{'  (same code)' if same else ''}",
              flush=True)
        if score == 1.0:
            break
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--spec", choices=SPECS, default="mine")
    ap.add_argument("--hint", action="store_true", help="state that the reset is asynchronous")
    ap.add_argument("--no-code", action="store_true", help="don't show the previous attempt")
    ap.add_argument("--explain", action="store_true", help="ask for a diagnosis before the fix")
    ap.add_argument("--principles", action="store_true", help="append general principles to checker feedback")
    ap.add_argument("--principles-v2", action="store_true",
                    help="add output-pattern detectors and principles (implies --principles)")
    ap.add_argument("--temperature", type=float, default=0.7)
    a = ap.parse_args()
    a.principles = a.principles or a.principles_v2
    checker.USE_PRINCIPLES = a.principles
    checker.USE_PRINCIPLES_V2 = a.principles_v2

    exp = a.spec + ("_hint" if a.hint else "") + ("_nocode" if a.no_code else "") \
          + ("_explain" if a.explain else "") + ("_principles" if a.principles else "") \
          + ("_v2" if a.principles_v2 else "") \
          + (f"_t{a.temperature}" if a.temperature != 0.7 else "")
    model = http("/models")["data"][0]["id"]
    spec = SPECS[a.spec].format(extra=EXTRA_HINT if a.hint else "")
    print(f"experiment: {exp}  model: {model}", flush=True)

    all_logs, solved, rounds_needed = [], 0, []
    for r in range(a.repeat):
        log = run_once(model, spec, a.rounds, r, exp, not a.no_code, a.explain, a.temperature)
        all_logs += log
        if log[-1]["score"] == 1.0:
            solved += 1
            rounds_needed.append(len(log))
    pathlib.Path("logs").mkdir(exist_ok=True)
    out = f"logs/lfsr_{exp}_{int(time.time())}.jsonl"
    pathlib.Path(out).write_text("\n".join(json.dumps(x) for x in all_logs))
    print(f"\nSOLVED {solved}/{a.repeat}   rounds to solve: {rounds_needed}   log: {out}")
