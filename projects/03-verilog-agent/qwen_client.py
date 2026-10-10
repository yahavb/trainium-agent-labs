"""
qwen_client.py
Sends Verilog to the local Qwen3-8B server and returns the optimized Verilog text.

Assumes serve.sh starts an OpenAI-compatible server (like vLLM).
Change these with environment variables if yours is different:
    QWEN_URL    default: http://localhost:8000/v1
    QWEN_MODEL  default: auto-detected from the server
"""

import json
import os
import re
import urllib.error
import urllib.request

QWEN_URL = os.environ.get("QWEN_URL", "http://localhost:8000/v1").rstrip("/")
QWEN_MODEL = os.environ.get("QWEN_MODEL")  # None = ask the server
TIMEOUT_SECONDS = 600

SYSTEM_PROMPT = (
    "You are an expert digital hardware engineer. You optimize Verilog to use "
    "fewer logic cells after synthesis while keeping the exact same behavior."
)

USER_PROMPT = """Optimize this Verilog so it synthesizes to FEWER cells.

Rules:
- Keep the EXACT same module name, port names, port order, and port widths.
- Keep the exact same functional behavior (cycle-accurate for sequential logic).
- Output ONLY the complete optimized Verilog inside one ```verilog code block.

```verilog
{code}
```
/no_think"""


def _post(path, payload):
    req = urllib.request.Request(
        QWEN_URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read())


def _get_model_name():
    if QWEN_MODEL:
        return QWEN_MODEL
    with urllib.request.urlopen(QWEN_URL + "/models", timeout=30) as resp:
        return json.loads(resp.read())["data"][0]["id"]


def extract_verilog(text):
    """Pull clean Verilog out of Qwen's reply. Returns None if none found."""
    # Remove Qwen3 thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Prefer a ```verilog / ```systemverilog / ``` block
    blocks = re.findall(r"```(?:verilog|systemverilog|v|sv)?\s*\n(.*?)```", text, re.DOTALL)
    for block in blocks:
        if "module" in block and "endmodule" in block:
            return block.strip() + "\n"

    # Fallback: grab from the first 'module' to the last 'endmodule'
    start = text.find("module")
    end = text.rfind("endmodule")
    if start != -1 and end != -1:
        return text[start:end + len("endmodule")].strip() + "\n"
    return None


def optimize_verilog(verilog_code):
    """Send Verilog to Qwen, return optimized Verilog text (or None on failure)."""
    try:
        payload = {
            "model": _get_model_name(),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(code=verilog_code)},
            ],
            "temperature": 0.2,
            "max_tokens": 4096,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            reply = _post("/chat/completions", payload)
        except urllib.error.HTTPError as e:
            if e.code not in (400, 422):
                raise
            # Server doesn't accept chat_template_kwargs: retry without it.
            # Any <think> block in the reply is still stripped by extract_verilog().
            print("[qwen] server rejected enable_thinking setting, retrying without it")
            del payload["chat_template_kwargs"]
            reply = _post("/chat/completions", payload)
        text = reply["choices"][0]["message"]["content"] or ""
    except Exception as e:
        print(f"[qwen] request failed (is serve.sh running at {QWEN_URL}?): {e}")
        return None

    code = extract_verilog(text)
    if code is None:
        print("[qwen] no Verilog found in reply")
    return code
