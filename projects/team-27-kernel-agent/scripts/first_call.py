import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import sys
from kagent import client
from kagent.extract import extract_code
from kagent.harness import verify
from kagent.levels import LEVELS

lv = LEVELS[int(sys.argv[1]) if len(sys.argv) > 1 else 1]
prompt = (f"Write a Python function `{lv.signature}` using numpy. {lv.task} "
          "Process the array in tiles of at most 128 rows x 512 columns with explicit python loops "
          "over the tiles; the last tile in each direction may be smaller. "
          "Reply with one ```python code block and nothing else.")
msgs = [{"role": "user", "content": prompt}]
r = client.chat(msgs)
print(f"prompt_tokens={r.prompt_tokens} completion={r.completion_tokens} finish={r.finish_reason} "
      f"latency={r.latency_s}s cached={r.cached} reasoning_chars={len(r.reasoning)}")
if r.problem():
    print("PROBLEM:", r.problem()); sys.exit(1)
code = extract_code(r.content)
print(code)
rep = verify(lv, code)
print(rep.text())
for v in rep.violations:
    print("FIX:", v.fix())
