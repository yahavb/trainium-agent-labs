#!/usr/bin/env python3
"""
try_rung.py — you are the agent. Type a prompt, see if the kernel passes.

    export GPTOSS_BASE_URL="https://..."
    python try_rung.py 2                          # uses a default prompt
    python try_rung.py 2 "write it with two nested loops and no numpy reductions"
    python try_rung.py 5 --repair                 # send the last failure back to the model

That's the whole loop an agent automates. Do it by hand a few times and you will feel
exactly where the difficulty is.
"""
import os, re, subprocess, sys, textwrap, warnings, inspect
import httpx
warnings.filterwarnings("ignore")
import kernelbench as kb

BASE = os.environ.get("GPTOSS_BASE_URL")
if not BASE:
    sys.exit("export GPTOSS_BASE_URL first")

args = [a for a in sys.argv[1:] if not a.startswith("--")]
repair = "--repair" in sys.argv
rung = int(args[0]) if args else 2
extra = args[1] if len(args) > 1 else ""
spec = kb.LADDER[rung]

if repair:
    # Do NOT silently fall back to a fresh attempt: you would think you were testing a
    # repair loop while actually testing first-shot generation, and draw wrong conclusions.
    if not os.path.exists(f"/tmp/rung{rung}.py"):
        sys.exit(f"--repair has nothing to repair: no previous attempt at rung {rung}.\n"
                 f"Run `python try_rung.py {rung}` first, then --repair.")
    prev = open(f"/tmp/rung{rung}.py").read()
    fail = open(f"/tmp/rung{rung}.fail").read() if os.path.exists(f"/tmp/rung{rung}.fail") else ""
    print(f"(repairing the previous attempt at rung {rung})")
    # The naive version pastes the failure verbatim, which says WHAT is wrong but never
    # WHAT TO DO -- and the model then reproduces the same violation. Passing a
    # prescriptive instruction is the whole difference. That translation step is the
    # agent's real job.
    instruction = extra or "Fix it."
    prompt = (f"This function is wrong:\n\n```python\n{prev.strip()}\n```\n\n"
              f"{fail}\n\n{instruction} Keep everything else identical. "
              f"Output only one ```python block.")
else:
    prompt = (f"Write a numpy function `kernel` matching this reference:\n\n"
              f"{inspect.getsource(spec['ref'])}\n"
              f"Loop over tiles of at most 128 rows and 512 columns. "
              f"{extra}\nOutput only one ```python block.")

print(f"RUNG {rung}: {spec['name']}\n  trap: {spec['trap']}")
print(f"\n--- prompt ({len(prompt)//4} est tokens) ---")
print(textwrap.indent(prompt.strip(), "  "))

r = httpx.Client(verify=False, timeout=900).post(
    f"{BASE}/agg/v1/chat/completions",
    json={"model": "gpt-oss-20b", "messages": [{"role": "user", "content": prompt}],
          "max_tokens": 3000}).json()
ch = r["choices"][0]
content = ch["message"].get("content") or ""
think = ch["message"].get("reasoning") or ""
print(f"\n--- model: finish={ch['finish_reason']}  thought={len(think)} chars  "
      f"answer={len(content)} chars  ({r['usage']['completion_tokens']} tokens) ---")
if not content.strip():
    print("  EMPTY ANSWER. It spent the whole budget thinking. Shorten the prompt.")
    print(f"  (last of its reasoning: ...{think[-200:]})")
    sys.exit(1)

m = re.search(r"```(?:python)?\s*(.*?)```", content, re.S)
src = m.group(1) if m else content
open(f"/tmp/rung{rung}.py", "w").write(src)
print(textwrap.indent(src.strip()[:800], "  "))

out = subprocess.run([sys.executable, "kernelbench.py", "--rung", str(rung),
                      "--check", f"/tmp/rung{rung}.py"], capture_output=True, text=True)
print("\n--- verdict ---")
print(out.stdout or out.stderr)
open(f"/tmp/rung{rung}.fail", "w").write(out.stdout)
if out.returncode:
    print(f"  try again:  python try_rung.py {rung} \"<a hint>\"")
    print(f"  or:         python try_rung.py {rung} --repair")
