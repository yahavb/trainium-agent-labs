#!/usr/bin/env python3
"""Generic ten-level NumPy challenge agent. Does not replace the NKI agent."""
import argparse
import collections
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import kernelbench as kb

ROOT = Path(__file__).resolve().parent


def contract(level):
    s = kb.LEVELS[level]
    guidance = (
        "Use explicit for loops over tiles at most 128 rows by 512 columns. "
        "Derive all bounds from input shapes, handle partial final tiles, and return a new output. "
        "Do not convert inputs with np.asarray/np.array; use the input dtype and values as provided. "
        "Do not solve the entire operation with one whole-array NumPy expression.\n"
    )
    if level == 1:
        guidance += (
            "For level 1, allocate out = np.empty_like(x), iterate over row and column tiles, "
            "compute each tile as y = a * x_tile + b, then store the ReLU result for that tile. "
            "The common wrong shortcut is return np.maximum(a * x + b, 0.0); do not use that.\n"
        )
    return (f"Implement def kernel{inspect.signature(s['ref'])}: using NumPy.\n"
            f"Operation: {s['name']}. Match this specification:\n"
            f"{inspect.getsource(s['ref'])}\n"
            + guidance)


def instruction(feedback):
    low = feedback.lower()
    if "banned" in low or "hides" in low:
        return "Replace the named whole-operation shortcut with explicit loops and scalar or slice arithmetic. Preserve the mathematical operation."
    if "mutation" in low:
        return "Allocate a new output and leave every input unchanged."
    if "wrong shape" in low:
        return "Derive output dimensions from the reference and input shapes; fix allocation and destination indices."
    if "non-finite" in low:
        return "Check initialization and denominators. For exponential normalization, subtract each row maximum before exp."
    if "partial" in low or "bounds" in low:
        return "Bound each final tile by the actual input extent, and use the matching output slice."
    if "no attribute" in low:
        return "Remove the unsupported API. Use ordinary loops and documented NumPy scalar/slice operations; do not substitute a spelling-similar name."
    return "Locate the reported mismatch or exception and repair its cause. Keep unrelated code unchanged."


def prompt_sections(level, source="", feedback="", ledger=()):
    parts = {"contract": contract(level)}
    if source:
        parts["code"] = f"Current candidate:\n```python\n{source}\n```\n"
    if feedback:
        parts["feedback"] = f"Latest checker report:\n{feedback[:1500]}\nRequired repair: {instruction(feedback)}\n"
    if ledger:
        parts["ledger"] = "Recent failed outcomes:\n" + "\n".join(ledger[-3:])[:600] + "\n"
    parts["output"] = "Return a FINAL ANSWER with one complete Python code block, imports and kernel only."
    return parts


def token_count(prompt, tokenizer=None):
    if tokenizer is not None:
        return len(tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                   tokenize=True, add_generation_prompt=True)) + 128
    # Explicitly a conservative reservation, NOT a measured token count.
    # Server usage is logged separately. Tokenizer option preferred.
    return len(prompt.encode("utf-8")) + 128


def fit_prompt(parts, context, max_tokens, tokenizer=None):
    parts = dict(parts)
    for drop in (None, "ledger"):
        if drop:
            parts.pop(drop, None)
        prompt = "\n".join(parts.values())
        reserved = token_count(prompt, tokenizer)
        if reserved + max_tokens <= context:
            return prompt, reserved, parts
    raise ValueError("Prompt plus reserved output exceeds context. No request sent. Shorten candidate/context or lower output budget deliberately.")


def request_model(a, prompt):
    body = dict(model=a.model, messages=[dict(role="user", content=prompt)], max_tokens=a.max_tokens,
                temperature=0.6, top_p=0.95)
    if a.qwen_no_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("KERNEL_AGENT_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(a.base.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=a.request_timeout) as res:
        payload = json.load(res)
    choice = payload["choices"][0]
    return choice["message"].get("content") or "", choice.get("finish_reason"), payload.get("usage", {})


def extract(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip() if text.lstrip().startswith(("import ", "def ", "from ")) else ""


def check(path, level, timeout, seed):
    # Isolation against accidental hangs, not malicious code. Run in a disposable pod.
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "SYSTEMROOT"}}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    try:
        r = subprocess.run([sys.executable, str(ROOT / "kernelbench.py"), "--json", "--level", str(level),
                            "--check", str(path.resolve()), "--seed", str(seed)],
                           capture_output=True, text=True, timeout=timeout, env=env, cwd=ROOT)
        return json.loads(r.stdout)
    except subprocess.TimeoutExpired:
        return dict(status="timeout", tests_passed=False, passed=0, feedback="Candidate exceeded evaluation timeout; inspect loop bounds and complexity.")
    except (ValueError, OSError):
        return dict(status="evaluator_error", tests_passed=False, passed=0, feedback="Evaluator failed to return valid JSON. Inspect candidate and evaluator independently.")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--level", type=int, choices=range(1, 11))
    sel.add_argument("--through", type=int, choices=range(1, 11))
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--max-tokens", type=int, default=2500)
    p.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL"))
    p.add_argument("--model", default=os.environ.get("KERNEL_AGENT_MODEL"))
    p.add_argument("--tokenizer", help="Local tokenizer directory for the serving model; no download")
    p.add_argument("--qwen-no-thinking", action="store_true")
    p.add_argument("--request-timeout", type=int, default=900)
    p.add_argument("--eval-timeout", type=int, default=120)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--pause", action="store_true", help="Pause after each attempt for inspection")
    p.add_argument("--dry-run", action="store_true", help="Build prompts only; no requests or candidate execution")
    p.add_argument("--output", type=Path, default=Path("runs"))
    a = p.parse_args(argv)
    if min(a.rounds, a.repeat, a.context, a.max_tokens, a.eval_timeout, a.request_timeout) < 1:
        p.error("counts, budgets and timeouts must be positive")
    if not a.dry_run and (not a.base or not a.model):
        p.error("Set --base to a complete API base (ending /v1 or /agg/v1), and --model")
    return a


def main():
    a = arguments()
    tokenizer = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
    levels = list(range(1, (a.through or 1) + 1)) if not a.level else [a.level]
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    out = a.output.resolve() / stamp
    out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.pop("base", None)  # endpoint URLs can contain credentials
    (out / "config.json").write_text(json.dumps(config, indent=2))
    summary, taxonomy = [], collections.Counter()
    print(f"NUMPY challenge ladder, levels {levels}. Artifacts: {out}", flush=True)
    with (out / "attempts.jsonl").open("a") as log:
        for rep in range(a.repeat):
            for level in levels:
                source, feedback, ledger, seen = "", "", [], {}
                passed = False
                for rnd in range(a.rounds):
                    start = time.monotonic()
                    prefix = out / f"r{rep+1}-level{level}-attempt{rnd+1}"
                    parts = prompt_sections(level, source, feedback, ledger)
                    try:
                        prompt, reserved, used = fit_prompt(parts, a.context, a.max_tokens, tokenizer)
                    except ValueError as e:
                        record = dict(level=level, repeat=rep+1, round=rnd+1, status="budget_blocked", feedback=str(e), tests_passed=False)
                        log.write(json.dumps(record)+"\n"); log.flush()
                        taxonomy["budget_blocked"] += 1
                        print(str(e), flush=True)
                        break
                    prefix.with_suffix(".prompt.txt").write_text(prompt)
                    print(f"level {level}, run {rep+1}, attempt {rnd+1}: input reservation {reserved}, output budget {a.max_tokens}", flush=True)
                    if a.dry_run:
                        break
                    try:
                        reply, finish, usage = request_model(a, prompt)
                    except Exception as e:
                        # Do not automatically retry paid/shared requests or hide environment failures.
                        log.write(json.dumps(dict(level=level, repeat=rep+1, round=rnd+1,
                            status="request_error", exception_type=type(e).__name__, tests_passed=False)) + "\n")
                        log.flush()
                        print(f"Model request failed ({type(e).__name__}). Check endpoint/auth/server logs. Artifacts: {out}", flush=True)
                        raise SystemExit(2)
                    prefix.with_suffix(".reply.txt").write_text(reply)
                    candidate = extract(reply)
                    path = prefix.with_suffix(".py")
                    if finish == "length":
                        result = dict(status="truncated", tests_passed=False, passed=0,
                                      feedback="Previous answer was truncated. Return a shorter complete kernel; omit commentary.")
                    elif not candidate:
                        result = dict(status="empty", tests_passed=False, passed=0, feedback="Return a complete Python kernel in your final answer, not reasoning only.")
                    else:
                        path.write_text(candidate)
                        digest = hashlib.sha256(candidate.encode()).hexdigest()
                        if digest in seen:
                            result = dict(status="duplicate", tests_passed=False, passed=0,
                                          feedback="This exact candidate already failed. Original failure: " + seen[digest]["feedback"])
                        else:
                            result = check(path, level, a.eval_timeout, a.seed)
                            seen[digest] = result
                        source = candidate
                    feedback = result["feedback"]
                    ledger.append(result["status"] + ": " + feedback[:150])
                    record = dict(level=level, repeat=rep+1, round=rnd+1, **result,
                                  finish_reason=finish, usage=usage, input_token_reservation=reserved,
                                  token_method="local_chat_template_plus_margin" if tokenizer else "utf8_byte_reservation_plus_margin",
                                  section_bytes={k: len(v.encode()) for k, v in used.items()},
                                  section_content_tokens=({k: len(tokenizer.encode(v, add_special_tokens=False))
                                                           for k, v in used.items()} if tokenizer else None),
                                  seconds=round(time.monotonic()-start, 3), candidate_path=str(path) if path.exists() else None)
                    log.write(json.dumps(record)+"\n"); log.flush()
                    taxonomy[result["status"]] += 1
                    print(f"  {result['status']}: {feedback[:700]}", flush=True)
                    passed = result["tests_passed"]
                    if passed:
                        (out / f"r{rep+1}-level{level}-tests-passed.py").write_text(candidate)
                    if a.pause and input("Enter = continue; q = stop for discussion: ").strip().lower() == "q":
                        print(f"Stopped by user; inspect {out}. No result claimed.")
                        return
                    if passed or result["status"] == "evaluator_error":
                        break
                summary.append(dict(level=level, repeat=rep+1, tests_passed=passed,
                                    rule_audit="pending_manual_review", dry_run=a.dry_run))
                (out / "summary.json").write_text(json.dumps(dict(results=summary, taxonomy=dict(taxonomy)), indent=2))
    if a.dry_run:
        print(f"Dry run complete: {len(levels)} level contracts built; no model calls or candidate tests. Artifacts: {out}")
        return
    print("\nTest pass rates (not full rule certification):")
    for level in levels:
        rows = [r for r in summary if r["level"] == level]
        print(f"  level {level}: {sum(r['tests_passed'] for r in rows)}/{len(rows)}")
    print(f"Logs and candidates: {out}")


if __name__ == "__main__":
    main()
