#!/usr/bin/env python3
"""
Verifier-grounded online RL controller for projects/02-kernel-agent  (v4).

Reuses agent.py (prompts, grader) and nkibench.py (verifier) UNCHANGED. Everything new lives here.

What it learns: which prompt strategy and sampling temperature earn the most verifier reward,
per level and per failure category. This is online policy learning over prompt strategies; it does
NOT update model weights. It can, however, export (prompt, reply, reward) rows for a later
rejection-sampling fine-tune (--export-sft).

Run from projects/02-kernel-agent:
  python rl_trace_agent.py --level 2 --rounds 8 --samples 4 --context 8192 --log rl_level2_v4.jsonl

See README-RL-TRACE.md for what changed from v3 and why.
"""

import argparse
import ast
import concurrent.futures as cf
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict

import agent as base_agent
import nkibench

SCHEMA_VERSION = 4
MODEL = os.environ.get("KERNEL_AGENT_MODEL", getattr(base_agent, "MODEL", "Qwen/Qwen3-8B"))

STRATEGIES = ("direct", "contract_trace", "counterexample", "repair_diagnosis", "exemplar")
TEMPERATURES = ("t=0.3", "t=0.6", "t=0.9")

TRACE_FIELDS = (
    "contract", "tile_strategy", "memory_flow", "edge_cases",
    "predicted_failure", "verification_checks", "diagnosis",
)
TRACE_RE = re.compile(r"<TRACE>\s*(\{.*?\})\s*</TRACE>", re.S)
TRACE_STRIP_RE = re.compile(r"<TRACE>.*?(?:</TRACE>|\Z)", re.S)
FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)(?:```|\Z)", re.S)
NON_CODE_LANGS = {"json", "text", "txt", "bash", "sh", "shell", "console", "yaml", "yml", "markdown"}

NKI_API_CARD = r"""
NKI API CONTRACT — follow this exactly:
- Use these imports when needed:
  import nki
  import nki.language as nl
  import nki.isa as nisa
- `nl` is only an alias for `nki.language`; NEVER write `import nl` or `import nki.nl`.
- `nl.sbuf`, `nl.psum`, and `nl.shared_hbm` are memory-region VALUES, not functions.
  Correct: `nl.ndarray(shape=(P, F), dtype=nl.float32, buffer=nl.sbuf)`.
  Incorrect: `nl.sbuf(...)`, `nl.shared_hbm(...)`, or `nl.psum(...)`.
- NKI tensor `reshape` takes ONE shape argument (a tuple), if supported:
  `x.reshape((P, F1, F2))`. NEVER call `x.reshape(P, F1, F2)`.
- Do not use NumPy to implement a kernel, and do not apply host-side `.T` to an input.
- Use `@nki.jit` on the required entry point and preserve its exact name/signature.
- Common documented primitives: `nl.ndarray`, `nl.affine_range`, `nl.ds`,
  `nisa.dma_copy(dst=..., src=...)`, `nisa.tensor_copy(dst=..., src=...)`.
- For data movement, allocate destination buffers with `nl.ndarray(..., buffer=...)`;
  pass the memory region as the `buffer` value.

FILE LAYOUT — the checker imports your file and then calls the kernel itself:
- The file may contain ONLY imports and the function definition(s).
- NEVER call the kernel, `nl.*`, or `nisa.*` at module level. No example usage, no test code,
  no `if __name__ == "__main__":` block, no `print`. Anything executed at import time fails with
  "No backend set", because the simulator is not running yet.
"""

LEVEL2_GUIDANCE = r"""
LEVEL 2 TRANSPOSE GUIDANCE:
The input is [P, F1*F2]. Each row is a flattened row-major F1-by-F2 matrix.
The output must flatten its transpose, with element mapping:
  output[p, f2*F1 + f1] = input[p, f1*F2 + f2].
Avoid calling `reshape` or `transpose` on NKI tensors for this task. A known-valid NKI
approach is to allocate input/output SBUF tiles, DMA-copy input to SBUF, then use nested
`nl.affine_range` loops and `nisa.tensor_copy` with one-element `nl.ds(start, 1)` slices
to scatter each element from source index `f1*F2+f2` to destination index `f2*F1+f1`,
then DMA-copy the result to the shared-HBM output. Handle all shapes supplied by the checker.
This is implementation guidance; still produce the complete kernel code.
"""

# Appended to every failure the verifier reports, keyed by category. The repo's own lesson is that
# an error message that names the fix beats a better model, so each of these says what to DO.
FEEDBACK_NOTES = {
    "no_backend": (
        "WHY: something at module level (a call to the kernel, to nl.*, or to nisa.*) ran while "
        "the file was being imported, before the simulator exists. FIX: the file must contain only "
        "imports and the @nki.jit function. Delete every top-level call, example, test, print, and "
        "`if __name__ == '__main__'` block. The checker calls the kernel for you."),
    "parse": (
        "FIX: reply with exactly one ```python fenced block that contains the complete file and "
        "nothing else inside the fence."),
    "truncated": (
        "The previous reply was CUT OFF by the token limit. Write the shortest correct kernel: no "
        "comments, no docstrings, no explanation outside the code block."),
}


def compact(value, limit=500):
    return str(value or "").strip().replace("\n", " ")[:limit]


# ------------------------------------------------------------------ extraction and hygiene

def parse_trace(reply):
    match = TRACE_RE.search(reply or "")
    if not match:
        return {}, False
    try:
        result = json.loads(match.group(1))
        return (result, True) if isinstance(result, dict) else ({}, False)
    except (json.JSONDecodeError, TypeError):
        return {}, False


def _is_json_object(text):
    try:
        return isinstance(json.loads(text), dict)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def extract_kernel_code(reply, entry):
    """Pick the fenced block that is the kernel.

    The v3 extractor took the LONGEST fenced block. When the reply also carried a JSON trace
    (fenced or not) a short kernel could lose to it, and the checker then reported "invalid syntax
    on line 1" -- blaming the model for the extractor's choice. Here the trace is cut out first,
    JSON/shell/prose blocks are ignored, and blocks are ranked by how much they look like the kernel.
    """
    text = TRACE_STRIP_RE.sub("", reply or "")
    scored = []
    for lang, body in FENCE_RE.findall(text):
        body = body.strip()
        if not body or lang.lower() in NON_CODE_LANGS or _is_json_object(body):
            continue
        score = 2 * (entry in body) + ("@nki.jit" in body) + ("import nki" in body)
        scored.append((score, len(body), body))
    if scored:
        code = max(scored)[2]
    else:
        code = base_agent.extract_code(text)
    lines = code.splitlines()
    if lines and lines[0].strip().lower() == "python":      # "```\npython\nimport nki"
        code = "\n".join(lines[1:])
    return code.strip()


def _is_static_expr(node):
    """Constants, tuples, arithmetic on them, and attribute reads like nl.float32. No calls."""
    if node is None:
        return True
    allowed = (ast.Constant, ast.Tuple, ast.List, ast.UnaryOp, ast.BinOp, ast.Name, ast.Attribute,
               ast.operator, ast.unaryop, ast.expr_context)
    return all(isinstance(n, allowed) for n in ast.walk(node))


def sanitize_module(source):
    """Drop module-level statements that execute at import time.

    The simulator is only active while the checker runs the kernel, so a top-level call such as
    `out = my_kernel(x)` or `nisa.dma_copy(...)` raises "No backend set" during import and the
    kernel never gets graded. Keeping only imports, definitions, docstrings and static constants
    makes the model's real work reachable by the checker. Returns (source, dropped_descriptions);
    the source is returned byte-for-byte unchanged when nothing had to go.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []
    keep, dropped = [], []
    for node in tree.body:
        is_docstring = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str))
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) or is_docstring:
            keep.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and _is_static_expr(node.value):
            keep.append(node)
        else:
            try:
                shown = ast.unparse(node).replace("\n", " ")[:80]
            except Exception:
                shown = type(node).__name__
            dropped.append(f"line {node.lineno}: `{shown}`")
    if not dropped:
        return source, []
    module = ast.Module(body=keep, type_ignores=[])
    return ast.unparse(ast.fix_missing_locations(module)) + "\n", dropped


def compact_exemplar(source):
    """Shrink a verified kernel for use as a prompt exemplar: no docstrings, no unused imports of
    numpy or nki.typing (they invite the model to import things the checker does not promise)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    def strip_docstring(body):
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            return body[1:] or [ast.Pass()]
        return body

    tree.body = strip_docstring(tree.body)
    kept = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.body = strip_docstring(node.body)
        if isinstance(node, ast.ImportFrom) and node.module == "nki.typing":
            continue
        if isinstance(node, ast.Import) and all(a.name == "numpy" and (a.asname or a.name) not in used
                                                for a in node.names):
            continue
        kept.append(node)
    tree.body = kept
    try:
        return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    except Exception:
        return source


def code_fingerprint(code):
    try:
        norm = ast.dump(ast.parse(code))
    except SyntaxError:
        norm = re.sub(r"\s+", " ", code).strip()
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ verifier feedback

def classify_feedback(feedback, passed, truncated=False):
    text = (feedback or "").lower()
    if passed:
        return "verified"
    if "no backend set" in text:
        return "no_backend"
    if "no module named 'nl'" in text or 'no module named "nl"' in text or "there is no module named" in text:
        return "bad_import"
    if "reshape() takes" in text or ("reshape" in text and "positional arguments" in text):
        return "reshape_api"
    if "does not parse" in text or "no code came back" in text:
        return "truncated" if truncated else "parse"
    if "rule violations" in text or "not decorated" in text:
        return "rules"
    if "memoryregion" in text or "object is not callable" in text:
        return "buffer_api"
    if any(t in text for t in ("has no attribute", "unexpected keyword argument",
                               "got multiple values", "must be in [")):
        return "api_name"
    if any(t in text for t in ("exceeds maximum", "must have at least 2 dimensions",
                               "same number of elements", "could not be broadcast",
                               "out-of-bound", "exceeds pmax")):
        return "tile_limits"
    if any(t in text for t in ("mismatch", "wrong shape", "shapes passed", "raised ",
                               "correct on cpu but wrong", "non-finite output")):
        return "correctness"
    if "cannot simulate" in text or "cannot import" in text:
        return "environment"
    return "other"


def enrich_feedback(feedback, category, dropped):
    parts = [feedback or ""]
    if category in FEEDBACK_NOTES:
        parts.append(FEEDBACK_NOTES[category])
    if dropped:
        parts.append("NOTE: these module-level statements were removed before grading because they "
                     "run at import time: " + "; ".join(dropped[:3]) + ". Do not write them.")
    return " ".join(p for p in parts if p)


# ------------------------------------------------------------------ prompts

def trace_instructions(use_trace):
    if not use_trace:
        return ""
    return """
Before the code, emit a short engineering record in this exact format:
<TRACE>
{"contract":"input/output shapes and operation","tile_strategy":"tile and loop plan",
 "memory_flow":"where data is allocated and copied","edge_cases":"partial/boundary tiles",
 "predicted_failure":"one likely risk","verification_checks":["shape check","correctness check"],
 "diagnosis":"initial hypothesis or latest failure diagnosis"}
</TRACE>
Keep fields concise and factual. Do not claim the checker passed before it runs.
Then give the code in ONE ```python block. Do not put the TRACE inside a code fence.
"""


def strategy_instructions(action, history):
    if action == "direct":
        return "Implement the operation directly; avoid unnecessary abstractions."
    if action == "contract_trace":
        return ("State the tensor shapes, dtype, output contract, loop bounds, and how partial "
                "tiles are handled (as short code comments). Keep the kernel simple.")
    if action == "counterexample":
        return ("Prioritize dimensions smaller than a tile, non-divisible dimensions, and final "
                "partial tiles. Do not read or write out of bounds.")
    if action == "repair_diagnosis":
        ledger = "\n".join("- " + compact(item, 200) for item in list(history)[-3:])
        return ("Use the actual checker feedback below to make a minimal repair. Do not repeat "
                "the same failed API usage:\n" + (ledger or "- No previous failures."))
    if action == "exemplar":
        return ("Follow the import style, buffer allocation, and nisa call conventions of the "
                "verified example below. Copy its conventions, not its algorithm.")
    return ""


def build_prompt(level, action, use_trace, anchor_code="", anchor_feedback="",
                 last_feedback="", last_regressed=False, history=(), exemplar=None, repeated=False):
    spec = nkibench.LEVELS[level]
    if anchor_code:
        core = base_agent.repair_prompt(level, anchor_code, anchor_feedback)
        if last_regressed and last_feedback:
            core += ("\n\nA LATER attempt made things worse and was discarded; do not do this: "
                     + compact(last_feedback, 300))
    else:
        core = base_agent.first_prompt(level, 1)
        if last_feedback:
            core += "\n\nYour previous reply failed: " + compact(last_feedback, 400)
    if repeated:
        core += ("\n\nYou already submitted this exact code once. Change the approach, not the "
                 "formatting.")

    exemplar_block = ""
    if action == "exemplar" and exemplar:
        ex_level, ex_code = exemplar
        exemplar_block = (f"\nVERIFIED WORKING EXAMPLE (a different operation: "
                          f"{nkibench.LEVELS[ex_level]['op']}):\n```python\n{ex_code.strip()}\n```\n")

    level_specific = LEVEL2_GUIDANCE if level == 2 else ""
    return (
        core + "\n\n" + NKI_API_CARD + "\n" + level_specific + "\n" + exemplar_block + "\n" +
        strategy_instructions(action, history) + "\n" + trace_instructions(use_trace) +
        f"\nRequired entry point: {spec['entry']}. Reply with ONE ```python code block containing "
        f"only the imports and the function."
    )


# ------------------------------------------------------------------ reward

def trace_score(trace, level, valid_format):
    """Field coverage of the structured note, not verbosity or claims of correctness."""
    if not valid_format:
        return 0.0, {"format": 0.0, "coverage": 0.0, "checks_quality": 0.0}
    coverage = sum(bool(str(trace.get(k, "")).strip()) for k in TRACE_FIELDS) / len(TRACE_FIELDS)
    checks = trace.get("verification_checks", [])
    if not isinstance(checks, list):
        checks = []
    check_text = " ".join(str(item).lower() for item in checks)
    required = ["shape", "correct"]
    if level >= 3:
        required.extend(["tile", "memory"])
    check_quality = sum(term in check_text for term in required) / len(required)
    claim_text = " ".join(str(trace.get(k, "")) for k in TRACE_FIELDS).lower()
    forbidden = ("all tests pass", "verified correct", "checker passed", "successfully tested")
    honesty = 0.0 if any(term in claim_text for term in forbidden) else 1.0
    score = 0.65 * coverage + 0.25 * check_quality + 0.10 * honesty
    return score, {"format": 1.0, "coverage": coverage,
                   "checks_quality": check_quality, "honesty": honesty}


def shaped_reward(kernel, trace, hygiene, improved, repeated, use_trace):
    """Verifier reward dominates. Small terms: hygiene (clean file, nothing stripped, not cut off),
    improved (beat this episode's best), and a penalty for resubmitting identical code. The v3
    'evidence' term is gone: it paid partial credit for failing in familiar ways."""
    if use_trace:
        r = 0.80 * kernel + 0.10 * trace + 0.05 * hygiene + 0.05 * improved
    else:
        r = 0.90 * kernel + 0.05 * hygiene + 0.05 * improved
    if repeated:
        r -= 0.10
    return min(1.0, max(0.0, r))


# ------------------------------------------------------------------ policy

class ShrunkUCB:
    """UCB bandit with hierarchical shrinkage.

    Value for (context, action) is estimated by shrinking toward the parent scope:
      global -> level -> level+failure_category.
    With few samples (8 rounds!) a fine context has almost no data of its own, so it borrows the
    level's and the global average instead of being forced to try every arm once (which is
    round-robin, not learning). The state is plain counts/sums, so it persists across runs.
    """
    def __init__(self, actions, exploration=0.5, prior_strength=2.0, prior_mean=0.5):
        self.actions = tuple(actions)
        self.c = float(exploration)
        self.k = float(prior_strength)
        self.prior = float(prior_mean)
        self.stats = defaultdict(lambda: [0, 0.0])          # "scope||action" -> [n, sum]

    @staticmethod
    def scopes(context):
        parts = context.split("|")
        return ["*"] + ["|".join(parts[:i]) for i in range(1, len(parts) + 1)]

    def _key(self, scope, action):
        return f"{scope}||{action}"

    def estimate(self, context, action):
        v, n = self.prior, 0
        for scope in self.scopes(context):
            n, s = self.stats[self._key(scope, action)]
            v = (s + self.k * v) / (n + self.k)
        return v, n

    def select(self, context, rng, legal=None):
        legal = tuple(legal or self.actions)
        total = sum(self.stats[self._key(context, a)][0] for a in self.actions)
        scores = {}
        for a in legal:
            v, n = self.estimate(context, a)
            scores[a] = v + self.c * math.sqrt(math.log(total + 2) / (n + 1))
        best = max(scores.values())
        return rng.choice([a for a, s in scores.items() if abs(s - best) < 1e-12])

    def update(self, context, action, reward):
        for scope in self.scopes(context):
            cell = self.stats[self._key(scope, action)]
            cell[0] += 1
            cell[1] += float(reward)

    def to_json(self):
        return {k: v for k, v in self.stats.items() if v[0] > 0}

    def load_json(self, blob):
        for k, v in (blob or {}).items():
            self.stats[k] = [int(v[0]), float(v[1])]

    def table(self, context):
        rows = []
        for a in self.actions:
            v, n = self.estimate(context, a)
            rows.append((a, v, n))
        return rows


class State:
    """Policy counts plus verified kernels, persisted as one JSON file across runs."""
    def __init__(self, path):
        self.path = path
        self.strategy = None
        self.temperature = None
        self.verified = {}                                   # str(level) -> source

    def load(self, strategy, temperature):
        self.strategy, self.temperature = strategy, temperature
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    blob = json.load(f)
                strategy.load_json(blob.get("strategy"))
                temperature.load_json(blob.get("temperature"))
                self.verified = dict(blob.get("verified", {}))
                pulls = sum(v[0] for k, v in strategy.stats.items() if k.startswith("*||"))
                print(f"loaded policy state from {self.path} ({pulls} prior strategy pulls, "
                      f"verified kernels for levels {sorted(self.verified) or 'none'})")
            except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
                print(f"warning: could not read {self.path} ({e}); starting fresh")

    def save(self):
        if not self.path:
            return
        blob = {"version": SCHEMA_VERSION, "strategy": self.strategy.to_json(),
                "temperature": self.temperature.to_json(), "verified": self.verified}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False)
        os.replace(tmp, self.path)


def load_exemplar_pool(state, level, seed_references, here):
    """Verified kernels from OTHER levels. Never the current level: that would be replaying the
    answer, which measures memory instead of the agent."""
    pool = {int(k): v for k, v in state.verified.items() if int(k) != level}
    if seed_references:
        for n in nkibench.LEVELS:
            path = os.path.join(here, f"reference_level{n}.py")
            if n != level and n not in pool and os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    pool[n] = f.read()
    return pool


def pick_exemplar(pool, level):
    if not pool:
        return None
    n = min(pool, key=lambda m: (abs(m - level), m))
    return n, compact_exemplar(pool[n])


# ------------------------------------------------------------------ the model

def ask_model(args, prompt, temperature):
    """Same request as agent.ask, but with the temperature as a policy decision and the finish
    reason returned, so a token-budget cut-off is not mistaken for a model failure."""
    import httpx
    est_prompt = len(prompt) // 4
    budget = min(args.max_tokens, max(256, args.context - est_prompt - 64))
    body = dict(model=args.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=budget, temperature=temperature, top_p=0.95,
                chat_template_kwargs={"enable_thinking": args.think})
    r = httpx.post(f"{args.base.rstrip('/')}/chat/completions", json=body,
                   timeout=900, verify=False)
    if r.status_code != 200:
        raise SystemExit(f"the endpoint returned HTTP {r.status_code}:\n{r.text[:600]}")
    ch = r.json()["choices"][0]
    content = (ch.get("message") or {}).get("content") or ""
    return content, ch.get("finish_reason"), budget, est_prompt


def generate(args, prompt, temperature, n):
    """n replies in parallel; the server batches them, so wall-clock is close to one call."""
    started = time.perf_counter()
    if n == 1:
        results = [ask_model(args, prompt, temperature)]
    else:
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            results = [f.result() for f in [ex.submit(ask_model, args, prompt, temperature)
                                            for _ in range(n)]]
    return results, time.perf_counter() - started


# ------------------------------------------------------------------ the loop

def run_episode(args, level, episode, run_id, strategy, temperature, state, log, sft, rng, here):
    spec = nkibench.LEVELS[level]
    print(f"\n========== RL trace agent v4: level {level} ({spec['op']}) "
          f"episode {episode + 1}/{args.episodes} ==========")
    pool = load_exemplar_pool(state, level, args.seed_references, here)
    best = {"kernel": 0.0, "code": "", "feedback": ""}       # the repair anchor: best parsed attempt
    last_feedback, last_category, last_regressed = "", "initial", False
    history, seen, verified = [], set(), False
    repeated_last = False

    for round_i in range(args.rounds):
        context = f"level={level}|failure={last_category}"
        legal = [a for a in STRATEGIES
                 if not (a == "repair_diagnosis" and not history)
                 and not (a == "exemplar" and not pool)]
        action = strategy.select(context, rng, legal)
        temp_name = temperature.select(context, rng)
        temp = float(temp_name.split("=")[1])
        exemplar = pick_exemplar(pool, level) if action == "exemplar" else None
        prompt = build_prompt(level, action, args.trace, best["code"], best["feedback"],
                              last_feedback, last_regressed, history, exemplar, repeated_last)

        replies, wall = generate(args, prompt, temp, args.samples)
        results = []
        for sample_i, (reply, finish, budget, est_prompt) in enumerate(replies):
            truncated = finish == "length"
            trace, trace_valid = parse_trace(reply)
            raw_code = extract_kernel_code(reply, spec["entry"])
            code, dropped = sanitize_module(raw_code)
            kernel, parts, feedback = base_agent.grade(code, level)
            passed = bool(parts.get("correct"))
            category = classify_feedback(feedback, passed, truncated)
            feedback = enrich_feedback(feedback, category, dropped) if not passed else feedback
            fp = code_fingerprint(code) if code.strip() else ""
            repeated = bool(fp) and fp in seen
            hygiene = 1.0 if (parts.get("parses") and not dropped and not truncated) else 0.0
            improved = 1.0 if kernel > best["kernel"] + 1e-9 else 0.0
            trace_reward, trace_parts = trace_score(trace, level, trace_valid)
            rl_reward = shaped_reward(kernel, trace_reward, hygiene, improved, repeated, args.trace)
            results.append(dict(
                reply=reply, finish=finish, truncated=truncated, trace=trace,
                trace_valid=trace_valid, trace_reward=trace_reward, trace_parts=trace_parts,
                raw_code=raw_code, code=code, dropped=dropped, kernel=float(kernel), parts=parts,
                feedback=feedback, category=category, fingerprint=fp, repeated=repeated,
                hygiene=hygiene, improved=improved, rl=rl_reward, passed=passed))

        group_mean = sum(r["rl"] for r in results) / len(results)
        for sample_i, r in enumerate(results):
            strategy.update(context, action, r["rl"])
            temperature.update(context, temp_name, r["rl"])
            row = {
                "schema_version": SCHEMA_VERSION, "run_id": run_id, "episode": episode,
                "level": level, "round": round_i, "sample": sample_i,
                "action": action, "temperature": temp, "context": context,
                "kernel_reward": r["kernel"], "trace_reward": r["trace_reward"],
                "rl_reward": r["rl"], "advantage": r["rl"] - group_mean,
                "hygiene": r["hygiene"], "improved": r["improved"], "repeated": r["repeated"],
                "trace_enabled": bool(args.trace), "trace_parts": r["trace_parts"],
                "trace": r["trace"], "trace_valid_json": r["trace_valid"],
                "verifier_parts": r["parts"], "failure_category": r["category"],
                "feedback": r["feedback"], "code": r["code"],
                "raw_code_differs": r["raw_code"] != r["code"], "sanitizer_dropped": r["dropped"],
                "truncated": r["truncated"], "finish_reason": r["finish"],
                "prompt": prompt, "reply": r["reply"][:12000],
                "elapsed_s": round(wall, 3),
            }
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            if sft and r["kernel"] >= args.sft_min_reward and r["code"].strip():
                sft.write(json.dumps({"messages": [{"role": "user", "content": prompt},
                                                   {"role": "assistant", "content": r["reply"]}],
                                      "reward": r["kernel"], "level": level}) + "\n")
                sft.flush()
            flags = "".join([" REPEAT" if r["repeated"] else "", " CUT" if r["truncated"] else "",
                             f" stripped={len(r['dropped'])}" if r["dropped"] else ""])
            print(f"  r{round_i}.{sample_i} {action:16s} {temp_name} kernel={r['kernel']:.2f} "
                  f"RL={r['rl']:.3f} adv={r['rl'] - group_mean:+.3f} {wall:.0f}s "
                  f"failure={r['category']}{flags}")
            print("    verifier:", compact(r["feedback"], 320))
            if args.verbose and (r["category"] in ("parse", "truncated") or not r["code"].strip()):
                print("    reply head:", compact(r["reply"], 240))

        top = max(results, key=lambda r: (r["kernel"], r["rl"]))
        regressed = bool(best["code"]) and top["kernel"] < best["kernel"] - 1e-9
        if top["kernel"] > 0 and top["kernel"] >= best["kernel"] - 1e-9 and top["code"].strip():
            best = {"kernel": top["kernel"], "code": top["code"], "feedback": top["feedback"]}
        for r in results:
            if r["fingerprint"]:
                seen.add(r["fingerprint"])
        last_feedback, last_category, last_regressed = top["feedback"], top["category"], regressed
        repeated_last = top["repeated"]
        history.append(top["feedback"])
        state.save()

        winner = next((r for r in results if r["passed"]), None)
        if winner:
            verified = True
            state.verified[str(level)] = winner["code"]
            state.save()
            print("  VERIFIED: every checker shape passed.")
            break

    print(f"level {level} episode {episode + 1}: best kernel reward={best['kernel']:.3f}"
          f"{'  (verified)' if verified else ''}")
    return best["kernel"], verified


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, choices=sorted(nkibench.LEVELS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1,
                        help="replies per round, generated in parallel; >1 gives a group baseline")
    parser.add_argument("--episodes", type=int, default=1,
                        help="independent attempts per level; the policy carries over between them")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL") or
                        os.environ.get("GPTOSS_BASE_URL"))
    parser.add_argument("--path", default="", help="optional endpoint suffix such as /agg/v1")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--think", action="store_true",
                        help="not recommended; may consume output budget")
    parser.add_argument("--trace", action="store_true",
                        help="ask for the structured TRACE note and include it in the reward "
                             "(off by default: it scored 1.00 every time, so it carried no signal)")
    parser.add_argument("--seed-references", action="store_true",
                        help="let the 'exemplar' strategy show reference_level{n}.py of OTHER "
                             "levels as API-usage examples")
    parser.add_argument("--state", default="rl_state.json",
                        help="policy counts + verified kernels persisted here; '' disables")
    parser.add_argument("--export-sft", default="",
                        help="write (prompt, reply) rows at or above --sft-min-reward here, for a "
                             "later rejection-sampling fine-tune")
    parser.add_argument("--sft-min-reward", type=float, default=0.999)
    parser.add_argument("--log", default="rl_trace_attempts.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--exploration", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.base:
        sys.exit("Set KERNEL_AGENT_BASE_URL (or GPTOSS_BASE_URL) or pass --base.")
    args.base = args.base.rstrip("/") + args.path
    if not args.base.startswith(("http://", "https://")):
        sys.exit(f"Invalid base URL: {args.base!r}")
    if min(args.rounds, args.samples, args.episodes) < 1:
        sys.exit("--rounds, --samples and --episodes must all be >= 1.")

    here = os.path.dirname(os.path.abspath(__file__))
    rng = random.Random(args.seed)
    strategy = ShrunkUCB(STRATEGIES, exploration=args.exploration)
    temperature = ShrunkUCB(TEMPERATURES, exploration=args.exploration)
    state = State(args.state)
    state.load(strategy, temperature)
    run_id = time.strftime("%Y%m%dT%H%M%S")
    levels = sorted(nkibench.LEVELS) if args.all else [args.level or 1]

    sft = open(args.export_sft, "a", encoding="utf-8") if args.export_sft else None
    try:
        with open(args.log, "a", encoding="utf-8") as log:
            for level in levels:
                for episode in range(args.episodes):
                    run_episode(args, level, episode, run_id, strategy, temperature, state,
                                log, sft, rng, here)
            print("\nLearned strategy values (shrunk estimate, pulls) per level:")
            for level in levels:
                ctx = f"level={level}"
                print(f"  level {level}: " + "  ".join(
                    f"{a}={v:.2f}({n})" for a, v, n in strategy.table(ctx)))
                print(f"           " + "  ".join(
                    f"{a}={v:.2f}({n})" for a, v, n in temperature.table(ctx)))
    finally:
        if sft:
            sft.close()

    print(f"\nAttempt log: {args.log}")
    if args.state:
        print(f"Policy state: {args.state}  (delete it for a from-scratch comparison)")
    print("This learns prompt-strategy values online; it does not update model weights.")


if __name__ == "__main__":
    main()