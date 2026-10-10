#!/usr/bin/env python3
"""
Verifier-grounded online RL controller for projects/02-kernel-agent.

This learns which prompt/repair strategy works best from the EXISTING nkibench verifier.
It does not fine-tune the model weights. It also asks for a short, auditable engineering
trace, not private chain-of-thought.

Run from projects/02-kernel-agent:
  python rl_trace_agent.py --level 2 --rounds 8 --samples 1 --context 8192
  python rl_trace_agent.py --level 4 --rounds 8 --samples 1 --context 8192
"""

import argparse
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

MODEL = os.environ.get("KERNEL_AGENT_MODEL", getattr(base_agent, "MODEL", "Qwen/Qwen3-8B"))
ACTIONS = ("direct", "contract_trace", "counterexample", "repair_diagnosis")
TRACE_FIELDS = (
    "contract", "tile_strategy", "memory_flow", "edge_cases",
    "predicted_failure", "verification_checks", "diagnosis",
)
TRACE_RE = re.compile(r"<TRACE>\s*(\{.*?\})\s*</TRACE>", re.S)

# Keep this deliberately short. Most important fix: spell out the three valid imports
# verbatim; `import nl` is invalid and must not be inferred from the alias.
IMPORT_CONTRACT = """MANDATORY imports, verbatim:
import nki
import nki.language as nl
import nki.isa as nisa

Never write `import nl`, `import nki.nl`, or `from nl import ...`.
`nl` is an alias created by `import nki.language as nl`; it is not a standalone module.
Allocate buffers with `nl.ndarray(shape, dtype=..., buffer=nl.sbuf)` or
`buffer=nl.shared_hbm` / `buffer=nl.psum`. These buffer names are values, not functions:
write `buffer=nl.sbuf`, NEVER `nl.sbuf(...)`.
Use the exact entry-point name and `@nki.jit` required by the checker.
Reply with one <TRACE> JSON record followed by exactly one ```python code block.
"""


def compact(value, limit=500):
    return str(value or "").strip().replace("\n", " ")[:limit]


def parse_trace(reply):
    match = TRACE_RE.search(reply or "")
    if not match:
        return {}, False
    try:
        result = json.loads(match.group(1))
        return (result, True) if isinstance(result, dict) else ({}, False)
    except (json.JSONDecodeError, TypeError):
        return {}, False


def trace_instructions(action, history):
    common = """
Before the code, emit a short engineering record in this exact format:
<TRACE>
{"contract":"input/output shapes and operation","tile_strategy":"tile and loop plan",
 "memory_flow":"where data is allocated and copied","edge_cases":"partial/boundary tiles",
 "predicted_failure":"one likely risk","verification_checks":["shape check","correctness check"],
 "diagnosis":"initial hypothesis or latest failure diagnosis"}
</TRACE>
Use concise, factual fields; do not claim the checker passed before it runs.
This is an auditable implementation note, not a private reasoning transcript.
"""
    if action == "direct":
        extra = "Implement the reference operation directly; avoid unnecessary abstractions."
    elif action == "contract_trace":
        extra = ("State the tensor shapes, dtype, output contract, loop bounds, and how any "
                 "partial final tile is handled. Keep the kernel itself simple.")
    elif action == "counterexample":
        extra = ("Pay special attention to dimensions smaller than a tile, non-divisible "
                 "dimensions, and final partial tiles. Do not read or write out of bounds.")
    else:
        ledger = "\n".join("- " + compact(item, 180) for item in list(history)[-3:])
        extra = ("Use the most recent checker evidence to make one minimal repair. Do not repeat "
                 "a failed import or buffer allocation. Recent checker feedback:\n" +
                 (ledger if ledger else "- No prior failures yet."))
    return extra + "\n" + common


def build_prompt(level, action, previous_code="", feedback="", history=()):
    spec = nkibench.LEVELS[level]
    if previous_code:
        core = base_agent.repair_prompt(level, previous_code, feedback)
    else:
        # Use the repository's short prompt; long API lectures have previously caused
        # these models to spend the budget reasoning instead of returning code.
        core = base_agent.first_prompt(level, 1)
    return (
        core + "\n\n" + IMPORT_CONTRACT + "\n" +
        trace_instructions(action, history) +
        f"\nRequired function name: {spec['entry']}. Keep all imports inside the code block."
    )


def trace_score(trace, level, valid_format):
    """Score trace coverage, never verbosity or self-reported correctness."""
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
    return score, {
        "format": 1.0, "coverage": coverage,
        "checks_quality": check_quality, "honesty": honesty,
    }


def classify_feedback(feedback, passed):
    text = (feedback or "").lower()
    if "no module named 'nl'" in text or 'no module named "nl"' in text:
        return "bad_import"
    if "does not parse" in text or "no code came back" in text:
        return "parse"
    if "rule violations" in text or "not decorated" in text:
        return "rules"
    if "memoryregion' object is not callable" in text or "memoryregion" in text:
        return "buffer_api"
    if any(term in text for term in (
        "mismatch", "wrong shape", "shapes passed", "out-of-bound", "raised ",
        "correct on cpu but wrong", "non-finite output",
    )):
        return "correctness"
    if "cannot simulate" in text or "cannot import" in text:
        return "environment"
    return "verified" if passed else "other"


def adjusted_reward(kernel_reward, trace_reward, feedback, passed):
    # Kernel verifier reward remains dominant. Evidence classification is based only on
    # the checker output, not on claims in the generated trace.
    category = classify_feedback(feedback, passed)
    evidence_reward = 1.0 if passed else (
        0.5 if category in ("bad_import", "parse", "rules", "buffer_api", "correctness") else 0.0
    )
    return 0.85 * float(kernel_reward) + 0.10 * float(trace_reward) + 0.05 * evidence_reward, category


class UCBPolicy:
    """Contextual UCB1 bandit: strategy choices conditioned on level + observed failure type."""
    def __init__(self, actions=ACTIONS, exploration=0.8):
        self.actions = tuple(actions)
        self.exploration = float(exploration)
        self.counts = defaultdict(lambda: defaultdict(int))
        self.values = defaultdict(lambda: defaultdict(float))
        self.total = defaultdict(int)

    def select(self, context, rng):
        for action in self.actions:
            if self.counts[context][action] == 0:
                return action
        total = max(1, self.total[context])
        scores = {
            action: self.values[context][action] +
                    self.exploration * math.sqrt(math.log(total + 1) / self.counts[context][action])
            for action in self.actions
        }
        best = max(scores.values())
        return rng.choice([action for action, score in scores.items() if abs(score - best) < 1e-12])

    def update(self, context, action, reward):
        old_n = self.counts[context][action]
        self.counts[context][action] += 1
        self.total[context] += 1
        self.values[context][action] += (reward - self.values[context][action]) / (old_n + 1)


def run_level(args, level, policy, log, rng):
    print(f"\n========== RL trace agent: level {level} ({nkibench.LEVELS[level]['op']}) ==========")
    best_kernel_reward = -1.0
    previous_code, previous_feedback = "", ""
    history = []
    failure_state = "initial"

    for round_i in range(args.rounds):
        context = f"level={level}|failure={failure_state}"
        action = policy.select(context, rng)
        prompt = build_prompt(level, action, previous_code, previous_feedback, history)
        candidates = []

        for sample_i in range(args.samples):
            started = time.perf_counter()
            reply = base_agent.ask(args, prompt)
            elapsed = time.perf_counter() - started
            trace, trace_valid = parse_trace(reply)
            code = base_agent.extract_code(reply)
            kernel_reward, parts, feedback = base_agent.grade(code, level)
            passed = bool(parts.get("correct"))
            trace_reward, trace_parts = trace_score(trace, level, trace_valid)
            rl_reward, category = adjusted_reward(kernel_reward, trace_reward, feedback, passed)
            row = {
                "schema_version": 2,
                "level": level, "round": round_i, "sample": sample_i,
                "action": action, "context": context,
                "kernel_reward": float(kernel_reward), "trace_reward": trace_reward,
                "rl_reward": rl_reward, "trace_parts": trace_parts,
                "trace": trace, "trace_valid_json": trace_valid,
                "verifier_parts": parts, "failure_category": category,
                "feedback": feedback, "code": code, "elapsed_s": round(elapsed, 3),
            }
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            candidates.append((rl_reward, kernel_reward, code, feedback, category, row))
            print(f"  sample {sample_i}: kernel={kernel_reward:.2f} trace={trace_reward:.2f} "
                  f"RL={rl_reward:.3f} elapsed={elapsed:.1f}s failure={category}")
            print("    verifier:", compact(feedback, 320))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        rl_reward, kernel_reward, code, feedback, category, row = candidates[0]
        policy.update(context, action, rl_reward)

        if kernel_reward > best_kernel_reward:
            best_kernel_reward = kernel_reward

        if code.strip():
            previous_code = code
        previous_feedback = feedback
        history.append(feedback)
        failure_state = category

        if bool(row["verifier_parts"].get("correct")):
            print("  VERIFIED: every checker shape passed.")
            break

    print(f"level {level}: best kernel reward={best_kernel_reward:.3f}")
    return best_kernel_reward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, choices=sorted(nkibench.LEVELS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL") or
                        os.environ.get("GPTOSS_BASE_URL"))
    parser.add_argument("--path", default="", help="optional endpoint suffix such as /agg/v1")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--think", action="store_true",
                        help="not recommended; repo measurements show it often consumes the budget")
    parser.add_argument("--log", default="rl_trace_attempts.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--exploration", type=float, default=0.8)
    args = parser.parse_args()

    if not args.base:
        sys.exit("Set KERNEL_AGENT_BASE_URL (the repo seat pod normally sets it already) or pass --base.")
    args.base = args.base.rstrip("/") + args.path
    if not args.base.startswith(("http://", "https://")):
        sys.exit(f"Invalid base URL: {args.base!r}")
    if args.rounds < 1 or args.samples < 1:
        sys.exit("--rounds and --samples must both be >= 1.")

    rng = random.Random(args.seed)
    policy = UCBPolicy(exploration=args.exploration)
    levels = sorted(nkibench.LEVELS) if args.all else [args.level or 1]
    with open(args.log, "a", encoding="utf-8") as log:
        for level in levels:
            run_level(args, level, policy, log, rng)

    print(f"\nAttempt log: {args.log}")
    print("This learns prompt-strategy values online; it does not update model weights.")


if __name__ == "__main__":
    main()
