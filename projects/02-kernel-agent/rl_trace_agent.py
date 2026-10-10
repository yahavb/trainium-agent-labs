#!/usr/bin/env python3
"""
rl_trace_agent.py — verifier-grounded RL controller for the existing NKI kernel agent.

This is an ONLINE agent-policy RL experiment, not a weight fine-tuning script:
the model endpoint in the original repo is inference-only. A contextual UCB bandit
learns which prompt/repair strategy works best for each NKI level from the actual
nkibench verifier reward. It also logs compact, structured, externally auditable
verification traces. No private/hidden chain-of-thought is requested or rewarded.

Run beside agent.py and nkibench.py:
  export KERNEL_AGENT_BASE_URL=http://localhost:8000/v1
  python rl_trace_agent.py --level 2 --rounds 8 --samples 2 --context 8192
  python rl_trace_agent.py --level 4 --rounds 8 --samples 2 --context 8192 --log rl_runs.jsonl
  python rl_trace_agent.py --all --rounds 6 --samples 2

The verifier remains authoritative: model-written explanations never count as proof.
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

ACTIONS = ("direct", "contract_trace", "counterexample", "repair_diagnosis")
TRACE_FIELDS = (
    "contract", "tile_strategy", "memory_flow", "edge_cases",
    "predicted_failure", "verification_checks", "diagnosis",
)
TRACE_RE = re.compile(r"<TRACE>\s*(\{.*?\})\s*</TRACE>", re.S)


def parse_trace(reply):
    """Parse the optional structured, visible engineering trace."""
    m = TRACE_RE.search(reply or "")
    if not m:
        return {}, False
    try:
        obj = json.loads(m.group(1))
        if not isinstance(obj, dict):
            return {}, False
        return obj, True
    except (json.JSONDecodeError, TypeError):
        return {}, False


def compact(s, limit=600):
    return str(s or "").strip().replace("\n", " ")[:limit]


def build_prompt(level, action, previous_code="", feedback="", history=()):
    """Action arms vary the engineering prompt, not the checker or reward."""
    spec = nkibench.LEVELS[level]
    if previous_code:
        core = base_agent.repair_prompt(level, previous_code, feedback)
    else:
        core = base_agent.first_prompt(level, 1)

    trace_instruction = """
Before the code, emit a compact, visible engineering record in this exact format:
<TRACE>
{"contract":"...","tile_strategy":"...","memory_flow":"...","edge_cases":"...",
 "predicted_failure":"...","verification_checks":["..."],"diagnosis":"..."}
</TRACE>
Keep each string short (one sentence). This is an auditable implementation note,
not a private reasoning transcript. Do not claim tests passed before the checker runs.
Then emit exactly one ```python code block containing the full NKI kernel.
"""
    if action == "direct":
        # Baseline arm: preserve the repo's short prompt behavior, add a concise trace.
        return core + "\n" + trace_instruction
    if action == "contract_trace":
        return (
            core + "\nFirst, make the tensor contract explicit in the trace: input/output "
            "shapes, dtype, operation, and tile limits. State the intended loop bounds and "
            "how the last partial tile is handled. Do not expand the explanation.\n"
            + trace_instruction
        )
    if action == "counterexample":
        return (
            core + "\nPrioritize hostile shapes: dimensions smaller than a tile, dimensions "
            "not divisible by a tile, and the final partial tile. In the trace, name the "
            "specific boundary risks and how the implementation avoids out-of-bounds access. "
            "Then implement the kernel.\n" + trace_instruction
        )
    if action == "repair_diagnosis":
        ledger = "\n".join("- " + compact(x, 220) for x in list(history)[-4:])
        return (
            core + "\nIn the trace, diagnose only the latest verifier evidence and identify "
            "one minimal code change that addresses it. Do not repeat failed approaches:\n"
            + (ledger or "- No prior failures recorded.") + "\n" + trace_instruction
        )
    raise ValueError(action)


def trace_score(trace, level, valid_format):
    """
    Conservative, deterministic trace rubric. It rewards a parseable, compact record
    covering independently checkable engineering obligations; it does NOT reward length,
    confidence, or claims that the kernel works.
    """
    if not valid_format:
        return 0.0, {"format": 0.0, "coverage": 0.0}
    coverage = sum(bool(str(trace.get(k, "")).strip()) for k in TRACE_FIELDS) / len(TRACE_FIELDS)
    checks = trace.get("verification_checks", [])
    if not isinstance(checks, list):
        checks = []
    checks_text = " ".join(str(x).lower() for x in checks)
    required = ["shape", "correct"]
    if level >= 3:
        required += ["tile", "memory"]
    checks_quality = sum(term in checks_text for term in required) / len(required)
    # The model must not pre-claim a pass. Such claims invalidate the trace.
    claim_text = " ".join(str(trace.get(k, "")) for k in TRACE_FIELDS).lower()
    forbidden = ("all tests pass", "verified correct", "checker passed", "successfully tested")
    honesty = 0.0 if any(x in claim_text for x in forbidden) else 1.0
    score = (0.65 * coverage + 0.25 * checks_quality + 0.10 * honesty)
    return score, {"format": 1.0, "coverage": coverage,
                   "checks_quality": checks_quality, "honesty": honesty}


def verifier_evidence(reward, parts, feedback):
    """Normalize evidence emitted by the trusted checker, not by the model."""
    passed = bool(parts.get("correct"))
    if "does not parse" in feedback.lower():
        category = "parse"
    elif "rule violations" in feedback.lower() or "not decorated" in feedback.lower():
        category = "static_rules"
    elif any(s in feedback.lower() for s in ("mismatch", "shapes passed", "out-of-bound",
                                               "raised ", "correct on cpu but wrong")):
        category = "execution_or_correctness"
    elif "cannot simulate" in feedback.lower():
        category = "environment"
    elif passed:
        category = "verified_pass"
    else:
        category = "other"
    return {"category": category, "verified_pass": passed,
            "reward": float(reward), "parts": parts, "feedback": feedback}


class UCBPolicy:
    """Per-level contextual bandit with UCB1 action selection and incremental means."""
    def __init__(self, actions=ACTIONS, exploration=0.8):
        self.actions = tuple(actions)
        self.exploration = float(exploration)
        self.counts = defaultdict(lambda: defaultdict(int))
        self.values = defaultdict(lambda: defaultdict(float))
        self.total = defaultdict(int)

    def select(self, context, rng):
        # Ensure each action is explored once per context before exploiting.
        for action in self.actions:
            if self.counts[context][action] == 0:
                return action
        t = max(1, self.total[context])
        def ucb(a):
            n = self.counts[context][a]
            return self.values[context][a] + self.exploration * math.sqrt(math.log(t + 1) / n)
        best = max(ucb(a) for a in self.actions)
        tied = [a for a in self.actions if abs(ucb(a) - best) < 1e-12]
        return rng.choice(tied)

    def update(self, context, action, reward):
        n = self.counts[context][action]
        self.counts[context][action] += 1
        self.total[context] += 1
        self.values[context][action] += (reward - self.values[context][action]) / (n + 1)

    def snapshot(self, context):
        return {a: {"n": self.counts[context][a],
                    "mean_reward": round(self.values[context][a], 4)}
                for a in self.actions}


def adjusted_reward(kernel_reward, trace_reward, parts, evidence):
    """
    Correctness dominates. At most 10% of reward is trace format/coverage, and 5%
    is evidence alignment. The evidence term is objective (checker category/pass).
    """
    # Give evidence alignment for a well-formed trace only; no model assertion of correctness.
    evidence_score = 1.0 if evidence["verified_pass"] else (
        0.5 if evidence["category"] in ("parse", "static_rules", "execution_or_correctness") else 0.0
    )
    return 0.85 * float(kernel_reward) + 0.10 * float(trace_reward) + 0.05 * evidence_score


def run_level(args, level, policy, log, rng):
    print(f"\n========== RL trace agent: level {level} ({nkibench.LEVELS[level]['op']}) ==========")
    best_reward, best_code = -1.0, ""
    last_code, last_feedback = "", ""
    history = []
    failure_state = "initial"
    for round_i in range(args.rounds):
        context = f"level={level}"  # enough repeated observations for online learning
        action = policy.select(context, rng)
        prompt = build_prompt(level, action, last_code, last_feedback, history)
        t0 = time.perf_counter()
        candidates = []
        for sample_i in range(args.samples):
            reply = base_agent.ask(args, prompt)
            trace, trace_ok = parse_trace(reply)
            code = base_agent.extract_code(reply)
            try:
                kernel_reward, parts, feedback = base_agent.grade(code, level)
            except SystemExit as exc:
                raise SystemExit(str(exc))
            evidence = verifier_evidence(kernel_reward, parts, feedback)
            trace_r, trace_parts = trace_score(trace, level, trace_ok)
            rl_reward = adjusted_reward(kernel_reward, trace_r, parts, evidence)
            row = {
                "schema_version": 1, "level": level, "round": round_i,
                "sample": sample_i, "action": action, "context": context,
                "failure_state_before_attempt": failure_state,
                "kernel_reward": kernel_reward, "trace_reward": trace_r,
                "rl_reward": rl_reward, "trace_parts": trace_parts,
                "trace": trace, "trace_valid_json": trace_ok,
                "verifier": evidence, "code": code,
                "elapsed_s": round(time.perf_counter() - t0, 3),
            }
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            candidates.append((rl_reward, kernel_reward, code, feedback, row))
        # Use the candidate with the highest RL reward, with kernel reward as tie-break.
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        rl_reward, kernel_reward, code, feedback, row = candidates[0]
        policy.update(context, action, rl_reward)
        print(f"round {round_i}: action={action:18s} kernel={kernel_reward:.2f} "
              f"trace={row['trace_reward']:.2f} rl={rl_reward:.3f} "
              f"({row['elapsed_s']}s)")
        print("  verifier:", compact(feedback, 260))
        if kernel_reward > best_reward:
            best_reward, best_code = kernel_reward, code
        if kernel_reward >= 0.999:
            print("  VERIFIED: all checker shapes passed.")
            last_code, last_feedback = code, feedback
            failure_state = "verified"
            break
        if code.strip():
            last_code = code
        last_feedback = feedback
        history.append(feedback)
        # Coarse, objective context; lets the policy learn separately for different failure types.
        if "parse" in feedback.lower():
            failure_state = "parse"
        elif "rule" in feedback.lower() or "decorated" in feedback.lower():
            failure_state = "rules"
        elif any(x in feedback.lower() for x in ("mismatch", "shapes passed", "raised ",
                                                  "out-of-bound", "correct on cpu")):
            failure_state = "correctness"
        elif not code.strip():
            failure_state = "empty"
        else:
            failure_state = "other"
    print(f"level {level}: best kernel reward={best_reward:.3f}")
    return best_reward, best_code


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, choices=sorted(nkibench.LEVELS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--model", default=base_agent.MODEL)
    ap.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL") or
                    os.environ.get("GPTOSS_BASE_URL"))
    ap.add_argument("--path", default="", help="optional endpoint suffix, e.g. /agg/v1")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--think", action="store_true",
                    help="normally leave off; repo experience shows this often consumes the budget")
    ap.add_argument("--log", default="rl_trace_attempts.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exploration", type=float, default=0.8)
    args = ap.parse_args()
    if not args.base:
        sys.exit("Set KERNEL_AGENT_BASE_URL, e.g. http://localhost:8000/v1, or pass --base.")
    args.base = args.base.rstrip("/") + args.path
    if not args.base.startswith(("http://", "https://")):
        sys.exit(f"Invalid base URL: {args.base!r}")
    rng = random.Random(args.seed)
    policy = UCBPolicy(exploration=args.exploration)
    levels = sorted(nkibench.LEVELS)[:4] if args.all else [args.level or 1]
    with open(args.log, "a", encoding="utf-8") as log:
        for level in levels:
            run_level(args, level, policy, log, rng)
    print(f"\nAttempt log: {args.log}")
    print("Note: this run learns prompt-strategy values online; it does not update LLM weights.")


if __name__ == "__main__":
    main()
