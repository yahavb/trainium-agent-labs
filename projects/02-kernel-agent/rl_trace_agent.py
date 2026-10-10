#!/usr/bin/env python3
"""
Verifier-grounded online RL controller for projects/02-kernel-agent.

Learns which prompt/repair strategy works best from the existing nkibench verifier.
This is online policy learning over prompt strategies; it does not fine-tune model weights.
It requests a short, auditable engineering record, not private chain-of-thought.

Run from projects/02-kernel-agent:
  python rl_trace_agent.py --level 2 --rounds 8 --samples 1 --context 8192
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
Keep fields concise and factual. Do not claim the checker passed before it runs.
This is an auditable implementation note, not a private reasoning transcript.
"""
    if action == "direct":
        extra = "Implement the operation directly; avoid unnecessary abstractions."
    elif action == "contract_trace":
        extra = ("State the tensor shapes, dtype, output contract, loop bounds, and how partial "
                 "tiles are handled. Keep the kernel simple.")
    elif action == "counterexample":
        extra = ("Prioritize dimensions smaller than a tile, non-divisible dimensions, and final "
                 "partial tiles. Do not read or write out of bounds.")
    else:
        ledger = "\n".join("- " + compact(item, 200) for item in list(history)[-3:])
        extra = ("Use the actual checker feedback below to make a minimal repair. Do not repeat "
                 "the same failed API usage:\n" + (ledger or "- No previous failures.") + "\n")
    return extra + "\n" + common


def build_prompt(level, action, previous_code="", feedback="", history=()):
    spec = nkibench.LEVELS[level]
    if previous_code:
        core = base_agent.repair_prompt(level, previous_code, feedback)
    else:
        core = base_agent.first_prompt(level, 1)

    level_specific = LEVEL2_GUIDANCE if level == 2 else ""
    return (
        core + "\n\n" + NKI_API_CARD + "\n" + level_specific + "\n" +
        trace_instructions(action, history) +
        f"\nRequired entry point: {spec['entry']}. Keep imports and implementation in the code block."
    )


def trace_score(trace, level, valid_format):
    """Score field coverage, not verbosity or model assertions of correctness."""
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
    if "reshape() takes" in text or "reshape" in text and "positional arguments" in text:
        return "reshape_api"
    if "does not parse" in text or "no code came back" in text:
        return "parse"
    if "rule violations" in text or "not decorated" in text:
        return "rules"
    if "memoryregion" in text or "object is not callable" in text:
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
    category = classify_feedback(feedback, passed)
    evidence_reward = 1.0 if passed else (
        0.5 if category in (
            "bad_import", "reshape_api", "parse", "rules", "buffer_api", "correctness"
        ) else 0.0
    )
    return 0.85 * float(kernel_reward) + 0.10 * float(trace_reward) + 0.05 * evidence_reward, category


class UCBPolicy:
    """Contextual UCB1 bandit conditioned on level and observed failure category."""
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
        return rng.choice([a for a, score in scores.items() if abs(score - best) < 1e-12])

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
                "schema_version": 3, "level": level, "round": round_i, "sample": sample_i,
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
        _, kernel_reward, code, feedback, category, row = candidates[0]
        policy.update(context, action, row["rl_reward"])
        best_kernel_reward = max(best_kernel_reward, kernel_reward)

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
                        help="not recommended; may consume output budget")
    parser.add_argument("--log", default="rl_trace_attempts.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--exploration", type=float, default=0.8)
    args = parser.parse_args()

    if not args.base:
        sys.exit("Set KERNEL_AGENT_BASE_URL (or GPTOSS_BASE_URL) or pass --base.")
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
