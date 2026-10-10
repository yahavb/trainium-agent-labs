"""Ask Qwen for a hypothesis and bounded operation graph, not executable Python."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

from math_program import OPERATIONS, lower
from math_tasks import TASKS, sources


def request(task, feedback, previous=None):
    info = TASKS[task]
    shapes = ({"displacement": [64, 8], "velocity": [64, 8], "stiffness": [64, 1],
               "damping": [64, 1]} if task == "spring" else {"forces": [64, 8]})
    system = (
        "You are an evidence-guided Trainium kernel optimization agent. Inspect the equation, "
        "shapes and baseline, choose a justified transformation and revise based on feedback. "
        "No technique is assigned by the controller. Preserve the equation, all inputs and output shape. "
        "Return ONLY JSON with fields hypothesis (brief string), expected_effect (brief string), "
        "program {operations: [{id: 'v0', op: ..., inputs: [...], engine: 'auto' or 'vector'}], result: 'v0'}. "
        f"Allowed operations: {OPERATIONS}. mul/add/sub take 2 values; neg/sum/serial_sum/copy take 1. "
        "mul_neg(a,b) computes -(a*b) with a fused instruction; its second operand must be (64,1). "
        "sum and serial_sum both sum the free axis and return (64,1). Binary operations allow a "
        "second operand of the same shape or (64,1). Use at most 16 operations and unique IDs v0..v99. "
        "Reference only inputs or earlier IDs. No Python, extra operations, constants or shape changes. "
        "These restrictions describe a bounded NKI code-generation space, not a recommendation "
        "to use any particular transformation. A shorter graph is not proof of better throughput. "
        "Do not claim success without checker and timing evidence."
    )
    user = (f"Equation: {info['equation']}\nInput shapes: {json.dumps(shapes)}\n"
            "Original baseline NKI:\n" + sources(task)["baseline"] +
            "\nMeasured feedback:\n" + feedback[:2500] +
            ("\nPrevious untrusted proposal:\n" + json.dumps(previous) if previous else ""))
    return dict(model="Qwen/Qwen3-8B", temperature=.4, max_tokens=1200,
                chat_template_kwargs=dict(enable_thinking=False),
                messages=[dict(role="system", content=system), dict(role="user", content=user)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    previous = json.loads(args.previous.read_text()) if args.previous else None
    payload = request(args.task, args.feedback.read_text(), previous)
    saved = json.dumps(payload, indent=2) + "\n"
    (args.out / "request.json").write_text(saved)
    record = dict(task=args.task, status="dry_run", correctness_score=None,
                  request_sha256=hashlib.sha256(saved.encode()).hexdigest())
    try:
        if not args.dry_run:
            headers = {"Content-Type": "application/json"}
            if os.environ.get("KERNEL_AGENT_API_KEY"):
                headers["Authorization"] = "Bearer " + os.environ["KERNEL_AGENT_API_KEY"]
            http = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions",
                json.dumps(payload).encode(), headers)
            started = time.monotonic()
            with urllib.request.urlopen(http, timeout=300) as response:
                raw = response.read()
            (args.out / "response.json").write_bytes(raw)
            response = json.loads(raw)
            choice = response["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("Truncated generation")
            content = choice["message"]["content"].strip()
            if content.startswith("```"):
                content = "\n".join(content.splitlines()[1:-1])
            proposal = json.loads(content)
            if not isinstance(proposal.get("hypothesis"), str) or not proposal["hypothesis"].strip():
                raise ValueError("Missing optimization hypothesis")
            source = lower(args.task, proposal["program"])
            (args.out / "proposal.json").write_text(json.dumps(proposal, indent=2) + "\n")
            (args.out / "candidate.py").write_text(source)
            record.update(status="generated_not_evaluated", seconds=time.monotonic() - started,
                          hypothesis=proposal["hypothesis"], usage=response.get("usage"),
                          candidate_sha256=hashlib.sha256(source.encode()).hexdigest())
    except Exception as exc:
        record.update(status="generation_error", error=f"{type(exc).__name__}: {exc}")
    (args.out / "generation-attempt.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record["status"])
    return int(record["status"] == "generation_error")


if __name__ == "__main__":
    raise SystemExit(main())
