"""Generate an auditable Qwen candidate; never execute untrusted generated code."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request
import uuid

from nki_guidance import sdk_context


SPEC = """Write a complete NKI Python module with exactly this entry point:
@nki.jit
def contact_batch(a_transposed, bias, initial, alpha, steps):

Solve independent regularized gripping force QPs: minimize .5*f.T*A*f+b.T*f,
subject to f>=0. These are pyramid-edge force coefficients, not direct xyz forces.
Input FP32 shapes: bias/initial/alpha=(16,B); a_transposed=(16,16*B), B=1/2/4/8/16.
World w has A.T in columns 16*w:16*(w+1). First eight rows are active; remaining
rows/columns are zero padding. initial is zero; alpha contains each world's
1/largest-eigenvalue step size repeated across rows. steps is a positive static
iteration budget. Return one FP32 HBM tensor (16,B), with zero padding.
All input arrays must remain unchanged. You may change the solving algorithm,
not the objective, friction model or accuracy gates. Do not specialize to saved
case answers. Do not use filesystem, network, Python/NumPy/SciPy host solvers,
or external modules; use import nki, import nki.language as nl, import nki.isa as nisa.
The baseline below demonstrates the installed NKI API; do not invent APIs.
Improve convergence first, then optimize throughput across the full public suite.
Return exactly one Python code block containing the complete module.
"""

PLANE_SPEC = """Write one complete NKI Python module with this entry point:
@nki.jit
def contact_batch(a_transposed, bias, initial, alpha, steps):

Task: improve throughput of independent frictionless contact impulse QPs.
These are synthetic plane/pair contact snapshots, not gripping or MuJoCo rollouts.
Preserve the baseline projected-gradient recurrence exactly in mathematics:
x_next = max(0, x - alpha*(A*x+b)), starting at initial and running exactly steps.
Do not add momentum, alter step sizes, reduce steps, or change the objective.
Primary workload: 33 active contacts padded to 64, B=8 worlds, steps=32.
FP32 bias/initial/alpha shapes=(64,B), a_transposed=(64,64*B).
World w stores A.T in columns 64*w:64*(w+1). alpha is each world's step size
repeated down its column. Return one FP32 HBM tensor (64,B), preserve inputs,
and keep padded output rows 33:64 zero. Do not hardcode case solutions or seeds.
Keep B=8 and the same workload for baseline and candidate timing comparisons.
Optimize scheduling, intermediate copies, instruction fusion or world layout.
The supplied baseline and installed SDK guidance establish the supported API.
Use only import nki, import nki.language as nl, import nki.isa as nisa.
No filesystem, network, host solvers, external modules or import-time side effects.
All worlds must pass fixed-update equivalence AND the independent physics checker
before any accepted-throughput claim. Simulation is not device timing.
Return exactly one Python code block with the complete module.
"""


def sha(value):
    return hashlib.sha256(value).hexdigest()


def same_program(first, second):
    """Detect identical syntax despite changes to comments or formatting."""
    return ast.dump(ast.parse(first), include_attributes=False) == ast.dump(ast.parse(second), include_attributes=False)


def extract_source(content):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", content, flags=re.DOTALL)
    if len(blocks) > 1:
        raise ValueError("Expected one complete code block, not several fragments")
    source = blocks[0] if blocks else content
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "contact_batch"]
    expected = ["a_transposed", "bias", "initial", "alpha", "steps"]
    if len(functions) != 1 or [arg.arg for arg in functions[0].args.args] != expected:
        raise ValueError("Wrong contact_batch entry point or argument names")
    return source.rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("gripping", "plane-throughput"), default="gripping")
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path(__file__).with_name("nki_contact_batch.py"))
    parser.add_argument("--previous", type=Path, help="Previous candidate as context only; never executed")
    parser.add_argument("--base-url", default=os.environ.get("KERNEL_AGENT_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-8B"))
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.base_url:
        parser.error("Set KERNEL_AGENT_BASE_URL or --base-url; no model endpoint is assumed")
    if args.max_tokens < 1:
        parser.error("Positive max-tokens required")
    if not 0 <= args.temperature <= 2:
        parser.error("Temperature must be in [0,2]")
    feedback = args.feedback.read_text()
    if len(feedback) > 7000:
        parser.error("Use the compact next-prompt.txt, not the old verbose report")
    baseline = args.baseline.read_text()
    previous = args.previous.read_text() if args.previous else ""
    nki_guide, sdk_metadata = sdk_context()
    retrieval = None
    if args.task == "plane-throughput":
        from optimization_knowledge import retrieve
        retrieved_text, retrieval = retrieve(feedback)
        nki_guide += "\n" + retrieved_text
    prefix = "qwen-grip" if args.task == "gripping" else "qwen-plane"
    out = args.out or Path("data") / f"{prefix}-{uuid.uuid4().hex}"
    out.mkdir(parents=True, exist_ok=False)
    request = dict(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens,
                   chat_template_kwargs=dict(enable_thinking=False),
                   messages=[dict(role="system", content=(SPEC if args.task == "gripping" else PLANE_SPEC) + "\n" + nki_guide),
                             dict(role="user", content="Baseline NKI source:\n" + baseline +
                                  ("\nPrevious candidate to revise (not a trusted API example):\n" + previous if previous else "") +
                                  "\nPublic checker feedback:\n" + feedback)])
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (out / "baseline.py").write_text(baseline)
    (out / "feedback.txt").write_text(feedback)
    (out / "nki-guidance.txt").write_text(nki_guide)
    (out / "sdk-context.json").write_text(json.dumps(sdk_metadata, indent=2) + "\n")
    if retrieval is not None:
        (out / "retrieval.json").write_text(json.dumps(retrieval, indent=2) + "\n")
    if args.previous:
        (out / "previous-candidate.py").write_text(previous)
    record = dict(attempt_id=out.name, task=args.task, model=args.model, status="dry_run" if args.dry_run else "pending",
                  temperature=args.temperature, max_tokens=args.max_tokens,
                  nki_guidance_sha256=sha(nki_guide.encode()),
                  sdk_context_sha256=sha((out / "sdk-context.json").read_bytes()),
                  physics_score=None, baseline_sha256=sha(baseline.encode()),
                  feedback_sha256=sha(feedback.encode()), request_sha256=sha((out / "request.json").read_bytes()))
    if args.previous:
        record["previous_candidate_sha256"] = sha(previous.encode())
    if retrieval is not None:
        record["retrieval_sha256"] = sha((out / "retrieval.json").read_bytes())
    try:
        if not args.dry_run:
            url = args.base_url.rstrip("/") + "/chat/completions"
            headers = {"Content-Type": "application/json"}
            if os.environ.get("KERNEL_AGENT_API_KEY"):
                headers["Authorization"] = "Bearer " + os.environ["KERNEL_AGENT_API_KEY"]
            started = time.perf_counter()
            http = urllib.request.Request(url, json.dumps(request).encode(), headers)
            with urllib.request.urlopen(http, timeout=300) as response:
                raw = response.read()
            record["generation_seconds"] = time.perf_counter() - started
            (out / "response.json").write_bytes(raw)
            payload = json.loads(raw)
            choice = payload["choices"][0]
            record["finish_reason"] = choice.get("finish_reason")
            record["usage"] = payload.get("usage")
            if choice.get("finish_reason") == "length":
                raise ValueError("Generation truncated; increase output/context budget before testing")
            source = extract_source(choice["message"].get("content") or "")
            (out / "candidate.py").write_text(source)
            record.update(status="generated_not_evaluated", candidate_sha256=sha(source.encode()))
            if previous and same_program(source, previous):
                record.update(status="duplicate_previous", revision_score=0,
                              feedback="Identical program to previous candidate; no new revision. Physics remains unmeasured for this attempt; reuse prior evidence only under identical execution settings.")
    except Exception as exc:
        record.update(status="generation_error", error=f"{type(exc).__name__}: {exc}")
    (out / "generation-attempt.json").write_text(json.dumps(record, indent=2) + "\n")
    with (out / "generation-attempts.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(f"{record['status']}: {out}; physics_score remains unmeasured")
    return int(record["status"] in ("generation_error", "duplicate_previous"))


if __name__ == "__main__":
    raise SystemExit(main())
