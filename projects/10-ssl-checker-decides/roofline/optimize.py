"""Can Qwen make a correct kernel faster, and does the roofline feedback help?

Starts from a CORRECT but slow level-4 matmul (reference_level4.py with TILE_N=128) and asks the model
for one change at a time. A candidate is kept only if it is still correct on all four nkibench shapes
and its predicted latency (compiler, no device) beats the best so far by more than 1%.

Feedback conditions, identical except for one paragraph:
  specific  the instruction counter's ONE named change (instcount.py) plus the floor in one line
  roofline  the roofline tool's diagnosis: floor, what binds, wasted bytes, busiest engine
  control   the predicted latency numbers only

Measured 2026-10-10 with `roofline` vs `control` (logs in run1-generic/): under `roofline` Qwen
returned the kernel unchanged in 24 of 24 attempts; `control` reached 1.03x. Hence `specific`.

    python optimize.py --condition specific --repeat 3 --log opt-specific.jsonl
    python optimize.py --condition control  --repeat 3 --log opt-control.jsonl
    python optimize.py --offline            # no model: replays the reference kernel once

Runs in a seat pod. Imports nkibench.py and agent.py from /workspace/projects/02-kernel-agent and
writes kernels only to fresh files under /tmp (agent.py's grader uses a fixed path, which a
concurrent agent.py run would race on).
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time
import types
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = "/workspace/projects/02-kernel-agent"
sys.path.insert(0, HERE)
sys.path.insert(1, BENCH)
sys.dont_write_bytecode = True
warnings.simplefilter("ignore")

import numpy as np  # noqa: E402

import agent  # noqa: E402
import nkibench  # noqa: E402
import roofline as rl  # noqa: E402
from instcount import simulate_and_count_all, specific_feedback  # noqa: E402
from predict import predict  # noqa: E402

LEVEL = 4
SHAPES = nkibench.LEVELS[LEVEL]["shapes"]
ENTRY = nkibench.LEVELS[LEVEL]["entry"]


def start_kernel():
    src = open(f"{BENCH}/reference_level4.py").read()
    old = "TILE_N = nl.tile_size.gemm_moving_fmax  # 512"
    assert old in src
    return src.replace(old, "TILE_N = 128")


def load(src):
    fd, path = tempfile.mkstemp(prefix="_opt_", suffix=".py", dir="/tmp")
    with os.fdopen(fd, "w") as f:
        f.write(src)
    return nkibench.load_kernel(path, ENTRY)


def evaluate(src):
    """Correctness on every shape first; only then the compiler's prediction per shape."""
    if not src.strip():
        return dict(ok=False, why="No code came back. Reply with one python code block.")
    try:
        compile(src, "<candidate>", "exec")
    except SyntaxError as e:
        return dict(ok=False, why=f"The code does not parse: {e.msg} on line {e.lineno}.")
    bad = nkibench.check_rules(src, LEVEL)
    if bad:
        return dict(ok=False, why="Rule violations: " + " ".join(bad))
    try:
        kernel = load(src)
    except Exception as e:
        return dict(ok=False, why=f"{ENTRY} could not be loaded: {type(e).__name__}: {e}")
    rows = []
    for spec in SHAPES:
        args, _ = nkibench.make_inputs(spec, LEVEL)
        before = [x.copy() for x in args]
        want = nkibench.ref_matmul(*args)
        lbl = nkibench.label(spec, LEVEL)
        try:
            got, counted = simulate_and_count_all(kernel, args)
        except Exception as e:
            return dict(ok=False, why=f"On {lbl}: " + agent.enrich(f"raised {type(e).__name__}: {e}"))
        m = nkibench.check_inputs_untouched(before, args) or nkibench.describe_mismatch(got, want)
        if m:
            return dict(ok=False, why=f"On {lbl}: {m}")
        try:
            p = predict(kernel, {"lhsT": args[0], "rhs": args[1]})
        except Exception as e:
            return dict(ok=False, why=f"Correct in the simulator but does not compile for the chip "
                                      f"on {lbl}: {type(e).__name__}: {str(e)[:400]}")
        op = rl.matmul(spec["M"], spec["K"], spec["N"], "float32")
        floor = rl.analyze(op, rl.MODEL)["floor_ns"]
        rows.append(dict(shape=lbl, spec=spec, op=op, pred_ns=p["ns"], util=p["util"],
                         bytes=counted["dma_bytes"], counts=counted, model_floor_ns=floor,
                         ratio=p["ns"] / floor))
    total = sum(r["pred_ns"] for r in rows)
    total_bytes = sum(r["bytes"] for r in rows)
    worst_waste = max(r["bytes"] / ((r["spec"]["M"] * r["spec"]["K"] + r["spec"]["K"] * r["spec"]["N"]
                                     + r["spec"]["M"] * r["spec"]["N"]) * 4) for r in rows)
    gmean = math.exp(sum(math.log(r["ratio"]) for r in rows) / len(rows))
    return dict(ok=True, rows=rows, total_ns=total, gmean_ratio=gmean, total_bytes=total_bytes,
                worst_waste=worst_waste)


def perf_feedback(ev, condition):
    per = ", ".join(f"{r['shape']}: {r['pred_ns'] / 1000:.1f} us" for r in ev["rows"])
    lines = [f"Correct on all 4 shapes. Predicted latency {ev['total_ns'] / 1000:.1f} us in total ({per})."]
    if condition == "specific" and getattr(perf_feedback, "objective", "") == "bytes":
        lines = [f"Correct on all 4 shapes. Worst HBM traffic is {ev['worst_waste']:.2f}x the byte floor; "
                 f"the target is at most 1.60x on every shape."]
    if condition == "specific":
        floor_total = sum(r["model_floor_ns"] for r in ev["rows"])
        lines.append(f"A perfect kernel would be predicted at about {floor_total / 1000:.1f} us in total.")
        # The largest shape shows the excess most clearly; fall back through the others.
        for r in sorted(ev["rows"], key=lambda r: -r["pred_ns"]):
            s = r["spec"]
            msg = specific_feedback(r["counts"], s["M"], s["K"], s["N"], r["shape"])
            if msg:
                lines.append(msg)
                break
    if condition == "roofline":
        worst = max(ev["rows"], key=lambda r: r["ratio"])
        floor_total = sum(r["model_floor_ns"] for r in ev["rows"])
        lines.append(f"A perfect kernel would be predicted at about {floor_total / 1000:.1f} us in total. "
                     f"The furthest from its floor is {worst['shape']}:")
        lines.append(rl.feedback(worst["op"], worst["pred_ns"], worst["bytes"], worst["util"]))
    return "\n".join(lines)


def opt_prompt(src, feedback, failed):
    tail = (f"\n\nYour previous change broke it, so it was discarded: {failed[:500]}" if failed else "")
    return (
        f"This NKI kernel computes a tiled matmul correctly (lhsT is [K, M], rhs is [K, N], result is "
        f"[M, N]; K and M are multiples of 128 and N a multiple of 512).\n\n"
        f"```python\n{src}\n```\n\n{feedback}{tail}\n\n"
        + ("Make it move fewer bytes between HBM and SBUF with ONE change (load each tile once and reuse "
           "it, rather than reloading it inside a loop that does not change it), keeping it correct for "
           "every such shape. "
           if getattr(perf_feedback, "objective", "") == "bytes" else
           "Make it faster with ONE change, keeping it correct for every such shape. ")
        + f"Hardware limits: a "
        f"tile's partition dimension is at most 128; nc_matmul's stationary free dimension at most 128 "
        f"and moving free dimension at most 512. Reply with ONE python code block containing the whole "
        f"kernel.")


def score(ev, a):
    return ev["total_bytes"] if a.objective == "bytes" else ev["total_ns"]


def run(a, rep, log):
    best_src = open(a.start).read() if a.start else start_kernel()
    best = evaluate(best_src)
    assert best["ok"], best
    start_total = best["total_ns"]
    feedback = perf_feedback(best, a.condition)
    failed = ""
    print(f"\n######## {a.condition} run {rep + 1}: start {start_total / 1000:.1f} us, "
          f"gmean {best['gmean_ratio']:.2f}x the model floor")
    stale = 0
    for rnd in range(a.rounds):
        if stale >= a.patience:
            print(f"  stopping: no improvement in {stale} rounds")
            break
        t0 = time.time()
        prompt = opt_prompt(best_src, feedback, failed)
        replies = ([f"```python\n{open(f'{BENCH}/reference_level4.py').read()}\n```"] if a.offline
                   else agent.ask_parallel(a, prompt, a.samples))
        cands = []
        for reply in replies:
            src = agent.extract_code(reply)
            ev = evaluate(src)
            cands.append((src, ev))
            log.write(json.dumps(dict(condition=a.condition, repeat=rep, round=rnd, ok=ev["ok"],
                                      total_ns=ev.get("total_ns"), gmean_ratio=ev.get("gmean_ratio"),
                                      per_shape_ns=[r["pred_ns"] for r in ev.get("rows", [])],
                                      why=ev.get("why"), prompt_chars=len(prompt), code=src)) + "\n")
        log.flush()
        good = [c for c in cands if c[1]["ok"]]
        n_ok = len(good)
        if good:
            src, ev = min(good, key=lambda c: score(c[1], a))
            if score(ev, a) < score(best, a) * 0.99:
                best_src, best = src, ev
                feedback = perf_feedback(best, a.condition)
                failed = ""
                stale = 0
                print(f"round {rnd}: KEPT worst traffic {ev['worst_waste']:.2f}x the floor, "
                      f"{ev['total_bytes']:,} bytes; {ev['total_ns'] / 1000:.1f} us "
                      f"({start_total / ev['total_ns']:.2f}x faster than the start, gmean "
                      f"{ev['gmean_ratio']:.2f}x the floor). {n_ok}/{len(cands)} correct. "
                      f"({time.time() - t0:.0f}s)")
                continue
        failed = next((c[1]["why"] for c in cands if not c[1]["ok"]), "") if not good else \
            "it was correct but not faster."
        stale += 1
        print(f"round {rnd}: no improvement, best {best['total_ns'] / 1000:.1f} us. "
              f"{n_ok}/{len(cands)} correct. ({time.time() - t0:.0f}s)")
        if not good:
            print(f"  {failed[:200]}")
    with open(f"/tmp/opt_best_{a.condition}_{rep}.py", "w") as f:
        f.write(best_src)
    return start_total, best["total_ns"], best["gmean_ratio"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=("specific", "roofline", "control"), default="specific")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--patience", type=int, default=3, help="end a run after this many rounds without gain")
    ap.add_argument("--log", default="/tmp/opt.jsonl")
    ap.add_argument("--max-tokens", type=int, default=agent.MIN_ANSWER_TOKENS)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--objective", choices=("ns", "bytes"), default="ns",
                    help="bytes: minimise HBM traffic (level 5's bar) instead of predicted latency")
    ap.add_argument("--start", help="start from this correct kernel instead of the TILE_N=128 one")
    a = ap.parse_args()
    perf_feedback.objective = a.objective
    a.model = agent.MODEL
    a.base = os.environ.get("KERNEL_AGENT_BASE_URL", "http://localhost:8000/v1")
    a.think = False
    if a.offline:
        a.rounds, a.repeat = 1, 1
    ref_total = sum(evaluate(open(f"{BENCH}/reference_level4.py").read())["rows"][i]["pred_ns"]
                    for i in range(len(SHAPES)))
    print(f"condition {a.condition}; reference_level4 (the tutorial's kernel) is {ref_total / 1000:.1f} us")
    results = []
    with open(a.log, "a") as log:
        for rep in range(a.repeat):
            results.append(run(a, rep, log))
    print(f"\n======== {a.condition}: {a.repeat} run(s) ========")
    for i, (s, b, g) in enumerate(results):
        print(f"  run {i + 1}: {s / 1000:.1f} -> {b / 1000:.1f} us  ({s / b:.2f}x faster, gmean "
              f"{g:.2f}x the model floor; reference is {ref_total / 1000:.1f} us)")
    print(f"  speedups: {[round(s / b, 2) for s, b, _ in results]}")


if __name__ == "__main__":
    main()
