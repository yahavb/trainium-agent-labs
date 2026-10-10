#!/usr/bin/env python3
"""
nearmiss_repair.py — targeted repair of a model-authored near-miss kernel.

The queue leaves near-misses behind: kernels that already measure far fewer HBM bytes but
break one specific thing. This driver starts from such a kernel, diagnoses the numerics
failure with hypothesis tests against the reference (behavioral diagnosis, not a static hint),
and asks for ONE named change per round until the kernel is valid and below the seed's 2.00x.

    python nearmiss_repair.py --kernel start_candidate.py --rounds 6 \
        --output runs/traffic/nearmiss_repair

Provenance: the starting kernel is a candidate produced by the agent loop in an earlier run;
every change after it is authored by the model in this loop. Logs every prompt/reply/evaluation.
"""
import argparse
import json
import os
import time

import numpy as np

import nkibench
import traffic_eval
from agent import extract_code
from traffic_agent import (MODEL, TokenCounter, ask, candidate_valid, demo_block,
                           enrich_feedback, kernel_block, progress)


def _close(got, pred, atol=2e-2):
    got = np.asarray(got, np.float64)
    pred = np.asarray(pred, np.float64)
    if got.shape != pred.shape:
        return False
    scale = float(np.sqrt((pred ** 2).mean())) or 1.0
    return bool(np.abs(got - pred).max() / scale <= atol)


def diagnose(got, want, args):
    """Name the arithmetic failure from the candidate's own output. '' when no hypothesis fits."""
    lhsT, rhs = args
    K, M = lhsT.shape
    _, N = rhs.shape
    g = np.asarray(got, np.float64)
    if not np.all(np.isfinite(g)):
        return ("your output contains non-finite values: a tile (almost certainly the PSUM "
                "accumulator) is read before it is written. Allocate the PSUM tile ONCE before "
                "the k loop and let nc_matmul accumulate into that same tile for every k chunk.")

    if K > 128 and K % 128 == 0:
        # H1: the stationary operand never advanced -- chunk 0 against every rhs chunk.
        # Sum of all rhs chunks has shape [128, N]; lhsT[:128, :].T @ that sum equals the
        # candidate's accumulation when the lhs tile is stuck on chunk 0.
        rhs_sum = rhs.reshape(K // 128, 128, N).sum(axis=0).astype(np.float32)
        pred = (lhsT[0:128, :].astype(np.float32).T @ rhs_sum).astype(lhsT.dtype)
        if _close(got, pred):
            return ("the numbers equal a computation in which the STATIONARY operand never "
                    "advanced past its first k-chunk: lhsT chunk 0 was multiplied against every "
                    "rhs chunk (your load index is stuck at 0*K, or the slice uses the wrong "
                    "axis). Give the stationary operand one tile per k chunk, and to keep the "
                    "bytes you already save, keep it reused across n: hold a stacked cache per m "
                    "block, shape [128, k_tiles * TILE_M], load chunk kk with "
                    "dma_copy(dst=cache[:, kk*TILE_M:(kk+1)*TILE_M], "
                    "src=lhsT[kk*128:(kk+1)*128, m*128:(m+1)*128]) OUTSIDE the n loop, and use "
                    "cache[:, kk*TILE_M:(kk+1)*TILE_M] as the stationary operand for chunk kk.")
        # H2: the moving operand is fine but only the LAST chunk contributed.
        last = rhs[(K // 128 - 1) * 128:, :]
        pred = (lhsT[0:128, :].astype(np.float32).T @ last.astype(np.float32)).astype(lhsT.dtype)
        if _close(got, pred):
            return ("the numbers equal only the LAST k-chunk contributing; every earlier chunk "
                    "was dropped. Accumulate every chunk into one PSUM tile.")
    return ""


def evaluate_with_diagnosis(path, level, seed):
    ev = traffic_eval.evaluate_file(path, level, 2e-2, seed)
    diag = ""
    if ev["rules_ok"]:
        for case in nkibench.LEVELS[level]["shapes"]:
            args, _ = nkibench.make_inputs(case, level, seed)
            want = nkibench.LEVELS[level]["ref"](*args)
            try:
                kernel = nkibench.load_kernel(path, nkibench.LEVELS[level]["entry"])
                got, _ = nkibench.simulate_and_count(kernel, args)
            except Exception:
                continue
            if nkibench.describe_mismatch(got, want) is None:
                continue
            d = diagnose(got, want, args)
            if d:
                diag = f"On {nkibench.label(case, level)}: {d}"
                break
    return ev, diag


def case_table(ev):
    lines = []
    for c in ev["per_case"]:
        w = f"{c['waste']:.2f}x" if c.get("waste") is not None else "?"
        line = f"- {c['case']}: {c.get('bytes')} bytes vs floor {c.get('floor')} = {w}"
        if c.get("failure"):
            line += f"  [not accepted: {' '.join(str(c['failure']).split())[:120]}]"
        lines.append(line)
    return "\n".join(lines)


def prompt_for(level, src, ev, extra, demo_path):
    parts = [demo_block(demo_path),
             kernel_block(level, src),
             "The current kernel measured:\n" + case_table(ev)]
    if extra:
        parts.append("The checker's diagnosis:\n" + extra)
    parts.append("Change exactly what the diagnosis names and keep everything else identical. "
                 "The bytes you already save must not regress. Reply with ONE complete python "
                 "code block (imports + the full kernel), then STRATEGY and CONFIDENCE lines.")
    return "\n\n".join(p for p in parts if p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="runs/traffic/nearmiss_repair")
    ap.add_argument("--demo", default="demo.json")
    ap.add_argument("--instruction", default="",
                    help="path to a text file appended to every prompt as an extra instruction")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL")
                    or os.environ.get("GPTOSS_BASE_URL"))
    a = ap.parse_args()
    if not a.base:
        raise SystemExit("set KERNEL_AGENT_BASE_URL to the model endpoint")

    os.makedirs(a.output, exist_ok=True)
    os.makedirs(os.path.join(a.output, "replies"), exist_ok=True)
    os.makedirs(os.path.join(a.output, "candidates"), exist_ok=True)
    counter = TokenCounter(a.base, a.model)
    logf = open(os.path.join(a.output, "rounds.jsonl"), "w")

    src = open(a.kernel).read()
    extra_instruction = open(a.instruction).read() if a.instruction else ""
    with open(os.path.join(a.output, "start_kernel.py"), "w") as f:
        f.write(src)
    print(f"start kernel: {a.kernel} (model-authored in an earlier run; copied to "
          f"{a.output}/start_kernel.py)")

    best = None
    for rnd in range(a.rounds):
        path = os.path.join(a.output, "candidates", f"r{rnd}.py")
        with open(path, "w") as f:
            f.write(src)
        t0 = time.perf_counter()
        ev, diag = evaluate_with_diagnosis(path, a.level, a.seed)
        prg = progress(ev)
        print(f"\nround {rnd}: worst={ev['worst_waste']} progress={prg} "
              f"valid={candidate_valid(ev)}")
        print(case_table(ev))
        if diag:
            print("diagnosis:", diag[:300])
        record = dict(round=rnd, worst=ev["worst_waste"], progress=prg, diagnosis=diag,
                      per_case=ev["per_case"], accepted=ev["accepted"],
                      elapsed=round(time.perf_counter() - t0, 1))
        if ev["accepted"] and ev["worst_waste"] is not None and ev["worst_waste"] < 2.0:
            print(f"\nSOLVED: verified improvement at {ev['worst_waste']:.2f}x (all shapes "
                  f"accepted). Winner written to {a.output}/winner.py")
            with open(os.path.join(a.output, "winner.py"), "w") as f:
                f.write(src)
            record["outcome"] = "solved"
            logf.write(json.dumps(record) + "\n")
            logf.close()
            return
        if ev["accepted"] and (ev["worst_waste"] or 9) >= 2.0:
            print("  (valid but not an improvement -- keep the byte-saving structure)")
        if best is None or prg > best[0]:
            best = (prg, src, ev)

        extra = diag or enrich_feedback(ev.get("feedback") or "")
        if extra_instruction:
            extra = (extra + "\n\n" if extra else "") + extra_instruction
        prompt = prompt_for(a.level, src, ev, extra, a.demo)
        ptok = counter.count(prompt)
        budget = min(a.max_tokens, max(256, a.context - ptok - 64))
        if ptok + budget + 64 > a.context:
            raise SystemExit("prompt does not fit the context")
        reply, usage = ask(a, prompt, budget)
        raw = os.path.join(a.output, "replies", f"r{rnd}.txt")
        with open(raw, "w") as f:
            f.write(reply)
        record["prompt_tokens"] = ptok
        record["completion_tokens"] = int((usage or {}).get("completion_tokens") or 0)
        record["reply"] = os.path.relpath(raw, a.output)
        new_src = extract_code(reply)
        if not new_src.strip():
            print("  (empty reply; keeping the current kernel)")
            record["outcome"] = "empty"
            logf.write(json.dumps(record) + "\n")
            logf.flush()
            continue
        # Adopt the new candidate; if it regresses, the next round's prompt shows the failure.
        src = new_src
        record["outcome"] = "candidate"
        logf.write(json.dumps(record) + "\n")
        logf.flush()

    # Rounds exhausted: report the best attempt honestly.
    with open(os.path.join(a.output, "best.py"), "w") as f:
        f.write(best[1])
    print(f"\nno verified improvement in {a.rounds} rounds; best progress {best[0]:.2f}, "
          f"best worst-case waste {best[2]['worst_waste']}. Best kernel written to "
          f"{a.output}/best.py -- reported as an honest failure.")
    logf.write(json.dumps(dict(outcome="unsolved", best_progress=best[0],
                               best_worst=best[2]["worst_waste"])) + "\n")
    logf.close()


if __name__ == "__main__":
    main()
