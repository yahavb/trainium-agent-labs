#!/usr/bin/env python3
"""guidance_repair.py — menu-guided repair after the Jev/System-One split.

The repair knowledge lives in a fixed menu of guidances (guidance_menu.py). Each round:

  1. CLASSIFIER: the model reads the state (per-shape results + the checker's behavioral
     diagnosis + what was already tried) and returns exactly one Choice -- the guidance id
     that should be applied next. It writes no code.
  2. APPLIER: the model receives only the chosen guidance's `exact` change text and the
     kernel, and returns the full updated kernel.

Adjudication is automatic: a deterministic rule maps the same state to the guidance we
would pick by hand, so every classifier decision is scored against that expectation. The
loop stops when the kernel is accepted at the level's traffic gate.

    python3 guidance_repair.py --kernel start.py --rounds 6 --output runs/traffic/guidance

Everything is logged: state, expected vs chosen guidance, confidence, both prompts' replies,
and the evaluation of every produced kernel.
"""
import argparse
import json
import os
import re
import time

import nkibench
import traffic_eval
from agent import extract_code
from traffic_agent import (MODEL, TokenCounter, ask, candidate_valid, demo_block,
                           kernel_block, progress)
from nearmiss_repair import case_table, evaluate_with_diagnosis


def load_menu(path):
    ns = {}
    with open(path) as f:
        exec(f.read(), ns)  # the menu file is trusted, local data
    return ns["MENU"]


def expected_guidance(src, ev, diag):
    """The guidance a human would pick from the same state -- the classifier's ground truth."""
    if diag:
        if "STATIONARY operand never advanced" in diag:
            return "stack_stationary_operand"
        if "non-finite" in diag or "LAST k-chunk" in diag:
            return "single_psum_accumulation"
    fb = " ".join(str(c.get("failure") or "") for c in ev.get("per_case") or [])
    low = fb.lower()
    if "same number of elements" in low:
        return "fix_cache_stride_and_width"
    if "contraction dimension" in low:
        return "split_k_contraction"
    if "must be in" in low:
        return "fix_tile_memory_region"
    if "could not be broadcast" in low:
        return "match_assignment_shapes"
    if "no attribute" in low:
        return "use_real_nki_names"
    if "multiple values for argument" in low:
        return "keyword_args"
    if "at least 2 dimensions" in low:
        return "two_dimensional_tiles"
    if "free dimension" in low and "exceeds" in low:
        return "keep_k_on_partition_axis"
    if "out of bounds" in low or "oob" in low:
        return "fix_slice_bounds"
    if "partition" in low:
        return "keep_k_on_partition_axis"
    if candidate_valid(ev) and ev.get("worst_waste") is not None:
        has_rhs = "rhs_cache" in src
        has_lhs = re.search(r"(?<!rhs_)\bcache\b", src) is not None
        if has_lhs and not has_rhs:
            return "stack_moving_operand_symmetrically"
        return "stack_stationary_operand"
    return "continue_from_best"


def parse_choice(reply, ids):
    m = re.search(r"CHOICE:\s*\**\s*`?([A-Za-z0-9_\-]+)", reply or "")
    cand = m.group(1).strip("`* ") if m else None
    if cand in ids:
        return cand
    if cand and cand.isdigit() and 1 <= int(cand) <= len(ids):
        return ids[int(cand) - 1]
    for i in ids:
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(i) + r"(?![A-Za-z0-9_])", reply or ""):
            return i
    return None


def parse_confidence(reply):
    m = re.search(r"CONFIDENCE:\s*\**\s*([1-5])", reply or "")
    return int(m.group(1)) if m else None


def parse_reason(reply):
    m = re.search(r"REASON:\s*(.+)", reply or "")
    return m.group(1).strip()[:200] if m else ""


def classifier_prompt(menu, ev, diag, tried):
    parts = [
        "You are the guidance classifier for a kernel-repair loop. Read the state of the "
        "current kernel and choose EXACTLY ONE guidance id from the menu that should be "
        "applied next. Decide from the evidence; do not write code.",
        "STATE -- the current kernel measured:\n" + case_table(ev),
    ]
    if diag:
        parts.append("The checker's behavioral diagnosis:\n" + diag)
    if tried:
        parts.append("Already applied this run (avoid repeating unless nothing else fits): "
                     + ", ".join(tried))
    lines = ["MENU -- choose one id:"]
    for m in menu:
        lines.append(f"- {m['id']}: {m['name']} | use when: {m['when']}")
    parts.append("\n".join(lines))
    parts.append("Reply with exactly these three lines and nothing else:\n"
                 "CHOICE: <one id from the menu>\n"
                 "REASON: <one short sentence>\n"
                 "CONFIDENCE: <a number 1-5>")
    return "\n\n".join(parts)


def applier_prompt(entry, src, demo_path, level):
    parts = [
        demo_block(demo_path),
        (f"The classifier chose guidance '{entry['id']}' ({entry['name']}).\n"
         f"Apply exactly this change and nothing else:\n\n{entry['exact']}"),
        kernel_block(level, src),
        "Reply with ONE complete python code block (imports + the full kernel), then "
        "STRATEGY and CONFIDENCE lines.",
    ]
    return "\n\n".join(p for p in parts if p)


def ask_budgeted(counter, a, prompt):
    ptok = counter.count(prompt)
    budget = min(a.max_tokens, max(256, a.context - ptok - 64))
    if ptok + budget + 64 > a.context:
        raise SystemExit("prompt does not fit the context")
    reply, usage = ask(a, prompt, budget)
    return reply, usage, ptok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--menu", default="guidance_menu.py")
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="runs/traffic/guidance")
    ap.add_argument("--demo", default="demo.json")
    ap.add_argument("--no-demo", action="store_true",
                    help="drop the worked-example block from the applier prompt "
                         "(simplification check)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL")
                    or os.environ.get("GPTOSS_BASE_URL"))
    a = ap.parse_args()
    if not a.base:
        raise SystemExit("set KERNEL_AGENT_BASE_URL to the model endpoint")

    menu = load_menu(a.menu)
    ids = [m["id"] for m in menu]
    by_id = {m["id"]: m for m in menu}

    os.makedirs(a.output, exist_ok=True)
    os.makedirs(os.path.join(a.output, "replies"), exist_ok=True)
    os.makedirs(os.path.join(a.output, "candidates"), exist_ok=True)
    counter = TokenCounter(a.base, a.model)
    logf = open(os.path.join(a.output, "rounds.jsonl"), "w")

    src = open(a.kernel).read()
    with open(os.path.join(a.output, "start_kernel.py"), "w") as f:
        f.write(src)
    print(f"start kernel: {a.kernel} -> {a.output}/start_kernel.py")
    print(f"menu: {a.menu} ({len(menu)} options)")

    best = None          # (progress, src)
    asked = matches = fallbacks = 0
    choices = []
    tried = []

    for rnd in range(a.rounds):
        path = os.path.join(a.output, "candidates", f"r{rnd}.py")
        with open(path, "w") as f:
            f.write(src)
        t0 = time.perf_counter()
        ev, diag = evaluate_with_diagnosis(path, a.level, a.seed)
        prg = progress(ev)
        valid = candidate_valid(ev)
        print(f"\nround {rnd}: worst={ev['worst_waste']} progress={prg} valid={valid} "
              f"accepted={ev['accepted']}")
        print(case_table(ev))
        if diag:
            print("diagnosis:", diag[:300])
        record = dict(round=rnd, worst=ev["worst_waste"], progress=prg, valid=valid,
                      accepted=ev["accepted"], diagnosis=diag, per_case=ev["per_case"],
                      elapsed=round(time.perf_counter() - t0, 1))

        if valid and ev.get("accepted"):
            with open(os.path.join(a.output, "winner.py"), "w") as f:
                f.write(src)
            record["outcome"] = "solved"
            logf.write(json.dumps(record) + "\n")
            acc = f"{matches}/{asked}" if asked else "n/a"
            print(f"\nSOLVED: accepted at worst {ev['worst_waste']:.3f}x. "
                  f"Classifier: {acc} matches, {fallbacks} parse fallbacks, "
                  f"choices={choices}")
            summary = dict(solved=True, rounds=rnd + 1, asked=asked, matches=matches,
                           fallbacks=fallbacks, choices=choices,
                           final_worst=ev["worst_waste"])
            json.dump(summary, open(os.path.join(a.output, "summary.json"), "w"), indent=2)
            logf.close()
            return

        if best is None or prg > best[0]:
            best = (prg, src, ev["worst_waste"])

        expected = expected_guidance(src, ev, diag)
        cprompt = classifier_prompt(menu, ev, diag, tried)
        reply, cusage, ctok = ask_budgeted(counter, a, cprompt)
        cpath = os.path.join(a.output, "replies", f"c{rnd}.txt")
        with open(cpath, "w") as f:
            f.write(reply)
        chosen = parse_choice(reply, ids)
        parse_failed = chosen is None
        if chosen is None:
            retry = (cprompt + "\n\nReply with exactly:\nCHOICE: <one id from the menu>\n"
                     "REASON: <one short sentence>\nCONFIDENCE: <1-5>")
            reply2, cusage2, ctok2 = ask_budgeted(counter, a, retry)
            with open(os.path.join(a.output, "replies", f"c{rnd}_retry.txt"), "w") as f:
                f.write(reply2)
            chosen = parse_choice(reply2, ids)
            if chosen is None:
                fallbacks += 1
                chosen = expected or "continue_from_best"
            reply = reply2
            ctok = ctok2
        asked += 1
        match = (chosen == expected) if (expected and not parse_failed) else None
        if match:
            matches += 1
        choices.append(chosen)
        tried.append(chosen)
        record.update(chosen=chosen, expected=expected, match=match,
                      parse_failed=parse_failed, fallback=parse_failed,
                      confidence=parse_confidence(reply), reason=parse_reason(reply),
                      classifier_reply=os.path.relpath(cpath, a.output),
                      classifier_tokens=ctok,
                      classifier_completion_tokens=int((cusage or {}).get("completion_tokens") or 0))
        print(f"classifier: chose {chosen} (expected {expected}, match={match}, "
              f"confidence={record['confidence']}) -- {record['reason']}")

        if chosen == "continue_from_best":
            if best is not None:
                src = best[1]
            record["outcome"] = "continue_from_best"
            logf.write(json.dumps(record) + "\n")
            logf.flush()
            continue

        entry = by_id.get(chosen) or by_id["continue_from_best"]
        # The worked-example block stays in the applier prompt: the --no-demo check
        # (evidence/guidance223_nodemo_*, GUIDANCE-CLASSIFIER.md) failed to replicate the
        # solve (botched geometry, six repeated choices), so its removal was rejected.
        aprompt = applier_prompt(entry, src, "" if a.no_demo else a.demo, a.level)
        areply, ausage, atok = ask_budgeted(counter, a, aprompt)
        apath = os.path.join(a.output, "replies", f"a{rnd}.txt")
        with open(apath, "w") as f:
            f.write(areply)
        record.update(applier_reply=os.path.relpath(apath, a.output),
                      applier_tokens=atok,
                      applier_completion_tokens=int((ausage or {}).get("completion_tokens") or 0))
        new_src = extract_code(areply)
        if not new_src.strip():
            record["outcome"] = "empty"
            logf.write(json.dumps(record) + "\n")
            logf.flush()
            print("  (empty applier reply; keeping the current kernel)")
            continue
        src = new_src
        record["outcome"] = "candidate"
        logf.write(json.dumps(record) + "\n")
        logf.flush()

    with open(os.path.join(a.output, "best.py"), "w") as f:
        f.write(best[1])
    acc = f"{matches}/{asked}" if asked else "n/a"
    print(f"\nno solve in {a.rounds} rounds; best progress {best[0]:.2f}, best worst-case "
          f"waste {best[2]}. Best kernel written to {a.output}/best.py -- honest failure.")
    print(f"classifier: {acc} matches, {fallbacks} parse fallbacks, choices={choices}")
    summary = dict(solved=False, rounds=a.rounds, asked=asked, matches=matches,
                   fallbacks=fallbacks, choices=choices, best_progress=best[0])
    json.dump(summary, open(os.path.join(a.output, "summary.json"), "w"), indent=2)
    logf.close()


if __name__ == "__main__":
    main()
