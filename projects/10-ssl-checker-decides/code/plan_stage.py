#!/usr/bin/env python3
"""
plan_stage.py -- H1: gate 1 (the level-4 tiling plan) asked ONE LOOP AT A TIME.

Asked for the whole plan at once, Qwen3-8B fails gate 1 every run (diagnostic feedback: 0/5,
results/2026-10-10-rishabh-plan-diagnostic.md). For the kernel body, the loop tool's staged recipe
got 5/5: a task line that spells out the steps, one piece per prompt, each piece frozen once it
passes. This file applies that recipe to the plan:

    stage m   the m loop: its count, out axis 0, lhsT axis 1          (the M axis)
    stage n   the n loop: its count, out axis 1, rhs axis 1           (the N axis)
    stage k   the k loop: its count, lhsT axis 0, rhs axis 0, accumulate_over   (the K axis)

Each stage is checked on its own, on Hypothesis-generated contract shapes: the axis is covered
exactly once, the tiles respect the hardware limits, the count is an integer, and the two slices of
the same axis agree. Once it passes it is frozen and shown to the next stage. The frozen pieces are
assembled into one plan and graded by the full plan_check.search, the same check gate 1 uses.

Honest test: the model never sees plan_check.REFERENCE or a formula. Feedback names what is wrong,
never the fix (the --selftest enforces both).

    python plan_stage.py --selftest
    python plan_stage.py --offline
    python plan_stage.py --rounds 8 --samples 1 --context 8192 --repeat 5 --log attempts-plan-stage.jsonl
"""

import argparse
import ast
import copy
import json
import os
import sys
import time

import numpy as np

try:
    from hypothesis import HealthCheck, Phase, given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover
    sys.exit("plan_stage.py needs Hypothesis:  pip install hypothesis")

import agent
import nkibench
import plan_check
from plan_check import PlanError

PMAX = nkibench.PMAX                       # 128, partition axis of every tile
STAT_FMAX = nkibench.GEMM_STATIONARY_FMAX  # 128, lhsT free axis
MOV_FMAX = nkibench.GEMM_MOVING_FMAX       # 512, rhs free axis and the out (PSUM) tile
# plan_check caps the whole plan at 100,000 iterations; 46**3 < 100,000, so a per-loop cap of 46
# means three passing loops can never trip the whole-plan cap.
LOOP_CAP = 46
ACC_ALL = False   # --acc-feedback all

# One entry per stage. Each slice is (tensor, axis, cap, what the axis is called).
STAGES = {
    "m": dict(dim="M", slices=[("out", 0, PMAX, "partition"),
                               ("lhsT", 1, STAT_FMAX, "stationary free")],
              about="output rows"),
    "n": dict(dim="N", slices=[("out", 1, MOV_FMAX, "free (PSUM)"),
                               ("rhs", 1, MOV_FMAX, "moving free")],
              about="output columns"),
    "k": dict(dim="K", slices=[("lhsT", 0, PMAX, "partition"),
                               ("rhs", 0, PMAX, "partition")],
              about="the shared K axis, summed into the same out tile"),
}
ORDER = ("m", "n", "k")


# ---------------------------------------------------------------- wording
#
# "steps" describes the method, in words, the way loop_tool.WORDING["steps"] did for the body. It
# never writes a formula. "loose" says only what to give (the ablation arm).

HEADER = (
    "We are planning the tiling for a matmul, out = lhsT.T @ rhs, before any code is written. lhsT is "
    "[K, M], rhs is [K, N] and out is [M, N]. K and M are multiples of 128, N is a multiple of 512.\n\n"
    "The kernel has three nested loops: m steps over tiles of out's rows (the M axis), n steps over "
    "tiles of out's columns (the N axis), and k steps over tiles of the K axis; the k steps' products "
    "are summed into the same out tile. We plan one loop at a time.\n\n"
    f"Hardware limits: every tile's first (partition) axis is at most {PMAX}; the lhsT tile's second "
    f"axis is at most {STAT_FMAX}; the rhs and out tiles' second axis is at most {MOV_FMAX}.")

FORMAT = {
    "m": '{"count": "<formula>", "out": ["<start>", "<stop>"], "lhsT": ["<start>", "<stop>"]}',
    "n": '{"count": "<formula>", "out": ["<start>", "<stop>"], "rhs": ["<start>", "<stop>"]}',
    "k": ('{"count": "<formula>", "lhsT": ["<start>", "<stop>"], "rhs": ["<start>", "<stop>"], '
          '"accumulate_over": ["<the loops whose steps sum into the same out tile>"]}'),
}

TASK = {
    "steps": {
        "m": ("Now plan the m loop. It tiles the M axis, which is out's axis 0 and lhsT's axis 1. Steps:\n"
              "1. The tile size along M is the largest size both limits allow: out's axis 0 is a "
              "partition axis, and lhsT's axis 1 is lhsT's second axis.\n"
              "2. The count is how many tiles of that size fit in M.\n"
              "3. m is the tile NUMBER: it runs 0, 1, ..., count-1. It is not a size. The slice for "
              "tile m starts at m times the tile size and stops (exclusive) one tile size later.\n"
              "4. out's axis 0 and lhsT's axis 1 are the same M rows of this tile, so give both the "
              "same slice."),
        "n": ("Now plan the n loop. It tiles the N axis, which is out's axis 1 and rhs's axis 1. Steps:\n"
              "1. The tile size along N is the largest size both limits allow: out's axis 1 and "
              "rhs's axis 1 are both second axes.\n"
              "2. The count is how many tiles of that size fit in N.\n"
              "3. n is the tile NUMBER: it runs 0, 1, ..., count-1. It is not a size. The slice for "
              "tile n starts at n times the tile size and stops (exclusive) one tile size later.\n"
              "4. out's axis 1 and rhs's axis 1 are the same N columns of this tile, so give both "
              "the same slice."),
        "k": ("Now plan the k loop. It tiles the K axis, which is lhsT's axis 0 and rhs's axis 0. Steps:\n"
              "1. The tile size along K is the largest size both limits allow: lhsT's axis 0 and "
              "rhs's axis 0 are both partition axes.\n"
              "2. The count is how many tiles of that size fit in K.\n"
              "3. k is the tile NUMBER: it runs 0, 1, ..., count-1. It is not a size. The slice for "
              "tile k starts at k times the tile size and stops (exclusive) one tile size later.\n"
              "4. lhsT's axis 0 and rhs's axis 0 are the same K rows of this step, so give both the "
              "same slice.\n"
              "5. accumulate_over lists the loops whose steps are summed into the same out tile."),
    },
    "loose": {
        "m": "Now plan the m loop: how many times it runs, and the slice of out's axis 0 and of lhsT's "
             "axis 1 that tile m covers.",
        "n": "Now plan the n loop: how many times it runs, and the slice of out's axis 1 and of rhs's "
             "axis 1 that tile n covers.",
        "k": "Now plan the k loop: how many times it runs, the slice of lhsT's axis 0 and of rhs's "
             "axis 0 that step k covers, and which loops accumulate into the same out tile.",
    },
}


def _frozen_text(frozen):
    if not frozen:
        return ""
    lines = "\n".join(f"  {s} loop: {json.dumps(frozen[s])}" for s in ORDER if s in frozen)
    return f"\n\nAlready planned and checked (frozen, do not change or repeat them):\n{lines}"


def stage_prompt(stage, frozen, wording="steps"):
    return (f"{HEADER}{_frozen_text(frozen)}\n\n{TASK[wording][stage]}\n\n"
            f"Formulas are strings and may use + - * / // %, ceil, min, max, numbers, M, K, N and "
            f"{stage}. Each slice is a JSON list of two strings, the start and the stop (exclusive).\n\n"
            f"Reply with ONE JSON object in this format and nothing else:\n{FORMAT[stage]}")


def stage_repair_prompt(stage, frozen, wording, last_text, feedback):
    """The stage's question again (so the step list is not lost), the last answer, and what is wrong."""
    return (f"{stage_prompt(stage, frozen, wording)}\n\nYour last answer was:\n```json\n{last_text}\n```\n"
            f"A checker that tries it on many shapes reports:\n{feedback}\n\n"
            f"Change what the checker names. Reply with ONE JSON object and nothing else.")


# ---------------------------------------------------------------- the per-stage check

def parse_piece(stage, obj):
    """Static checks on one stage's answer. Returns the compiled piece or raises PlanError."""
    if not isinstance(obj, dict):
        raise PlanError(f"the answer must be ONE JSON object: {FORMAT[stage]}")
    spec = STAGES[stage]
    need = ["count"] + [t for t, *_ in spec["slices"]] + (["accumulate_over"] if stage == "k" else [])
    missing = [k for k in need if k not in obj]
    if missing:
        raise PlanError(f"the answer has no {', '.join(repr(k) for k in missing)}. It needs "
                        f"{', '.join(need)}.")
    piece = dict(count=plan_check._compile(obj["count"], f"{stage}.count"), slices={})
    for t, axis, _, _ in spec["slices"]:
        s = obj[t]
        if (isinstance(s, list) and len(s) == 1 and isinstance(s[0], list)):
            s = s[0]
        if not (isinstance(s, list) and len(s) == 2 and all(isinstance(x, (str, int)) for x in s)):
            raise PlanError(f"{t}: expected ONE pair [\"<start>\", \"<stop>\"] for {t}'s axis {axis} "
                            f"only, got {json.dumps(s)}")
        piece["slices"][t] = (plan_check._compile(s[0], f"{t}.start"),
                              plan_check._compile(s[1], f"{t}.stop"))
    allowed = {"M", "K", "N", stage} | set(plan_check._FUNCS)
    for where, expr in [("count", piece["count"])] + [
            (f"{t} {'start' if j == 0 else 'stop'}", e)
            for t, pair in piece["slices"].items() for j, e in enumerate(pair)]:
        for node in ast.walk(ast.parse(expr[0], mode="eval")):
            if isinstance(node, ast.Name) and node.id not in allowed:
                raise PlanError(f"{where}: '{expr[0]}' uses '{node.id}', which is not defined here. "
                                f"Allowed names: M, K, N, {stage}.")
    if stage == "k":
        acc = obj["accumulate_over"]
        if isinstance(acc, str):
            acc = [acc]
        if not isinstance(acc, list):
            raise PlanError("accumulate_over must be a JSON list of loop names")
        piece["acc"] = [str(x) for x in acc]
    return piece


def _ev(expr, env, where):
    """plan_check._eval without its hint ("Use // or ceil()"): the message says what, not how."""
    try:
        return plan_check._eval(expr, env, where)
    except PlanError as e:
        raise PlanError(str(e).split(" Use // or ceil()")[0])


def check_piece(stage, piece, M, K, N):
    """One stage's piece on one shape. None if right, else a message that names what is wrong."""
    spec = STAGES[stage]
    shape = dict(M=M, K=K, N=N)
    D = shape[spec["dim"]]
    v = stage
    try:
        if stage == "k":
            acc = piece["acc"]
            other = [x for x in acc if x in ("m", "n")]
            probs = []
            if other:
                probs.append(f"accumulate_over lists {other}. Each step of the {other[0]} loop is a "
                             f"different out tile, so its steps do not sum into the same tile.")
            if "k" not in acc:
                probs.append("accumulate_over does not list k. Every k step reads a different K "
                             "slice for the same out tile; if they are not summed, the out tile "
                             "holds only one step's product.")
            if probs:
                # v1 (default) reports the first problem only; v2 (--acc-feedback all) reports all.
                raise PlanError(" ".join(probs) if ACC_ALL else probs[0])
            unknown = [x for x in acc if x not in ("m", "n", "k")]
            if unknown:
                raise PlanError(f"accumulate_over lists {unknown}, which are not loops (m, n, k)")
        c = _ev(piece["count"], shape, f"the {v} loop's count")
        if c < 1:
            raise PlanError(f"the {v} loop's count '{piece['count'][0]}' is {c} at "
                            f"{spec['dim']}={D}, so the loop never runs and nothing is covered.")
        if c > LOOP_CAP:
            per = (f"once per element of {spec['dim']}, not once per tile" if c == D else
                   f"more than the {LOOP_CAP} a loop may run (100,000 iterations over the three loops)")
            raise PlanError(f"the {v} loop runs {c} times at {spec['dim']}={D} "
                            f"('{piece['count'][0]}'): {per}.")
        cover = {t: np.zeros(D, np.int32) for t, *_ in spec["slices"]}
        for i in range(c):
            env = {**shape, v: i}
            got = {}
            for t, axis, cap, what in spec["slices"]:
                start, stop = piece["slices"][t]
                a = _ev(start, env, f"{t} axis {axis} start")
                b = _ev(stop, env, f"{t} axis {axis} stop")
                where = (f"{t} axis {axis} ({spec['dim']}): for tile {v}={i} at {spec['dim']}={D}, "
                         f"the slice from {a} to {b} (stop exclusive), from '{start[0]}' and "
                         f"'{stop[0]}',")
                if b <= a:
                    raise PlanError(f"{where} is empty.")
                if a < 0:
                    raise PlanError(f"{where} starts before 0.")
                if b > D:
                    raise PlanError(f"{where} runs past the end of {spec['dim']}={D}.")
                if b - a > cap:
                    raise PlanError(f"{where} is {b - a} wide; {t}'s axis {axis} is its {what} axis, "
                                    f"and the limit there is {cap}.")
                got[t] = (a, b)
                cover[t][a:b] += 1
            (t1, s1), (t2, s2) = list(got.items())
            if s1 != s2:
                raise PlanError(f"for tile {v}={i} at {spec['dim']}={D}, {t1} covers {spec['dim']} "
                                f"{s1[0]}..{s1[1] - 1} but {t2} covers {s2[0]}..{s2[1] - 1}. They are "
                                f"the same {spec['dim']} positions of the same tile, so they must agree.")
        for t, cv in cover.items():
            if np.all(cv == 1):
                continue
            bad = np.flatnonzero(cv != 1)
            lo = bad[0]
            hi = lo
            while hi + 1 < D and cv[hi + 1] == cv[lo]:
                hi += 1
            kind = "never covered" if cv[lo] == 0 else f"covered {cv[lo]} times"
            raise PlanError(f"at {spec['dim']}={D}, {spec['dim']} positions {lo}..{hi} of {t} are "
                            f"{kind}: the {v} loop runs {c} time(s) ('{piece['count'][0]}'). Its "
                            f"tiles must cover 0..{D - 1} exactly once.")
    except PlanError as e:
        return str(e)
    return None


def search_piece(stage, piece, examples=300):
    """Hypothesis over the contract shapes (as plan_check.search), shrinking to the smallest failure."""
    stats = dict(tried=0, fail=None)

    @settings(max_examples=examples, derandomize=True, database=None, deadline=None,
              phases=(Phase.generate, Phase.shrink), report_multiple_bugs=False,
              print_blob=False, suppress_health_check=list(HealthCheck))
    @given(**plan_check._contract())
    def prop(M, K, N):
        stats["tried"] += 1
        m = check_piece(stage, piece, M, K, N)
        if m is not None:
            stats["fail"] = ((M, K, N), m)
            raise AssertionError(m)

    try:
        prop()
        return True, f"the {stage} loop is right on {stats['tried']} generated shapes.", stats["tried"]
    except AssertionError:
        (M, K, N), m = stats["fail"]
        return False, f"Smallest failing shape: M={M} K={K} N={N}. {m}", stats["tried"]


def grade_piece(stage, reply):
    """(ok, feedback, piece_obj or None, json_text)."""
    text = agent.extract_json(reply)
    if not text:
        return False, f"No JSON object came back. Reply with ONE JSON object: {FORMAT[stage]}", None, ""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"The answer is not valid JSON: {e.msg} at line {e.lineno} column {e.colno}.", None, text
    try:
        piece = parse_piece(stage, obj)
    except PlanError as e:
        return False, f"The answer cannot be read: {e}", obj, text
    ok, msg, _ = search_piece(stage, piece)
    return ok, msg, obj, text


def assemble(frozen):
    m, n, k = frozen["m"], frozen["n"], frozen["k"]
    acc = k["accumulate_over"]
    return {"loops": {"m": m["count"], "n": n["count"], "k": k["count"]},
            "accumulate_over": [acc] if isinstance(acc, str) else acc,
            "reads": {"lhsT": [k["lhsT"], m["lhsT"]], "rhs": [k["rhs"], n["rhs"]]},
            "writes": {"out": [m["out"], n["out"]]}}


def _flat(s):
    return s[0] if isinstance(s, list) and len(s) == 1 and isinstance(s[0], list) else s


def _normalise(stage, obj):
    """The frozen piece as the model wrote it, with only the keys the stage asks for."""
    keep = ["count"] + [t for t, *_ in STAGES[stage]["slices"]] + (["accumulate_over"] if stage == "k" else [])
    return {k: (_flat(obj[k]) if k not in ("count", "accumulate_over") else obj[k]) for k in keep}


# ---------------------------------------------------------------- offline answers

def _piece(stage, w, count=None, a=None, b=None, acc=("k",)):
    """A canned piece for tile width w. Built here, never shown to the model."""
    dim = STAGES[stage]["dim"]
    s = [a or f"{stage}*{w}", b or f"({stage}+1)*{w}"]
    obj = {"count": count or f"{dim} // {w}"}
    for t, *_ in STAGES[stage]["slices"]:
        obj[t] = list(s)
    if stage == "k":
        obj["accumulate_over"] = list(acc)
    return obj


REF_W = dict(m=PMAX, n=MOV_FMAX, k=PMAX)


def offline_reply(stage, rnd):
    """Round 0: the measured confusion (one step per element, the loop index as the stop). Then right."""
    if rnd == 0:
        obj = _piece(stage, 1, count=STAGES[stage]["dim"], a="0", b=f"tile_{stage}")
    else:
        obj = _piece(stage, REF_W[stage])
    return json.dumps(obj)


# ---------------------------------------------------------------- the loop

def run_once(a, log, run):
    """One run: stage m, n, k, then the full plan_check.search. Returns a summary dict."""
    ap = copy.copy(a)
    ap.max_tokens = a.plan_max_tokens
    frozen, rounds, stuck = {}, {}, None
    for stage in ORDER:
        prompt, last_text, seen, passed = stage_prompt(stage, frozen, a.wording), None, {}, False
        for rnd in range(a.rounds):
            t0 = time.perf_counter()
            reply = offline_reply(stage, rnd) if a.offline else agent.ask_parallel(ap, prompt, 1)[0]
            ok, fb, obj, text = grade_piece(stage, reply)
            log.write(json.dumps(dict(level=4, run=run, stage=stage, round=rnd, ok=ok, piece=obj,
                                      feedback=fb, prompt_chars=len(prompt), reply_chars=len(reply),
                                      reply=reply[:4000], wording=a.wording)) + "\n")
            log.flush()
            print(f"  stage {stage} round {rnd}: {'PASS' if ok else 'fail'}  "
                  f"({time.perf_counter() - t0:.1f}s)  {text[:160]}")
            if ok:
                frozen[stage] = _normalise(stage, obj)
                rounds[stage] = rnd + 1
                passed = True
                print(f"    frozen: {json.dumps(frozen[stage])}")
                break
            print(f"    {fb[:300]}")
            seen[fb] = seen.get(fb, 0) + 1
            if a.give_up_after and seen[fb] >= a.give_up_after:
                print(f"    STOPPING stage {stage}: the same failure {seen[fb]} times.")
                break
            if text:
                last_text = text
            prompt = (stage_repair_prompt(stage, frozen, a.wording, last_text, fb) if last_text
                      else stage_prompt(stage, frozen, a.wording) + f"\n\nYour last reply could not be "
                                                                     f"used: {fb}")
        if not passed:
            rounds[stage] = None
            stuck = dict(stage=stage, feedback=fb)
            break
    out = dict(run=run, rounds=rounds, stuck=stuck, plan=None, plan_ok=False, plan_message=None)
    if stuck is None:
        plan = assemble(frozen)
        out["plan"] = plan
        try:
            r = plan_check.search(plan_check.parse_plan(plan), 300)
            out["plan_ok"], out["plan_message"] = r["ok"], r["message"]
        except PlanError as e:
            out["plan_message"] = f"PLAN REJECTED: {e}"
        print(f"  assembled plan: {json.dumps(plan)}")
        print(f"  full plan_check.search: {out['plan_message'][:300]}")
        if out["plan_ok"] and a.plan_out:
            path = a.plan_out.format(run=run)
            json.dump(plan, open(path, "w"), indent=1)
            print(f"  passing plan written to {path}")
    log.write(json.dumps(dict(level=4, run=run, stage="full", ok=out["plan_ok"], piece=out["plan"],
                              feedback=out["plan_message"] or (stuck and f"stuck at stage {stuck['stage']}"),
                              rounds=rounds, wording=a.wording)) + "\n")
    log.flush()
    return out


# ---------------------------------------------------------------- ablation: all three loops at once

def whole_prompt(wording="steps"):
    """The same header, task wording and per-loop checks, but one prompt for all three loops."""
    tasks = "\n\n".join(TASK[wording][x].replace("Now plan", "Plan", 1) for x in ORDER)
    fmt = "{\n" + ",\n".join(f' "{x}": {FORMAT[x]}' for x in ORDER) + "\n}"
    return (f"{HEADER}\n\nPlan all three loops in one answer.\n\n{tasks}\n\n"
            f"Formulas are strings and may use + - * / // %, ceil, min, max, numbers, M, K, N and the "
            f"loop's own name (m, n or k). Each slice is a JSON list of two strings, the start and the "
            f"stop (exclusive).\n\nReply with ONE JSON object in this format and nothing else:\n{fmt}")


def grade_whole(reply):
    """(ok, feedback, obj, text): each loop by its stage check, in order, then the full search."""
    text = agent.extract_json(reply)
    if not text:
        return False, "No JSON object came back. Reply with ONE JSON object.", None, ""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"The answer is not valid JSON: {e.msg} at line {e.lineno} column {e.colno}.", None, text
    if not isinstance(obj, dict):
        return False, "The answer must be ONE JSON object with parts m, n and k.", None, text
    for x in ORDER:
        if x not in obj:
            return False, f"The answer has no '{x}' part. It needs m, n and k.", obj, text
        ok, msg, _, _ = grade_piece(x, json.dumps(obj[x]))
        if not ok:
            return False, f"The {x} loop: {msg}", obj, text
    plan = assemble({x: _normalise(x, obj[x]) for x in ORDER})
    try:
        r = plan_check.search(plan_check.parse_plan(plan), 300)
    except PlanError as e:
        return False, f"PLAN REJECTED: {e}", obj, text
    return r["ok"], r["message"].split(" Fix:")[0], obj, text


def run_whole(a, log, run):
    ap = copy.copy(a)
    ap.max_tokens = a.plan_max_tokens
    prompt, last_text = whole_prompt(a.wording), None
    out = dict(run=run, rounds={"all": None}, stuck=None, plan=None, plan_ok=False, plan_message=None)
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        reply = (json.dumps({x: json.loads(offline_reply(x, rnd)) for x in ORDER}) if a.offline
                 else agent.ask_parallel(ap, prompt, 1)[0])
        ok, fb, obj, text = grade_whole(reply)
        log.write(json.dumps(dict(level=4, run=run, stage="all", round=rnd, ok=ok, piece=obj,
                                  feedback=fb, prompt_chars=len(prompt), reply_chars=len(reply),
                                  reply=reply[:4000], wording=a.wording, unstaged=True)) + "\n")
        log.flush()
        print(f"  all loops round {rnd}: {'PASS' if ok else 'fail'}  ({time.perf_counter() - t0:.1f}s)  "
              f"{text[:200]}")
        if ok:
            out.update(rounds={"all": rnd + 1}, plan=assemble({x: _normalise(x, obj[x]) for x in ORDER}),
                       plan_ok=True, plan_message=fb)
            return out
        print(f"    {fb[:300]}")
        if text:
            last_text = text
        prompt = (f"{whole_prompt(a.wording)}\n\nYour last answer was:\n```json\n{last_text}\n```\n"
                  f"A checker that tries it on many shapes reports:\n{fb}\n\nChange what the checker "
                  f"names. Reply with ONE JSON object and nothing else." if last_text
                  else whole_prompt(a.wording) + f"\n\nYour last reply could not be used: {fb}")
    out["stuck"] = dict(stage="all", feedback=fb)
    return out


# ---------------------------------------------------------------- selftest

FORBIDDEN = ["M // 128", "N // 512", "K // 128", "M//128", "N//512", "K//128", "*128", "*512",
             "Fix:", "Use // or ceil"]


def selftest():
    print("Proving the per-stage checker before trusting it.\n")
    rc = 0

    def line(what, ok, extra=""):
        nonlocal rc
        rc |= 0 if ok else 1
        print(f"  {what:<58} -> {'ok' if ok else 'FAIL'}{'  ' + extra if extra else ''}")

    ref = {s: _piece(s, REF_W[s]) for s in ORDER}
    for s in ORDER:
        ok, msg, _, _ = grade_piece(s, json.dumps(ref[s]))
        line(f"stage {s}: the reference piece passes", ok, msg[:80])
    ok, msg, _, _ = grade_piece("m", json.dumps(_piece("m", 64)))
    line("stage m: tiles of 64 pass too (legal, just smaller)", ok)
    plan = assemble({s: _normalise(s, ref[s]) for s in ORDER})
    r = plan_check.search(plan_check.parse_plan(plan), 300)
    line("assembled reference pieces == plan_check.REFERENCE, passes", r["ok"] and plan == plan_check.REFERENCE)

    print("\n  mutants (each must fail with the named symptom and no fix):\n")
    mutants = [
        ("m", "measured: loop runs M times, stop is 'tile_m'",
         _piece("m", 1, count="M", a="0", b="tile_m"), "not defined"),
        ("m", "measured: loop runs M times, slices per element",
         _piece("m", 1, count="M"), "once per element"),
        ("m", "one tile for the whole axis", _piece("m", 1, count="1", a="0", b="M"), "limit there is 128"),
        ("n", "rhs/out free tile 1024 wide", _piece("n", 1, count="1", a="0", b="N"), "limit there is 512"),
        ("k", "count is not an integer", _piece("k", 128, count="K / 100"), "not an integer"),
        ("k", "count too small: K rows never covered", _piece("k", 128, count="K // 256"), "never runs"),
        ("m", "tiles overlap (stride 64, width 128)",
         _piece("m", 128, count="M // 64 - 1", a="m*64", b="m*64 + 128"), "covered 2 times"),
        ("n", "off-by-one stop", _piece("n", 512, b="(n+1)*512 - 1"), "never covered"),
        ("k", "accumulate_over missing k", _piece("k", 128, acc=()), "does not list k"),
        ("k", "accumulate_over lists m", _piece("k", 128, acc=("m", "k")), "do not sum"),
    ]
    mis = _piece("m", 128)
    mis["lhsT"] = ["0", "128"]
    mutants.append(("m", "out and lhsT slices disagree", mis, "must agree"))
    for s, name, obj, must in mutants:
        ok, msg, _, _ = grade_piece(s, json.dumps(obj))
        clean = not any(f in msg for f in ("Fix:", "Use // or ceil", "Split that axis"))
        line(f"stage {s}: {name}", (not ok) and must in msg and clean, "" if (not ok and must in msg)
             else msg[:200])
        print(f"      {msg[:220]}")

    print("\n  honesty: no prompt shows a reference formula or a fix:\n")
    frozen = {s: _normalise(s, ref[s]) for s in ORDER}
    # Prompts for later stages show the model's OWN frozen pieces; use a non-reference piece there,
    # so the scan tests the wording and not the frozen echo.
    alt = {s: _normalise(s, _piece(s, 64)) for s in ORDER}
    prompts = []
    for wording in TASK:
        for i, s in enumerate(ORDER):
            fz = {x: alt[x] for x in ORDER[:i]}
            prompts.append(stage_prompt(s, fz, wording))
            prompts.append(stage_repair_prompt(s, fz, wording, '{"count": "M"}', "something is wrong"))
    prompts += [whole_prompt(w) for w in TASK]
    leaks = [f for f in FORBIDDEN for p in prompts if f in p]
    line(f"{len(prompts)} prompts scanned for {len(FORBIDDEN)} forbidden strings", not leaks, str(set(leaks)))
    del frozen

    print("\n  offline run through all three stages:\n")
    ns = argparse.Namespace(offline=True, rounds=8, wording="steps", give_up_after=0, plan_max_tokens=800,
                            plan_out=None)
    import io
    buf = io.StringIO()
    out = run_once(ns, buf, 0)
    line("offline run: each stage passes in round 1, full plan passes",
         out["plan_ok"] and out["rounds"] == dict(m=2, n=2, k=2))
    rows = [json.loads(x) for x in buf.getvalue().splitlines()]
    line("offline run logs 7 rows (2 per stage + the full plan)", len(rows) == 7)
    out = run_whole(ns, io.StringIO(), 0)
    line("offline unstaged run: fails round 0, passes round 1", out["plan_ok"] and out["rounds"] == {"all": 2})

    print("\nPLAN STAGE SELFTEST " + ("PASSED" if rc == 0 else "FAILED"))
    return rc


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description="H1: gate 1 asked one loop at a time")
    ap.add_argument("--rounds", type=int, default=8, help="rounds PER STAGE")
    ap.add_argument("--samples", type=int, default=1, help="only 1 is supported (the 0/5 setting)")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--plan-max-tokens", type=int, default=800,
                    help="answer budget per stage; the same as gate 1's")
    ap.add_argument("--model", default=agent.MODEL)
    ap.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL"))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--give-up-after", type=int, default=0,
                    help="stop a stage after this many identical failures; 0 = never (gate 1 ran all 8)")
    ap.add_argument("--wording", choices=sorted(TASK), default="steps")
    ap.add_argument("--acc-feedback", choices=("first", "all"), default="first",
                    help="v1 'first': report the first accumulate_over problem; v2 'all': every one")
    ap.add_argument("--unstaged", action="store_true",
                    help="ablation: all three loops in one prompt, same wording and checks; --rounds total")
    ap.add_argument("--log", default="attempts-plan-stage.jsonl")
    ap.add_argument("--plan-out", default="plan-stage-run{run}.json",
                    help="where each passing assembled plan is written ('' to skip)")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    a.think, a.max_tokens = False, a.plan_max_tokens
    global ACC_ALL
    ACC_ALL = a.acc_feedback == "all"
    if a.selftest:
        sys.exit(selftest())
    if a.samples != 1:
        sys.exit("--samples 1 only: the 0/5 comparison used one sample per round.")
    if not a.offline:
        if not (a.base or "").strip():
            sys.exit("KERNEL_AGENT_BASE_URL is unset; the seat pods set it. Or pass --offline.")
        a.base = a.base.strip().rstrip("/")
        print(f"endpoint {a.base}  model {a.model}  wording {a.wording}")
    else:
        print("*** OFFLINE: replaying canned answers. Numbers are meaningless. ***")

    runs = []
    t0 = time.time()
    with open(a.log, "a") as log:
        for run in range(a.repeat):
            print(f"\n################ run {run + 1} of {a.repeat} ################")
            runs.append((run_whole if a.unstaged else run_once)(a, log, run))

    mode = "all loops in one prompt" if a.unstaged else "one loop at a time"
    print(f"\n=========== summary (level 4 gate 1, {mode}, wording={a.wording}) ===========")
    for r in runs:
        where = ("plan PASSES plan_check.search" if r["plan_ok"] else
                 f"stuck at stage {r['stuck']['stage']}" if r["stuck"] else "assembled plan FAILS")
        print(f"  run {r['run'] + 1}: rounds per stage {r['rounds']}; {where}")
    print(f"  plans passing: {sum(r['plan_ok'] for r in runs)}/{len(runs)}   "
          f"(gate 1 whole-plan, diagnostic: 0/5)   wall {time.time() - t0:.0f}s")
    print(f"attempts logged to {a.log}")


if __name__ == "__main__":
    main()
