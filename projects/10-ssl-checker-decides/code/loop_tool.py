#!/usr/bin/env python3
"""
loop_tool.py -- plan first, then a tool writes the loops and the model writes only the tile body.

Level 4 sticks at 0.62 because the model has to choose the tiling AND write it in NKI in one go.
`agent.py --plan-first` splits the choice out (gate 1: a JSON plan graded by plan_check.py). This
file goes one step further, the way project 1 gave its model a calculator: once a plan passes, a
TOOL turns it into the kernel's loops, slices and PSUM accumulator, and the model writes only what
happens to one tile:

    (A) inside the accumulate loop: bring this step's lhsT and rhs slices into SBUF, matmul into acc
    (B) after it: move acc out to this tile's slice of the result

The model still makes every decision -- the plan is its own -- and the tool does the part it keeps
getting wrong. Graded by the same agent.grade as the baseline, so the score compares with 0.62.

    python loop_tool.py --selftest                 # plan -> skeleton -> reference body, no model
    python loop_tool.py --offline                  # the whole loop, no model
    python loop_tool.py --rounds 8 --samples 4 --context 8192 --repeat 5 --log attempts-loop-tool.jsonl
    python loop_tool.py --given-plan reference ...    # skip gate 1: does the tile body alone pass?

Level 4 only (plan_check.py knows matmul). Contract shapes only: K, M multiples of 128, N of 512.
"""

import argparse
import ast
import inspect
import json
import os
import re
import sys
import tempfile
import textwrap
import time

import agent
import nkibench
import plan_check

LEVEL = 4
ENTRY = nkibench.LEVELS[LEVEL]["entry"]
FULL = sum(agent.WEIGHTS.values())
RESERVED = {"lhsT", "rhs", "result", "acc", "nl", "nisa", "nki", "ceil", "int",
            "lhsT_slice", "rhs_slice", "out_slice", "M", "K", "N"}


class ToolError(Exception):
    """The plan passed gate 1 but cannot be turned into loops. Its text goes back to the model."""


# ---------------------------------------------------------------- isolated grading
#
# agent.grade writes every candidate to the fixed path /tmp/_agent_level4.py, so two level-4 runs in
# one pod grade each other's kernels. Run the same function against a private file instead.

def _private_grade():
    src = textwrap.dedent(inspect.getsource(agent.grade))
    old = 'path = f"/tmp/_agent_level{level}.py"'
    assert old in src, "agent.grade changed; update loop_tool._private_grade"
    fd, path = tempfile.mkstemp(prefix="_loop_tool_", suffix=".py", dir="/tmp")
    os.close(fd)
    g = dict(vars(agent))
    g["_GRADE_PATH"] = path
    exec(compile(src.replace(old, "path = _GRADE_PATH"), "<agent.grade>", "exec"), g)
    return g["grade"]


grade = _private_grade()


# ---------------------------------------------------------------- the tool: plan -> skeleton

def _floor_div(expr):
    """A slice bound in kernel syntax. Loop variables are symbolic inside the kernel, so '/' (which
    plan_check allows when it divides evenly) becomes '//'."""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            node.op = ast.FloorDiv()
    return ast.unparse(tree)


def _tile_shape(p, name, shapes):
    """The tile's (rows, cols), which must be the same for every tile on every contract shape --
    the tool allocates one SBUF/PSUM buffer of that size."""
    seen = set()
    for s in shapes:
        env = dict(M=s["M"], K=s["K"], N=s["N"])
        counts = {v: plan_check._eval(e, env, f"loops.{v}") for v, e in p["loops"].items()}
        for pick in ("first", "last"):
            e = dict(env, **{v: (0 if pick == "first" else max(c - 1, 0)) for v, c in counts.items()})
            seen.add(tuple(plan_check._eval(b, e, name) - plan_check._eval(a, e, name)
                           for a, b in p[name]))
    if len(seen) != 1:
        raise ToolError(f"the {name} tile changes size between iterations or shapes "
                        f"({sorted(seen)}); every {name} tile must be the same size.")
    return seen.pop()


def skeleton(plan):
    """Plan (a dict that passed gate 1) -> (kernel source with holes '# (A)' and '# (B)', info)."""
    p = plan_check.parse_plan(plan)
    clash = [v for v in p["loops"] if v in RESERVED]
    if clash:
        raise ToolError(f"loop name(s) {clash} clash with names in the kernel; use m, n, k.")
    shapes = nkibench.LEVELS[LEVEL]["shapes"]
    tiles = {t: _tile_shape(p, t, shapes) for t in ("lhsT", "rhs", "out")}
    outer = [v for v in p["loops"] if v not in p["acc"]]
    inner = [v for v in p["loops"] if v in p["acc"]]

    def sl(t):
        return ", ".join(f"{_floor_div(a[0])}:{_floor_div(b[0])}" for a, b in p[t])

    L = ["import nki", "import nki.isa as nisa", "import nki.language as nl",
         "from math import ceil", "", "",
         "@nki.jit", f"def {ENTRY}(lhsT, rhs):",
         "    K, M = lhsT.shape", "    _, N = rhs.shape",
         "    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)", "",
         "    # Loops and slices generated from your tiling plan. They are correct: do not change them."]
    ind = "    "
    for v in outer:
        L.append(f"{ind}for {v} in nl.affine_range(int({p['loops'][v][0]})):")
        ind += "    "
    r, c = tiles["out"]
    L.append(f"{ind}acc = nl.ndarray(({r}, {c}), dtype=nl.float32, buffer=nl.psum)"
             f"  # PSUM accumulator for this output tile")
    hole_b = ind
    for v in inner:
        L.append(f"{ind}for {v} in nl.affine_range(int({p['loops'][v][0]})):")
        ind += "    "
    (lr, lc), (rr, rc) = tiles["lhsT"], tiles["rhs"]
    L += [f"{ind}lhsT_slice = lhsT[{sl('lhsT')}]  # {lr} x {lc}, in HBM",
          f"{ind}rhs_slice = rhs[{sl('rhs')}]  # {rr} x {rc}, in HBM",
          f"{ind}# (A)",
          f"{hole_b}out_slice = result[{sl('out')}]  # {r} x {c}, in HBM",
          f"{hole_b}# (B)",
          "    return result", ""]
    acc_loop = inner[-1] if inner else "innermost"
    return "\n".join(L), dict(tiles=tiles, outer=outer, inner=inner, acc_loop=acc_loop)


def splice(skel, a_code, b_code):
    out = []
    for line in skel.splitlines():
        s = line.strip()
        if s in ("# (A)", "# (B)"):
            body = a_code if s == "# (A)" else b_code
            ind = line[:len(line) - len(line.lstrip())]
            out.append(textwrap.indent(body, ind))
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def _clean(block):
    """A hole's code: dedented, with any imports dropped (the skeleton has them)."""
    lines = [ln for ln in block.splitlines()
             if ln.strip() and not re.match(r"^\s*(import |from )", ln)]
    # Measured on the staged run: the model sends its first line flush left and the rest indented as
    # in the skeleton, so a plain dedent leaves mixed indentation and the spliced kernel fails to parse.
    # Normalise to the smallest indent among the lines after the first.
    # A first line that opens a block (ends with ':') legitimately indents what follows.
    if len(lines) > 1 and not lines[0][:1].isspace() and not lines[0].rstrip().endswith(":"):
        rest = textwrap.dedent("\n".join(lines[1:])).splitlines()
        lines = [lines[0]] + rest
    return textwrap.dedent("\n".join(lines)).strip() or "pass"


def extract_holes(text):
    """(A, B) from a reply: two code blocks, or one block split at a '(B)' comment. None if neither."""
    blocks = agent.CODE_BLOCK.findall(text or "")
    if len(blocks) >= 2:
        return _clean(blocks[0]), _clean(blocks[1])
    if len(blocks) == 1:
        parts = re.split(r"^\s*#.*\(B\).*$", blocks[0], maxsplit=1, flags=re.M)
        if len(parts) == 2:
            return _clean(re.sub(r"^\s*#.*\(A\).*$", "", parts[0], flags=re.M)), _clean(parts[1])
    return None


# ---------------------------------------------------------------- structural check (--structure-check)
#
# Measured 2026-10-10 (results/2026-10-10-rishabh-loop-tool-api-messages.md): the first correctly
# allocated body put the whole pipeline in (B), after the accumulate loop, so only the last K tile was
# multiplied -- right on K=128, wrong elsewhere, and the numerical-mismatch message never said where
# the matmul belongs. The tool knows which hole is which, so it can say so. Names in these messages
# are the kernel's own (acc, out_slice, the model's tile names): placeholders get copied literally.

def _tile_for(code, slice_name):
    m = re.search(r"dma_copy\(\s*dst\s*=\s*(\w+)\s*,\s*src\s*=\s*" + slice_name + r"\b", code)
    return m.group(1) if m else None


def structure_hint(a_code, b_code, info):
    """One named fix when the body's structure is wrong for the skeleton, else None."""
    k = info["acc_loop"]
    both = a_code + "\n" + b_code
    lt, rt = _tile_for(both, "lhsT_slice"), _tile_for(both, "rhs_slice")
    operands = (f"stationary={lt}, moving={rt}" if lt and rt else
                "stationary= the SBUF tile holding lhsT_slice, moving= the SBUF tile holding rhs_slice")
    mm_a = re.search(r"nc_matmul\(\s*dst\s*=\s*(\w+)", a_code)
    if not mm_a:
        where = ("is in (B), after the " + k + " loop, so only the last K tile is multiplied"
                 if "nc_matmul" in b_code else "is missing from (A)")
        return (f"STRUCTURE: nc_matmul {where}. Every K tile must be multiplied and summed, so (A), "
                f"inside the {k} loop, must load lhsT_slice and rhs_slice into SBUF tiles and call "
                f"nisa.nc_matmul(dst=acc, {operands}). (B) only copies acc out: tensor_copy acc into an "
                f"SBUF tile, then nisa.dma_copy(dst=out_slice, src=that SBUF tile).")
    if mm_a.group(1) != "acc":
        return (f"STRUCTURE: nc_matmul in (A) writes dst={mm_a.group(1)}, not acc, so the {k} loop's "
                f"partial products are never summed. Use nisa.nc_matmul(dst=acc, {operands}): acc is "
                f"the PSUM accumulator the skeleton allocated before the {k} loop.")
    if "out_slice" not in b_code:
        return ("STRUCTURE: (B) never writes out_slice, so the result never reaches HBM. In (B): "
                "tensor_copy acc into an SBUF tile, then nisa.dma_copy(dst=out_slice, src=that SBUF tile).")
    return None


# ---------------------------------------------------------------- gate 2 with the tool

# The two ways of asking for each hole. "loose" is what the unstaged prompt used; "steps" is what the
# staged prompt used. The --wording ablation swaps them, so staging and wording can be told apart.
WORDING = {
    "loose": dict(A="bring lhsT_slice and rhs_slice into SBUF and accumulate their product into acc.",
                  B="move acc out to out_slice."),
    "steps": dict(A="allocate SBUF tiles for lhsT_slice and rhs_slice, dma_copy each slice into its "
                    "tile, then accumulate their product into acc with nisa.nc_matmul. Each allocation "
                    "goes directly above its load.",
                  B="move acc out to out_slice (acc is in PSUM; out_slice is in HBM)."),
}


def body_prompt(skel, info, wording="loose"):
    w = WORDING[wording]
    return (
        f"Write the body of an AWS Neuron NKI kernel for a tiled matmul: result = lhsT.T @ rhs.\n"
        f"The loops, the slices and the PSUM accumulator are already written from a checked tiling "
        f"plan, and they are correct. Do not change them.\n\n```python\n{skel}```\n\n"
        f"Fill in the two marked places:\n"
        f"  (A) inside the {info['acc_loop']} loop: {w['A']}\n"
        f"  (B) after the {info['acc_loop']} loop: {w['B']}\n\n"
        f"{agent.API_CARD}\n\n"
        f"Reply with exactly TWO python code blocks: the first is the code for (A), the second the "
        f"code for (B). Only those lines: no imports, no loops, no def.")


def body_repair_prompt(kernel, feedback):
    return (
        f"This NKI kernel is not right yet. Its loops, slices and acc were generated from a checked "
        f"plan and are correct; only the code at (A) and (B) is yours.\n\n```python\n{kernel}```\n\n"
        f"A checker reports:\n{feedback}\n\n"
        f"Change what the checker names. Reply with exactly TWO python code blocks: the new code for "
        f"(A), then for (B). No imports, no loops, no def.")


REF_A = """lhsT_tile = nl.ndarray(lhsT_slice.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
rhs_tile = nl.ndarray(rhs_slice.shape, dtype=rhs.dtype, buffer=nl.sbuf)
nisa.dma_copy(dst=lhsT_tile, src=lhsT_slice)
nisa.dma_copy(dst=rhs_tile, src=rhs_slice)
nisa.nc_matmul(dst=acc, stationary=lhsT_tile, moving=rhs_tile)"""

REF_B = """res_sb = nl.ndarray(acc.shape, dtype=result.dtype, buffer=nl.sbuf)
nisa.tensor_copy(dst=res_sb, src=acc)
nisa.dma_copy(dst=out_slice, src=res_sb)"""


def offline_bodies(n, rnd):
    """No model. Round 0 matmuls straight from HBM (a real mistake), then the reference body."""
    if rnd == 0:
        bad = "nisa.nc_matmul(dst=acc, stationary=lhsT_slice, moving=rhs_slice)"
        return [f"```python\n{bad}\n```\n```python\n{REF_B}\n```"] * n
    return [f"```python\n{REF_A}\n```\n```python\n{REF_B}\n```"] * n


def gate2_tool(a, plan, log, run):
    print(f"\n----------- gate 2 (loop tool): tile body for level {LEVEL} -----------")
    try:
        skel, info = skeleton(plan)
    except (ToolError, plan_check.PlanError) as e:
        print(f"  TOOL CANNOT BUILD LOOPS from the passing plan: {e}")
        log.write(json.dumps(dict(level=LEVEL, run=run, gate=2, tool=True, tool_error=str(e))) + "\n")
        return 0.0, 0, f"tool: {e}"
    print(textwrap.indent(skel, "    | "))
    wording = a.wording or "loose"
    prompt = body_prompt(skel, info, wording)
    best, seen, latest, fb = (0.0, ""), {}, None, "no rounds run"
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        replies = (offline_bodies(a.samples, rnd) if a.offline
                   else agent.ask_parallel(a, prompt, a.samples))
        graded = []
        for reply in replies:
            holes = extract_holes(reply)
            if holes is None:
                kernel, reward, fb = "", 0.0, ("Reply with exactly TWO python code blocks: the code "
                                               "for (A), then the code for (B). Not the whole kernel.")
            else:
                kernel = splice(skel, *holes)
                reward, _, fb = grade(kernel, LEVEL)
            hint = (structure_hint(*holes, info) if a.structure_check and holes is not None
                    and reward < FULL - 1e-9 else None)
            if hint:
                fb = hint + "\n" + fb
            graded.append((reward, kernel, fb))
            log.write(json.dumps(dict(level=LEVEL, run=run, gate=2, tool=True, round=rnd,
                                      reward=reward, format_ok=holes is not None, code=kernel,
                                      structure_hint=bool(hint),
                                      feedback=fb, prompt_chars=len(prompt),
                                      reply_chars=len(reply))) + "\n")
        log.flush()
        graded.sort(key=lambda g: g[0], reverse=True)
        reward, kernel, fb = graded[0]
        if reward > best[0]:
            best = (reward, fb)
        print(f"body round {rnd}: this round {reward:.2f}  best so far {best[0]:.2f}  "
              f"({time.perf_counter() - t0:.1f}s)")
        print(f"  {fb[:400]}")
        if reward >= FULL - 1e-9:
            print(f"  SOLVED on body round {rnd}.")
            print(textwrap.indent(kernel, "    | "))
            return reward, rnd + 1, None
        seen[fb] = seen.get(fb, 0) + 1
        if seen[fb] >= a.give_up_after:
            print(f"  STOPPING gate 2: the same failure {seen[fb]} times.")
            return best[0], rnd + 1, fb
        if kernel:
            latest = kernel
        prompt = body_repair_prompt(latest, fb) if latest else body_prompt(skel, info, wording)
    return best[0], a.rounds, fb


# ---------------------------------------------------------------- staged gate 2 (--staged)
#
# Measured 2026-10-10 (results/2026-10-10-rishabh-loop-tool-v21-struct.md): every named fix was
# followed (structure 15/16), but only the LATEST one stuck -- fixing the matmul's hole broke the
# allocation, and back. So keep what is right in code, not in the prompt: (A) alone first, frozen once
# it passes, then (B) alone against the frozen (A).
#
# Checking (A) alone needs some (B) to run the kernel, so the harness supplies REF_B as a test stub.
# The model never sees it: its own (B) is written and graded in stage B against its frozen (A).

def _one_block(text):
    blocks = agent.CODE_BLOCK.findall(text or "")
    if not blocks:
        return None
    code = _clean(blocks[0])
    return None if re.search(r"^\s*(def |@nki)", code, re.M) else code


def stage_prompt(skel, info, stage, frozen_a=None, wording="steps"):
    k, w = info["acc_loop"], WORDING[wording]
    if stage == "A":
        shown = skel.replace("# (B)", "# (B) -- not part of this step")
        task = f"Write ONLY the code for (A), inside the {k} loop: {w['A']}"
    else:
        shown = splice(skel, frozen_a, "# (B)")
        task = (f"(A) is done and correct -- it is checked and frozen, do not repeat it. Write ONLY the "
                f"code for (B), after the {k} loop: {w['B']}")
    return (f"Write part of an AWS Neuron NKI kernel for a tiled matmul: result = lhsT.T @ rhs.\n"
            f"The loops, the slices and the PSUM accumulator are already written and correct.\n\n"
            f"```python\n{shown}```\n\n{task}\n\n{agent.API_CARD}\n\n"
            f"Reply with ONE python code block containing only those lines: no imports, no loops, no def.")


def stage_repair_prompt(code, stage, feedback):
    return (f"Your code for ({stage}) is not right yet:\n\n```python\n{code}\n```\n\n"
            f"A checker reports:\n{feedback}\n\nChange what the checker names. Reply with ONE python "
            f"code block: only the code for ({stage}), no imports, no loops, no def.")


def offline_stage(stage, rnd):
    if stage == "A":
        code = ("nisa.nc_matmul(dst=acc, stationary=lhsT_slice, moving=rhs_slice)" if rnd == 0
                else REF_A)
    else:
        code = REF_B
    return f"```python\n{code}\n```"


def gate2_staged(a, plan, log, run):
    print(f"\n----------- gate 2 (loop tool, staged): (A) then (B) for level {LEVEL} -----------")
    try:
        skel, info = skeleton(plan)
    except (ToolError, plan_check.PlanError) as e:
        print(f"  TOOL CANNOT BUILD LOOPS from the passing plan: {e}")
        return 0.0, 0, f"tool: {e}"
    best, frozen_a, fb, used = 0.0, None, "no rounds run", 0
    wording = a.wording or "steps"
    for stage in ("A", "B"):
        prompt, latest, seen = stage_prompt(skel, info, stage, frozen_a, wording), None, {}
        while used < a.rounds:
            rnd, used = used, used + 1
            t0 = time.perf_counter()
            reply = (offline_stage(stage, rnd) if a.offline else agent.ask_parallel(a, prompt, 1)[0])
            code = _one_block(reply)
            if code is None:
                kernel, reward, fb = "", 0.0, (f"Reply with ONE python code block holding only the code "
                                               f"for ({stage}). Not the whole kernel.")
            else:
                hole_a, hole_b = (code, REF_B) if stage == "A" else (frozen_a, code)
                kernel = splice(skel, hole_a, hole_b)
                reward, _, fb = grade(kernel, LEVEL)
                hint = (structure_hint(hole_a, hole_b, info)
                        if a.structure_check and reward < FULL - 1e-9 else None)
                if hint:
                    fb = hint + "\n" + fb
            # stage A's score uses the harness's (B), so it is not the model's kernel; report the
            # model's own score only from stage B (stage A rows are logged with model_kernel=False).
            if stage == "B":
                best = max(best, reward)
            log.write(json.dumps(dict(level=LEVEL, run=run, gate=2, tool=True, staged=stage,
                                      round=rnd, reward=reward, model_kernel=stage == "B",
                                      format_ok=code is not None, code=kernel, feedback=fb,
                                      prompt_chars=len(prompt), reply_chars=len(reply))) + "\n")
            log.flush()
            print(f"stage {stage} round {rnd}: {reward:.2f}  ({time.perf_counter() - t0:.1f}s)")
            print(f"  {fb[:400]}")
            if reward >= FULL - 1e-9:
                if stage == "A":
                    frozen_a = code
                    print("  (A) PASSES and is frozen.")
                    break
                print(f"  SOLVED in stage B, round {rnd}.")
                print(textwrap.indent(kernel, "    | "))
                return reward, used, None
            seen[fb] = seen.get(fb, 0) + 1
            if seen[fb] >= a.give_up_after:
                print(f"  STOPPING stage {stage}: the same failure {seen[fb]} times.")
                return best, used, f"stage {stage}: {fb}"
            if code:
                latest = code
            prompt = (stage_repair_prompt(latest, stage, fb) if latest
                      else stage_prompt(skel, info, stage, frozen_a, wording))
        if frozen_a is None:
            return best, used, f"stage A: {fb}"
    return best, used, f"stage B: {fb}"


# ---------------------------------------------------------------- selftest and main

def selftest():
    """The tool must turn a correct plan into a correct kernel, or nothing it reports means anything."""
    ok = True
    for name, plan in [("reference plan", plan_check.REFERENCE),
                       ("same plan, n loop before m", {**plan_check.REFERENCE, "loops": {
                           "n": "N // 512", "m": "M // 128", "k": "K // 128"}})]:
        skel, _ = skeleton(plan)
        reward, _, fb = grade(splice(skel, REF_A, REF_B), LEVEL)
        good = reward >= FULL - 1e-9
        ok &= good
        print(f"  {name:<32} + reference body -> {reward:.2f} {'ok' if good else 'FAIL: ' + fb[:300]}")
    skel, _ = skeleton(plan_check.REFERENCE)
    reward, _, fb = grade(splice(skel, "nisa.nc_matmul(dst=acc, stationary=lhsT_slice, "
                                       "moving=rhs_slice)", REF_B), LEVEL)
    bad = reward < FULL - 1e-9
    ok &= bad
    print(f"  {'matmul straight from HBM':<32} -> {reward:.2f} {'rejected, ok' if bad else 'FAIL: accepted'}")
    print(f"     {fb[:300]}")
    _, info = skeleton(plan_check.REFERENCE)
    alloc = ("t1 = nl.ndarray((128, 128), dtype=lhsT.dtype, buffer=nl.sbuf)\n"
             "t2 = nl.ndarray((128, 512), dtype=rhs.dtype, buffer=nl.sbuf)")
    late = ("t_acc = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)\n"
            "nisa.dma_copy(dst=t1, src=lhsT_slice)\nnisa.dma_copy(dst=t2, src=rhs_slice)\n"
            "nisa.nc_matmul(dst=t_acc, stationary=t1, moving=t2)\n" + REF_B.replace("acc", "t_acc"))
    h = structure_hint(alloc, late, info)
    good = h is not None and "is in (B)" in h and "stationary=t1, moving=t2" in h
    ok &= good
    print(f"  {'matmul after the k loop (measured)':<32} -> {'hinted, ok' if good else 'FAIL: ' + str(h)}")
    good = structure_hint(REF_A, REF_B, info) is None
    ok &= good
    print(f"  {'reference body':<32} -> {'no hint, ok' if good else 'FAIL: hinted'}")
    print("\nLOOP TOOL SELFTEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="plan first, then a tool writes the loops")
    ap.add_argument("--rounds", type=int, default=8, help="gate-2 (body) rounds")
    ap.add_argument("--plan-rounds", type=int, default=4, help="gate-1 (plan) rounds")
    ap.add_argument("--plan-max-tokens", type=int, default=800,
                    help="gate-1 answer budget; same default as agent.py --plan-first")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=agent.MIN_ANSWER_TOKENS)
    ap.add_argument("--model", default=agent.MODEL)
    ap.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL"))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--give-up-after", type=int, default=4)
    ap.add_argument("--log", default="attempts-loop-tool.jsonl")
    ap.add_argument("--given-plan", metavar="reference|PLAN.json",
                    help="skip gate 1 and give the tool this plan; 'reference' is "
                         "plan_check.REFERENCE, the same plan as agent.py --plan-file reference")
    ap.add_argument("--wording", choices=sorted(WORDING),
                    help="how each hole is asked for; default loose unstaged, steps staged (ablation)")
    ap.add_argument("--staged", action="store_true",
                    help="write (A) alone, freeze it once it passes, then (B) alone; --rounds is the "
                         "total across both stages")
    ap.add_argument("--structure-check", action="store_true",
                    help="prepend a named fix when the matmul is in the wrong hole, does not write acc, "
                         "or (B) never writes out_slice")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    a.think, a.terse = False, 0
    if a.selftest:
        sys.exit(selftest())
    if not a.offline:
        if not (a.base or "").strip():
            sys.exit("KERNEL_AGENT_BASE_URL is unset; the seat pods set it. Or pass --offline.")
        a.base = a.base.strip().rstrip("/")
        print(f"endpoint {a.base}  model {a.model}")
    else:
        print("*** OFFLINE: replaying fixed answers. Numbers are meaningless. ***")

    given = None
    if a.given_plan:
        given = (plan_check.REFERENCE if a.given_plan == "reference"
                 else json.load(open(a.given_plan)))
        r = plan_check.search(plan_check.parse_plan(given), 300)
        if not r["ok"]:
            sys.exit(f"--given-plan does not pass plan_check, so gate 2 would test a wrong plan:\n"
                     f"{r['message']}")
        print(f"*** PLAN GIVEN ({a.given_plan}): gate 1 skipped. Not comparable with the baseline as "
              f"a fix -- it measures whether the model can write the tile body. ***")

    runs = []
    with open(a.log, "a") as log:
        for run in range(a.repeat):
            print(f"\n################ run {run + 1} of {a.repeat} ################")
            if given is not None:
                plan_text, plan_rounds, msg = json.dumps(given), 0, ""
            else:
                plan_text, plan_rounds, msg = agent.gate1(a, LEVEL, log)
            if plan_text is None:
                runs.append(dict(reward=0.0, plan_rounds=plan_rounds, stuck_at=1, stuck_on=msg))
            else:
                reward, body_rounds, stuck = (gate2_staged if a.staged else gate2_tool)(
                    a, json.loads(plan_text), log, run)
                runs.append(dict(reward=reward, plan_rounds=plan_rounds, body_rounds=body_rounds,
                                 stuck_at=None if stuck is None else 2, stuck_on=stuck))
            r = runs[-1]
            print(f"\n  run {run + 1}: reward {r['reward']:.2f}, plan in {r['plan_rounds']} round(s)"
                  + (", SOLVED" if r["stuck_at"] is None else f", stuck at gate {r['stuck_at']}"))

    mode = f"plan given ({a.given_plan})" if given is not None else "plan first"
    print(f"\n=========== summary (level 4, {mode} + loop tool) ===========")
    print(f"  all = {[round(r['reward'], 2) for r in runs]}")
    print(f"  solved {sum(r['stuck_at'] is None for r in runs)}/{len(runs)}   "
          f"stuck at gate 1: {sum(r['stuck_at'] == 1 for r in runs)}   "
          f"stuck at gate 2: {sum(r['stuck_at'] == 2 for r in runs)}")


if __name__ == "__main__":
    main()
