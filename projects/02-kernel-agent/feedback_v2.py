"""Feedback v2 for the kernel agent: the same loop and checker, better messages.

Run it instead of agent.py, from the event repo's projects/02-kernel-agent directory:

    FEEDBACK_V2=full   PYTHONPATH=. python /path/to/feedback_v2.py --all ...   # line + new messages
    FEEDBACK_V2=locate PYTHONPATH=. python /path/to/feedback_v2.py --all ...   # line only

It imports their agent.py unchanged and swaps in one function, grade(), whose only difference is the
feedback text. Prompts, sampling, rounds, rewards and stopping rules are all theirs.

Two changes, both keyed on the error and the offending line, never on the level number, so they
should carry over to levels nobody has seen:

1. LOCATE. Every error that the simulator raises gets prefixed with the line of the model's own code
   that raised it, e.g. "Line 25: `nisa.tensor_copy(dst=sbuf_lhsT, src=psum)`". The traceback always
   reaches that line (checked on 30 baseline failures), and their messages never said where.
2. MESSAGES. New instructions for the errors the baseline got stuck on, each naming one change. Two
   replace messages that pointed at the wrong fix:
     - a shape mismatch inside nisa.tensor_copy was blamed on a Python slice assignment;
     - "cannot reshape" inside nisa.nc_matmul was answered with "do not reshape", but the model had
       not reshaped anything: its psum tile had the wrong shape.
   Anything not matched here falls back to their enrich().
"""
import os
import re
import sys
import traceback

import numpy as np

import agent      # their agent.py, unchanged
import nkibench   # their checker, unchanged

MODE = os.environ.get("FEEDBACK_V2", "full")
if MODE not in ("full", "locate"):
    sys.exit(f"FEEDBACK_V2 must be 'full' or 'locate', got {MODE!r}")

REDUCERS = r"(?:nl|nisa)\.(?:sum|max|min|mean|prod|all|any|tensor_reduce)\("


def locate(exc, path):
    """The last frame of the traceback inside the candidate file: (line number, stripped source)."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == path]
    if not frames:
        return None
    return frames[-1].lineno, (frames[-1].line or "").strip()


def _kw(line, name):
    """The expression passed as name=... on this line, e.g. _kw(line, 'dst') -> 'sbuf_lhsT'."""
    m = re.search(rf"\b{name}\s*=\s*([\w.]+(?:\[[^\]]*\])?)", line)
    return m.group(1) if m else None


def _matmul_operands(source):
    """Names passed as stationary= / moving= anywhere in the kernel."""
    names = set()
    for m in re.finditer(r"nc_matmul\(([^)]*)\)", source, re.S):
        for kw in ("stationary", "moving"):
            v = _kw(m.group(1), kw)
            if v:
                names.add(v.split("[")[0])
    return names


def advise(err, line, source):
    """A new instruction for this error on this line, or None to fall back to their enrich()."""
    # 1. A reduction over axis 0 leaves fewer than 2 dimensions.
    if "must have at least 2 dimensions" in err:
        m = re.search(REDUCERS + r".*axis\s*=\s*\[?([\d,\s]+)", line)
        if m and "0" in [a.strip() for a in m.group(1).split(",")]:
            return ("This line reduces over axis 0, the partition axis, so the result has fewer than "
                    "2 dimensions. Never reduce axis 0. Put the dimension you want to keep on axis 0 "
                    "(up to 128 of it per tile) and reduce only over axes 1 and up with "
                    "keepdims=True, for example nl.sum(tile, axis=[1], keepdims=True) on a (rows, n) "
                    "tile gives a (rows, 1) result.")
        if m and not re.search(r"keepdims\s*=\s*True", line):
            return ("This reduction drops the axes it reduces, so the result has fewer than 2 "
                    "dimensions. Add keepdims=True to this call, e.g. "
                    "nl.sum(tile, axis=[1], keepdims=True) on a (rows, n) tile gives (rows, 1).")
        if "nl.ndarray" in line:
            return ("The shape on this line has fewer than 2 dimensions. Every sbuf and psum tile is "
                    "(partition, free): write the shape as a 2-element tuple, for example (n, 1) for a "
                    "column of n <= 128 values or (1, n) for a row.")
        return None

    # 2. Writing a result into a tile of a different size.
    m = re.search(r"value array of shape \((\d+),?\) could not be broadcast to indexing result of "
                  r"shape \((\d+),?\)", err)
    if m and re.search(r"nisa\.\w+\(", line):
        got, room = int(m.group(1)), int(m.group(2))
        dst, src = _kw(line, "dst"), _kw(line, "src") or _kw(line, "data")
        src_shape = f"{src}.shape" if src else "the source's shape"
        return (f"`{dst}` holds {room} elements, but the result written into it holds {got}. A "
                f"destination must have exactly the shape of what goes into it, so do not reuse a "
                f"tile you allocated for something else (such as an input). Allocate a new tile for "
                f"this result, e.g. res = nl.ndarray({src_shape}, dtype=<output dtype>, "
                f"buffer=nl.sbuf), and use it from this line on.")

    # 3. nc_matmul wrote its result into a psum tile of the wrong shape.
    m = re.search(r"cannot reshape array of size (\d+) into shape \(([\d, ]+)\)", err)
    if m and "nc_matmul" in line:
        size, shape = int(m.group(1)), m.group(2)
        dst = _kw(line, "dst") or "the psum tile"
        return (f"nc_matmul produces a result of shape (stationary.shape[1], moving.shape[1]), here "
                f"{size} elements, but `{dst}` has shape ({shape}). Allocate `{dst}` as "
                f"nl.ndarray((stationary.shape[1], moving.shape[1]), dtype=nl.float32, "
                f"buffer=nl.psum), using the shapes of the two tiles you pass on this line.")

    # 4. A matmul operand with more than 128 rows: that axis is the contraction K.
    m = re.search(r"dma_copy dst partition dimension (\d+) exceeds maximum (\d+)", err)
    if m:
        k, mx = int(m.group(1)), int(m.group(2))
        dst = (_kw(line, "dst") or "").split("[")[0]
        if dst and dst in _matmul_operands(source):
            return (f"`{dst}` is a matmul operand, so its first axis is the contraction dimension K = "
                    f"{k}, and a tile holds at most {mx} rows. Split K: allocate the psum result tile "
                    f"once, then loop k over {k} // {mx} chunks. Inside the loop, dma_copy rows "
                    f"[k*{mx}:(k+1)*{mx}] of each input into its own {mx}-row tile and call nc_matmul "
                    f"into that same psum tile. Matmuls into one psum tile add up, so after the loop "
                    f"it holds the full product.")
        return None

    # 5. The stationary operand is wider than 128: split the output rows M.
    m = re.search(r"Matmul stationary free dimension (\d+) exceeds gemm_stationary_fmax=(\d+)", err)
    if m:
        width, mx = int(m.group(1)), int(m.group(2))
        return (f"The stationary operand's second axis is M = {width}, and one nc_matmul takes at "
                f"most {mx} there. Split M into an outer loop: for each block m of {mx} columns of "
                f"lhsT, use stationary=<lhsT tile>[:, m*{mx}:(m+1)*{mx}], give that block its own "
                f"psum tile of shape ({mx}, N), and write it to rows [m*{mx}:(m+1)*{mx}] of the "
                f"output.")

    # 5b. The moving operand is wider than 512: split the output columns N.
    m = re.search(r"Matmul moving free dimension (\d+) exceeds max (\d+)", err)
    if m:
        width, mx = int(m.group(1)), int(m.group(2))
        return (f"The moving operand's second axis is N = {width}, and one nc_matmul takes at most "
                f"{mx} there. Split N into a loop: for each block n of {mx} columns of rhs, use "
                f"moving=<rhs tile>[:, n*{mx}:(n+1)*{mx}], give that block its own psum tile of "
                f"shape (M, {mx}), and write it to columns [n*{mx}:(n+1)*{mx}] of the output.")

    # 6. dma_copy cannot read psum.
    if "dma_copy requires HBM or SBUF tensors" in err and "psum" in err:
        return ("dma_copy cannot read psum. Copy the psum tile into an sbuf tile of the same shape "
                "with nisa.tensor_copy first, then dma_copy that sbuf tile to the output.")

    # 7. A real function, called through the wrong module.
    m = re.search(r"module '(nki\.(?:isa|language))' has no attribute '(\w+)'", err)
    if m:
        here, name = m.groups()
        other = "nki.language" if here == "nki.isa" else "nki.isa"
        short = {"nki.language": "nl", "nki.isa": "nisa"}
        try:
            mod = __import__(other, fromlist=["x"])
        except Exception:
            return None
        if hasattr(mod, name):
            return (f"`{name}` lives in {other}, not {here}: write {short[other]}.{name} instead of "
                    f"{short[here]}.{name}.")
        return None
    return None


def feedback(exc, path, source):
    """What the model is told about an exception raised while simulating its kernel."""
    err = f"{type(exc).__name__}: {exc}"
    where = locate(exc, path)
    head = f"Line {where[0]}: `{where[1]}` raised {err}" if where else f"raised {err}"
    if MODE == "full" and where:
        tip = advise(str(exc), where[1], source)
        if tip:
            return f"{head} {tip}"
    # Fall back to their enrich() on the same text, so nothing they handle is lost.
    base = agent.enrich(f"raised {err}")
    return head + base[len(f"raised {err}"):]


def grade(source, level):
    """Their agent.grade(), identical except where marked V2: the feedback for simulator errors."""
    W = agent.WEIGHTS
    parts = dict(parses=False, rules=False, runs=False, correct=False)
    if not source.strip():
        return 0.0, parts, ("No code came back. Reply with one python code block containing the "
                            "kernel and nothing else.")
    try:
        compile(source, "<candidate>", "exec")
        parts["parses"] = True
    except SyntaxError as e:
        return (W["parses"] * 0, parts,
                f"The code does not parse: {e.msg} on line {e.lineno}. Send one complete python "
                f"code block.")

    violations = nkibench.check_rules(source, level)
    if violations:
        extra = ""
        if any("no function named" in v for v in violations):
            import inspect
            ref = nkibench.LEVELS[level]["ref"]
            args = ", ".join(inspect.signature(ref).parameters)
            extra = (f" Start the function with exactly this line:  "
                     f"def {nkibench.LEVELS[level]['entry']}({args}):  "
                     f"and put @nki.jit on the line above it.")
        return (sum(W[k] for k, v in parts.items() if v), parts,
                "Rule violations, which score zero however fast the kernel is. Fix exactly "
                "these: " + " ".join(violations) + extra)
    parts["rules"] = True

    spec = nkibench.LEVELS[level]
    # Task 05: one file per process, so agents sharing a machine cannot overwrite each other's
    # candidate between writing and loading it. Their agent.py uses the fixed path above.
    path = f"/tmp/_agent_{os.getpid()}_level{level}.py"
    with open(path, "w") as f:
        f.write(source)
    try:
        kernel = nkibench.load_kernel(path, spec["entry"])
    except ModuleNotFoundError as e:
        try:
            import nki  # noqa: F401
            sdk_present = True
        except ImportError:
            sdk_present = False
        if not sdk_present:
            raise SystemExit(f"cannot import {e.name!r}, so no NKI kernel can be loaded here.") from e
        return (sum(W[k] for k, v in parts.items() if v), parts,
                f"There is no module named {e.name!r}. The only imports that exist are: "
                f"`import nki`, `import nki.language as nl`, and `import nki.isa as nisa`. "
                f"Use exactly those three.")
    except Exception as e:
        return (sum(W[k] for k, v in parts.items() if v), parts,
                f"The file imports but {spec['entry']} could not be loaded: "
                f"{type(e).__name__}: {e}")

    failures, passed, intensity = [], 0, None
    for case in spec["shapes"]:
        args, _ = nkibench.make_inputs(case, level)
        before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
        want = spec["ref"](*args)
        try:
            got, counted = nkibench.simulate_and_count(kernel, args)
        except nkibench.NkiMissing as e:
            return (sum(W[k] for k, v in parts.items() if v), parts, f"CANNOT SIMULATE: {e}")
        except Exception as e:
            failures.append((nkibench.label(case, level), feedback(e, path, source)))  # V2
            continue
        parts["runs"] = True
        m = (nkibench.check_inputs_untouched(before, args)
             or nkibench.describe_mismatch(got, want)
             or nkibench.check_traffic_bar(level, counted, args, want))
        hazards = [w for w in counted.get("warnings", [])
                   if "incorrect results on hardware" in w]
        if hazards and not m:
            m = ("CORRECT ON CPU BUT WRONG ON HARDWARE: " + hazards[0]
                 + ". Fix that before anything else -- the simulator agrees with the reference here "
                   "and the device would not.")
        if m:
            failures.append((nkibench.label(case, level), m))
            continue
        passed += 1
        # Task 07: their grade() computes a MATMUL roofline for every level >= 3, so on level 8
        # (attention) and on our added levels the first shape that passes raised KeyError: 'M' and
        # killed the agent. Only matmul shapes have M, K and N.
        if level >= 3 and counted["bytes"] and all(d in case for d in ("M", "K", "N")):
            intensity = nkibench.roofline(
                nkibench.matmul_flops(case["M"], case["K"], case["N"]), counted["bytes"])

    if failures:
        lbl, first = failures[0]
        return (sum(W[k] for k, v in parts.items() if v)
                + W["correct"] * passed / len(spec["shapes"]), parts,
                f"{passed} of {len(spec['shapes'])} shapes passed. On {lbl}: {first}")

    parts["correct"] = True
    reward = sum(W.values())
    note = "Correct on every shape."
    if intensity:
        note += " " + nkibench.explain_roofline(intensity)
    return reward, parts, note


agent.grade = grade   # solve() looks grade up in agent's namespace at call time

if __name__ == "__main__":
    print(f"feedback v2, mode={MODE}")
    agent.main()
