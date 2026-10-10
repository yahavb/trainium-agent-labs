#!/usr/bin/env python3
"""
agent.py — the loop: a model writes an NKI kernel, the harness grades it, the reason goes back,
it tries again.

    controller  picks a level                          nkibench.py
    generator   the model writes a kernel             the shared gpt-oss endpoint
    checker     rules, then simulate, then roofline   nkibench.py
    loop        the failure becomes the next prompt   this file

Everything here runs on the CPU, so it does NOT need the Trainium device -- which is the point,
because the device is what Project 2 needs later for real timings, and a Neuron device cannot be
shared by two processes. The model therefore comes from the shared endpoint, not from a server on
your own chip.

    export GPTOSS_BASE_URL="https://..."
    python agent.py --level 1
    python agent.py --level 4 --rounds 6 --samples 4
    python agent.py --all
    python agent.py --offline --level 4        # no model; replays the reference kernel

Every attempt is appended to a JSONL file with its reward, so the log is the deliverable.
"""

import argparse
import json
import os
import re
import sys
import textwrap
import time
import traceback

import numpy as np

import nkibench

MODEL = os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-8B")

# The model writes to a hidden reasoning channel before it writes any answer. Measured on this
# endpoint: a coding task burned 900 tokens thinking and returned EMPTY content. See gptoss/README.
MIN_ANSWER_TOKENS = 2500

REASONING_KEYS = ("reasoning", "reasoning_content")


# ---------------------------------------------------------------- reward
#
# Graded, not pass/fail, so a near miss is distinguishable from nonsense and the loop has
# something to climb. Matches Project 1's shape: correctness dominates, and nothing else counts
# until the kernel is right.

WEIGHTS = dict(parses=0.1, rules=0.2, runs=0.2, correct=0.5)


# ---------------------------------------------------------------- locating the failure
#
# The checker used to hand back only the exception text: "cannot reshape array of size 32768 into
# shape (1,64)". True, and it never says WHICH of the model's lines did it. So the model rewrites
# something else and the same error returns -- levels 1, 3 and 4 each repeat one error on every run.
# "located" feedback quotes the model's own failing line back to it.
#
#   raw       the bare exception, nothing added        (the ablation's floor)
#   enriched  the exception plus the fix instruction   (the repo's original behaviour)
#   located   enriched, plus the failing line quoted   (the default)
#   directed  located, plus messages rewritten from what real runs showed the model doing:
#             the actual shapes in a failed copy, and where a misplaced name really lives

FEEDBACK_MODE = "located"
LOCATED = " The failing line is line "

# Each mode adds to the one before it, so "at least located" is a meaningful test.
#   directed2  directed, plus the real signature of the NKI function called on the failing line.
#              Measured on a seat: the model wrote nisa.tensor_scalar(dst=tile, data=0.5,
#              op0=nisa.multiply) -- a real function with the arguments in the wrong roles.
#   directed3  directed2, with the two messages behind the level 3 wall replaced (see directed()).
MODES = ("raw", "enriched", "located", "directed", "directed2", "directed3", "directed4")

# The model's failing line, set by explain() from directed4 on, so that a message can depend on what
# the line IS (a copy, an assignment) and name the variable on it.
FAILING_LINE = ""


def at_least(mode):
    return MODES.index(FEEDBACK_MODE) >= MODES.index(mode)

# Used only when the traceback has no frame inside the candidate file. (error pattern, pattern to
# look for in the source). `{0}` is the first group captured from the error.
GUESSES = [
    (r"cannot reshape", r"reshape"),
    (r"'MemoryRegion' object is not callable", r"nl\.(?:sbuf|psum|shared_hbm|hbm)\s*\("),
    (r"has no attribute '(\w+)'", r"\.{0}\b"),
    (r"got an unexpected keyword argument '(\w+)'", r"\b{0}\s*="),
]


def locate(exc, path, source):
    """(line number, line text) of the model's own line that raised, or None.

    Reads line text from `source`, never from linecache: every candidate is written to the same
    path, so linecache would quote a PREVIOUS candidate's line under this one's number.
    """
    lines = source.splitlines()
    hit, seen, e = None, set(), exc
    while e is not None and id(e) not in seen and hit is None:
        seen.add(id(e))
        for frame, lineno in traceback.walk_tb(e.__traceback__):
            if frame.f_code.co_filename == path and 1 <= lineno <= len(lines):
                hit = lineno                      # keep walking: the innermost frame wins
        e = e.__cause__ or e.__context__
    if hit is None:
        # No frame of ours in the traceback. Guess from the error text, but ONLY when exactly one
        # line matches: pointing at the wrong line is worse than pointing at none.
        text = str(exc)
        for err_pat, src_pat in GUESSES:
            m = re.search(err_pat, text)
            if not m:
                continue
            pat = re.compile(src_pat.format(*[re.escape(g) for g in m.groups()]))
            matches = [i for i, ln in enumerate(lines, 1)
                       if pat.search(ln) and not ln.lstrip().startswith("#")]
            if len(matches) == 1:
                hit = matches[0]
            break
    if hit is None or not lines[hit - 1].strip():
        return None
    return hit, lines[hit - 1].strip()[:160]


def explain(exc, path, source, prefix="raised ", add_fix=True):
    """Turn an exception from the candidate into feedback, at the current FEEDBACK_MODE."""
    global FAILING_LINE
    text = f"{prefix}{type(exc).__name__}: {exc}"
    if FEEDBACK_MODE == "raw":
        return text
    where = locate(exc, path, source) if at_least("located") else None
    FAILING_LINE = where[1] if where and at_least("directed4") else ""
    if add_fix:
        text = enrich(text)
    FAILING_LINE = ""
    if at_least("located"):
        if where:
            # A size mismatch can be fixed at either end, so do not point at the copy alone.
            close = ("Change that line, or the line that allocates its destination, so the two "
                     "shapes match."
                     if at_least("directed") and "same number of elements" in text
                     else "That is the line to change.")
            text += f"{LOCATED}{where[0]} of your kernel: `{where[1]}`. {close}"
            if at_least("directed2"):
                text += called_names_on(where[1]) + signatures_on(where[1])
        elif not explain.warned:
            explain.warned = True
            frames = [f"{os.path.basename(f.f_code.co_filename)}:{n}"
                      for f, n in traceback.walk_tb(exc.__traceback__)]
            print(f"    (could not locate the failing line for {type(exc).__name__}; the traceback "
                  f"passes through {' > '.join(frames[-6:]) or 'no frames'}. Shown once per run.)")
    return text


explain.warned = False


def called_names_on(line):
    """Names on a line that are CALLED but are not functions, as a sentence, or ''.

    Seen on a seat: `data=nl.float32(0.5)` raised "'str' object is not callable" two rounds
    running. nl.float32 is the NAME of a dtype. Nothing in that error says so.
    """
    import importlib
    out = []
    for alias, name in dict.fromkeys(re.findall(r"\b(nl|nisa)\.(\w+)\s*\(", line)):
        try:
            mod = importlib.import_module("nki.language" if alias == "nl" else "nki.isa")
        except Exception:
            continue
        obj = getattr(mod, name, None)
        if obj is not None and not callable(obj):
            what = "the name of a dtype" if isinstance(obj, str) else "a value"
            out.append(f" `{alias}.{name}` is {what}, not a function, so it cannot be called. "
                       f"Remove the call: write the plain number where you wrote "
                       f"`{alias}.{name}(...)`.")
    return "".join(out)


def signatures_on(line):
    """The real signatures of the nl/nisa functions called on a line, as a sentence, or ''."""
    sigs = []
    for name in dict.fromkeys(re.findall(r"\b(?:nl|nisa)\.(\w+)\s*\(", line)):
        sig = real_signature(name)
        # real_signature names the module "isa" / "language"; the model writes nisa / nl.
        for long, short in (("isa.", "nisa."), ("language.", "nl.")):
            if sig.startswith(long):
                sig = short + sig[len(long):]
        if "(" in sig and sig not in sigs:
            sigs.append(sig if len(sig) <= 400 else sig[:397] + "...")
    if not sigs:
        return ""
    return (" For reference, the real signature" + (" is " if len(sigs) == 1 else "s are ")
            + " and ".join(sigs[:2]) + ". Pass every argument by keyword, in those roles.")


def signature(feedback):
    """The failure with the quoted line removed, for the repeat detector and the ledger.

    The quoted line changes whenever the code does, so comparing whole messages would never see a
    repeat and the give-up rule would stop firing. Two attempts hitting the same error on different
    lines are the same failure.
    """
    return feedback.split(LOCATED)[0]


def grade(source, level):
    """Returns (reward, parts, feedback). Feedback is an INSTRUCTION, never just a verdict."""
    parts = dict(parses=False, rules=False, runs=False, correct=False)

    if not source.strip():
        return 0.0, parts, ("No code came back. Reply with one python code block containing the "
                            "kernel and nothing else.")
    try:
        compile(source, "<candidate>", "exec")
        parts["parses"] = True
    except SyntaxError as e:
        return (WEIGHTS["parses"] * 0, parts,
                f"The code does not parse: {e.msg} on line {e.lineno}. Send one complete python "
                f"code block.")

    violations = nkibench.check_rules(source, level)
    if violations:
        extra = ""
        if any("no function named" in v for v in violations):
            # Measured: this repeated 15 rounds running, because "there is no function named X"
            # never said what the function should look like. Hand over the exact line.
            import inspect
            ref = nkibench.LEVELS[level]["ref"]
            args = ", ".join(inspect.signature(ref).parameters)
            extra = (f" Start the function with exactly this line:  "
                     f"def {nkibench.LEVELS[level]['entry']}({args}):  "
                     f"and put @nki.jit on the line above it.")
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                "Rule violations, which score zero however fast the kernel is. Fix exactly "
                "these: " + " ".join(violations) + extra)
    parts["rules"] = True

    spec = nkibench.LEVELS[level]
    # One file per PROCESS. With a shared name, two runs in the same pod (a baseline and an
    # experiment, say) overwrite each other's candidate between the write and the load.
    path = f"/tmp/_agent_{os.getpid()}_level{level}.py"
    with open(path, "w") as f:
        f.write(source)
    try:
        kernel = nkibench.load_kernel(path, spec["entry"])
    except ModuleNotFoundError as e:
        # Two very different causes, and conflating them aborted a whole run: either the SDK is
        # absent (an environment problem the model cannot fix), or the model imported a module it
        # invented -- observed: `import nki.nl`. Tell them apart by asking whether nki itself is
        # importable.
        try:
            import nki  # noqa: F401
            sdk_present = True
        except ImportError:
            sdk_present = False
        if not sdk_present:
            raise SystemExit(
                f"cannot import {e.name!r}, so no NKI kernel can be loaded here. Run this where "
                f"the Neuron SDK exists -- k8s/kernel-agent-job.yaml does that -- or use "
                f"--offline to exercise the loop without a kernel ever running.") from e
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                f"There is no module named {e.name!r}. The only imports that exist are: "
                f"`import nki`, `import nki.language as nl`, and `import nki.isa as nisa`. "
                f"Use exactly those three.")
    except Exception as e:
        return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                explain(e, path, source,
                        prefix=f"The file imports but {spec['entry']} could not be loaded: ",
                        add_fix=False))

    failures, passed, intensity = [], 0, None
    for case in spec["shapes"]:
        args, _ = nkibench.make_inputs(case, level)
        before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
        want = spec["ref"](*args)
        try:
            got, counted = nkibench.simulate_and_count(kernel, args)
        except nkibench.NkiMissing as e:
            return (sum(WEIGHTS[k] for k, v in parts.items() if v), parts,
                    f"CANNOT SIMULATE: {e}")
        except Exception as e:
            failures.append((nkibench.label(case, level), explain(e, path, source)))
            continue
        parts["runs"] = True
        m = (nkibench.check_inputs_untouched(before, args)
             or nkibench.describe_mismatch(got, want)
             or nkibench.check_traffic_bar(level, counted, args, want))
        if m and at_least("directed3"):
            m = directed_mismatch(m, got, want, args, spec)
        # A simulator warning about a hardware-correctness hazard counts as a failure even when the
        # numbers happen to match on CPU: the kernel would be wrong on the device.
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
        if level >= 3 and counted["bytes"]:
            intensity = nkibench.roofline(
                nkibench.matmul_flops(case["M"], case["K"], case["N"]), counted["bytes"])

    if failures:
        lbl, first = failures[0]
        return (sum(WEIGHTS[k] for k, v in parts.items() if v)
                + WEIGHTS["correct"] * passed / len(spec["shapes"]), parts,
                f"{passed} of {len(spec['shapes'])} shapes passed. On {lbl}: {first}")

    parts["correct"] = True
    reward = sum(WEIGHTS.values())
    note = "Correct on every shape."
    if intensity:
        note += " " + nkibench.explain_roofline(intensity)
    return reward, parts, note


# ---------------------------------------------------------------- prompting

# Every name here appears in the three shipped tutorial kernels, so none of it is invented. The
# model does not know this API and guesses plausible names -- nl.scalar, nl.value, nl.dot,
# tile.mean -- none of which exist. A short card of what IS real costs ~200 tokens and is
# documentation rather than the answer.
API_CARD = """Available NKI functions:

  @nki.jit                                  decorate the entry point
  nl.ndarray(shape, dtype=..., buffer=b)    allocate; b is nl.sbuf, nl.psum or nl.shared_hbm
  nl.affine_range(n)                        the loop
  nl.sum(view, axis=[i, j])                 reduce; axis is a list
  nl.float32, nl.bfloat16                   dtypes
  nisa.dma_copy(dst=, src=)                 move data between HBM and SBUF
  nisa.nc_matmul(dst=, stationary=, moving=)   matmul into a PSUM tile
  nisa.tensor_copy(dst=, src=)              copy, e.g. PSUM to SBUF
  nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=0.5)   scale by a constant
  tile.ap([[stride, count], ...])           a strided view, for reductions

nisa.nc_matmul has strict memory rules: dst must live in nl.psum, while stationary and moving must
both live in nl.sbuf. So the pattern is: dma_copy both operands from HBM into sbuf tiles, allocate a
psum tile for the result, call nc_matmul(dst=psum_tile, stationary=..., moving=...), then
tensor_copy from psum into an sbuf tile, then dma_copy that out to the shared_hbm output.
The left operand arrives already transposed, with K on the partition axis.

Slice tiles with ranges, e.g. a[0:128, 0:64]. A complete kernel looks like this:

import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def copy_kernel(a):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    tile = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=a)
    nisa.dma_copy(dst=out, src=tile)
    return out

"""


def available_names(dotted):
    """Turn 'no attribute X' into 'here are the real ones'.

    Measured: told only `module 'nki.language' has no attribute 'value'`, the model guessed
    another invented name every round -- value, scalar, sbuf_scalar, dot. A verdict names the
    mistake and never the fix, so list what actually exists and let it choose.
    """
    import difflib
    import importlib
    mod_name, _, attr = dotted.rpartition(".")
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        return ""
    names = [n for n in dir(mod) if not n.startswith("_")]
    close = difflib.get_close_matches(attr, names, n=6, cutoff=0.4)
    if close:
        return (f" `{mod_name}` has no `{attr}`. The closest real names are: "
                f"{', '.join(close)}. Pick one of those or use a different approach.")
    return (f" `{mod_name}` has no `{attr}`, and nothing similar exists. Its real names include: "
            f"{', '.join(sorted(names)[:25])}.")


def real_signature(func_name):
    """The actual signature of an NKI function, for when the model invents arguments."""
    import inspect
    for mod_name in ("nki.language", "nki.isa", "nki"):
        try:
            mod = __import__(mod_name, fromlist=["x"])
        except Exception:
            continue
        fn = getattr(mod, func_name, None)
        if fn is None:
            continue
        try:
            return f"{mod_name.split('.')[-1]}.{func_name}{inspect.signature(fn)}"
        except (TypeError, ValueError):
            return f"{mod_name.split('.')[-1]}.{func_name}"
    return ""


ALIAS = {"nki.language": "nl", "nki.isa": "nisa", "nki": "nki"}


def _shape(x):
    try:
        return tuple(int(v) for v in x)
    except Exception:
        return None


def directed(error_text):
    """Feedback written from what the model was SEEN to do on the chip, or None.

    Both cases were measured on a seat, level 1, with the failing line already quoted back:

    * `dma_copy requires src and dst to have the same number of elements, got src=4, dst=16384`.
      Four rounds, no progress. The message gave element counts and an unrelated 128x512 example.
      The model needs the two SHAPES, which only the checker can see.
    * `module 'nki.isa' has no attribute 'multiply'`. The old message listed the first 25 names
      of nki.isa alphabetically. The name exists -- in nki.language. Say that.
    """
    if at_least("directed4"):
        # Seen on a seat, level 3, the first directed3 run (session 20261010-193145). directed3 got
        # the model past both old level 3 walls in one round each, to a kernel that is correct but
        # for one line:
        #     nisa.tensor_copy(dst=sbuf_lhsT, src=psum, engine=nisa.engine.scalar)
        # psum is (64, 512); sbuf_lhsT was allocated for the (128, 64) input. The simulator raises
        # "value array of shape (32768,) could not be broadcast to indexing result of shape (8192,)"
        # and the repo's advice for that text is about assignments in a loop: "index the
        # destination to match, e.g. out[i*128:(i+1)*128, :] = tile". There is no assignment and
        # no loop. Four rounds, seven different kernels, the line never changed.
        m = re.search(r"value array of shape \((\d+),?\) could not be broadcast to "
                      r"indexing result of shape \((\d+),?\)", error_text)
        call = re.search(r"\b(?:nisa|nl)\.(\w+)\(", FAILING_LINE)
        if m and call:
            dst = re.search(r"\bdst\s*=\s*([A-Za-z_]\w*)", FAILING_LINE)
            name = f"`{dst.group(1)}`" if dst else "the dst"
            return (error_text + f" The failing line is a call to nisa.{call.group(1)}, not an "
                    f"assignment. Its result has {m.group(1)} elements and its destination "
                    f"{name} holds {m.group(2)}. The destination and what goes into it must "
                    f"have IDENTICAL shapes. {name[0].upper() + name[1:]} was allocated for "
                    f"something else and is being reused here: do not reuse it. Allocate a new "
                    f"tile for this result, with exactly the shape of what is written into it, "
                    f"and use that as dst. Then check that every later line that reads this "
                    f"result, and the array you return, has that same shape.")
    if at_least("directed3"):
        # THE LEVEL 3 WALL. The repo records "cannot reshape array of size 32768 into shape (1,64)"
        # on every run and reads it as "it reshapes instead of slicing". Seen on a seat, in order:
        #   round 0  psum = nl.ndarray(shape=lhsT.shape[1:], ...)  -> "must have at least 2
        #            dimensions ... give a length-N vector the shape (1, N)"
        #   round 1  psum = nl.ndarray(shape=(1, lhsT.shape[1]), ...)   as told
        #            nisa.nc_matmul(dst=psum, ...) -> "cannot reshape array of size 32768 into shape
        #            (1,64) Do not reshape." -- four rounds running
        # The model never calls reshape. The simulator does, inside nc_matmul, to put a (64, 512)
        # result into the (1, 64) tile the FIRST message told the model to make. One message
        # steered it to the wrong shape and the next misnamed the consequence.
        m = re.search(r"cannot reshape array of size (\d+) into shape \(([\d, ]+)\)", error_text)
        if m:
            dims = tuple(int(x) for x in m.group(2).split(",") if x.strip())
            return (error_text + f" Your code does not call reshape, and removing one will not "
                    f"help. This is raised inside the NKI call on the failing line: its result has "
                    f"{m.group(1)} elements and the destination tile you gave it has shape {dims}, "
                    f"which holds {int(np.prod(dims))}. The destination has the wrong shape. For "
                    f"nisa.nc_matmul with stationary of shape (K, M) and moving of shape (K, N), "
                    f"dst must have shape (M, N). Allocate the destination with the full shape of "
                    f"the result.")
        if "must have at least 2 dimensions" in error_text:
            return (error_text + " The shape on the failing line has one dimension and a tile "
                    "needs two. Putting a 1 in front is only right when the data really is a "
                    "single row. Otherwise give the tile the full shape of the data it will hold: "
                    "a tile that receives the result of nisa.nc_matmul needs shape (M, N), where "
                    "stationary is (K, M) and moving is (K, N); a tile that receives a copy needs "
                    "exactly the shape of what is copied into it.")
        # FIVE MORE WALLS, all from one log: the five-repeat `located` run on a seat (session
        # 20261010-163428, 340 attempts). Each is quoted as the model met it.
        #
        # 1. Levels 2 and 3, a two-step cycle that ran 7 rounds on each:
        #      tile = nl.ndarray((128, 128)) for a (32, 12) input -> "holds 16384, you copied 384"
        #      so it copied x[0:128, 0:128]   -> "index range [0, 127] exceed dimension size of 12"
        #      so it went back to the first kernel.
        #    Each message is true and each sends the model to the other. The second one has to say
        #    which side must give way: the tile shrinks to the data, the slice never grows.
        m = re.search(r"Out-of-bound access for tensor .*? on dimension (\d+): "
                      r"index range \[(\d+), (\d+)\] exceed dimension size of (\d+)", error_text)
        if m:
            dim, hi, size = m.group(1), int(m.group(3)), int(m.group(4))
            return (error_text + f" Dimension {dim} of that tensor has only {size} entries, so "
                    f"the widest slice it allows is 0:{size}, and you asked for 0:{hi + 1}. The "
                    f"slice cannot grow to fit a tile. The tile has to shrink to fit the data: "
                    f"take every bound from the tensor's own .shape, never a fixed number, and "
                    f"allocate the tile that receives it with exactly that same shape. A tile "
                    f"bigger than the data cannot be used either, because nisa.dma_copy needs "
                    f"identical shapes.")
        # 2. Level 2, four rounds unchanged, with no advice at all after the quoted line:
        #      tile = nl.ndarray((F1, F2)) ... tile[j, i] = tile[i, j]
        #      -> "index 3 exceed dimension size of 3"
        m = re.search(r"Out-of-bound access for tensor .*? on dimension (\d+): "
                      r"index (\d+) exceed dimension size of (\d+)", error_text)
        if m:
            dim, idx, size = m.group(1), int(m.group(2)), int(m.group(3))
            return (error_text + f" Dimension {dim} has {size} entries, numbered 0 to "
                    f"{size - 1}, and the failing line used {idx}. An index on that line runs "
                    f"over the range of a different dimension. Check which dimension each loop "
                    f"variable was sized for. If you are writing a result whose shape differs "
                    f"from the source's, for example with two dimensions swapped, it cannot be "
                    f"written back into the source tile: allocate a separate destination tile "
                    f"with the result's shape and write into that.")
        # 3. Level 3, four rounds unchanged, no advice:
        #      out = nl.ndarray(shape=lhsT.shape[1:], ...)   ... 0:out.shape[1]
        #      -> "IndexError: tuple index out of range"
        if "tuple index out of range" in error_text:
            return (error_text + " A .shape[...] on the failing line asks for a dimension that "
                    "array does not have. Find the array whose shape is being indexed and look at "
                    "the line that allocated it: a shape written as a slice of another shape, "
                    "such as x.shape[1:], has fewer entries than x.shape. Allocate that array "
                    "with every dimension written out. If it holds the result of nisa.nc_matmul "
                    "with stationary (K, M) and moving (K, N), its shape is (M, N).")
        # 4. Level 4, at 0.62, two rounds unchanged, no advice:
        #      nisa.nc_matmul(dst=psum_tile, stationary=sbuf_lhsT[start_K:end_K, :], moving=sbuf_rhs)
        #      -> "contraction dimension mismatch: stationary[0]=128 != moving[0]=256"
        m = re.search(r"contraction dimension mismatch: stationary\[0\]=(\d+) != moving\[0\]=(\d+)",
                      error_text)
        if m:
            return (error_text + f" The first dimension of both operands is K, the one being "
                    f"contracted, and the two must be equal in every call. Here stationary has "
                    f"{m.group(1)} rows and moving has {m.group(2)}: one operand was cut into a K "
                    f"chunk and the other was passed whole. Slice BOTH with the same K range, "
                    f"stationary=...[k0:k1, ...] and moving=...[k0:k1, ...].")
        # 5. Level 4. After a NaN verdict the model invented nisa.psum_init; the checker offered
        #    "the closest real names: gpsimd_engine, dma_engine"; the model called
        #    nisa.gpsimd_engine(psum, 0.0) -> "'engine' object is not callable", and then went
        #    round that loop twice more. The suggestion list offered things that are not functions.
        m = re.search(r"'(\w+)' object is not callable", error_text)
        if m and m.group(1) != "MemoryRegion":
            return (error_text + " The name called on the failing line is not a function. The "
                    "nisa.*_engine names are constants that say which engine runs an "
                    "instruction; they do nothing when called. Delete that line. No function "
                    "zeroes or initialises a tile and none is needed: a tile gets its contents "
                    "by being the dst of a copy or of a computation.")
    if "dma_copy requires src and dst to have the same number of elements" in error_text:
        dst, src = _shape(nkibench.LAST_DMA.get("dst")), _shape(nkibench.LAST_DMA.get("src"))
        if dst and src and at_least("directed2"):
            # Seen on a seat, level 3 under `directed`: one (128, 512) tile used for BOTH operands,
            # (128, 64) and (128, 512). Told "allocate it as (128, 64), or copy a slice of shape
            # (128, 512)", the model flipped between the two for six rounds, because the second
            # option is impossible when the source is the smaller one, and neither says the real
            # fix: one tile per tensor.
            if int(np.prod(src)) < int(np.prod(dst)):
                return (error_text + f" The destination has shape {dst} but the piece copied into "
                        f"it has shape {src}, which is smaller. nisa.dma_copy needs IDENTICAL "
                        f"shapes and cannot fill part of a tile. Allocate a separate tile for this "
                        f"copy with exactly the source's shape: nl.ndarray({src}, dtype=..., "
                        f"buffer=nl.sbuf). If one tile is being used for two different tensors, "
                        f"stop doing that: every tensor you load needs its own tile, allocated "
                        f"with that tensor's own shape.")
            if at_least("directed4"):
                # The older wording below says "allocate the destination as nl.ndarray(...,
                # buffer=nl.sbuf)". When the destination is the array the kernel returns, that is
                # an instruction to move the output out of HBM. Say the shape and nothing else.
                return (error_text + f" The destination has shape {dst} but the piece copied "
                        f"into it has shape {src}, which is larger. nisa.dma_copy needs IDENTICAL "
                        f"shapes. If the source is the whole result, the destination was "
                        f"allocated too small: allocate it with shape {src}, with every "
                        f"dimension written out, and keep its buffer as it is. Only if you meant "
                        f"to copy a part, slice the source down to {dst}.")
            return (error_text + f" The destination has shape {dst} but the piece copied into it "
                    f"has shape {src}, which is larger. nisa.dma_copy needs IDENTICAL shapes. "
                    f"Either allocate the destination as nl.ndarray({src}, dtype=..., "
                    f"buffer=nl.sbuf), or copy only a slice of the source whose shape is {dst}.")
        if dst and src:
            return (error_text + f" The destination you allocated has shape {dst} and the piece "
                    f"you are copying into it has shape {src}. nisa.dma_copy needs the two shapes "
                    f"to be IDENTICAL. Make them match: either allocate the destination as "
                    f"nl.ndarray({src}, dtype=..., buffer=nl.sbuf), or copy a slice whose shape "
                    f"is {dst}.")
    if at_least("directed2"):
        # Seen on a seat, level 4 under `directed`: the model chunked K correctly and reached 0.75
        # (2 of 4 shapes), then hit this with no advice at all beyond the quoted line.
        m = re.search(r"Matmul (stationary|moving) free dimension (\d+) exceeds", error_text)
        if m:
            sm, mm = nkibench.GEMM_STATIONARY_FMAX, nkibench.GEMM_MOVING_FMAX
            return (error_text + f" One nisa.nc_matmul call accepts at most {sm} on the stationary "
                    f"operand's free dimension (M) and at most {mm} on the moving operand's free "
                    f"dimension (N); you passed {m.group(2)} on the {m.group(1)} one. Keep your "
                    f"loop over K as it is and add loops over the output: split M into chunks of "
                    f"at most {sm} and N into chunks of at most {mm}. For each (M chunk, N chunk) "
                    f"pair, allocate a psum tile of that chunk's shape, run the K loop into it "
                    f"using only the matching columns of each operand -- stationary=...[k0:k1, "
                    f"m0:m1] and moving=...[k0:k1, n0:n1] -- then copy that result to its own "
                    f"slice of the output, out[m0:m1, n0:n1].")
        # Seen on a seat, level 1 under `directed`, one per round, none with usable advice:
        #   tensor_scalar() missing 1 required positional argument: 'operand0'
        #   'float' object has no attribute 'shape'      (old advice: "use the nl/nisa functions",
        #                                                 so it wrapped the number in nl.float32())
        m = re.search(r"(\w+)\(\) missing \d+ required (?:positional |keyword-only )?arguments?: "
                      r"(.+)$", error_text)
        if m:
            return (error_text + f" `{m.group(1)}` was called without {m.group(2)}. Add "
                    f"{'it' if ' and ' not in m.group(2) and ',' not in m.group(2) else 'them'} "
                    f"by keyword and leave the other arguments as they are.")
        m = re.search(r"'(float|int)' object has no attribute '(\w+)'", error_text)
        if m:
            return (error_text + f" A plain number was passed where a tile is required. On that "
                    f"line, the argument that takes a tile must be given a tile you allocated "
                    f"with nl.ndarray, and the plain number belongs in the argument that takes a "
                    f"constant. Do not wrap the number in anything.")
    m = re.search(r"module '([\w.]+)' has no attribute '(\w+)'", error_text)
    if m:
        import difflib
        import importlib
        mod_name, attr = m.groups()
        here = ALIAS.get(mod_name, mod_name)
        found, pool = [], {}
        for other in ("nki.language", "nki.isa"):
            try:
                mod = importlib.import_module(other)
            except Exception:
                continue
            if other != mod_name and hasattr(mod, attr):
                found.append(f"{ALIAS[other]}.{attr}")
            for n in dir(mod):
                if n.startswith("_"):
                    continue
                # From directed3 on, only suggest things that can be called. Seen on a seat: for
                # the invented nisa.psum_init the list offered gpsimd_engine and dma_engine, which
                # are constants; the model called one and lost three more rounds to it.
                if at_least("directed3") and not callable(getattr(mod, n, None)):
                    continue
                pool.setdefault(n, f"{ALIAS[other]}.{n}")
        if found:
            return (error_text + f" `{attr}` is not in `{here}`. It exists as `{found[0]}`. "
                    f"Write `{found[0]}` instead of `{here}.{attr}`.")
        if at_least("directed3") and re.search(r"init|zero|fill|clear|reset", attr, re.I):
            return (error_text + f" Nothing called `{attr}` exists in nl or nisa, and nothing "
                    f"like it is needed. No function zeroes or initialises a tile: a tile gets "
                    f"its contents by being the dst of a copy or of a computation. Delete that "
                    f"line.")
        close = difflib.get_close_matches(attr, list(pool), n=5, cutoff=0.5)
        if close:
            return (error_text + f" Nothing called `{attr}` exists in nl or nisa. The closest real "
                    f"names are: {', '.join(pool[c] for c in close)}. Use one of those.")
    return None


def moved_values(got, want, args, spec):
    """For a task that only MOVES values: where each one belongs, read off this test case, or "".

    Seen on a seat, level 2 (session 20261010-193145): once the kernel ran, the verdict was
    "NUMERICAL MISMATCH ... most elements are wrong, so this is the core arithmetic or the operand
    layout". The model sent back the identical kernel four times and the level was stopped. There
    is no arithmetic in a transpose. The checker holds the input and the expected output, so it
    can say which input element belongs at which output position, and which one the kernel put
    there. Applies only when the expected output is a rearrangement of one input.
    """
    try:
        want = np.asarray(want)
        got = np.asarray(got)
        src = next((np.asarray(a) for a in args if isinstance(a, np.ndarray)
                    and a.size == want.size and a.ndim == 2), None)
        if src is None or want.ndim != 2 or got.shape != want.shape:
            return ""
        key = lambda v: np.float32(v).tobytes()
        index = {}
        for pos in np.ndindex(src.shape):
            index.setdefault(key(src[pos]), []).append(pos)
        if any(len(v) != 1 for v in index.values()):
            return ""                                   # repeated values: sources are ambiguous
        origin = {pos: index.get(key(want[pos])) for pos in np.ndindex(want.shape)}
        if any(v is None for v in origin.values()):
            return ""                                   # not a pure rearrangement
        import inspect
        name = next(iter(inspect.signature(spec["ref"]).parameters))
        moved = [pos for pos in np.ndindex(want.shape) if origin[pos][0] != pos]
        if not moved:
            return ""
        show = moved[:4] + moved[len(moved) // 2:len(moved) // 2 + 2]
        fmt = lambda p: "[" + ", ".join(str(int(i)) for i in p) + "]"
        right = "; ".join(f"out{fmt(p)} = {name}{fmt(origin[p][0])}" for p in show)
        yours = []
        for p in show:
            g = index.get(key(got[p]))
            yours.append(f"out{fmt(p)} holds {name}{fmt(g[0])}" if g
                         else f"out{fmt(p)} holds a value that is not in {name}")
        same_row = all(origin[p][0][0] == p[0] for p in origin)
        text = (f" Nothing is computed in this task: the correct output holds exactly the values "
                f"of `{name}`, moved. On this test case ({name} has shape {tuple(src.shape)}) the "
                f"correct positions are: {right}. Your kernel gave: {'; '.join(yours)}.")
        if same_row:
            text += (" In the correct output every value stays in its own row; only its position "
                     "along the second dimension changes.")
        return text
    except Exception:
        return ""


def directed_mismatch(message, got, want=None, args=None, spec=None):
    """A wrong-answer verdict rewritten from what the output actually contains, or unchanged.

    Seen on a seat, level 4 (session 20261010-163428): the model deleted the copy into the output
    and returned an array nothing had written. The verdict was "NON-FINITE OUTPUT: 65536 NaN ...
    Usually an uninitialised PSUM or SBUF tile being read". Every element was NaN, which the
    checker can see and did not say. The model looked for a way to initialise a tile, invented
    nisa.psum_init, and fell from 0.62 to 0.30 for the rest of the run.
    """
    if (message.startswith("NUMERICAL MISMATCH") and at_least("directed4")
            and want is not None and args is not None and spec is not None):
        return message + moved_values(got, want, args, spec)
    if not message.startswith("NON-FINITE OUTPUT"):
        return message
    try:
        arr = np.asarray(got, np.float64)
        bad, total = int((~np.isfinite(arr)).sum()), int(arr.size)
    except Exception:
        return message
    head = message.split(" Usually ")[0]
    if bad == total:
        return (head + f" Every one of the {total} output elements is non-finite, which means the "
                f"array you return was never written: no line copies a result into it. This is "
                f"not about initialising a tile, and no function for that exists. After the "
                f"computation, copy the result into the array you return with "
                f"nisa.dma_copy(dst=<that array>[...], src=<an sbuf tile>). A psum tile cannot "
                f"be copied to HBM directly: move it into an sbuf tile with nisa.tensor_copy "
                f"first.")
    return (head + f" {bad} of the {total} output elements were never written with a real "
            f"value; the rest are finite. Part of the output is being skipped: check that your "
            f"loops cover every row and column of the output and that each chunk is copied to "
            f"its own slice of it.")


def enrich(error_text):
    """Add the real names when the failure is an invented API call."""
    if at_least("directed"):
        better = directed(error_text)
        if better:
            return better
    if "'MemoryRegion' object is not callable" in error_text:
        return (error_text + " nl.sbuf, nl.psum and nl.shared_hbm are memory regions, not "
                "functions. Do not call them. Allocate with "
                "nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf) and pass the region as the "
                "buffer= argument.")
    m = re.search(r"(\w+)\(\) got an unexpected keyword argument '(\w+)'", error_text)
    if m:
        sig = real_signature(m.group(1))
        return (error_text + f" Remove the `{m.group(2)}=` argument."
                + (f" The real signature is {sig}." if sig else ""))
    if "unsupported operand type(s) for" in error_text and "NkiTensor" in error_text:
        return (error_text + " A tile is not a number, so Python operators like += do not work on "
                "one. Accumulate by allocating a PSUM tile with "
                "nl.ndarray(shape, nl.float32, buffer=nl.psum) and letting nisa.nc_matmul add into "
                "it across the loop, or combine two tiles with a nisa op rather than a Python "
                "operator.")
    m = re.search(r"dma_copy requires src and dst to have the same number of elements, "
                  r"got src=(\d+), dst=(\d+)", error_text)
    if m:
        src, dst = int(m.group(1)), int(m.group(2))
        return (error_text + f" The tile you allocated holds {dst} elements but you copied {src} "
                f"into it. nisa.dma_copy does not slice or broadcast: allocate the destination with "
                f"EXACTLY the shape of the slice you are moving. If you want a 128x512 piece of a "
                f"bigger tensor, write "
                f"t = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf) and then "
                f"nisa.dma_copy(dst=t, src=a[0:128, 0:512]) -- the slice on the right must have the "
                f"same shape as the tile on the left.")
    m = re.search(r"dma_copy (\w+) partition dimension (\d+) exceeds maximum (\d+)", error_text)
    if m:
        which, got, mx = m.group(1), int(m.group(2)), int(m.group(3))
        return (error_text + f" A tile may have at most {mx} rows, and you asked for {got}. Do not "
                f"allocate one tile for the whole tensor: loop over the partition dimension in "
                f"chunks of at most {mx} with nl.affine_range, allocate the tile inside the loop with "
                f"the chunk's own size, and copy one chunk at a time, e.g. "
                f"src=a[i*{mx}:(i+1)*{mx}, :]. If a dimension is already {mx} or smaller, use it "
                f"whole -- do NOT pad it up to {mx}, that reads past the end of the tensor. The same "
                f"applies to where you write the result back.")
    m = re.search(r"value array of shape \((\d+),?\) could not be broadcast to "
                  r"indexing result of shape \((\d+),?\)", error_text)
    if m:
        val, dst = int(m.group(1)), int(m.group(2))
        return (error_text + f" You assigned {val} elements into a slice that holds {dst}. Assignment "
                f"does not reshape or broadcast either: the slice on the left and the value on the "
                f"right must have the SAME shape. If the value is bigger, you are writing a whole tile "
                f"where a slice belongs -- index the destination to match, e.g. "
                f"out[i*128:(i+1)*128, :] = tile. If it is smaller, you are looping over the wrong "
                f"dimension.")
    m = re.search(r"Out-of-bound access for tensor .*? on dimension (\d+): "
                  r"index range \[(\d+), (\d+)\] exceed dimension size of (\d+)",
                  error_text)
    if m:
        dim, hi, size = m.group(1), int(m.group(3)), int(m.group(4))
        return (error_text + f" You indexed up to {hi} on dimension {dim}, which is only {size} "
                f"long. Tile limits are a MAXIMUM, not a target. Derive every bound from the tensor's "
                f"own shape -- use min(limit, size) and let the final chunk be partial -- rather than "
                f"writing a fixed number. Note the two limits differ: the partition dimension (first) "
                f"allows at most 128, the free dimension allows more.")
    m = re.search(r"Matmul contraction dimension (\d+) exceeds pmax=(\d+)", error_text)
    if m:
        k, mx = int(m.group(1)), int(m.group(2))
        return (error_text + f" The contraction dimension K is {k} and one nc_matmul can only "
                f"contract {mx}. Split K into chunks of {mx} and accumulate: allocate ONE psum tile "
                f"OUTSIDE the K loop, call nisa.nc_matmul into that same psum tile once per chunk so "
                f"the partial products add up there, and only after the loop copy it out with "
                f"nisa.tensor_copy. Do not allocate a new psum tile per chunk and do not write partial "
                f"results to HBM.")
    m = re.search(r"(\w+) (?:dst|src)? ?must be in \['sbuf', 'psum'\], got shared_hbm", error_text)
    if m:
        return (error_text + f" `nisa.{m.group(1)}` only moves data between on-chip buffers, sbuf and "
                f"psum. To reach HBM -- the tensor you allocated with buffer=nl.shared_hbm and will "
                f"return -- use nisa.dma_copy instead. The usual sequence is nc_matmul into psum, "
                f"tensor_copy psum to sbuf, then dma_copy sbuf to the shared_hbm output.")
    m = re.search(r"(\w+) must be in \['(\w+)'\], got (\w+)", error_text)
    if m:
        which, needed, got = m.groups()
        place = {"psum": "nl.psum", "sbuf": "nl.sbuf"}.get(needed, needed)
        return (error_text + f" Allocate the `{which}` tile with buffer={place} instead of "
                f"nl.{got}. For nisa.nc_matmul: dst must be in nl.psum, and stationary and moving "
                f"must both be in nl.sbuf. Copy between them with nisa.tensor_copy.")
    if "got multiple values for argument" in error_text:
        return (error_text + " Pass every argument by keyword, e.g. "
                "nisa.nc_matmul(dst=..., stationary=..., moving=...), so none is bound twice.")
    if "must have at least 2 dimensions" in error_text:
        return (error_text + " Every SBUF and PSUM tile needs two dimensions: a partition dimension "
                "first, then a free dimension. A 1-D tile is not allowed, so write "
                "nl.ndarray((rows, cols), ...) and give a length-N vector the shape (1, N) or "
                "(N, 1) depending on which axis you are reducing over.")
    if "cannot reshape array of size" in error_text:
        return (error_text + " Do not reshape. Work with the shapes you were given and slice "
                "them into tiles, e.g. src=a[0:128, 0:64].")
    m = re.search(r"module '([\w.]+)' has no attribute '(\w+)'", error_text)
    if m:
        return error_text + available_names(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"'(\w+)' object has no attribute '(\w+)'", error_text)
    if m:
        return (error_text + f" A {m.group(1)} is not a numpy array, so it has no "
                f"`{m.group(2)}`. Use the nl/nisa functions instead.")
    return error_text

def first_prompt(level, terse=0):
    """Deliberately short, and it does NOT list the rules.

    Measured twice in this repo: hand a model an enumerated list of prohibitions and it audits
    itself against each one and returns nothing, while a bigger budget only buys more thinking.
    So the rules live in the checker. Generate freely, let the checker object, then send back one
    named change.
    """
    s = nkibench.LEVELS[level]
    import inspect
    if terse >= 2:
        # Last resort. Measured on this endpoint: one-sentence prompts answered in 300-700
        # tokens while every structured, rule-carrying prompt spiralled.
        return (f"Write a Python function `{s['entry']}` decorated with @nki.jit that computes "
                f"the same thing as this, using nki.language as nl and nki.isa as nisa:\n\n"
                f"{inspect.getsource(s['ref'])}\n"
                f"Reply with one python code block.")
    if terse >= 1:
        # The matmul memory rules are the substance of levels 3 and 4, and the short prompt has to
        # carry them: measured, the agent cycled between "dst must be in ['psum']" and "moving must
        # be in ['sbuf']" because nothing told it where the operands live.
        mm = ("nisa.nc_matmul(dst=, stationary=, moving=) needs dst in nl.psum and both operands "
              "in nl.sbuf. So: dma_copy the operands HBM->sbuf, allocate a psum tile, nc_matmul "
              "into it, tensor_copy psum->sbuf, then dma_copy sbuf->the shared_hbm output you "
              "return. The left operand is already transposed, with K on the partition axis.\n"
              if level >= 3 else "")
        return (f"Write an AWS Neuron NKI kernel: a function `{s['entry']}` decorated with "
                f"@nki.jit that computes what this reference computes.\n\n"
                f"{inspect.getsource(s['ref'])}\n"
                f"Allocate with nl.ndarray(shape, dtype=..., buffer=nl.sbuf), move data with "
                f"nisa.dma_copy(dst=, src=), loop with nl.affine_range(n). A tile's partition "
                f"dimension is at most {nkibench.PMAX}.\n{mm}\n"
                f"Reply with one python code block.")
    return (
        f"Write an AWS Neuron NKI kernel.\n\n"
        f"Operation: {s['op']}\n"
        f"Entry point: a function named `{s['entry']}`, decorated with `@nki.jit`.\n"
        f"It must compute exactly what this NumPy reference computes:\n\n"
        f"{inspect.getsource(s['ref'])}\n"
        f"Hardware limits: a tile's partition dimension is at most {nkibench.PMAX}. For matmul, "
        f"the stationary free dimension is at most {nkibench.GEMM_STATIONARY_FMAX} and the "
        f"moving free dimension at most {nkibench.GEMM_MOVING_FMAX}.\n\n"
        f"Import nki, nki.language as nl, and nki.isa as nisa.\n\n{API_CARD}\n\n"
        f"Reply with ONE python code block containing the imports and the function. No prose.")


def repair_prompt(level, source, feedback):
    """One named change, and the previous code. No rules list, no reference re-sent.

    The lesson this whole repo keeps re-learning: feeding a verifier's report back verbatim
    reproduces the same mistake, because a report says what is wrong and never what to do.
    """
    return (
        f"This NKI kernel for {nkibench.LEVELS[level]['op']} is not right yet.\n\n"
        f"```python\n{source}\n```\n\n"
        f"A checker reports:\n{feedback}\n\n"
        f"Change exactly what the checker names and keep everything else identical. Reply with "
        f"ONE python code block.")


def prompt_shape(prompt, kind, level, terse=0, code="", feedback="", ledger=""):
    """Where this prompt's characters went, for the token-budget instrumentation.

    reference = the NumPy reference, docs = the API card, code = the previous kernel,
    feedback = the checker's message, ledger = the list of failed attempts,
    instructions = everything else (the task wording).
    """
    import inspect
    shape = dict(kind=kind, total=len(prompt), reference=0, docs=0,
                 code=len(code), feedback=len(feedback), ledger=len(ledger))
    if kind == "first":
        shape["reference"] = len(inspect.getsource(nkibench.LEVELS[level]["ref"]))
        shape["docs"] = len(API_CARD) if terse == 0 else 0
    shape["instructions"] = max(0, shape["total"] - shape["reference"] - shape["docs"]
                                - shape["code"] - shape["feedback"] - shape["ledger"])
    return shape


# Measured on a seat, level 1, eight rounds of four samples at temperature 0.6: the four answers
# were byte-identical in seven rounds and two distinct in the eighth. So four samples of one prompt
# are one attempt paid for four times. The server runs four sequences at once at no extra
# wall-clock, so --diverse spends those slots on four differently worded requests instead.
# Each variant is a way of working, not a piece of the answer.
FIRST_VARIANTS = (
    "",
    "\n\nKeep it as simple as you can. When a whole tensor fits in one tile (first dimension at "
    "most 128), load it whole instead of looping over small pieces.",
    "\n\nOn every line that allocates a tile, add a comment giving its shape, and make sure every "
    "copy into or out of it has exactly that shape.",
    "\n\nUse as few different NKI functions as possible, and only ones named above.",
)
REPAIR_VARIANTS = (
    "",
    "\n\nAfter fixing that, check every other line that uses the same function or the same tile "
    "for the same mistake, and fix those too.",
    "\n\nDo not patch this attempt. Write the kernel again from the start in the simplest form "
    "that could work, keeping only the parts the checker did not complain about.",
    "\n\nOn every line that allocates a tile, add a comment giving its shape, and make sure every "
    "copy into or out of it has exactly that shape.",
)


def variants(prompt, kind, n):
    """n prompts: the original, then the same request with a different way of working added."""
    extra = FIRST_VARIANTS if kind == "first" else REPAIR_VARIANTS
    return [prompt + extra[i % len(extra)] for i in range(n)]


CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_code(text):
    """Pull out the code, and return NOTHING rather than prose.

    The old fallback returned the whole reply whenever it contained "def ", so a numbered list
    or a sentence reached the compiler and produced "invalid decimal literal on line 2" -- a
    parse error that blamed the model for the extractor's mistake.
    """
    text = text or ""
    blocks = CODE_BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    # No fence: start at the first line that can legally begin a module and keep the rest.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*(import |from |@nki|def )", line):
            return "\n".join(lines[i:]).strip()
    return ""


# ---------------------------------------------------------------- the model

class Reply(str):
    """The answer text, carrying what the server reported about it (tokens, finish reason).

    A str subclass so everything that already treats a reply as text keeps working.
    """
    def __new__(cls, text, **meta):
        obj = super().__new__(cls, text)
        obj.meta = meta
        return obj


def ask(a, prompt):
    import httpx
    # enable_thinking=False matters. Qwen3 reasons before answering, and with thinking on it
    # spent the whole budget there: the first cluster run returned "No code came back" at 54.7s
    # over and over, plus truncated fragments (invalid decimal literal, unterminated string).
    # Keep prompt + answer inside the server's context, or the answer is silently cut off and
    # every parse error below is really a budget error. Repair prompts grow with the kernel.
    est_prompt = len(prompt) // 4
    budget = min(a.max_tokens, max(256, a.context - est_prompt - 64))
    if budget < a.max_tokens:
        print(f"    (prompt is ~{est_prompt} tokens, so the answer budget is capped at {budget} "
              f"to stay inside the {a.context}-token context)")
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=budget, temperature=0.6, top_p=0.95,
                chat_template_kwargs={"enable_thinking": a.think})
    r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body,
                   timeout=900, verify=False)
    if r.status_code != 200:
        raise SystemExit(f"the endpoint returned HTTP {r.status_code}:\n{r.text[:600]}")
    payload = r.json()
    ch = payload["choices"][0]
    msg = ch.get("message", {})
    reasoning = next((msg[k] for k in REASONING_KEYS if msg.get(k)), "")
    content = msg.get("content") or ""
    finish = ch.get("finish_reason")
    if finish == "length":
        # Do not let a budget problem look like a model failure.
        print(f"    (TRUNCATED: finish_reason=length after {len(content)} chars. The answer was "
              f"cut off, so any parse error below is the budget, not the model. Prompt is "
              f"{len(prompt)} chars; server context is the ceiling.)")
    if not content.strip() and reasoning:
        # The single most common surprise on this endpoint, so name it rather than reporting
        # an empty answer as a model failure.
        print(f"    (empty answer, {len(reasoning)} chars of hidden reasoning, "
              f"finish={ch.get('finish_reason')} — shorten the prompt rather than raising the "
              f"budget)")
    usage = payload.get("usage") or {}
    return Reply(content, prompt_tokens=usage.get("prompt_tokens"),
                 completion_tokens=usage.get("completion_tokens"),
                 finish=finish, reasoning_chars=len(reasoning))


def ask_parallel(a, prompt, n):
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        return [f.result() for f in [ex.submit(ask, a, prompt) for _ in range(n)]]


def ask_each(a, prompts):
    """One request per prompt, all at once."""
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        return [f.result() for f in [ex.submit(ask, a, p) for p in prompts]]


def offline_answers(level, n, rnd):
    """No model. Replays the shipped reference, preceded by a deliberately broken version, so the
    loop and the feedback path can be exercised with no endpoint. Never report a number."""
    ref = open(f"reference_level{level}.py").read()
    if rnd == 0:
        broken = ref.replace("@nki.jit", "", 1)
        return [f"```python\n{broken}\n```"] * n
    return [f"```python\n{ref}\n```"] * n


# ---------------------------------------------------------------- the loop

def solve(a, level, log):
    print(f"\n=========== level {level}: {nkibench.LEVELS[level]['op']} ===========")
    terse = a.terse
    prompt = first_prompt(level, terse)
    shape = prompt_shape(prompt, "first", level, terse)
    best = (0.0, None, "")
    tried, streak, seen = [], 0, {}
    latest = ("", "")
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        diverse = getattr(a, "diverse", False)
        prompts = (variants(prompt, shape["kind"], a.samples) if diverse
                   else [prompt] * a.samples)
        replies = (offline_answers(level, a.samples, rnd) if a.offline
                   else ask_each(a, prompts) if diverse
                   else ask_parallel(a, prompt, a.samples))
        gen_s = round(time.perf_counter() - t0, 1)
        graded = []
        for i, reply in enumerate(replies):
            src = extract_code(reply)
            reward, parts, feedback = grade(src, level)
            graded.append((reward, src, feedback, parts))
            meta = getattr(reply, "meta", {})
            log.write(json.dumps(dict(level=level, round=rnd, reward=reward, parts=parts,
                                      prompt_chars=len(prompts[i]), reply_chars=len(reply),
                                      variant=(i % len(REPAIR_VARIANTS)) if diverse else 0,
                                      diverse=diverse, commit=getattr(a, "commit", ""),
                                      code=src, feedback=feedback,
                                      session=getattr(a, "session", ""),
                                      run=getattr(a, "run", 0), sample=i, model=a.model,
                                      feedback_mode=FEEDBACK_MODE,
                                      failure=signature(feedback),
                                      located=LOCATED in feedback,
                                      prompt_shape=shape, gen_seconds=gen_s,
                                      prompt_tokens=meta.get("prompt_tokens"),
                                      completion_tokens=meta.get("completion_tokens"),
                                      finish=meta.get("finish"),
                                      reasoning_chars=meta.get("reasoning_chars"))) + "\n")
        log.flush()
        if diverse:
            # Rewards often tie (four attempts at 0.30). Among equals, repair from the attempt whose
            # failure has not been seen before: a new failure is movement, the old one is a cycle.
            graded.sort(key=lambda g: (g[0], signature(g[2]) not in seen), reverse=True)
        else:
            graded.sort(key=lambda g: g[0], reverse=True)
        top = graded[0]
        if top[0] > best[0]:
            best = (top[0], top[1], top[2])
        # Repair the LATEST attempt, not the best one. Rebuilding from the best attempt with the
        # best attempt's feedback is a fixed point: once a round scores worse, the prompt stops
        # changing, and a greedy model then returns the same answer forever. Measured: level 2
        # stuck at 0.10 for four rounds while the prompt still carried the 0.50 code.
        if (top[1] or "").strip():
            latest = (top[1], top[2])
        sig = signature(top[2])
        same = sig == (tried[-1] if tried else None)
        if same:
            # Collapse. Fifteen identical multi-line blocks is noise, not information.
            print(f"round {rnd}: same failure again ({top[0]:.2f}, best so far {best[0]:.2f})")
        else:
            print(f"round {rnd}: this round {top[0]:.2f}  best so far {best[0]:.2f}  "
                  f"({time.perf_counter() - t0:.1f}s)")
            print(f"  {top[2][:400]}")
        if top[0] >= sum(WEIGHTS.values()) - 1e-9:
            print(f"  SOLVED on round {rnd}. {top[2]}")
            print("  ---------------- the kernel ----------------")
            print(textwrap.indent(top[1], "  "))
            print("  -------------------------------------------")
            return top[0], rnd + 1
        seen[sig] = seen.get(sig, 0) + 1
        streak = streak + 1 if same else 1
        if seen[sig] >= a.give_up_after:
            how = ("the identical failure %d rounds running" % streak if streak >= a.give_up_after
                   else "this failure for the %dth time, alternating with %d other(s)"
                        % (seen[sig], len(seen) - 1))
            print(f"  STOPPING this level: {how}. The agent is cycling between a fixed set of "
                  f"mistakes rather than converging, so more rounds will not help. Failures seen:")
            for f, n in sorted(seen.items(), key=lambda kv: -kv[1]):
                print(f"    {n}x  {f[:110]}")
            return best[0], rnd + 1
        tried.append(sig)
        repeats = streak
        if repeats >= 2 and (best[1] or "").strip():
            # Sampling on this endpoint is greedy, so an unchanged prompt returns an unchanged
            # answer. Measured: the same TypeError 19 rounds running. Changing the prompt is the
            # only thing that can change the answer, so say what has already been tried.
            ledger = "\n".join(f"- {t[:160]}" for t in dict.fromkeys(tried))
            prompt = (repair_prompt(level, latest[0], latest[1])
                      + f"\n\nThese approaches have already failed, so do something different:\n"
                        f"{ledger}")
            shape = prompt_shape(prompt, "repair+ledger", level, code=latest[0],
                                 feedback=latest[1], ledger=ledger)
            print(f"  same failure {repeats}x — adding a ledger of {len(set(tried))} failed "
                  f"attempts to break the repeat")
            continue
        if not (latest[0] or "").strip():
            # Nothing came back to repair. Asking it to "fix" an empty code block produced a
            # 202-character prompt and, under greedy sampling, the identical non-answer six
            # rounds running. Shorten and re-ask instead.
            terse = min(terse + 1, 2)
            prompt = first_prompt(level, terse)
            shape = prompt_shape(prompt, "first", level, terse)
            print(f"  no code yet, so re-asking with a shorter prompt (terseness {terse})")
        else:
            prompt = repair_prompt(level, latest[0], latest[1])
            shape = prompt_shape(prompt, "repair", level, code=latest[0], feedback=latest[1])
    print(f"  not solved in {a.rounds} rounds; best reward {best[0]:.2f}")
    return best[0], a.rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, choices=sorted(nkibench.LEVELS))
    ap.add_argument("--all", action="store_true", help="levels 1 to 4")
    ap.add_argument("--levels", help="several levels, comma-separated, e.g. 3,4. Chip time is the "
                                     "scarce thing; this spends it where the question is.")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=MIN_ANSWER_TOKENS)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL")
                    or os.environ.get("GPTOSS_BASE_URL"))
    ap.add_argument("--path", default="", help="path to append, e.g. /agg/v1 for gpt-oss")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run the whole thing N times and report a solve RATE. One run is not a "
                         "result: measured, the same config scored 1.00, 1.00 and 0.50 on level 2 "
                         "across three runs with no code change.")
    ap.add_argument("--log", default="attempts.jsonl")
    ap.add_argument("--give-up-after", type=int, default=4,
                    help="stop a level after this many identical failures in a row. Measured: 15 "
                         "was pure waste, because the prompt had stopped changing.")
    ap.add_argument("--terse", type=int, default=0, choices=(0, 1, 2),
                    help="starting prompt length. Measured on gpt-oss-20b: 0 produced 13,245 "
                         "chars of hidden reasoning and no answer, while 1 answered with code. "
                         "Qwen3-8B is fine at 0.")
    ap.add_argument("--context", type=int, default=4096,
                    help="the server's max-model-len; prompt + answer must fit inside it")
    ap.add_argument("--think", action="store_true",
                    help="let the model reason first; costs budget, and it ran out")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--feedback", default="located",
                    choices=MODES,
                    help="what the checker sends back when the kernel raises. raw: the bare "
                         "exception. enriched: plus the fix instruction (the original behaviour). "
                         "located: plus the model's own failing line, quoted. directed: plus "
                         "messages rewritten from what real runs showed (actual shapes, where a "
                         "name really lives). Run the same level under each to measure what the "
                         "feedback is worth.")
    ap.add_argument("--diverse", action="store_true",
                    help="send --samples DIFFERENTLY WORDED requests each round instead of the "
                         "same prompt --samples times. Measured on a seat: four samples of one "
                         "prompt came back identical in seven rounds of eight.")
    a = ap.parse_args()
    global FEEDBACK_MODE
    FEEDBACK_MODE = a.feedback
    a.session = time.strftime("%Y%m%d-%H%M%S")
    try:
        import subprocess
        a.commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                  text=True, timeout=5,
                                  cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        a.commit = ""

    if not a.offline:
        # Validate before the first request. An empty or scheme-less value produces a hostname
        # starting with "." and 30 lines of httpx/idna traceback that say nothing about the cause.
        raw = (a.base or "").strip()
        if not raw:
            sys.exit("KERNEL_AGENT_BASE_URL is empty or unset. Point it at a model:\n"
                     "  export KERNEL_AGENT_BASE_URL=http://localhost:8000/v1\n"
                     "  export KERNEL_AGENT_MODEL=Qwen/Qwen3-8B\n"
                     "That is the server ./serve.sh started. The seat pods set these for you; if they are "
                     "missing, you are not in a seat pod. Or pass --offline to run with no model.\n"
                     "(Separately: the shared gpt-oss endpoint is reached by setting "
                     "GPTOSS_BASE_URL and passing --path /agg/v1 -- not needed for the local run.)")
        from urllib.parse import urlparse
        u = urlparse(raw)
        if u.scheme not in ("http", "https") or not u.netloc:
            sys.exit(f"base URL {raw!r} is not usable. It needs a scheme and a host, e.g. "
                     f"http://qwen3-8b:8000/v1.")
        a.base = raw.rstrip("/") + a.path
        print(f"endpoint {a.base}  model {a.model}  feedback {FEEDBACK_MODE}  "
              f"diverse {a.diverse}  commit {a.commit or '?'}  session {a.session}")
    else:
        print("*** OFFLINE: replaying the reference kernel. Numbers are meaningless. ***")

    if a.levels:
        try:
            levels = [int(x) for x in a.levels.split(",") if x.strip()]
        except ValueError:
            sys.exit(f"--levels wants numbers separated by commas, e.g. 3,4 -- got {a.levels!r}")
        unknown = [lv for lv in levels if lv not in nkibench.LEVELS]
        if unknown or not levels:
            sys.exit(f"--levels: no such level(s) {unknown}. Known: {sorted(nkibench.LEVELS)}")
    else:
        levels = sorted(nkibench.LEVELS)[:4] if a.all else [a.level or 1]
    full = sum(WEIGHTS.values())
    history = {lv: [] for lv in levels}

    with open(a.log, "a") as log:
        for rep in range(a.repeat):
            if a.repeat > 1:
                print(f"\n################ run {rep + 1} of {a.repeat} ################")
            a.run = rep
            results = []
            for level in levels:
                results.append((level,) + solve(a, level, log))
                history[level].append(results[-1][1])

            print("\n=========== summary ===========")
            for level, reward, rounds in results:
                print(f"  level {level}  reward {reward:.2f} after {rounds} round(s)"
                      + ("  SOLVED" if reward >= full - 1e-9 else ""))
            print(f"  solved {sum(1 for _, r, _ in results if r >= full - 1e-9)}/{len(results)}")

    if a.repeat > 1:
        # The number that actually means something. A solve rate over N runs survives the variance
        # that makes any single run uninterpretable.
        print(f"\n=========== over {a.repeat} runs ===========")
        for lv in levels:
            got = history[lv]
            solves = sum(1 for r in got if r >= full - 1e-9)
            print(f"  level {lv}: solved {solves}/{len(got)}  "
                  f"best {max(got):.2f}  worst {min(got):.2f}  "
                  f"mean {sum(got) / len(got):.2f}  all={[round(r, 2) for r in got]}")
        print("\n  Report the rate, not your best run. A level that solves 1 in 3 times is not solved.")
    print(f"\nattempts logged to {a.log}")


if __name__ == "__main__":
    main()
