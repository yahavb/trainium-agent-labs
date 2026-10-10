"""Feedback v4 for the kernel agent: task 03's breakages fixed, plus an optional API-card addition.

Run from the event repo's projects/02-kernel-agent directory, with that directory and this file's
directory on PYTHONPATH (this imports feedback_v3 and feedback_v2, which sit next to it):

    MESSAGES=v4 REPAIR_PROMPT=restructure CARD=theirs  python feedback_v4.py --all ...
    MESSAGES=v4 REPAIR_PROMPT=restructure CARD=reduce  python feedback_v4.py --all ...
    MESSAGES=v3 REPAIR_PROMPT=restructure CARD=theirs  python feedback_v4.py --all ...   # = task 03's B

MESSAGES=v4 changes four messages and keeps everything else from v3 (and v2 behind it):

  R4K+  split K, as in v3, plus the copy-out when the old kernel staged its result in an operand
        tile. In task 03, 9 of 34 pasted R4K loops broke because that staging tile became a 128-row
        chunk and the model then wrote psum straight to HBM.
  R8+   the dma_copy shape fix, anchored: "replace line N with these two lines", indented like the
        original. In task 03 the model pasted v3's single allocation line at the top of the kernel,
        above the loop defining its variables (12 of 21 pastes: UnboundLocalError).
  R3+   the psum shape fix, as code: the allocation line built from the kernel's inputs. v2's prose
        version was ignored 17 of 23 times in task 03 and is now level 3's main wall. Only when both
        operands are passed whole (no slicing); otherwise v2's prose.
  R9    new: a Python operator on a tile ("unsupported operand type(s) for /: 'NkiTensor' and
        'int'"). Their message for this talks about accumulating in psum with nc_matmul, which is the
        wrong fix for `tile / 4`. R9 gives the nisa.tensor_scalar or nisa.tensor_tensor call, with
        the operands from the line. Level 1 hit this 43 times in task 03.

CARD=reduce appends real elementwise calls and one worked reduction (a row mean) to their API card,
which the first prompt carries. Level 1 didn't move in 45 runs of tasks 01-03 while the model invented
APIs (nisa.ADD, nisa.nc_reduce, NC_OP_ADD). The example is a mean over the free axis, not pooling: it
shows the calls, not level 1's windowing. CARD=theirs leaves the card alone. Only the untersed first
prompt carries the card; repair prompts never did.
"""
import os
import re
import sys

MESSAGES = os.environ.get("MESSAGES", "v4")
CARD = os.environ.get("CARD", "theirs")
if MESSAGES not in ("v3", "v4") or CARD not in ("theirs", "reduce"):
    sys.exit("MESSAGES must be v3 or v4; CARD must be theirs or reduce")
os.environ["MESSAGES"] = "v3"      # feedback_v3 installs its messages; v4 then wraps them
import agent            # their agent.py, unchanged
import feedback_v2 as v2
import feedback_v3 as v3

v3_advise = v3.advise_v3


def _indent_of(source, line):
    for l in source.splitlines():
        if l.strip() == line.strip():
            return l[:len(l) - len(l.lstrip())]
    return ""


def _lineno_of(source, line):
    for i, l in enumerate(source.splitlines(), 1):
        if l.strip() == line.strip():
            return i
    return None


def _top_level_split(expr, op):
    """Split `expr` at the one top-level occurrence of `op` (outside brackets), else None."""
    depth, hits = 0, []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and expr.startswith(op, i):
            # skip ** and // and the second char of ops like +=
            if not (op in "*/" and (expr[i + 1:i + 2] == op or expr[i - 1:i] == op)):
                hits.append(i)
        i += 1
    if len(hits) != 1:
        return None
    return expr[:hits[0]].strip(), expr[hits[0] + len(op):].strip()


def advise_v4(err, line, source):
    ps = v3.params(source) + ["lhsT", "rhs"]
    ind = _indent_of(source, line)

    # R4K+: v3's split-K, plus the copy-out when an operand tile was staging the result.
    tip = v3_advise(err, line, source)
    if tip and "hold the whole contraction dimension" in tip:
        names = v3.matmul_names(source, line if "nc_matmul" in line else None)
        S, Mv, P = names["stationary"], names["moving"], names["dst"]
        out = v3.output_name(source)
        staged = re.search(rf"tensor_copy\(\s*dst\s*=\s*({re.escape(S)}|{re.escape(Mv)})\b", source) or \
            re.search(rf"dma_copy\(\s*dst\s*=\s*{re.escape(out)}\b[^\n]*src\s*=\s*({re.escape(S)}|"
                      rf"{re.escape(Mv)})\b", source)
        if staged:
            tip += (f"\nAfter this loop `{staged.group(1)}` is only a 128-row chunk, so it can no "
                    f"longer hold the result. Copy `{P}` out through a tile of its own instead, after "
                    f"the loop:\n\n"
                    f"    res = nl.ndarray({P}.shape, dtype={out}.dtype, buffer=nl.sbuf)\n"
                    f"    nisa.tensor_copy(dst=res, src={P})\n"
                    f"    nisa.dma_copy(dst={out}, src=res)\n")
        return tip

    # R8+: dma_copy shape fix, anchored to the line.
    m = re.search(r"dma_copy requires src and dst to have the same number of elements, "
                  r"got src=(\d+), dst=(\d+)", err)
    if m and "dma_copy" in line:
        dst, src = v2._kw(line, "dst"), v2._kw(line, "src")
        base = v3._base(src or "")
        n = _lineno_of(source, line)
        if dst and src and "[" not in dst and base in v3.params(source) and n:
            return (f"`{dst}` holds {m.group(2)} elements but `{src}` holds {m.group(1)}. Replace "
                    f"line {n} with these two lines, so the tile is allocated with exactly the shape "
                    f"of what goes into it, right where it is filled:\n\n"
                    f"{ind}{dst} = nl.ndarray({src}.shape, dtype={base}.dtype, buffer=nl.sbuf)\n"
                    f"{ind}{line.strip()}\n")

    # R3+: nc_matmul writes into a psum tile of the wrong shape -> the allocation line, as code.
    m = re.search(r"cannot reshape array of size (\d+) into shape \(([\d, ]+)\)", err)
    if m and "nc_matmul" in line:
        names = v3.matmul_names(source, line)
        if names and "[" not in names["stationary_expr"] and "[" not in names["moving_expr"]:
            S, Mv, P = names["stationary"], names["moving"], names["dst"]
            sS = v3.hbm_source(source, S, ps[0])
            sM = v3.hbm_source(source, Mv, ps[1])
            if sS == sM:      # both operands from one input: not a matmul whose shape we can derive
                return tip
            return (f"`{P}` has shape ({m.group(2)}), but nc_matmul writes a result of shape "
                    f"(stationary.shape[1], moving.shape[1]), here {m.group(1)} elements. Change the "
                    f"line that allocates `{P}` to:\n\n"
                    f"    {P} = nl.ndarray(({sS}.shape[1], {sM}.shape[1]), dtype=nl.float32, "
                    f"buffer=nl.psum)\n")

    # R9: a Python operator on a tile.
    m = re.search(r"unsupported operand type\(s\) for (\S+): '(\w+)' and '(\w+)'", err)
    if m:
        op, left, right = m.group(1), m.group(2), m.group(3)
        aug = op.endswith("=")
        sym = op.rstrip("=")
        name = {"+": "nl.add", "-": "nl.subtract", "*": "nl.multiply", "/": "nl.multiply"}.get(sym)
        if name and left == "NkiTensor":
            target, A, B = None, "<tile>", "<value>"
            simple = r"^[A-Za-z_][\w.]*(\[.*\])?$"
            if aug:
                lhs, _, rhs = line.partition(op)
                if re.match(simple, lhs.strip()):
                    target, A, B = lhs.strip(), lhs.strip(), rhs.strip()
            elif "=" in line:
                lhs, _, rhs = line.partition("=")
                parts = _top_level_split(rhs, sym)
                if parts and re.match(simple, lhs.strip()):
                    target, (A, B) = lhs.strip(), parts
            into, copy_out = target or "res", ""
            if target and "[" in target:          # an indexed slice, maybe HBM: compute into res first
                into, copy_out = "res", f"{ind}nisa.dma_copy(dst={target}, src=res)\n"
            if target is None:                    # the operator sits inside a call's argument
                call = re.match(r"\s*([\w.]+)\(", line)
                args = v3._call_args(line, call.group(1).split(".")[-1]) if call else None
                for a in v3._split_top(args or ""):
                    kv = re.match(r"(\w+)\s*=\s*(.+)$", a, re.S)
                    parts = _top_level_split(kv.group(2) if kv else a, sym)
                    if parts:
                        A, B = parts
                        new_line = line.strip().replace(kv.group(2) if kv else a, "res", 1)
                        copy_out = f"{ind}{new_line}\n"
                        break
            if right == "NkiTensor":
                code = f"nisa.tensor_tensor(dst={into}, data1={A}, data2={B}, op={name})"
            else:
                operand = f"1.0 / {B}" if sym == "/" else B
                code = f"nisa.tensor_scalar(dst={into}, data={A}, op0={name}, operand0={operand})"
            alloc = ("" if aug and into == target else
                     f"{ind}{into} = nl.ndarray({A}.shape, dtype={A}.dtype, buffer=nl.sbuf)\n")
            n = _lineno_of(source, line)
            where = f"Replace line {n} with" if n else "Write it as"
            return (f"Python operators like `{op}` do not work on tiles. {where}:\n\n{alloc}{ind}{code}\n{copy_out}\n"
                    f"nisa.tensor_scalar combines a tile with a number; nisa.tensor_tensor combines two "
                    f"tiles of the same shape. Both write into dst.")
    return tip


CARD_ADDITION = """
More real functions, and one worked example:

  nisa.tensor_tensor(dst=, data1=, data2=, op=nl.add)            two tiles of the same shape, elementwise
  nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=0.5) a tile and a number, elementwise
  Python operators (+ - * /) do not work on tiles; use the two calls above. The ops are nl.add,
  nl.subtract, nl.multiply and nl.maximum.

Mean over the free axis of a (rows, n) tile, rows <= 128:

    t = nl.ndarray((rows, n), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x[0:rows, 0:n])
    s = nl.sum(t, axis=[1], keepdims=True)        # (rows, 1). Never reduce axis 0, the partition axis.
    m = nl.ndarray((rows, 1), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m, data=s, op0=nl.multiply, operand0=1.0 / n)
"""

def advise_safe(err, line, source):
    """advise_v4, but a bug in it can never stop a run: fall back to v3's message and say so."""
    try:
        return advise_v4(err, line, source)
    except Exception as e:  # noqa: BLE001
        print(f"    [v4 advise failed, using v3: {type(e).__name__}: {e}]")
        return v3_advise(err, line, source)


if MESSAGES == "v4":
    v2.advise = advise_safe          # v2.feedback() looks advise up at call time
if CARD == "reduce":
    agent.API_CARD = agent.API_CARD + CARD_ADDITION   # first_prompt() reads it at call time

if __name__ == "__main__":
    print(f"feedback v4: MESSAGES={MESSAGES} REPAIR_PROMPT={v3.PROMPT} CARD={CARD}")
    agent.main()
