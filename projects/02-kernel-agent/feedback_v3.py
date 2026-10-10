"""Feedback v3 for the kernel agent: task 02's lessons applied, plus an optional repair-prompt change.

Run from the event repo's projects/02-kernel-agent directory, with that directory and this file's
directory on PYTHONPATH (this imports feedback_v2, which must sit next to it):

    MESSAGES=v3 REPAIR_PROMPT=theirs      python feedback_v3.py --all ...
    MESSAGES=v2 REPAIR_PROMPT=restructure python feedback_v3.py --all ...

MESSAGES=v2 is task 02's `full` mode, unchanged. MESSAGES=v3 changes four kinds of message; every
other error still gets the v2 message, or their enrich() behind it:

  R4+K  contraction too long (either "dma_copy dst partition dimension > 128" on a matmul operand, or
        "Matmul contraction dimension K exceeds pmax"): one message, and it gives the loop AS CODE in
        the model's own variable names. Task 02 found that prose asking for a new loop is ignored
        (R5: 20 of 24 next-round samples were byte-identical), while R2, which gave the exact
        allocation line, was followed and solved level 3.
  R5    stationary too wide (split M): AS CODE.
  R5b   moving too wide (split N): AS CODE.
  R7    a name called through the wrong module: names the one TOKEN and gives the corrected line.
        Task 02's version named only the module, and the model flipped every nisa./nl. prefix on the
        line, back and forth (22 of 23 times the error came back).
  R8    new: dma_copy element-count mismatch when the source is known. Gives the allocation line for
        a tile with exactly the source's shape. This was level 3's new final wall in task 02 (3 of
        5 unsolved runs).

The loops in the code use shape expressions such as lhsT.shape[0] // 128, never the numbers from the
failing shape, so the same code is right on every test shape.

REPAIR_PROMPT=restructure replaces their repair prompt's last line ("Change exactly what the
checker names and keep everything else identical") with one that allows restructuring. That line is
the suspected reason R5 was ignored. Everything else in the prompt, and the ledger, is unchanged.
"""
import os
import re
import sys

os.environ["FEEDBACK_V2"] = "full"
import agent          # their agent.py, unchanged
import nkibench       # their checker, unchanged
import feedback_v2 as v2

MESSAGES = os.environ.get("MESSAGES", "v3")
PROMPT = os.environ.get("REPAIR_PROMPT", "theirs")
if MESSAGES not in ("v2", "v3") or PROMPT not in ("theirs", "restructure"):
    sys.exit("MESSAGES must be v2 or v3; REPAIR_PROMPT must be theirs or restructure")

v2_advise = v2.advise


# ------------------------------------------------------------------ reading the model's kernel

def _call_args(source, func):
    """The argument text of the first call to `func` in source, with nested brackets balanced."""
    i = source.find(func + "(")
    if i < 0:
        return None
    i += len(func) + 1
    depth, j = 1, i
    while j < len(source) and depth:
        depth += {"(": 1, "[": 1, ")": -1, "]": -1}.get(source[j], 0)
        j += 1
    return source[i:j - 1]


def _split_top(text):
    """Split on commas that are not inside brackets."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _base(expr):
    """`sbuf_lhsT[:, 0:128]` -> `sbuf_lhsT`."""
    return re.match(r"[\w.]+", expr.strip()).group(0) if expr and re.match(r"[\w.]+", expr.strip()) else None


def matmul_names(source, line=None):
    """dst / stationary / moving base names of the nc_matmul on `line`, else the first in source."""
    for text in ([line] if line and "nc_matmul" in line else []) + [source]:
        got = _parse_matmul(_call_args(text, "nc_matmul"))
        if got:
            return got
    return None


def _parse_matmul(args):
    if args is None:
        return None
    parts = _split_top(args)
    named = {}
    for p in parts:
        m = re.match(r"(\w+)\s*=\s*(.+)", p, re.S)
        if m:
            named[m.group(1)] = m.group(2)
    pos = [p for p in parts if not re.match(r"\w+\s*=", p)]
    for i, key in enumerate(("dst", "stationary", "moving")):
        if key not in named and i < len(pos):
            named[key] = pos[i]
    if not all(k in named for k in ("dst", "stationary", "moving")):
        return None
    got = {k: _base(named[k]) for k in ("dst", "stationary", "moving")}
    got.update({k + "_expr": " ".join(named[k].split()) for k in ("stationary", "moving")})
    return got


def with_free_slice(expr, sl):
    """Set the second (free) index of a tile expression, keeping its first (partition) index:
    `t` -> `t[:, sl]`, `t[i:j]` -> `t[i:j, sl]`, `t[i:j, :]` -> `t[i:j, sl]`."""
    m = re.match(r"^([\w.]+)\[(.*)\]$", expr.strip(), re.S)
    if not m:
        return f"{expr.strip()}[:, {sl}]"
    idx = _split_top(m.group(2))
    idx = idx[:1] + [sl] + idx[2:]
    return f"{m.group(1)}[{', '.join(idx)}]"


def params(source):
    m = re.search(r"def\s+\w+\(([^)]*)\)", source)
    return [p.split("=")[0].strip() for p in m.group(1).split(",") if p.strip()] if m else []


def hbm_source(source, tile, fallback):
    """The kernel parameter whose data is loaded into `tile`, following one `view = param[...]` hop."""
    ps = params(source)
    for line in source.splitlines():
        if "dma_copy" in line and re.search(rf"\bdst\s*=\s*{re.escape(tile)}\b", line):
            src = _base(v2._kw(line, "src") or "")
            if src in ps:
                return src
            m = re.search(rf"^\s*{re.escape(src or '#')}\s*=\s*(\w+)\[", source, re.M)
            if m and m.group(1) in ps:
                return m.group(1)
    return fallback


def output_name(source):
    m = re.search(r"(\w+)\s*=\s*nl\.ndarray\([^\n]*shared_hbm", source)
    return m.group(1) if m else "out"


# ------------------------------------------------------------------ the v3 messages

def advise_v3(err, line, source):
    ps = params(source) + ["lhsT", "rhs"]

    # R4+K: the contraction dimension is longer than 128.
    k_err = re.search(r"Matmul contraction dimension (\d+) exceeds pmax=(\d+)", err)
    p_err = re.search(r"dma_copy dst partition dimension (\d+) exceeds maximum (\d+)", err)
    names = matmul_names(source, line if k_err else None)
    is_operand = (p_err and names and (v2._kw(line, "dst") or "").split("[")[0]
                  in (names["stationary"], names["moving"]))
    if names and (k_err or is_operand):
        S, Mv, P = names["stationary"], names["moving"], names["dst"]
        sS, sM = hbm_source(source, S, ps[0]), hbm_source(source, Mv, ps[1])
        k = int((k_err or p_err).group(1))
        return (f"`{S}` and `{Mv}` hold the whole contraction dimension K = {k}, but nc_matmul "
                f"takes at most 128 rows of K at a time. Feed K in 128-row chunks that all add into "
                f"`{P}`: allocate `{P}` once, before the loop, and replace the loads of `{S}` and "
                f"`{Mv}` and the nc_matmul with this loop:\n\n"
                f"    for k in nl.affine_range({sS}.shape[0] // 128):\n"
                f"        {S} = nl.ndarray((128, {sS}.shape[1]), dtype={sS}.dtype, buffer=nl.sbuf)\n"
                f"        {Mv} = nl.ndarray((128, {sM}.shape[1]), dtype={sM}.dtype, buffer=nl.sbuf)\n"
                f"        nisa.dma_copy(dst={S}, src={sS}[k * 128:(k + 1) * 128, :])\n"
                f"        nisa.dma_copy(dst={Mv}, src={sM}[k * 128:(k + 1) * 128, :])\n"
                f"        nisa.nc_matmul(dst={P}, stationary={S}, moving={Mv})\n")

    # R5: the stationary operand is wider than 128 -> split the output rows M.
    m = re.search(r"Matmul stationary free dimension (\d+) exceeds gemm_stationary_fmax=(\d+)", err)
    if m and matmul_names(source, line):
        names = matmul_names(source, line)
        S, Mv, P = names["stationary"], names["moving"], names["dst"]
        sS, sM = hbm_source(source, S, ps[0]), hbm_source(source, Mv, ps[1])
        out = output_name(source)
        return (f"`{S}` is {m.group(1)} columns wide, but nc_matmul takes at most 128 there. Compute "
                f"the output 128 rows at a time: put your K loop inside a loop over m, give each m "
                f"its own `{P}`, take a 128-column slice of `{S}`, and write that block of rows out. "
                f"The code has this shape:\n\n"
                f"    for m in nl.affine_range({sS}.shape[1] // 128):\n"
                f"        {P} = nl.ndarray((128, {sM}.shape[1]), dtype=nl.float32, buffer=nl.psum)\n"
                f"        for k in ...:   # your K loop, with only the nc_matmul changed to:\n"
                f"            nisa.nc_matmul(dst={P}, stationary="
                f"{with_free_slice(names['stationary_expr'], 'm * 128:(m + 1) * 128')}, "
                f"moving={names['moving_expr']})\n"
                f"        res = nl.ndarray({P}.shape, dtype={out}.dtype, buffer=nl.sbuf)\n"
                f"        nisa.tensor_copy(dst=res, src={P})\n"
                f"        nisa.dma_copy(dst={out}[m * 128:(m + 1) * 128, :], src=res)\n")

    # R5b: the moving operand is wider than 512 -> split the output columns N.
    m = re.search(r"Matmul moving free dimension (\d+) exceeds max (\d+)", err)
    if m and matmul_names(source, line):
        names = matmul_names(source, line)
        S, Mv, P = names["stationary"], names["moving"], names["dst"]
        sM = hbm_source(source, Mv, ps[1])
        out = output_name(source)
        return (f"`{Mv}` is {m.group(1)} columns wide, but nc_matmul takes at most 512 there. Split "
                f"the output columns the same way as the rows: inside the loop that owns `{P}`, loop "
                f"n over {sM}.shape[1] // 512, allocate `{P}` with 512 columns, pass a 512-column "
                f"slice of `{Mv}`, and write to columns [n * 512:(n + 1) * 512] of the output:\n\n"
                f"    for n in nl.affine_range({sM}.shape[1] // 512):\n"
                f"        {P} = nl.ndarray((<rows of this block>, 512), dtype=nl.float32, "
                f"buffer=nl.psum)\n"
                f"        for k in ...:   # your K loop, with only the nc_matmul changed to:\n"
                f"            nisa.nc_matmul(dst={P}, stationary={names['stationary_expr']}, "
                f"moving={with_free_slice(names['moving_expr'], 'n * 512:(n + 1) * 512')})\n"
                f"        ...   # then copy {P} out to {out}[<rows of this block>, "
                f"n * 512:(n + 1) * 512]\n")

    # R7: one token called through the wrong module.
    m = re.search(r"module '(nki\.(?:isa|language))' has no attribute '(\w+)'", err)
    if m:
        here, name = m.groups()
        other = "nki.language" if here == "nki.isa" else "nki.isa"
        short = {"nki.language": "nl", "nki.isa": "nisa"}
        try:
            mod = __import__(other, fromlist=["x"])
        except Exception:
            mod = None
        tok = re.search(rf"\b(?:nki\.isa|nki\.language|nisa|nl|isa|language)\.{name}\b", line)
        if mod is not None and hasattr(mod, name) and tok:
            fixed = line.replace(tok.group(0), f"{short[other]}.{name}", 1)
            return (f"Only the token `{tok.group(0)}` is wrong: `{name}` is in {other}. Change that "
                    f"one token and nothing else on the line, so it reads:\n\n    {fixed}\n")

    # R8: dma_copy into a tile whose element count differs from the source's.
    m = re.search(r"dma_copy requires src and dst to have the same number of elements, "
                  r"got src=(\d+), dst=(\d+)", err)
    if m and "dma_copy" in line:
        dst, src = v2._kw(line, "dst"), v2._kw(line, "src")
        base = _base(src or "")
        if dst and src and "[" not in dst and base in params(source):
            return (f"`{dst}` holds {m.group(2)} elements but `{src}` holds {m.group(1)}. Allocate "
                    f"`{dst}` with exactly the shape of what you copy into it, right before this "
                    f"line:\n\n"
                    f"    {dst} = nl.ndarray({src}.shape, dtype={base}.dtype, buffer=nl.sbuf)\n\n"
                    f"If `{dst}` is used for something else as well, give this copy its own tile "
                    f"instead.")

    return v2_advise(err, line, source)


# ------------------------------------------------------------------ the repair prompt

their_repair_prompt = agent.repair_prompt


def repair_prompt_restructure(level, source, feedback):
    """Theirs, with the last instruction changed from 'keep everything else identical'."""
    return (
        f"This NKI kernel for {nkibench.LEVELS[level]['op']} is not right yet.\n\n"
        f"```python\n{source}\n```\n\n"
        f"A checker reports:\n{feedback}\n\n"
        f"Make the change the checker describes. If it gives code, use that code. If the change "
        f"needs new loops or new tiles, restructure around them; otherwise keep the rest of the "
        f"kernel as it is. Reply with ONE python code block.")


if MESSAGES == "v3":
    v2.advise = advise_v3            # v2.feedback() looks advise up at call time
if PROMPT == "restructure":
    agent.repair_prompt = repair_prompt_restructure   # solve() looks it up at call time

if __name__ == "__main__":
    print(f"feedback v3: MESSAGES={MESSAGES} REPAIR_PROMPT={PROMPT}")
    agent.main()
