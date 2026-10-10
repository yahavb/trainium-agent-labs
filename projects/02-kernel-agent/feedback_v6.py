"""Feedback v6: messages for the error classes attention (level 8) hits, plus three new held-out
operations and a grading timeout.

Run from the event repo's projects/02-kernel-agent directory, with that directory and this file's
directory on PYTHONPATH, and run THIS file from this folder so Python finds the copies next to it:

    CARD=category MESSAGES=v5 SAMPLING=qwen REPAIR_PROMPT=restructure python feedback_v6.py --level 9 ...
    CARD=category MESSAGES=v6 SAMPLING=qwen REPAIR_PROMPT=restructure python feedback_v6.py --level 9 ...

What importing this does, whatever MESSAGES is:
  * registers levels 9-11 (ops07.py): row softmax, layer norm, gated SiLU. They are held out: v6
    was written from level 8's logged failures only.
  * grades with a timeout, GRADE_TIMEOUT seconds (default 120). In task 06 one kernel took 680 s per
    grade. A timeout scores 0.30 (it parsed and passed the rules) with a message saying why.
  * uses task 07's fixes in the copies next to it: our grade() no longer crashes on the first shape a
    non-matmul level passes (their grade() raises KeyError 'M' there, on level 8 too), and v5's tiling
    message handles one tile passed as both operands and gives the output allocation itself.

MESSAGES=v6 is v5 plus these, all designed from the 304 level-8 samples of task 06:

  R14  a reduction used as an elementwise op (op=nl.max in tensor_tensor or tensor_scalar):
       the row reduction as code, `m = nl.max(t, axis=[1], keepdims=True)`, and the elementwise
       operator's real name (nl.maximum). 13 samples.
  R15  nl functions or dtypes applied to plain Python numbers (`nl.sqrt(dim)`, `nl.float32(...)`):
       the line rewritten in plain Python (`dim ** 0.5`, `float(...)`). 48 samples.
  R10  tensor_tensor with a (rows, 1) operand whose row count differs ("data1 shape (64, 64) is not
       compatible with data2 shape (128, 1)"): nisa.tensor_scalar with that tile as operand0, which
       applies one value per row, and the row count to use. Division becomes a reciprocal and a
       multiply. The simulator lets tensor_tensor broadcast a column when the rows match, and accepts
       nl.divide, but neuronx-cc rejects both for trn2, so the advice is the compilable form. 28 samples
       had this error class; in all of them the op was nl.max, so R14 answers them first.
  R12  nc_matmul operands with different partition sizes: what nc_matmul contracts, and the two
       transposed loads (nisa.dma_transpose) that compute A @ B.T. 23 samples, plus 13 that invented
       a transpose argument for nc_matmul.
  R13  a dma_copy slice with a hard-coded bound past the tensor's end (`q[0:128, 0:128]` on a
       (128, 64) input): the bound from the tensor's shape, and the tile allocated to match. 41 samples.
  R9/  `a / b` with b a tile: a reciprocal and a multiply. v4 told the model to MULTIPLY by b.
  API  when the error is an invented name or argument, also list every other invented name, argument
       or missing required argument in the kernel, checked against nki's real modules and signatures.

Every v6 message falls back to v5's if it can't read the kernel, and a bug in v6 prints
"[v6 advise failed, using v5: ...]" and uses v5's message.
"""
import ast
import difflib
import importlib
import inspect
import os
import re
import signal
import sys
import threading

MESSAGES = os.environ.get("MESSAGES", "v5")
GRADE_TIMEOUT = int(os.environ.get("GRADE_TIMEOUT", "120"))
if MESSAGES not in ("v4", "v5", "v6"):
    sys.exit(f"MESSAGES must be v4, v5 or v6, got {MESSAGES!r}")
os.environ["MESSAGES"] = "v5" if MESSAGES == "v6" else MESSAGES
import ops07                # registers levels 9-11  # noqa: E402,F401
import feedback_v5 as v5    # noqa: E402  (imports v4, v3, v2 and installs v5's switches)
os.environ["MESSAGES"] = MESSAGES
v2, v3, v4 = v5.v2, v5.v3, v5.v4
agent, nkibench = v5.agent, v5.nkibench


# ------------------------------------------------------------------ reading a line

def kw_full(line, name):
    """The whole expression passed as name=..., brackets balanced (v2._kw stops at the first ']')."""
    m = re.search(rf"\b{name}\s*=\s*", line)
    if not m:
        return None
    i, depth = m.end(), 0
    for j in range(i, len(line)):
        ch = line[j]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                return line[i:j].strip()
            depth -= 1
        elif ch == "," and depth == 0:
            return line[i:j].strip()
    return line[i:].strip()


def _where(source, line):
    return v4._lineno_of(source, line), v4._indent_of(source, line)


def _replace(n, ind, code_lines):
    body = "".join(f"{ind}{c}\n" for c in code_lines)
    return (f"Replace line {n} with:\n\n{body}\n" if len(code_lines) == 1 else
            f"Replace line {n} with these {len(code_lines)} lines:\n\n{body}\n")


def _parses(lines):
    try:
        ast.parse("\n".join(l.strip() for l in lines))
        return True
    except SyntaxError:
        return False


# ------------------------------------------------------------------ R14: a reduction used as an elementwise op

WORD = {"max": "maximum", "min": "minimum", "sum": "sum", "mean": "mean"}
ELEMENTWISE = {"max": "nl.maximum", "min": "nl.minimum", "sum": "nl.add", "mean": None}


R14_ERRORS = ("Expected binary operator", "is not compatible with data2", "could not be broadcast")


def r14(err, line, source):
    if not any(e in err for e in R14_ERRORS):     # only the errors this mistake causes
        return None
    m = re.search(r"\bop[01]?\s*=\s*nl\.(max|min|sum|mean)\b", line)
    call = re.search(r"nisa\.(tensor_tensor|tensor_scalar)\(", line)
    if not (m and call):
        return None
    red = m.group(1)
    data = kw_full(line, "data") or kw_full(line, "data1")
    dst = kw_full(line, "dst")
    n, ind = _where(source, line)
    if not (data and dst and n) or not re.match(r"^[A-Za-z_]\w*$", dst):
        return None
    code = f"{dst} = nl.{red}({data}, axis=[1], keepdims=True)"
    msg = (f"`nl.{red}` is a reduction over axes, not an elementwise operator, so it can't be the op of "
           f"nisa.{call.group(1)}. For the {WORD[red]} of each row, {_replace(n, ind, [code])}\n"
           f"That gives a tile with one value per row, shape ({data}.shape[0], 1).")
    if ELEMENTWISE[red]:
        msg += f" For an elementwise {WORD[red]} of two tiles, the operator is {ELEMENTWISE[red]}."
    return msg


# ------------------------------------------------------------------ R15: nl applied to plain Python numbers

DTYPES = {"float32": "float", "float16": "float", "bfloat16": "float", "tfloat32": "float",
          "int32": "int", "int16": "int", "int8": "int", "uint8": "int", "uint16": "int", "uint32": "int"}
NUMBER_FUNCS = {"sqrt": "({}) ** 0.5", "rsqrt": "({}) ** -0.5", "exp": "math.exp({})",
                "log": "math.log({})", "abs": "abs({})", "power": "({}) ** ({})"}


def _rewrite_numbers(line):
    """Replace nl.<dtype>(x) with float(x)/int(x) and nl.sqrt(x) etc. with Python, innermost first."""
    pat = re.compile(r"\bnl\.(" + "|".join(list(DTYPES) + list(NUMBER_FUNCS)) + r")\(")
    for _ in range(12):
        hits = list(pat.finditer(line))
        if not hits:
            break
        h = hits[-1]
        i, depth = h.end(), 1
        j = i
        while j < len(line) and depth:
            depth += {"(": 1, ")": -1}.get(line[j], 0)
            j += 1
        if depth:
            return None
        inner = line[i:j - 1]
        name = h.group(1)
        if name in DTYPES:
            rep = f"{DTYPES[name]}({inner})"
        else:
            args = v3._split_top(inner)
            tmpl = NUMBER_FUNCS[name]
            if tmpl.count("{}") != len(args):
                return None
            rep = tmpl.format(*[a.strip() for a in args])
        line = line[:h.start()] + rep + line[j:]
    # `x = nisa.tensor_scalar(<one number>)`: a nisa call wrapped around a plain number; drop it.
    m = re.match(r"^(\s*[\w.]+\s*=\s*)nisa\.\w+\((.*)\)\s*$", line)
    if m and len(v3._split_top(m.group(2))) == 1 and not re.match(r"\s*\w+\s*=", m.group(2)):
        line = m.group(1) + m.group(2).strip()
    return line


def r15(err, line, source):
    if not ("'str' object is not callable" in err
            or re.search(r"'(int|float)' object has no attribute", err)):
        return None
    new = _rewrite_numbers(line)
    n, ind = _where(source, line)
    if not new or new.strip() == line.strip() or not n or not _parses([new]):
        return None
    why = ("`nl.float32` and the other dtypes are names, not functions, and `nl` functions such as "
           "nl.sqrt work on tiles, not on numbers."
           if "'str' object is not callable" in err else
           "`nl` functions such as nl.sqrt work on tiles, not on numbers.")
    tail = ("\nAlso add `import math` at the top of the file."
            if "math." in new and not re.search(r"^\s*import math\b", source, re.M) else "")
    return (f"Shapes are plain Python ints, so a number computed from them is plain Python too. {why} "
            f"{_replace(n, ind, [new.strip()])}{tail}")


# ------------------------------------------------------------------ R10: one value per row, via tensor_scalar

def r10(err, line, source):
    m = re.search(r"data1 shape \((\d+), (\d+)\) is not compatible with data2 shape \((\d+), (\d+)\)", err)
    if not m or "tensor_tensor" not in line:
        return None
    a0, a1, b0, b1 = map(int, m.groups())
    d1, d2, dst, op = (kw_full(line, k) for k in ("data1", "data2", "dst", "op"))
    n, ind = _where(source, line)
    if not (d1 and d2 and dst and op and n) or "nl.ndarray(" in d1 + d2:
        return None
    if b1 == 1 and a1 > 1:
        big, small, rb, rs, reverse = d1, d2, a0, b0, False
    elif a1 == 1 and b1 > 1:
        big, small, rb, rs, reverse = d2, d1, b0, a0, True
    else:
        return None
    if op == "nl.divide":
        if reverse:
            return None
        code = ["inv = nl.ndarray(" + small + ".shape, dtype=nl.float32, buffer=nl.sbuf)",
                f"nisa.reciprocal(dst=inv, data={small})",
                f"nisa.tensor_scalar(dst={dst}, data={big}, op0=nl.multiply, operand0=inv)"]
        note = (" Division is a reciprocal and a multiply: neuronx-cc rejects nl.divide for the chip, "
                "although the simulator accepts it.")
    else:
        rev = ", reverse0=True" if reverse and op in ("nl.subtract",) else ""
        code = [f"nisa.tensor_scalar(dst={dst}, data={big}, op0={op}, operand0={small}{rev})"]
        note = ""
    rows = (f" Also, one value per row needs the same number of rows: `{small}` has {rs} and `{big}` has "
            f"{rb}, so allocate `{small}` with {big}.shape[0] rows." if rs != rb else "")
    return (f"`{small}` has one column: one value per row. nisa.tensor_scalar applies one value per "
            f"row when its operand0 is a (rows, 1) tile; nisa.tensor_tensor is for two tiles of the same "
            f"shape (the simulator lets it broadcast a column, but neuronx-cc rejects that for the "
            f"chip). {_replace(n, ind, code)}{note.strip()}{rows}")


# ------------------------------------------------------------------ R12: what nc_matmul contracts, and transposed loads

def _transposed_load(tile, src):
    return [f"{tile} = nl.ndarray(({src}.shape[1], {src}.shape[0]), dtype={src}.dtype, buffer=nl.sbuf)",
            f"nisa.dma_transpose(dst={tile}, src={src})"]


def r12(err, line, source):
    mm = re.search(r"Matmul contraction dimension mismatch: stationary\[0\]=(\d+) != moving\[0\]=(\d+)", err)
    tk = re.search(r"nc_matmul\(\) got an unexpected keyword argument '(\w*transpose\w*)'", err)
    if not (mm or tk) or "nc_matmul" not in line:
        return None
    names = v3.matmul_names(source, line)
    if not names:
        return None
    S, Mv, P = names["stationary"], names["moving"], names["dst"]
    sS, sM = v3.hbm_source(source, S, None), v3.hbm_source(source, Mv, None)
    head = (f"nc_matmul computes stationary.T @ moving. Both operands need the contraction axis K first, "
            f"on the partitions: stationary is [K, M], moving is [K, N], and the result is [M, N]. ")
    if mm:
        head += f"Here `{S}` has {mm.group(1)} rows and `{Mv}` has {mm.group(2)}. "
    else:
        head = (f"nc_matmul has no `{tk.group(1)}` argument and cannot transpose an operand itself. " + head)
    if sS and sM and sS != sM and S != Mv and "[" not in names["stationary_expr"] \
            and "[" not in names["moving_expr"]:
        code = (_transposed_load(S, sS) + _transposed_load(Mv, sM)
                + [f"{P} = nl.ndarray(({sS}.shape[0], {sM}.shape[0]), dtype=nl.float32, buffer=nl.psum)",
                   f"nisa.nc_matmul(dst={P}, stationary={S}, moving={Mv})"])
        return (head + f"To compute {sS} @ {sM}.T, where both have K as their last axis, load both "
                f"transposed so K comes first. Replace the loads of `{S}` and `{Mv}`, the allocation of "
                f"`{P}` and the nc_matmul with:\n\n" + "".join(f"    {c}\n" for c in code)
                + f"\nIf instead you want {sS} @ {sM} (K is {sM}'s FIRST axis), transpose only `{S}`'s load.")
    return (head + "To transpose a tile that is already on chip, use the tensor engine and copy the "
            "result back to sbuf:\n\n"
            f"    t_ps = nl.ndarray(({S}.shape[1], {S}.shape[0]), dtype=nl.float32, buffer=nl.psum)\n"
            f"    nisa.nc_transpose(dst=t_ps, data={S})\n"
            f"    t = nl.ndarray(({S}.shape[1], {S}.shape[0]), dtype=nl.float32, buffer=nl.sbuf)\n"
            f"    nisa.tensor_copy(dst=t, src=t_ps)\n\n"
            f"and pass `t` as the operand. To load an input transposed, use "
            f"nisa.dma_transpose(dst=<[cols, rows] tile>, src=<input>).")


# ------------------------------------------------------------------ R13: a hard-coded bound past the tensor's end

def r13(err, line, source):
    m = re.search(r"Out-of-bound access for tensor .*? on dimension (\d+): index range \[(\d+), (\d+)\] "
                  r"exceed dimension size of (\d+)", err)
    if not m or "dma_copy" not in line:
        return None
    d, hi, size = int(m.group(1)), int(m.group(3)), int(m.group(4))
    tensors = v3.params(source) + [v3.output_name(source)]
    n, ind = _where(source, line)
    for side in ("src", "dst"):
        expr = kw_full(line, side) or ""
        mm = re.match(r"^([\w.]+)\[(.*)\]$", expr, re.S)
        if not mm or mm.group(1) not in tensors:
            continue
        base, idx = mm.group(1), v3._split_top(mm.group(2))
        if d >= len(idx):
            continue
        c = re.match(r"^\s*(0?)\s*:\s*(\d+)\s*$", idx[d])
        if not c or int(c.group(2)) <= size:
            continue
        idx[d] = f"0:min(128, {base}.shape[0])" if d == 0 else f"0:{base}.shape[{d}]"
        new = f"{base}[{', '.join(i.strip() for i in idx)}]"
        other = kw_full(line, "dst" if side == "src" else "src") or ""
        fixed = line.strip().replace(expr, new, 1)
        if side == "src" and re.match(r"^[A-Za-z_]\w*$", other):
            code = [f"{other} = nl.ndarray({new}.shape, dtype={base}.dtype, buffer=nl.sbuf)", fixed]
        else:
            code = [fixed]
        if not n or not _parses(code):
            return None
        return (f"`{expr}` runs past the end of `{base}`: its dimension {d} is {size} long, and the slice "
                f"asks for {hi + 1}. Take the bound from the tensor's shape instead of writing a number, "
                f"and allocate the tile to match. {_replace(n, ind, code)}")
    return None


# ------------------------------------------------------------------ R9 for a tile divisor

def is_column(source, name):
    """Whether the kernel makes `name` as one value per row: a keepdims reduction over axis 1, or an
    allocation whose second dimension is 1."""
    n = re.escape(name)
    return bool(re.search(rf"^\s*{n}\s*=\s*nl\.(sum|max|min|mean)\([^\n]*keepdims\s*=\s*True", source, re.M)
                or re.search(rf"^\s*{n}\s*=\s*nl\.ndarray\(\(\s*[^,()]+,\s*1\s*\)", source, re.M)
                or re.search(rf"tensor_reduce\([^\n]*dst\s*=\s*{n}\b[^\n]*keepdims\s*=\s*True", source))


def r9_divide(err, line, source):
    if "unsupported operand type(s) for /: 'NkiTensor' and 'NkiTensor'" not in err:
        return None
    lhs, _, rhs = line.partition("=")
    parts = v4._top_level_split(rhs, "/")
    n, ind = _where(source, line)
    if not parts or not n or not re.match(r"^[A-Za-z_]\w*$", lhs.strip()):
        return None
    A, B = parts
    tgt = lhs.strip()
    col = re.match(r"^[A-Za-z_]\w*$", B) and is_column(source, B)
    last = (f"nisa.tensor_scalar(dst={tgt}, data={A}, op0=nl.multiply, operand0=inv)" if col else
            f"nisa.tensor_tensor(dst={tgt}, data1={A}, data2=inv, op=nl.multiply)")
    code = [f"inv = nl.ndarray({B}.shape, dtype=nl.float32, buffer=nl.sbuf)",
            f"nisa.reciprocal(dst=inv, data={B})",
            f"{tgt} = nl.ndarray({A}.shape, dtype={A}.dtype, buffer=nl.sbuf)", last]
    why = (f"`{B}` holds one value per row, so the multiply is nisa.tensor_scalar with `inv` as operand0."
           if col else
           f"If `{B}` holds one value per row, shape (rows, 1), make the last line "
           f"nisa.tensor_scalar(dst={tgt}, data={A}, op0=nl.multiply, operand0=inv) instead.")
    return (f"Python operators like `/` do not work on tiles, and dividing by a tile is a reciprocal and "
            f"a multiply (neuronx-cc rejects nl.divide for the chip, although the simulator accepts it). "
            f"{_replace(n, ind, code)}{why}")


# ------------------------------------------------------------------ API: every invented name at once

def _nki_aliases(tree):
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("nki"):
                    out[a.asname or a.name] = a.name
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("nki"):
            for a in node.names:
                out[a.asname or a.name] = f"{node.module}.{a.name}"
    mods = {}
    for alias, name in out.items():
        try:
            mods[alias] = importlib.import_module(name)
        except Exception:  # noqa: BLE001
            pass
    return mods


def api_problems(source):
    """(line, text) for every nl./nisa. name that does not exist, every keyword argument a real
    function does not take, and every required argument a call leaves out."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    mods = _nki_aliases(tree)
    other = {"nki.language": "nki.isa", "nki.isa": "nki.language"}
    short = {"nki.language": "nl", "nki.isa": "nisa"}
    probs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in mods:
            mod = mods[node.value.id]
            if hasattr(mod, node.attr):
                continue
            name = f"{node.value.id}.{node.attr}"
            alt = other.get(mod.__name__)
            if alt and hasattr(importlib.import_module(alt), node.attr):
                probs.append((node.lineno, f"`{name}` does not exist; it is `{short[alt]}.{node.attr}`."))
                continue
            close = difflib.get_close_matches(node.attr, [n for n in dir(mod) if not n.startswith("_")],
                                              n=4, cutoff=0.5)
            probs.append((node.lineno, f"`{name}` does not exist" +
                          (f"; the closest real names are {', '.join(short.get(mod.__name__, node.value.id) + '.' + c for c in close)}."
                           if close else ".")))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id in mods:
            fn = getattr(mods[node.func.value.id], node.func.attr, None)
            if not callable(fn):
                continue
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                continue
            params = sig.parameters
            if any(p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL) for p in params.values()):
                continue
            name = f"{node.func.value.id}.{node.func.attr}"
            bad = [k.arg for k in node.keywords if k.arg and k.arg not in params]
            given = {k.arg for k in node.keywords if k.arg}
            positional = [p for p in params.values()
                          if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            given |= {p.name for p in positional[:len(node.args)]}
            missing = ([p.name for p in params.values() if p.default is p.empty
                        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.name not in given]
                       if mods[node.func.value.id].__name__ == "nki.isa" else [])
            if bad or missing:
                what = []
                if bad:
                    what.append("takes no " + ", ".join(f"`{b}=`" for b in bad))
                if missing:
                    what.append("is missing " + ", ".join(f"`{m}=`" for m in missing))
                probs.append((node.lineno, f"`{name}(...)` {' and '.join(what)}. Its real signature is "
                              f"{name}{sig}."))
    return sorted(set(probs))


INVENTED = re.compile(r"has no attribute|unexpected keyword argument|missing \d+ required")
API_MAX = 10   # task 07 desktop patch: most entries listed


def with_api_list(err, line, source, tip):
    if not INVENTED.search(err):
        return tip
    here = v4._lineno_of(source, line)
    rest = [(ln, t) for ln, t in api_problems(source) if ln != here]
    if not rest:
        return tip
    if not tip:      # keep their enrich() advice for this error, then add the list
        tip = agent.enrich(err)[len(err):].strip()
    # Task 07 desktop patch: cap the listing. A truncated, repetitive reply (one bad line ~200 times) gave
    # 214 entries, a 39k-character message and an 8,193-token prompt, so the server returned HTTP 400
    # and the agent exited (V_L11_1). Every other listing logged had 1-3 entries, so the cap changes
    # nothing else.
    listing = "\n".join(f"- line {ln}: {t}" for ln, t in rest[:API_MAX])
    if len(rest) > API_MAX:
        listing += f"\n- ... and {len(rest) - API_MAX} more like these"
    return ((tip or "") + f"\n\nThe kernel has {len(rest)} more name{'s' if len(rest) > 1 else ''} or "
            f"argument{'s' if len(rest) > 1 else ''} that would fail next. Fix them in the same change:\n"
            + listing)


# ------------------------------------------------------------------ v6

def advise_v6(err, line, source):
    try:
        for rule in (r14, r15, r10, r12, r13, r9_divide):
            tip = rule(err, line, source)
            if tip:
                break
        else:
            tip = v5.advise_v5(err, line, source)
        return with_api_list(err, line, source, tip)
    except Exception as e:  # noqa: BLE001
        print(f"    [v6 advise failed, using v5: {type(e).__name__}: {e}]")
        return v5.advise_v5(err, line, source)


# ------------------------------------------------------------------ grading timeout (every condition)

class GradeTimeout(BaseException):
    """BaseException, so the `except Exception` blocks inside grade() can't swallow it."""


def _alarm(signum, frame):
    raise GradeTimeout()


_grade = agent.grade


def grade_with_timeout(source, level):
    if GRADE_TIMEOUT <= 0 or threading.current_thread() is not threading.main_thread():
        return _grade(source, level)
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(GRADE_TIMEOUT)
    try:
        return _grade(source, level)
    except GradeTimeout:
        parts = dict(parses=True, rules=True, runs=False, correct=False)
        reward = agent.WEIGHTS["parses"] + agent.WEIGHTS["rules"]
        return reward, parts, (
            f"GRADING TIMED OUT: simulating the kernel took more than {GRADE_TIMEOUT} s, so it was "
            f"stopped. That usually means Python loops over single elements. Work on whole tiles: load "
            f"a block with one nisa.dma_copy and apply one nisa operation to the whole tile.")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


if MESSAGES == "v6":
    v2.advise = advise_v6          # v2.feedback() looks advise up at call time
agent.grade = grade_with_timeout   # solve() looks grade up at call time

if __name__ == "__main__":
    print(f"feedback v6: MESSAGES={MESSAGES} CARD={v5.CARD} LOOP={v5.LOOP} SAMPLING={v5.SAMPLING} "
          f"THINK={v5.THINK} REPAIR_PROMPT={v3.PROMPT} GRADE_TIMEOUT={GRADE_TIMEOUT}"
          + (f" USAGE_LOG={v5.USAGE_LOG}" if v5.USAGE_LOG else ""))
    agent.main()
