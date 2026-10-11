"""
verdicts.py -- turn a checker VERDICT into an INSTRUCTION, and name the failure class.

The lesson this repo keeps re-learning: "X is wrong" teaches the model nothing, "change exactly Y"
does. upstream agent.py did this in `enrich()`, a hard-coded chain of regexes. This module is the
same knowledge as an explicit table, so that

  * every failure lands in a named BUCKET -- the failure taxonomy (scripts/taxonomy.py) and the
    error-keyed doc retrieval (doc_slices.json) are both keyed by it;
  * each rule is unit-tested (python verdicts.py --selftest);
  * adding a translation is one row, not another elif.

    bucket, text = translate(raw_error)   # raw simulator/loader error -> (bucket, error + instruction)
    bucket = classify(feedback)           # any feedback string agent.grade() produced -> bucket

The instructions are upstream's wording unless a comment says otherwise; moving them here does not
change what the model is told.
"""

import ast
import re


def _available_names(dotted):
    """Turn 'no attribute X' into 'here are the real ones'. Needs the SDK; empty without it."""
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


def _real_signature(func_name):
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


# ---------------------------------------------------------------- v2: code-aware translations
#
# Added on seat 73 from E1 transcripts (LOG.md). Behind agent.py --v2-verdicts so the A/B rows
# measured without them stay comparable. Each exists because the v1 instruction was TRUE BUT NOT
# THE FIX for a failure we actually saw:
#   * the model followed the skeleton exactly and wrote lhsT[m0:.., k0:..] (axes swapped); v1 said
#     "derive bounds with min()", which it already did.
#   * the model looped over the partition axis and indexed it away (in_tile[c, ...]); v1 had no
#     translation at all for "Partition dim size must be preserved".
V2 = False

_INPUT_AXES = {lv: {"lhsT": ("K", "M"), "rhs": ("K", "N")} for lv in (3, 4, 5, 6, 7)}


def _swapped_axes(code, level):
    """Find a slice of an input whose first index variable names the WRONG axis, e.g. lhsT[m0:...]."""
    out = []
    for name, axes in _INPUT_AXES.get(level, {}).items():
        for m in re.finditer(rf"\b{name}\[\s*([A-Za-z_]\w*)[^,\]]*,\s*([A-Za-z_]\w*)", code or ""):
            first, second = m.group(1).lower(), m.group(2).lower()
            if first.startswith(axes[1].lower()) and second.startswith(axes[0].lower()):
                a0, a1 = axes[0].lower(), axes[1].lower()
                good = f"{name}[{a0}0:{a0}0 + {a0}_sz, {a1}0:{a1}0 + {a1}_sz]"
                out.append((name, axes, m.group(0), good))
    return out


def _v2(bucket, m, text, code, level):
    """Return (bucket, better instruction), or None to keep v1's."""
    if bucket == "out_of_bounds":
        sw = _swapped_axes(code, level)
        if sw:
            name, axes, seen, good = sw[0]
            return ("swapped_axes",
                    f" The slice `{seen}...` lists the axes in the wrong order. `{name}` has shape "
                    f"[{axes[0]}, {axes[1]}], so its FIRST index must be the {axes[0]} range: write "
                    f"`{good}`. Change only that slice; everything else is right.")
    return None


def code_hint(code, level):
    """v2, for failures that do NOT raise: a swapped slice on a square tile computes silently wrong
    numbers instead of going out of bounds. Measured in E1: all 5 skeleton runs of level 4 wrote
    lhsT[m0:.., k0:..] and got NUMERICAL MISMATCH on every multi-tile shape. Returns '' if none."""
    if not V2:
        return ""
    sw = _swapped_axes(code, level)
    if not sw:
        return ""
    name, axes, seen, good = sw[0]
    return (f"\n  CAUSE: the slice `{seen}...` lists the axes in the wrong order. `{name}` has "
            f"shape [{axes[0]}, {axes[1]}], so its FIRST index must be the {axes[0]} range: write "
            f"`{good}`. On square tiles this does not crash, it silently computes the wrong product. "
            f"Change only that slice.")


# v3 (C2): level 1's remaining wall, read from E1/C1 transcripts. The model computes each pooling
# window as a SCALAR (nl.sum(window) / .reshape) and assigns it element-wise with Python `=`
# (sum_tile[c, h, w] = ...). That is what produces both the 1-D tile and the collapsed partition
# axis; v2's "keep the partition axis whole" fixes half of it. The missing piece is ONE API idiom:
# reduce a window into a (C, 1) tile with nisa.tensor_reduce and tensor_copy it into a column of the
# output tile. Source: neuron-nki-writing/references/api-translation.md (Reduction Operations row).
V3 = False
_SCALAR_REDUCE = re.compile(r"\bnl\.sum\(|\.reshape\(|\w+\[[^\]]+\]\s*=\s*(?!nl\.ndarray)")


def _i_window_reduce(code):
    return (" Each pooling window must be reduced into a TILE, not a scalar, and tiles are never "
            "assigned with `=`. For window (h, w): allocate s = nl.ndarray((C, 1), dtype=nl.float32, "
            "buffer=nl.sbuf), then nisa.tensor_reduce(dst=s, data=in_tile[:, h*p:(h+1)*p, "
            "w*p:(w+1)*p], op=nl.add, axis=(1, 2)), then nisa.tensor_copy(dst=sum_tile[:, h, "
            "w:w+1], src=s). Keep the partition axis whole (':') and drop any loop over it.")


# v4 (C3): level 5's verdict "hoist the operand loads out of the innermost loop" is ambiguous for
# the structure the model actually has (loop m > n > k, both loads inside k): the innermost loop is
# k, but the load that must move is lhsT, OUT OF THE n LOOP, because lhsT[k.., m..] does not depend
# on n. Proved before use: the skeleton with exactly this change passes level 5 4/4 under the bar,
# and every held-out case (handwritten/sk5_hoisted.py).
V4 = False


# v5 (C5): one instruction for all of levels 5-7. Computed by hand for the largest public shape
# (K=256 M=512 N=1024): reloads inside loops cost 2.00x (L4), hoisting lhsT 1.86x (L5 only),
# hoisting rhs 1.14x (L5+L6), loading every tile exactly once 1.00x (L5+L6+L7). Proved before
# use: the load-once kernel passes levels 5, 6 and 7 4/4 at 1.00x and every held-out case
# (handwritten/sk7_load_once.py). It needs ~1.5 MB of SBUF for the largest test shape (24 MiB per
# NeuronCore on trn2): "block = the whole matrix" is the end of level 6's block-size search space
# that fits here. Larger matrices would need real blocking -- stated in the write-up.
V5 = False

_LOAD_ONCE = ("\n  CAUSE: the same lhsT/rhs tiles are copied from HBM more than once. Load every operand "
              "tile exactly ONCE and keep it in SBUF: before any compute loop, build two 2-D Python "
              "lists, a[k][m] = the (k_sz, m_sz) tile lhsT[k0:k0 + k_sz, m0:m0 + m_sz] and b[k][n] = "
              "the (k_sz, n_sz) tile rhs[k0:k0 + k_sz, n0:n0 + n_sz], each filled by one "
              "nisa.dma_copy. Then the m and n loops contain no dma_copy from lhsT or rhs at all: "
              "for each (m, n) allocate one psum tile and call nisa.nc_matmul(dst=acc, "
              "stationary=a[k][m], moving=b[k][n]) for every k, then copy it out once. Use plain "
              "Python range() for the loading loops. Keep the min() tile sizes.")


def traffic_hint(code, level):
    if V5 and level in (5, 6, 7):
        return _LOAD_ONCE
    if not V4 or level != 5:
        return ""
    lines = (code or "").splitlines()
    n_loop = next((i for i, l in enumerate(lines) if re.search(r"for\s+n\b", l)), None)
    lhs_load = next((i for i, l in enumerate(lines) if "dma_copy" in l and "lhsT[" in l), None)
    if n_loop is None or lhs_load is None or lhs_load < n_loop:
        return ""
    return ("\n  CAUSE: the lhsT load is inside the n loop, but lhsT[k0:.., m0:..] does not depend on "
            "n, so every n re-reads the same lhsT tiles. Move it OUT of the n loop: right after "
            "computing m0 and m_sz, loop over k once, allocate each (k_sz, m_sz) lhsT tile, "
            "dma_copy it, and append it to a Python list; inside the n loop use that list's k-th "
            "tile as stationary. Keep loading the rhs tile inside the k loop. Change nothing else.")


# v6 (C6): nkibench says NON-FINITE output is "usually an uninitialised tile". For a kernel that
# calls exp() with no row-max subtraction the cause is OVERFLOW (measured: the naive L8 kernel
# passes every public shape and returns 16384 NaN on x30 inputs).
V6 = False


# v7 (C7): measured in c5_L6_r0. After the load-once instruction the model built the a[k][m] /
# b[k][n] lists of ALLOCATED tiles but never dma_copy'd lhsT/rhs into them, then repeated the same
# kernel: the v1 verdict ("usually an uninitialised tile") is true but names neither the tile nor
# the missing call.
V7 = False


def unloaded_hint(code, level):
    if not V7 or not (3 <= level <= 7):
        return ""
    missing = [n for n in ("lhsT", "rhs")
               if not re.search(rf"dma_copy\([^)]*src\s*=\s*{n}\s*\[", code or "", re.S)]
    if not missing:
        return ""
    names = " and ".join(f"`{n}`" for n in missing)
    return (f"\n  CAUSE: {names} is never copied into SBUF: the tiles are allocated with nl.ndarray "
            f"but nothing writes them, so nc_matmul reads garbage. Right after allocating each "
            f"tile, fill it: nisa.dma_copy(dst=tile, src=lhsT[k0:k0 + k_sz, m0:m0 + m_sz]) for "
            f"lhsT tiles and nisa.dma_copy(dst=tile, src=rhs[k0:k0 + k_sz, n0:n0 + n_sz]) for rhs "
            f"tiles. Build both lists with the k index first: a[k][m] and b[k][n].")


def nonfinite_hint(code):
    if not V6 or "exp" not in (code or "") or (re.search(r"nl.maximum", code or "")
                                                and re.search(r"nl.subtract", code or "")):
        return ""
    return ("\n  CAUSE: exp() overflowed. Before the exp, compute each row's maximum with "
            "nisa.tensor_reduce(dst=mx, data=scores, op=nl.maximum, axis=(1,)) into a (rows, 1) tile "
            "and subtract it: nisa.tensor_scalar(dst=shifted, data=scores, op0=nl.subtract, "
            "operand0=mx). Then exp the shifted scores. The result is mathematically identical.")


# v8 (C8), from two root-cause analyses of the E-logs (LOG 15:2x):
#  * L6: after load-once, the level-6 traffic sentence ("block M and N ... reused across iterations")
#    pulls the model to put the k loop OUTSIDE m and n, allocating a fresh PSUM tile per k: nothing
#    accumulates. 26/38 failing L6 attempts; all 14 under v7. kloop_hint names it (levels 5-7).
#  * L8: (a) one name for the PSUM tile and the SBUF tile (31/31 skeleton-arm attempts), after which
#    v1's "allocate it in sbuf" made it transpose INTO sbuf -> an untranslated Vector-engine error;
#    (b) without the skeleton, q/k/v used as matmul operands straight from HBM ("already transposed",
#    which the API card says about levels 3-7 only). v8a/v8b cover 45/50 failing L8 attempts.
V8 = False

_KLOOP_TEXT = ("\n  CAUSE: the contraction loop over k is OUTSIDE the m and n loops, so a new PSUM tile "
               "is allocated for every k and nothing accumulates across k; the copy-out then reads "
               "whichever `acc` was allocated last. Reorder the compute loops to m, then n, then k "
               "innermost: inside the n loop allocate ONE psum tile `acc`, then for k: "
               "nisa.nc_matmul(dst=acc, stationary=a[k][m], moving=b[k][n]), then -- still inside the "
               "n loop, after the k loop -- tensor_copy `acc` to SBUF and dma_copy it to "
               "result[m0:m0 + m_sz, n0:n0 + n_sz]. Keep the loading loops and the a[k][m] / b[k][n] "
               "lists unchanged.")


def _kloop_outside(code):
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return False

    def walk(node, loops):
        for ch in ast.iter_child_nodes(node):
            st = loops + [ch] if isinstance(ch, ast.For) else loops
            if isinstance(ch, ast.Call) and ast.unparse(ch.func).endswith("nc_matmul"):
                s = next((k.value for k in ch.keywords if k.arg == "stationary"), None)
                while isinstance(s, ast.Subscript) and isinstance(s.value, ast.Subscript):
                    s = s.value
                kv = s.slice.id if isinstance(s, ast.Subscript) and isinstance(s.slice, ast.Name) else None
                names = [l.target.id for l in loops if isinstance(l.target, ast.Name)]
                if kv in names:
                    kl = loops[names.index(kv)]
                    psum_in_k = any(isinstance(c, ast.Call) and ast.unparse(c.func).endswith("ndarray")
                                    and "psum" in ast.unparse(c) for c in ast.walk(kl))
                    if psum_in_k or names.index(kv) != len(names) - 1:
                        return True
            if walk(ch, st):
                return True
        return False
    return walk(tree, [])


def kloop_hint(code, level):
    if not V8 or not (5 <= level <= 7):
        return ""
    return _KLOOP_TEXT if _kloop_outside(code) else ""


def _kw(call, name):
    for k in call.keywords:
        if k.arg == name and isinstance(k.value, ast.Name):
            return k.value.id
    return None

def _fn(call):
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else None

def scan(code):
    """Walk statements in order; buffers[name] = last buffer allocated for that name."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)), None)
    params = {a.arg for a in fn.args.args} if fn else set()
    buf, events = {}, []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and _fn(node.value) == "ndarray":
            for k in node.value.keywords:
                if k.arg == "buffer" and isinstance(k.value, ast.Attribute):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            events.append((node.lineno, "alloc", t.id, k.value.attr))
        if isinstance(node, ast.Call) and _fn(node) in ("nc_matmul", "nc_transpose", "tensor_copy"):
            events.append((node.lineno, _fn(node), node, None))
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript) \
                and isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name) \
                and node.value.value.id in params:
            events.append((node.lineno, "elementwise", node.value.value.id, None))
    events.sort(key=lambda e: e[0])
    return params, events

# ---- v8a: a PSUM tile is used as the SBUF tile too (one name for both) -------------------
def v8a_offenders(code):
    s = scan(code)
    if not s:
        return []
    params, events = s
    buf, tdst, bad = {}, set(), []
    for ln, kind, a, b in events:
        if kind == "alloc":
            buf[a] = b
        elif kind == "nc_transpose":
            d = _kw(a, "dst")
            if d and buf.get(d) == "sbuf":
                bad.append(("transpose_into_sbuf", d))
            tdst.add(d)
        elif kind == "nc_matmul":
            for role in ("stationary", "moving"):
                n = _kw(a, role)
                if n and buf.get(n) == "psum":
                    bad.append(("psum_operand", n))
        elif kind == "tensor_copy":
            d, s_ = _kw(a, "dst"), _kw(a, "src")
            if d and d == s_:
                bad.append(("self_copy", d))
    return bad

V8A_ERR = re.compile(r"(stationary|moving) must be in \['sbuf'\], got psum|"
                     r"Vector engine transpose requires shape")

def v8a(feedback, code):
    if not V8:
        return ""
    if not V8A_ERR.search(feedback or ""):
        return ""
    bad = v8a_offenders(code)
    if not bad:
        return ""
    x = bad[0][1]
    return (f"\n  CAUSE: `{x}` is used as BOTH the PSUM tile and the SBUF tile. These must be TWO "
            f"tiles with two names. nisa.nc_transpose and nisa.nc_matmul write ONLY into a PSUM "
            f"tile, and nc_matmul reads its stationary and moving ONLY from SBUF tiles. So: allocate "
            f"`{x}_p` with buffer=nl.psum and pass it as dst=; allocate a separate `{x}` of the "
            f"same shape with buffer=nl.sbuf; nisa.tensor_copy(dst={x}, src={x}_p); then use "
            f"`{x}` as the matmul operand. Do the same for every transpose and matmul result "
            f"(never tensor_copy a tile onto itself, never reuse q_t/k_t/v_t as a destination).")

# ---- v8b: an HBM input goes straight into nc_matmul, or is transposed element by element --
V8B_ERR = re.compile(r"(stationary|moving) must be in \['sbuf'\], got (private_hbm|shared_hbm)|"
                     r"Out-of-bound access for tensor")

def v8b_offenders(code):
    s = scan(code)
    if not s:
        return []
    params, events = s
    bad = []
    for ln, kind, a, b in events:
        if kind == "nc_matmul":
            for role in ("stationary", "moving"):
                n = _kw(a, role)
                if n in params:
                    bad.append(("hbm_operand", n))
        if kind == "elementwise":
            bad.append(("elementwise_copy", a))
    has_transpose = any(k == "nc_transpose" for _, k, _, _ in events)
    return bad if (bad and not has_transpose) else []

def v8b(feedback, code):
    if not V8:
        return ""
    if not V8B_ERR.search(feedback or ""):
        return ""
    bad = v8b_offenders(code)
    if not bad:
        return ""
    return ("\n  CAUSE: q, k and v are [seq, dim] tensors in HBM and none of them arrives transposed "
            "(the 'already transposed' note applies only to the matmul levels). nc_matmul contracts "
            "over the PARTITION axis, so q.kT needs dim on the partition axis of both operands. "
            "dma_copy q and k whole into (S, D) SBUF tiles, then for each: nisa.nc_transpose("
            "dst=<(D, S) psum tile>, data=<the (S, D) sbuf tile>) and nisa.tensor_copy into a new "
            "(D, S) SBUF tile; pass those as stationary and moving. Never pass q, k or v to nc_matmul "
            "and never copy tiles element by element in a Python loop.")



def l8_hint(feedback, code, level):
    if level != 8:
        return ""
    return v8a(feedback, code) or v8b(feedback, code)



# v9 (C10), from the L8 analysis of c8/c9 (LOG 16:0x): from round 2 every L8 run loops on
# tensor_scalar's argument ROLES (data= the tile vs operand0= the scalar), and 7 error kinds had no
# translation. The simulator surfaces one error per round, so a kernel with five API misuses needs
# five rounds. signature_lint() reports ALL nisa/nl call misuses in one round, before simulation.
V9 = False
_NL_OPS = {"multiply", "subtract", "add", "divide", "maximum", "minimum", "exp", "power"}


def signature_lint(code):
    """[(line, message)] for every nisa.*/nl.* call whose keywords do not fit the real signature, plus
    tensor_scalar role swaps. Needs the SDK; returns [] without it."""
    import inspect
    try:
        import nki.isa as nisa_mod
        import nki.language as nl_mod
        tree = ast.parse(code or "")
    except Exception:
        return []
    mods = {"nisa": nisa_mod, "nl": nl_mod}
    out = []
    for c in ast.walk(tree):
        if not (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and isinstance(c.func.value, ast.Name) and c.func.value.id in mods):
            continue
        fn = getattr(mods[c.func.value.id], c.func.attr, None)
        if fn is None:
            continue                       # invented names are handled by the simulator rule
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        params = sig.parameters
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            continue
        given = [k.arg for k in c.keywords if k.arg]
        bad = [k for k in given if k not in params]
        nposit = len(c.args)
        names = list(params)
        bound = set(names[:nposit]) | set(given)
        missing = [n for n, p in params.items() if p.default is p.empty and
                   p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and n not in bound]
        call = f"{c.func.value.id}.{c.func.attr}"
        if bad:
            out.append((c.lineno, f"line {c.lineno}: {call}() has no argument(s) {bad}; its real "
                                  f"signature is {call}{sig}"))
        if missing:
            out.append((c.lineno, f"line {c.lineno}: {call}() is missing required argument(s) "
                                  f"{missing}; its real signature is {call}{sig}"))
        if c.func.attr == "tensor_scalar":
            kw = {k.arg: k.value for k in c.keywords}
            isop = lambda v: isinstance(v, ast.Attribute) and v.attr in _NL_OPS
            isnum = lambda v: isinstance(v, (ast.Constant, ast.BinOp)) and not isinstance(v, ast.Name)
            if "data" in kw and (isnum(kw["data"]) or isop(kw["data"])):
                out.append((c.lineno, f"line {c.lineno}: tensor_scalar(data=...) must be the TILE being "
                                      f"transformed, not a number or an operation; the number goes in "
                                      f"operand0= and the nl operation in op0="))
            if "operand0" in kw and isop(kw["operand0"]):
                out.append((c.lineno, f"line {c.lineno}: tensor_scalar(operand0=...) is the number or "
                                      f"(P, 1) tile; the nl operation goes in op0="))
        if V10:
            kw = {k.arg: k.value for k in c.keywords}
            FIX = {"sum": "nl.add", "max": "nl.maximum"}
            for r in ("op", "op0", "op1"):
                v = kw.get(r)
                if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and \
                        (v.value.id == "nisa" or (c.func.attr == "tensor_reduce" and v.attr in FIX)):
                    good = FIX.get(v.attr, f"nl.{v.attr}")
                    out.append((c.lineno, f"line {c.lineno}: write {r}={good}; operations come from nl, "
                                          f"and a row sum is nl.add, a row max nl.maximum"))
            if c.func.attr == "tensor_scalar" and "data" in kw and not isinstance(kw["data"], ast.Name) \
                    and not isinstance(kw["data"], ast.Subscript):
                d = ast.unparse(kw["dst"]) if "dst" in kw else "the tile"
                out.append((c.lineno, f"line {c.lineno}: tensor_scalar data= must be the NAME of an existing "
                                      f"tile (normally data={d}); never build a tile inline"))
    return out


V10 = False


def softmax_hint(code, level):
    """L8: no attempt of 115 ever used reciprocal or nl.maximum (LOG 16:4x): the API card offers only
    nl.sum. Name the exact op sequence once the code reaches the softmax."""
    if not V10 or level != 8 or "tensor_reduce" not in (code or ""):
        return ""
    if "reciprocal(" in code and "nl.maximum" in code and "nl.subtract" in code:
        return ""
    return ("\n  CAUSE: the softmax must be exactly: tensor_reduce op=nl.maximum -> (S,1) mx; tensor_scalar "
            "op0=nl.subtract operand0=mx; activation op=nl.exp; tensor_reduce op=nl.add -> (S,1) sm; "
            "nisa.reciprocal(dst=rs, data=sm); tensor_scalar op0=nl.multiply operand0=rs. Every (S,1) "
            "tile goes in operand0=, never data=. Do not negate or add the sum.")


def _i_missing(m, t):
    sig = _real_signature(m.group(1))
    return (f" Pass every argument of {m.group(1)} by keyword." + (f" Real signature: {sig}." if sig else "")
            + " For nisa.tensor_scalar: data= is the TILE being transformed, op0= the nl operation "
            "(nl.multiply, nl.subtract), operand0= the Python number or the (P, 1) per-row tile.")


def _i_undef(m, t):
    if m.group(1) == "np":
        return " numpy is not available inside the kernel. Write the scale in plain Python: 1.0 / (D ** 0.5)."
    return f" `{m.group(1)}` is never defined. Define it before its first use."


def _i_unbound(m, t):
    return (f" `{m.group(1)}` is read before the line that allocates it. Remove it from the earlier call, "
            f"or move its nl.ndarray and the call that fills it above that use.")


def mm_hint(feedback, code, level):
    """Level 8: the scores matmul given the (S, D) tile as loaded instead of its transpose."""
    if not V9 or level != 8 or "contraction dimension mismatch" not in (feedback or ""):
        return ""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return ""
    loaded = set()
    for c in ast.walk(tree):
        if isinstance(c, ast.Call) and ast.unparse(c.func).endswith("dma_copy"):
            kw = {k.arg: k.value for k in c.keywords}
            if isinstance(kw.get("dst"), ast.Name) and isinstance(kw.get("src"), ast.Name) and \
                    kw["src"].id in ("q", "k", "v"):
                loaded.add(kw["dst"].id)
    for c in ast.walk(tree):
        if isinstance(c, ast.Call) and ast.unparse(c.func).endswith("nc_matmul"):
            kw = {k.arg: k.value for k in c.keywords}
            st = kw.get("stationary")
            if isinstance(st, ast.Name) and st.id in loaded and not st.id.startswith("v"):
                return (f"\n  CAUSE: stationary=`{st.id}` is the (S, D) tile as loaded. The scores "
                        f"contract over D, so pass the transposed (D, S) SBUF tiles you built with "
                        f"nc_transpose: stationary=qT, moving=kT.")
    return ""


_PARTITION_COLLAPSED = re.compile(r"Partition dim size must be preserved, got (\d+) -> (\d+)")


def _i_partition_collapsed(m, t):
    return (" You indexed the PARTITION axis (the first one) with a single index, e.g. "
            "in_tile[c, ...], which collapses it. Never loop over the partition axis and never "
            "index it with one number: keep it whole with `:` and let the engine process every "
            "partition at once, e.g. in_tile[:, h*p:(h+1)*p, w*p:(w+1)*p]. Remove the loop over "
            "that axis. Do not reshape tiles.")


# ---------------------------------------------------------------- raw error -> instruction
#
# (bucket, pattern, instruction builder). First match wins, so order matters exactly as it did in
# enrich(): the specific dma/psum patterns sit above the generic "must be in" one.

def _i_memregion(m, t):
    return (" nl.sbuf, nl.psum and nl.shared_hbm are memory regions, not functions. Do not call "
            "them. Allocate with nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf) and pass the "
            "region as the buffer= argument.")


def _i_kwarg(m, t):
    sig = _real_signature(m.group(1))
    return (f" Remove the `{m.group(2)}=` argument."
            + (f" The real signature is {sig}." if sig else ""))


def _i_operator(m, t):
    return (" A tile is not a number, so Python operators like += do not work on one. Accumulate by "
            "allocating a PSUM tile with nl.ndarray(shape, nl.float32, buffer=nl.psum) and letting "
            "nisa.nc_matmul add into it across the loop, or combine two tiles with a nisa op rather "
            "than a Python operator.")


def _i_dma_count(m, t):
    src, dst = int(m.group(1)), int(m.group(2))
    return (f" The tile you allocated holds {dst} elements but you copied {src} into it. "
            f"nisa.dma_copy does not slice or broadcast: allocate the destination with EXACTLY the "
            f"shape of the slice you are moving. If you want a 128x512 piece of a bigger tensor, "
            f"write t = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf) and then "
            f"nisa.dma_copy(dst=t, src=a[0:128, 0:512]) -- the slice on the right must have the "
            f"same shape as the tile on the left.")


def _i_partition(m, t):
    got, mx = int(m.group(2)), int(m.group(3))
    return (f" A tile may have at most {mx} rows, and you asked for {got}. Do not allocate one tile "
            f"for the whole tensor: loop over the partition dimension in chunks of at most {mx} with "
            f"nl.affine_range, allocate the tile inside the loop with the chunk's own size, and copy "
            f"one chunk at a time, e.g. src=a[i*{mx}:(i+1)*{mx}, :]. If a dimension is already {mx} "
            f"or smaller, use it whole -- do NOT pad it up to {mx}, that reads past the end of the "
            f"tensor. The same applies to where you write the result back.")


def _i_broadcast(m, t):
    val, dst = int(m.group(1)), int(m.group(2))
    return (f" You assigned {val} elements into a slice that holds {dst}. Assignment does not "
            f"reshape or broadcast either: the slice on the left and the value on the right must "
            f"have the SAME shape. If the value is bigger, you are writing a whole tile where a "
            f"slice belongs -- index the destination to match, e.g. out[i*128:(i+1)*128, :] = tile. "
            f"If it is smaller, you are looping over the wrong dimension.")


def _i_oob(m, t):
    dim, hi, size = m.group(1), int(m.group(3)), int(m.group(4))
    return (f" You indexed up to {hi} on dimension {dim}, which is only {size} long. Tile limits are "
            f"a MAXIMUM, not a target. Derive every bound from the tensor's own shape -- use "
            f"min(limit, size) and let the final chunk be partial -- rather than writing a fixed "
            f"number. Note the two limits differ: the partition dimension (first) allows at most "
            f"128, the free dimension allows more.")


def _i_contraction(m, t):
    k, mx = int(m.group(1)), int(m.group(2))
    return (f" The contraction dimension K is {k} and one nc_matmul can only contract {mx}. Split K "
            f"into chunks of {mx} and accumulate: allocate ONE psum tile OUTSIDE the K loop, call "
            f"nisa.nc_matmul into that same psum tile once per chunk so the partial products add up "
            f"there, and only after the loop copy it out with nisa.tensor_copy. Do not allocate a "
            f"new psum tile per chunk and do not write partial results to HBM.")


def _i_hbm_engine(m, t):
    return (f" `nisa.{m.group(1)}` only moves data between on-chip buffers, sbuf and psum. To reach "
            f"HBM -- the tensor you allocated with buffer=nl.shared_hbm and will return -- use "
            f"nisa.dma_copy instead. The usual sequence is nc_matmul into psum, tensor_copy psum to "
            f"sbuf, then dma_copy sbuf to the shared_hbm output.")


def _i_buffer(m, t):
    which, needed, got = m.groups()
    place = {"psum": "nl.psum", "sbuf": "nl.sbuf"}.get(needed, needed)
    return (f" Allocate the `{which}` tile with buffer={place} instead of nl.{got}. For "
            f"nisa.nc_matmul: dst must be in nl.psum, and stationary and moving must both be in "
            f"nl.sbuf. Copy between them with nisa.tensor_copy.")


def _i_contraction_mismatch(m, t):
    # ADDED on seat 73: the baseline's level-2 (transpose) runs reached for nc_matmul and hit this
    # with no translation at all -- the API card's matmul paragraph pulls non-matmul levels to it.
    s, mv = int(m.group(1)), int(m.group(2))
    return (f" nc_matmul contracts over the PARTITION (first) dimension, so stationary and moving "
            f"must have the same first dimension; you passed {s} and {mv}. If this operation is not "
            f"a matrix multiply, do not use nc_matmul at all: move data with nisa.dma_copy or "
            f"nisa.tensor_copy between slices instead.")


def _i_multi(m, t):
    return (" Pass every argument by keyword, e.g. nisa.nc_matmul(dst=..., stationary=..., "
            "moving=...), so none is bound twice.")


def _i_1d(m, t):
    return (" Every SBUF and PSUM tile needs two dimensions: a partition dimension first, then a "
            "free dimension. A 1-D tile is not allowed, so write nl.ndarray((rows, cols), ...) and "
            "give a length-N vector the shape (1, N) or (N, 1) depending on which axis you are "
            "reducing over.")


def _i_reshape(m, t):
    return (" Do not reshape. Work with the shapes you were given and slice them into tiles, e.g. "
            "src=a[0:128, 0:64].")


def _i_mod_attr(m, t):
    return _available_names(f"{m.group(1)}.{m.group(2)}")


def _i_obj_attr(m, t):
    return (f" A {m.group(1)} is not a numpy array, so it has no `{m.group(2)}`. Use the nl/nisa "
            f"functions instead.")


RULES = [
    ("call_memory_region", r"'MemoryRegion' object is not callable", _i_memregion),
    ("invented_kwarg", r"(\w+)\(\) got an unexpected keyword argument '(\w+)'", _i_kwarg),
    ("python_operator_on_tile", r"unsupported operand type\(s\) for.*NkiTensor", _i_operator),
    ("dma_shape_mismatch", r"dma_copy requires src and dst to have the same number of elements, "
                           r"got src=(\d+), dst=(\d+)", _i_dma_count),
    ("partition_over_128", r"dma_copy (\w+) partition dimension (\d+) exceeds maximum (\d+)",
     _i_partition),
    ("assign_shape_mismatch", r"value array of shape \((\d+),?\) could not be broadcast to "
                              r"indexing result of shape \((\d+),?\)", _i_broadcast),
    ("out_of_bounds", r"Out-of-bound access for tensor .*? on dimension (\d+): "
                      r"index range \[(\d+), (\d+)\] exceed dimension size of (\d+)", _i_oob),
    ("contraction_over_128", r"Matmul contraction dimension (\d+) exceeds pmax=(\d+)",
     _i_contraction),
    ("engine_cannot_reach_hbm", r"(\w+) (?:dst|src)? ?must be in \['sbuf', 'psum'\], got shared_hbm",
     _i_hbm_engine),
    ("wrong_buffer", r"(\w+) must be in \['(\w+)'\], got (\w+)", _i_buffer),
    ("missing_positional", r"(\w+)\(\) missing \d+ required positional arguments?: '(\w+)'",
     lambda m, t: _i_missing(m, t) if V9 else ""),
    ("undefined_name", r"NameError: name '(\w+)' is not defined", lambda m, t: _i_undef(m, t) if V9 else ""),
    ("used_before_assigned", r"cannot access local variable '(\w+)' where it is not associated",
     lambda m, t: _i_unbound(m, t) if V9 else ""),
    ("contraction_mismatch", r"Matmul contraction dimension mismatch: stationary\[0\]=(\d+) != "
                             r"moving\[0\]=(\d+)", _i_contraction_mismatch),
    ("arg_bound_twice", r"got multiple values for argument", _i_multi),
    ("tile_1d", r"must have at least 2 dimensions", _i_1d),
    ("reshape_instead_of_slice", r"cannot reshape array of size", _i_reshape),
    ("invented_api", r"module '([\w.]+)' has no attribute '(\w+)'", _i_mod_attr),
    ("numpy_method_on_tile", r"'(\w+)' object has no attribute '(\w+)'", _i_obj_attr),
]
_COMPILED = [(b, re.compile(p, re.S), f) for b, p, f in RULES]


def translate(error_text, code=None, level=None):
    """(bucket, error_text + instruction). Unknown errors pass through with bucket 'raised_other'.
    With V2 on and the code/level supplied, code-aware rules may replace the v1 instruction."""
    if V3 and level == 1 and code and _SCALAR_REDUCE.search(code) and (
            _PARTITION_COLLAPSED.search(error_text) or "at least 2 dimensions" in error_text):
        return "scalar_window_reduce", error_text + _i_window_reduce(code)
    if V2:
        m = _PARTITION_COLLAPSED.search(error_text)
        if m:
            return "partition_collapsed", error_text + _i_partition_collapsed(m, error_text)
    for bucket, rx, build in _COMPILED:
        m = rx.search(error_text)
        if m:
            if V2 and code is not None:
                better = _v2(bucket, m, error_text, code, level)
                if better:
                    return better[0], error_text + better[1]
            return bucket, error_text + build(m, error_text)
    return "raised_other", error_text


def enrich(error_text, code=None, level=None):
    """Drop-in for upstream agent.enrich(); code/level enable the v2 rules when V2 is on."""
    return translate(error_text, code, level)[1]


# ---------------------------------------------------------------- any feedback -> bucket
#
# agent.grade() produces feedback from several stages. classify() maps the WHOLE string to one
# bucket so the taxonomy can count it. Stage prefixes first, then the raw-error table.

_STAGES = [
    ("no_code", r"^No code came back"),
    ("syntax_error", r"^The code does not parse"),
    ("rule_wrong_entry_name", r"^Rule violations.*no function named"),
    ("rule_missing_jit", r"^Rule violations.*not decorated with `@nki\.jit`"),
    ("rule_partition_over_128", r"^Rule violations.*partition dimension \d+ exceeds"),
    ("rule_framework_call", r"^Rule violations.*(calls `|uses `|operator)"),
    ("rule_device_comprehension", r"^Rule violations.*not supported by the Neuron device compiler"),
    ("rule_other", r"^Rule violations"),
    ("invented_import", r"^There is no module named"),
    ("load_error", r"could not be loaded"),
    ("cannot_simulate", r"^CANNOT SIMULATE"),
    ("solved", r"^Correct on every shape"),
]
_NUMERIC = [
    ("hardware_hazard", r"CORRECT ON CPU BUT WRONG ON HARDWARE"),
    ("traffic_over_bar", r"TOO MUCH HBM TRAFFIC"),
    ("mutates_input", r"(?i)input.*(modified|overwr|changed)"),
    ("wrong_output_shape", r"WRONG SHAPE"),
    ("non_finite_output", r"NON-FINITE OUTPUT"),
    ("output_not_written", r"OUTPUT IS \d+% ZEROS"),
    ("partial_coverage", r"% of the output is zero, in the block"),
    ("ragged_edge", r"NUMERICAL MISMATCH.*ragged edge is the likely"),
    ("wrong_arithmetic", r"NUMERICAL MISMATCH"),
]


def classify(feedback):
    fb = feedback or ""
    for bucket, pat in _STAGES:
        if re.search(pat, fb, re.S):
            return bucket
    if "must be reduced into a TILE, not a scalar" in fb:
        return "scalar_window_reduce"
    if fb.startswith("API misuse"):
        return "api_misuse_lint"
    if "the contraction loop over k is OUTSIDE" in fb:
        return "kloop_outside"
    if "is used as BOTH the PSUM tile and the SBUF tile" in fb:
        return "psum_sbuf_conflated"
    if "none of them arrives transposed" in fb:
        return "assumed_pretransposed"
    if "is never copied into SBUF" in fb:
        return "tiles_never_loaded"
    if "exp() overflowed" in fb:
        return "exp_overflow"
    if "Load every operand tile exactly ONCE" in fb:
        return "traffic_reloads"
    if "the lhsT load is inside the n loop" in fb:
        return "traffic_lhs_not_hoisted"
    if "lists the axes in the wrong order" in fb:
        return "swapped_axes"
    if _PARTITION_COLLAPSED.search(fb):
        return "partition_collapsed"
    m = re.search(r"raised (\w+): (.*)", fb, re.S)
    if m:
        return translate(m.group(0))[0]
    for bucket, pat in _NUMERIC:
        if re.search(pat, fb, re.S):
            return bucket
    return "other"


# ---------------------------------------------------------------- selftest

_CASES = [
    ("TypeError: 'MemoryRegion' object is not callable", "call_memory_region"),
    ("TypeError: nc_matmul() got an unexpected keyword argument 'transpose_moving'", "invented_kwarg"),
    ("TypeError: unsupported operand type(s) for +=: 'NkiTensor' and 'NkiTensor'",
     "python_operator_on_tile"),
    ("dma_copy requires src and dst to have the same number of elements, got src=4096, dst=128",
     "dma_shape_mismatch"),
    ("AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128", "partition_over_128"),
    ("Out-of-bound access for tensor a on dimension 0: index range [0, 127] exceed dimension size "
     "of 32", "out_of_bounds"),
    ("Matmul contraction dimension 256 exceeds pmax=128", "contraction_over_128"),
    ("tensor_copy dst must be in ['sbuf', 'psum'], got shared_hbm", "engine_cannot_reach_hbm"),
    ("AssertionError: dst must be in ['psum'], got sbuf", "wrong_buffer"),
    ("SBUF and PSUM tensors must have at least 2 dimensions", "tile_1d"),
    ("ValueError: cannot reshape array of size 32768 into shape (1,64)", "reshape_instead_of_slice"),
    ("AttributeError: module 'nki.language' has no attribute 'dot'", "invented_api"),
    ("AttributeError: 'NkiTensor' object has no attribute 'mean'", "numpy_method_on_tile"),
    ("AssertionError: Matmul contraction dimension mismatch: stationary[0]=3 != moving[0]=4",
     "contraction_mismatch"),
    ("ZeroDivisionError: division by zero", "raised_other"),
]
_FEEDBACK = [
    ("No code came back. Reply with one python code block", "no_code"),
    ("The code does not parse: invalid syntax on line 3.", "syntax_error"),
    ("Rule violations, which score zero however fast the kernel is. Fix exactly these: no function "
     "named `x` is defined", "rule_wrong_entry_name"),
    ("Rule violations, which score zero however fast the kernel is. Fix exactly these: `k` is not "
     "decorated with `@nki.jit`, so it", "rule_missing_jit"),
    ("0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised AssertionError: dst must be in "
     "['psum'], got sbuf Allocate", "wrong_buffer"),
    ("1 of 4 shapes passed. On K=256: NUMERICAL MISMATCH: worst ... this is in the final partial "
     "PARTITION tile (partition 3 of 4) -- the ragged edge is the likely cause", "ragged_edge"),
    ("1 of 4 shapes passed. On K=256: NUMERICAL MISMATCH: worst ... most elements are wrong",
     "wrong_arithmetic"),
    ("3 of 4 shapes passed. On K=256 M=512 N=1024: CORRECT, BUT TOO MUCH HBM TRAFFIC FOR THIS "
     "LEVEL", "traffic_over_bar"),
    ("Correct on every shape.", "solved"),
]


def _sdk():
    try:
        import nki  # noqa: F401
        return True
    except ImportError:
        return False


def selftest():
    bad = 0
    for err, want in _CASES:
        got, text = translate(err)
        # invented_api lists real names from the SDK, so it can only add text where nki imports.
        needs_sdk = want == "invented_api" and not _sdk()
        ok = got == want and (want == "raised_other" or needs_sdk or len(text) > len(err))
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} translate -> {got:28s} {err[:60]}")
    for fb, want in _FEEDBACK:
        got = classify(fb)
        bad += got != want
        print(f"  {'ok  ' if got == want else 'FAIL'} classify  -> {got:28s} {fb[:60]}")
    global V2
    V2 = True
    code = "nisa.dma_copy(dst=t, src=lhsT[m0:m0 + m_sz, k0:k0 + k_sz])"
    err = ("AssertionError: Out-of-bound access for tensor `unnamed` on dimension 1: index range "
           "[0, 127] exceed dimension size of 64")
    b, txt = translate(err, code, 3)
    ok = b == "swapped_axes" and "lhsT[k0:k0 + k_sz, m0:m0 + m_sz]" in txt
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} v2 swapped axes        -> {b}")
    h = code_hint(code, 4)
    ok = "lhsT[k0:k0 + k_sz, m0:m0 + m_sz]" in h and classify("NUMERICAL MISMATCH ..." + h) == "swapped_axes"
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} v2 code_hint on a numeric failure")
    b, _ = translate(err, "lhsT[k0:k0 + k_sz, m0:m0 + m_sz]", 3)
    ok = b == "out_of_bounds"
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} v2 correct order kept  -> {b}")
    b, txt = translate("AssertionError: Partition dim size must be preserved, got 1 -> 4")
    ok = b == "partition_collapsed" and "keep it whole" in txt
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} v2 partition collapsed -> {b}")
    ok = classify("0 of 4 shapes passed. On x: raised AssertionError: Partition dim size must be "
                  "preserved, got 1 -> 4 You indexed") == "partition_collapsed"
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} v2 classify partition_collapsed")
    global V3
    V3 = True
    l1 = ("for c in nl.affine_range(C):\n    sum_tile[c, h, w] = nl.sum(window.reshape((4,)))")
    b, txt = translate("AssertionError: SBUF and PSUM tensors must have at least 2 dimensions", l1, 1)
    ok = b == "scalar_window_reduce" and "tensor_reduce" in txt and classify(txt) == b
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} v3 scalar window reduce -> {b}")
    b, _ = translate("AssertionError: SBUF and PSUM tensors must have at least 2 dimensions", l1, 3)
    ok = b == "tile_1d"
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} v3 only on level 1       -> {b}")
    global V4
    V4 = True
    l5 = ("for m in nl.affine_range(M):\n    for n in nl.affine_range(N):\n        for k in r:\n"
          "            nisa.dma_copy(dst=a, src=lhsT[k0:k0 + k_sz, m0:m0 + m_sz])\n")
    h = traffic_hint(l5, 5)
    ok = "OUT of the n loop" in h and classify("CORRECT, BUT TOO MUCH" + h) == "traffic_lhs_not_hoisted"
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} v4 traffic hint (lhsT inside n loop)")
    ok = traffic_hint(l5, 4) == "" and traffic_hint("for n in r:\n  pass\n", 5) == ""
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} v4 silent off-level / when already hoisted")
    global V5
    V5 = True
    ok = all("exactly ONCE" in traffic_hint("x", lv) for lv in (5, 6, 7)) and traffic_hint("x", 4) == ""
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} v5 load-once hint on levels 5-7 only")
    ok = classify("CORRECT, BUT TOO MUCH" + traffic_hint("x", 6)) == "traffic_reloads"
    bad += not ok
    V5 = False
    V4 = False
    V3 = False
    V2 = False
    b, _ = translate(err, code, 3)
    ok = b == "out_of_bounds"
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} v2 off -> v1 unchanged ({b})")
    print("VERDICTS SELFTEST " + ("PASSED" if not bad else f"FAILED ({bad})"))
    return bad


if __name__ == "__main__":
    import sys
    sys.exit(1 if selftest() else 0)
