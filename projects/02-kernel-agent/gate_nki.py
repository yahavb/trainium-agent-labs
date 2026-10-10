"""gate_nki.py: a kernel the simulator accepts but the trn2 compiler rejects is not solved (GATE=static or
GATE=compile; off by default). feedback_v7 wraps agent.grade with it.

Task 08: the simulator accepted 39 distinct solves of new operations; 4 compiled and matched in birsim.
34 of the 35 failures used one of two forms, and the model kept writing them even when the prompt said
not to (nl.divide in 37 of 80 samples). So on a kernel that passes every shape:
  * GATE=static  finds those forms in the source and rewrites the line as code, the way the NumPy
                 messages rewrite a banned call (pasted literally in 210 of 212 samples, task 10):
                   nl.divide(A, B)            -> nisa.reciprocal into `inv`, then a multiply
                   tensor_tensor with a (rows, 1) column operand -> nisa.tensor_scalar with it as operand0
  * GATE=compile also lowers a kernel with no such form for trn2 (when neuronx-cc is on PATH), and
                 passes on the compiler's reason.
The kernel then scores FULL - 0.05: not solved, still the best so far, so the next round repairs it.
"""
import ast
import os
import re
import shutil

import feedback_v6 as v6

MARK = "the trn2 compiler rejects"


def _col(code, name):
    return bool(re.match(r"^[A-Za-z_]\w*$", name or "")) and v6.is_column(code, name)


def _is(call, mod, name):
    f = call.func
    return isinstance(f, ast.Attribute) and f.attr == name and isinstance(f.value, ast.Name) and f.value.id == mod


def fixes(code):
    """[(first line, last line, the original lines, replacement lines or None, why)] for the forms the
    trn2 compiler rejects. Parsed with ast, so arguments like `a[i, j]` and calls over several lines work."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    lines = code.splitlines()
    seg = lambda n: ast.get_source_segment(code, n)    # noqa: E731
    out, done = [], set()
    for st in ast.walk(tree):
        if not isinstance(st, (ast.Assign, ast.Expr)) or not isinstance(st.value, ast.Call):
            continue
        call, a0, a1 = st.value, st.lineno, st.end_lineno
        ind = re.match(r"\s*", lines[a0 - 1]).group(0)
        old = "\n".join(lines[a0 - 1:a1])
        if _is(call, "nl", "divide") and len(call.args) == 2 and isinstance(st, ast.Assign) \
                and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            x, a, b = st.targets[0].id, seg(call.args[0]), seg(call.args[1])
            simple_b = isinstance(call.args[1], ast.Name)
            if isinstance(call.args[0], ast.Constant) and call.args[0].value == 1:
                new = [f"{x} = nl.ndarray({b}.shape, dtype=nl.float32, buffer=nl.sbuf)",
                       f"nisa.reciprocal(dst={x}, data={b})"] if simple_b else None
            elif simple_b and _col(code, b):
                new = [f"inv = nl.ndarray({b}.shape, dtype=nl.float32, buffer=nl.sbuf)",
                       f"nisa.reciprocal(dst=inv, data={b})",
                       f"{x} = nl.ndarray({a}.shape, dtype={a}.dtype, buffer=nl.sbuf)",
                       f"nisa.tensor_scalar(dst={x}, data={a}, op0=nl.multiply, operand0=inv)"]
            elif simple_b:
                new = [f"inv = nl.ndarray({b}.shape, dtype=nl.float32, buffer=nl.sbuf)",
                       f"nisa.reciprocal(dst=inv, data={b})",
                       f"{x} = nl.multiply({a}, inv)"]
            else:
                new = None
            out.append((a0, a1, old, [ind + l for l in new] if new else None,
                        "nl.divide: compute the reciprocal with nisa.reciprocal, then multiply"))
            done.add(a0)
        elif (_is(call, "nisa", "tensor_scalar") or _is(call, "nisa", "tensor_tensor")) and any(
                k.arg in ("op", "op0") and seg(k.value).replace(" ", "") == "nl.divide" for k in call.keywords):
            kw = {k.arg: k.value for k in call.keywords if k.arg}
            opk, dk = ("op0", "operand0") if "op0" in kw else ("op", "data2")
            den = kw.get(dk)
            if isinstance(den, ast.Constant) and isinstance(den.value, (int, float)) and opk == "op0":
                body = re.sub(r"op0\s*=\s*nl\.divide", "op0=nl.multiply", old, count=1)
                body = re.sub(rf"operand0\s*=\s*{re.escape(seg(den))}", f"operand0={1.0 / den.value!r}", body, count=1)
                out.append((a0, a1, old, body.splitlines(), "nl.divide as an operation: multiply by the reciprocal"))
            elif isinstance(den, ast.Name) and not any(k.arg == "reverse0" for k in call.keywords):
                body = re.sub(rf"{opk}\s*=\s*nl\.divide", f"{opk}=nl.multiply", old, count=1)
                body = re.sub(rf"{dk}\s*=\s*{den.id}\b", f"{dk}=inv", body, count=1)
                out.append((a0, a1, old, [f"{ind}inv = nl.ndarray({den.id}.shape, dtype=nl.float32, buffer=nl.sbuf)",
                                          f"{ind}nisa.reciprocal(dst=inv, data={den.id})"] + body.splitlines(),
                            "nl.divide as an operation: compute the reciprocal with nisa.reciprocal, then multiply"))
            else:
                out.append((a0, a1, old, None, "nl.divide as an operation: compute the reciprocal with "
                                               "nisa.reciprocal, then use nl.multiply"))
            done.add(a0)
        elif _is(call, "nisa", "tensor_tensor"):
            kw = {k.arg: k.value for k in call.keywords if k.arg}
            d1, d2 = kw.get("data1"), kw.get("data2")
            if "dst" not in kw or "op" not in kw:
                continue
            dst, op = seg(kw["dst"]), seg(kw["op"])
            n1 = d1.id if isinstance(d1, ast.Name) else None
            n2 = d2.id if isinstance(d2, ast.Name) else None
            if n2 and _col(code, n2):
                new = [f"{ind}nisa.tensor_scalar(dst={dst}, data={seg(d1)}, op0={op}, operand0={n2})"]
                why = f"`{n2}` is a (rows, 1) column; tensor_tensor with a broadcast column"
            elif n1 and _col(code, n1):
                new = [f"{ind}nisa.tensor_scalar(dst={dst}, data={seg(d2)}, op0={op}, operand0={n1}, reverse0=True)"]
                why = f"`{n1}` is a (rows, 1) column; tensor_tensor with a broadcast column"
            else:
                continue
            out.append((a0, a1, old, new, why))
            done.add(a0)
    for node in ast.walk(tree):                        # any other nl.divide: name it, no code
        if isinstance(node, ast.Call) and _is(node, "nl", "divide") and node.lineno not in done:
            out.append((node.lineno, node.lineno, lines[node.lineno - 1], None,
                        "nl.divide: compute the reciprocal with nisa.reciprocal, then multiply "
                        "(nisa.tensor_scalar with it as operand0 when it has one value per row)"))
            done.add(node.lineno)
    return sorted(out)


def lower_error(kernel_src, level, agent):
    """The compiler's reason if this kernel fails to lower for trn2, else None (also None if no compiler)."""
    if not shutil.which("neuronx-cc"):
        return None
    try:
        import nkibench
        from nkitool import analyze
        path = f"/tmp/_gate_{os.getpid()}.py"
        open(path, "w").write(kernel_src)
        k = nkibench.load_kernel(path, nkibench.LEVELS[level]["entry"])
        args, _ = nkibench.make_inputs(nkibench.LEVELS[level]["shapes"][0], level)
        analyze(k, *args, neff=False)
        return None
    except ImportError:
        return None
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")
        j = msg.find("error:")
        return msg[j:j + 200] if j >= 0 else msg[:200]


def message(code, found):
    parts = [f"Correct in the simulator on every shape, but {MARK} it: the chip has no such instruction, so "
             f"this kernel cannot run on the device. Fix exactly these lines:"]
    for a0, a1, old, new, why in found:
        where = f"Line {a0}" if a0 == a1 else f"Lines {a0}-{a1}"
        if new:
            parts.append(f"{where} ({why}). Replace:\n{old}\nwith:\n" + "\n".join(new))
        else:
            parts.append(f"{where}: {why}.\n{old.strip()}")
    parts.append("Keep everything else identical.")
    return "\n\n".join(parts)


def install(agent, mode):
    their = agent.grade
    full = sum(agent.WEIGHTS.values())

    def gated(source, level):
        reward, parts, note = their(source, level)
        if reward < full - 1e-9:
            return reward, parts, note
        found = fixes(source)
        if found:
            return full - 0.05, dict(parts, correct=False), message(source, found)
        if mode == "compile":
            why = lower_error(source, level, agent)
            if why:
                return full - 0.05, dict(parts, correct=False), (
                    f"Correct in the simulator on every shape, but {MARK} it for trn2: {why}\n"
                    f"Change the line the error points at; keep everything else identical.")
        return reward, parts, note

    agent.grade = gated
