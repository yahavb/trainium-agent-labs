"""Lower the ns-level IR to NKI Python source (nl.* / nisa.* names)."""

from __future__ import annotations

import keyword
import re

from . import ir
from .expr import Aff, to_py
from .hw import INSTRS, NC_DEFAULT, HardwareConfig
from .lower import check_hw, infer_kinds, select_copies


class EmitError(Exception):
    pass


_RANGE = {"affine": "nl.affine_range", "sequential": "nl.sequential_range", "static": "nl.static_range"}
_DT = {"f32": "nl.float32", "bf16": "nl.bfloat16", "f16": "nl.float16"}
_BUF = {ir.SBUF: "nl.sbuf", ir.PSUM: "nl.psum"}


def ident(name: str) -> str:
    n = re.sub(r"\W", "_", name)
    if keyword.iskeyword(n) or n[0].isdigit():
        n = "v_" + n
    return n


def dtype_src(d: str) -> str:
    return f"{ident(d[5:])}.dtype" if d.startswith("like:") else _DT[d]


def aff(e: Aff) -> str:
    return to_py(e, ident)


def shape_src(shape) -> str:
    parts = [aff(d) for d in shape]
    return "(" + ", ".join(parts) + ("," if len(parts) == 1 else "") + ")"


def window_src(w: ir.Window, bufs: dict) -> str:
    b = bufs[w.buf]
    if w.is_full(bufs):
        return ident(w.buf)
    parts = []
    for i, (l, z) in enumerate(zip(w.lo, w.size)):
        parts.append(aff(l) if i in w.points else f"{aff(l)}:{aff(l + z)}")
    return f"{ident(w.buf)}[{', '.join(parts)}]"


def sbuf_footprint(proc: ir.Proc) -> Aff:
    """Upper bound on SBUF bytes per partition: every SBUF allocation counted once (no liveness
    reuse), 4 bytes per element worst case since dtypes follow the inputs."""
    total = Aff(0)
    for s in ir.walk(proc.body):
        if isinstance(s, ir.Alloc) and s.buf.mem == ir.SBUF:
            el = Aff(1)
            for d in s.buf.shape[1:]:
                el = el * d
            total = total + el * 4
    return total


def emit_nki(proc: ir.Proc, hw: HardwareConfig = NC_DEFAULT) -> str:
    """Finalize (copy selection, loop-kind inference, hardware legality) and print NKI source."""
    proc = select_copies(proc)
    proc = infer_kinds(proc)
    check_hw(proc, hw)
    bufs = ir.buffers_of(proc)
    lines = ["import nki", "import nki.isa as nisa", "import nki.language as nl", "", "", "@nki.jit"]
    args = [b for b in proc.args if b.role == "arg"]
    outs = [b for b in proc.args if b.role == "out"]
    lines.append(f"def {ident(proc.name)}({', '.join(ident(a.name) for a in args)}):")
    ind = "  "

    # sizes
    by_arg = {}
    for sym_, argname, dim in proc.sizes:
        by_arg.setdefault(argname, {})[dim] = sym_
    for a in args:
        if a.name in by_arg:
            nd = len(a.shape)
            names = [by_arg[a.name].get(d, "_") for d in range(nd)]
            lines.append(f"{ind}{', '.join(names)} = {ident(a.name)}.shape")
    for name, e in proc.params:
        lines.append(f"{ind}{ident(name)} = {aff(e)}")
    for a in proc.assumptions:
        lines.append(f'{ind}assert {aff(a.expr)} % {a.mod} == 0, "expected {a.expr} to be a multiple of {a.mod}"')
    foot = sbuf_footprint(proc)
    if foot.is_const:
        if foot.const > hw.sbuf_bytes_per_partition:
            raise EmitError(f"SBUF footprint {foot.const} B/partition exceeds the {hw.sbuf_bytes_per_partition} B budget")
    else:
        lines.append(f'{ind}assert {aff(foot)} <= {hw.sbuf_bytes_per_partition}, "SBUF footprint (bytes/partition, f32 worst case) exceeds the budget; block the loops"')
    for o in outs:
        lines.append(f"{ind}{ident(o.name)} = nl.ndarray({shape_src(o.shape)}, dtype={dtype_src(o.dtype)}, buffer=nl.shared_hbm)")
    lines.append("")

    def emit(stmts, depth):
        pad = ind * depth
        for s in stmts:
            if isinstance(s, ir.Alloc):
                b = s.buf
                buf = "nl.hbm" if b.mem == ir.HBM else _BUF[b.mem]
                lines.append(f"{pad}{ident(b.name)} = nl.ndarray({shape_src(b.shape)}, dtype={dtype_src(b.dtype)}, buffer={buf})")
            elif isinstance(s, ir.For):
                if s.kind == "auto":
                    raise EmitError(f"loop {s.var} has no resolved kind")
                lines.append(f"{pad}for {ident(s.var)} in {_RANGE[s.kind]}({aff(s.extent)}):")
                emit(s.body, depth + 1)
            elif isinstance(s, ir.Call):
                ins = INSTRS[s.instr]
                kw = ", ".join(f"{r}={window_src(s.arg(r), bufs)}" for r in ins.roles)
                extra = ""
                for k, v in s.attrs:
                    if k == "accumulate" and v is not None:
                        extra += f", accumulate={v}"
                    elif k == "engine":
                        extra += f", engine=nisa.{v}_engine"
                lines.append(f"{pad}nisa.{ins.nisa}({kw}{extra})")
            else:
                raise EmitError(
                    f"scalar statement left in the program ({type(s).__name__} on {s.buf}); every loop nest must be "
                    f"lowered to an instruction (replace(...)) before emission"
                )

    # unwrap fresh output allocs already printed
    emit(proc.body, 1)
    lines.append("")
    lines.append(f"{ind}return {', '.join(ident(o.name) for o in outs)}")
    return "\n".join(lines) + "\n"
