"""Torch frontend: a plain torch function is both the compiler input (traced with torch.export and
expanded into the loop IR) and the correctness oracle (run eagerly).

The exported ATen graph is walked node by node. Every tensor value is represented as a *virtual*
tensor: a symbolic shape, a dtype and a function from indices to a data expression. Views and exact
casts (transpose, permute, widening `.to`) just compose that function, so they cost nothing and never
become loops. A matmul is the only op that *materializes* a stage (an f32 accumulator buffer with an
init nest and a reduction nest); the graph output becomes a final pointwise stage that writes the
kernel output. The resulting program is the naive, breadth-first loop nest every schedule starts from.

Supported ops (anything else fails at trace time and is named in the error): aten.mm / aten.matmul on
2-D operands, t / permute / transpose / numpy_T on 2-D values, dtype casts (`.to`, `.float()`, ...).

Dtypes: buffers whose dtype equals the dtype of a kernel argument are emitted as "like:<arg>", so one
traced kernel follows the dtype of its inputs (f32 or bf16 alike).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import ir
from .expr import Aff, Sym, var

_TORCH_DT = {"torch.float32": "f32", "torch.bfloat16": "bf16", "torch.float16": "f16"}
# casts a -> b that are exact (every a value is representable in b), so they can be elided
_EXACT = {("f32", "f32"), ("bf16", "bf16"), ("f16", "f16"), ("bf16", "f32"), ("f16", "f32")}


class TraceError(Exception):
    pass


@dataclass(frozen=True)
class ArgSpec:
    dims: tuple  # symbolic dim names (str) or constants (int), e.g. ("K", "M")
    dtype: str = "f32"
    example: tuple = None  # concrete example shape used for tracing / the oracle

    def torch_dtype(self):
        import torch

        return {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[self.dtype]


def arg(dims, dtype="f32", example=None) -> ArgSpec:
    return ArgSpec(tuple(dims), dtype, tuple(example) if example is not None else None)


@dataclass(frozen=True)
class _Val:
    """A virtual tensor: shape (tuple of Aff), dtype, and idx-tuple -> data expression."""

    shape: tuple
    dtype: str
    elem: Callable


class _Builder:
    def __init__(self, args: dict, output: str):
        self.args = args
        self.output = output
        self.stmts = []
        self.used = set(args) | {output}

    # ---- naming ----------------------------------------------------------------
    def dtype_ref(self, dt: str) -> str:
        for name, spec in self.args.items():
            if spec.dtype == dt:
                return f"like:{name}"
        return dt

    def unique(self, base: str) -> str:
        name, i = base, 1
        while name in self.used:
            name = f"{base}_{i}"
            i += 1
        self.used.add(name)
        return name

    @staticmethod
    def dim_names(shape, fallback):
        out, seen = [], set()
        for d, fb in zip(shape, fallback):
            base = fb
            if len(d.terms) == 1 and d.const == 0 and d.terms[0][1] == 1 and isinstance(d.terms[0][0], Sym):
                base = d.terms[0][0].name.lower()
            n, i = base, 1
            while n in seen:
                n = f"{base}{i}"
                i += 1
            seen.add(n)
            out.append(n)
        return out

    # ---- loop nests ------------------------------------------------------------
    @staticmethod
    def nest(stage, loops, body):
        """loops: [(varname, extent)] outermost first."""
        for v, ext in reversed(loops):
            body = (ir.For(v, ext, tuple(body), stage),)
        return body[0]

    # ---- ops -------------------------------------------------------------------
    def view_transpose(self, x: _Val) -> _Val:
        if len(x.shape) != 2:
            raise TraceError("transpose is only supported on 2-D values")
        return _Val(x.shape[::-1], x.dtype, lambda idx, x=x: x.elem(tuple(idx)[::-1]))

    def cast(self, x: _Val, dt: str) -> _Val:
        if (x.dtype, dt) in _EXACT:
            return _Val(x.shape, dt, x.elem)
        return _Val(x.shape, dt, lambda idx, x=x, dt=dt: ir.CastE(self.dtype_ref(dt), x.elem(idx)))

    def matmul(self, node, x: _Val, y: _Val, res_dt: str) -> _Val:
        if len(x.shape) != 2 or len(y.shape) != 2:
            raise TraceError("matmul is only supported on 2-D operands")
        if x.shape[1] != y.shape[0]:
            raise TraceError(f"contraction dims differ: {x.shape[1]} vs {y.shape[0]}")
        M, K, N = x.shape[0], x.shape[1], y.shape[1]
        name = self.unique(node.name)
        m, n, k = self.dim_names((M, N, K), ("m", "n", "k"))
        buf = ir.Buffer(name, (M, N), "f32", ir.HBM, "temp")
        vm, vn, vk = (var(f"{name}.{d}") for d in (m, n, k))
        zm, zn = var(f"{name}.init.{m}"), var(f"{name}.init.{n}")
        init = self.nest(f"{name}.init", [(zm.terms[0][0].name, M), (zn.terms[0][0].name, N)],
                         (ir.Assign(name, (zm, zn), ir.Lit(0.0)),))
        upd = self.nest(name, [(f"{name}.{m}", M), (f"{name}.{n}", N), (f"{name}.{k}", K)],
                        (ir.Reduce(name, (vm, vn), ir.Mul(x.elem((vm, vk)), y.elem((vk, vn)))),))
        self.stmts += [ir.Alloc(buf), init, upd]
        # torch returns the operands' dtype; the accumulator is f32, so narrow if needed
        out = _Val((M, N), "f32", lambda idx: ir.Read(name, tuple(idx)))
        return out if res_dt == "f32" else self.cast(out, res_dt)

    def finish(self, v: _Val) -> ir.Buffer:
        dims = self.dim_names(v.shape, tuple(f"d{i}" for i in range(len(v.shape))))
        idx = tuple(var(f"{self.output}.{d}") for d in dims)
        out = ir.Buffer(self.output, v.shape, self.dtype_ref(v.dtype), ir.HBM, "out")
        self.stmts.append(self.nest(self.output, [(f"{self.output}.{d}", e) for d, e in zip(dims, v.shape)],
                                    (ir.Assign(self.output, idx, v.elem(idx)),)))
        return out


def _dt(torch_dtype) -> str:
    try:
        return _TORCH_DT[str(torch_dtype)]
    except KeyError:
        raise TraceError(f"unsupported dtype {torch_dtype} (supported: float32, bfloat16, float16)") from None


def trace(fn, args: dict, name: str = "kernel", output: str = "C") -> ir.Proc:
    """Trace `fn(*args)` with torch.export and expand it to the naive loop IR.

    args: ordered {argument name: ArgSpec}, in the order of fn's parameters.
    name: entry-point name of the kernel; output: name of the output buffer."""
    import torch

    examples = []
    for spec in args.values():
        if spec.example is None:
            raise TraceError("ArgSpec.example (a concrete example shape) is required for tracing")
        examples.append(torch.randn(*spec.example).to(spec.torch_dtype()))

    class _Spec(torch.nn.Module):
        def forward(self, *a):
            return fn(*a)

    graph = torch.export.export(_Spec(), tuple(examples)).graph_module.graph
    b = _Builder(args, output)
    env = {}
    placeholders = iter(args.items())
    result = None

    for node in graph.nodes:
        if node.op == "placeholder":
            aname, spec = next(placeholders)
            shape = tuple(Aff.of(d) for d in spec.dims)
            env[node] = _Val(shape, spec.dtype, lambda idx, aname=aname: ir.Read(aname, tuple(idx)))
        elif node.op == "output":
            result = env[node.args[0][0]]
        elif node.op == "call_function":
            t = str(node.target)
            a = node.args
            if t == "aten._assert_tensor_metadata.default":
                continue
            elif t in ("aten.t.default", "aten.numpy_T.default"):
                env[node] = b.view_transpose(env[a[0]])
            elif t == "aten.transpose.int" and sorted(int(i) % 2 for i in a[1:3]) == [0, 1]:
                env[node] = b.view_transpose(env[a[0]])
            elif t == "aten.permute.default" and list(a[1]) == [1, 0]:
                env[node] = b.view_transpose(env[a[0]])
            elif t in ("aten.to.dtype", "aten._to_copy.default"):
                dt = a[1] if len(a) > 1 else node.kwargs.get("dtype")
                env[node] = b.cast(env[a[0]], _dt(dt))
            elif t in ("aten.mm.default", "aten.matmul.default"):
                env[node] = b.matmul(node, env[a[0]], env[a[1]], _dt(node.meta["val"].dtype))
            else:
                raise TraceError(f"unsupported torch op in spec: {t} (supported: mm/matmul, t/permute/transpose, dtype casts)")
    if result is None:
        raise TraceError("the spec has no tensor output")

    out = b.finish(result)
    buffers = tuple(
        ir.Buffer(aname, tuple(Aff.of(d) for d in spec.dims), f"like:{aname}", ir.HBM, "arg") for aname, spec in args.items()
    ) + (out,)
    sizes, seen = [], set()
    for aname, spec in args.items():
        for i, d in enumerate(spec.dims):
            if isinstance(d, str) and d not in seen:
                seen.add(d)
                sizes.append((d, aname, i))
    return ir.Proc(name=name, args=buffers, sizes=tuple(sizes), body=tuple(b.stmts))
