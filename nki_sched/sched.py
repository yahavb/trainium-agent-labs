"""Schedule primitives. Each primitive is a checked rewrite of the ns-level loop IR (Exo style):
it validates its own precondition, rebuilds the affected part of the immutable program, and
records a history entry. v0 handles are unique loop / buffer names resolved against the current
program on every call (a renamed or consumed name raises an error that says what replaced it).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

from . import ir
from .analysis import AnalysisError, const_int, dim_interval, injective, loop_ranges, region
from .emit import emit_nki
from .expr import Aff, Var, var
from .hw import INSTRS, NC_DEFAULT, HardwareConfig
from .lower import HardwareError, check_hw, copy_nest_to_call, infer_kinds, nest_window


class ScheduleError(Exception):
    pass


def _chain(loop: ir.For):
    """Maximal perfect loop chain starting at `loop` (each body exactly one For)."""
    out = [loop]
    while len(out[-1].body) == 1 and isinstance(out[-1].body[0], ir.For):
        out.append(out[-1].body[0])
    return out


def _rebuild_chain(loops, inner_body, order):
    body = inner_body
    for l in reversed(order):
        body = (replace(l, body=tuple(body)),)
    return body[0]


class Sched:
    def __init__(self, proc: ir.Proc, hw: HardwareConfig = NC_DEFAULT, check: Optional[Callable] = None):
        self.proc = proc
        self.hw = hw
        self.history = [("init", proc)]
        self._renamed = {}
        self._check = check
        self.notes = []  # e.g. reassociation notices

    # ------------------------------------------------------------------ plumbing
    def _commit(self, label: str, proc: ir.Proc):
        try:
            check_hw(proc, self.hw)
        except HardwareError as e:
            raise ScheduleError(f"{label}: {e}") from None
        if self._check is not None:
            try:
                self._check(proc)
            except Exception as e:  # semantic regression detected by differential testing
                raise ScheduleError(f"{label}: result no longer matches the oracle: {e}") from None
        self.proc = proc
        self.history.append((label, proc))

    def _find(self, var_name: str) -> ir.For:
        l = ir.find_loop(self.proc.body, var_name)
        if l is None:
            if var_name in self._renamed:
                raise ScheduleError(f"loop '{var_name}' no longer exists: it was split into {self._renamed[var_name]}; use one of those")
            names = ir.loop_vars(self.proc.body)
            raise ScheduleError(f"no loop named '{var_name}'. Loops: {names}")
        return l

    def _buf(self, name: str) -> ir.Buffer:
        b = ir.buffers_of(self.proc).get(name)
        if b is None:
            raise ScheduleError(f"no buffer named '{name}'. Buffers: {list(ir.buffers_of(self.proc))}")
        return b

    @staticmethod
    def _const(e: Aff, what: str) -> int:
        try:
            return const_int(e, what)
        except AnalysisError as err:
            raise ScheduleError(f"{err}. Tile the loops so the window has a fixed size (e.g. split a symbolic loop).") from None

    def _fresh(self, name: str):
        if name in ir.buffers_of(self.proc) or name in ir.loop_vars(self.proc.body):
            raise ScheduleError(f"name '{name}' is already in use")

    def _rewrite_loop(self, var_name, fn):
        """Replace the For named var_name by fn(For) -> tuple of statements."""
        done = []

        def rec(stmts):
            out = []
            for s in stmts:
                if isinstance(s, ir.For):
                    if s.var == var_name:
                        done.append(1)
                        out.extend(fn(s))
                        continue
                    s = replace(s, body=tuple(rec(s.body)))
                out.append(s)
            return out

        body = tuple(rec(self.proc.body))
        if not done:
            raise ScheduleError(f"no loop named '{var_name}'")
        return self.proc.with_(body=body)

    # ------------------------------------------------------------------ inspection
    def loops(self, stage: str):
        """Loop variable names of a stage's update nest, outermost first."""
        for s in ir.walk(self.proc.body):
            if isinstance(s, ir.For) and s.stage == stage:
                return [l.var for l in _chain(s)]
        raise ScheduleError(f"no stage '{stage}'")

    def show(self):
        return str(self.proc)

    def source(self) -> str:
        return emit_nki(self.proc, self.hw)

    # ------------------------------------------------------------------ split
    def split(self, loop: str, factor: int, names=None, perfect: bool = False):
        """i -> (io, ii) with i = io*factor + ii. If the extent is not provably divisible by
        `factor`, perfect=True records the assumption `extent % factor == 0` (emitted as an assert)."""
        if not isinstance(factor, int) or factor < 1:
            raise ScheduleError("split factor must be a positive int")
        l = self._find(loop)
        o, i = names or (f"{loop}.o", f"{loop}.i")
        self._fresh(o)
        self._fresh(i)
        ext = l.extent
        provable = (ext.is_const and ext.const % factor == 0) or (
            ext.terms and ext.const % factor == 0 and all(c % factor == 0 for _, c in ext.terms))
        assumptions = self.proc.assumptions
        if ext.is_const and ext.const % factor:
            raise ScheduleError(f"split({loop}, {factor}): the constant extent {ext.const} is not a multiple of {factor} (tail='cut' not implemented yet)")
        if not provable:
            if not perfect:
                raise ScheduleError(
                    f"split({loop}, {factor}): cannot prove {ext} is a multiple of {factor}. "
                    f"Pass perfect=True to assume it (the kernel will assert it), or use tail='cut' (not implemented yet)."
                )
            a = ir.Assumption(ext, factor)
            if a not in assumptions:
                assumptions = assumptions + (a,)

        def fn(f: ir.For):
            body = tuple(ir.subst_vars(b, {loop: var(o) * factor + var(i)}) for b in f.body)
            inner = ir.For(i, Aff(factor), body, f.stage, f.kind)
            return (ir.For(o, ext // factor, (inner,), f.stage, f.kind),)

        p = self._rewrite_loop(loop, fn).with_(assumptions=assumptions)
        self._renamed[loop] = (o, i)
        self._commit(f"split({loop}, {factor})", p)
        return o, i

    # ------------------------------------------------------------------ reorder
    def reorder(self, *loops: str):
        """Permute a perfectly nested chain of loops into the given order (outermost first)."""
        if len(set(loops)) != len(loops) or len(loops) < 2:
            raise ScheduleError("reorder needs two or more distinct loops")
        ls = [self._find(n) for n in loops]
        # outermost = the one that contains all the others
        top = None
        for cand in ls:
            if {l.var for l in _chain(cand)} >= set(loops):
                top = cand
                break
        if top is None:
            raise ScheduleError(f"reorder{loops}: loops are not a perfectly nested contiguous chain "
                                f"(each must be the only statement of the one above)")
        chain = _chain(top)
        names = [l.var for l in chain]
        n = len(loops)
        if set(names[:n]) != set(loops):
            raise ScheduleError(f"reorder{loops}: loops must be the outermost {n} loops of the chain {names}")
        head = chain[:n]
        inner_body = chain[n - 1].body
        by_name = {l.var: l for l in head}
        # rectangular: no loop extent depends on another loop var of the chain
        for l in head:
            if l.extent.free_vars() & set(loops):
                raise ScheduleError(f"reorder: extent of {l.var} ({l.extent}) depends on another loop in the nest")
        self._check_reorder_legal(head, inner_body, loops)
        new = _rebuild_chain(head, inner_body, [by_name[x] for x in loops])
        p = self._rewrite_loop(top.var, lambda f: (new,))
        self._commit(f"reorder{tuple(loops)}", p)

    def _check_reorder_legal(self, head, inner_body, new_order):
        P = {l.var: l.extent for l in head}
        if [l.var for l in head] == list(new_order):
            return
        bufs = ir.buffers_of(self.proc)
        inner_allocs = {s.buf.name for s in ir.walk(inner_body) if isinstance(s, ir.Alloc)}
        acc = {}
        for s in ir.walk(inner_body):
            for mode, buf, loc in ir.accesses(s):
                acc.setdefault(buf, []).append((mode, loc))
        for buf, lst in acc.items():
            if buf in inner_allocs or not any("w" in m for m, _ in lst):
                continue
            keys = {(loc.lo, loc.size) if isinstance(loc, ir.Window) else tuple(loc) for _, loc in lst}
            if len(keys) != 1:
                raise ScheduleError(f"reorder{tuple(new_order)}: cannot prove independence — '{buf}' is accessed at different indices "
                                    f"inside the nest: {sorted(str(k) for k in keys)}")
            key = next(iter(keys))
            idx = key[0] if isinstance(lst[0][1], ir.Window) else key
            appears = {v: P[v] for v in P if any(v in e.free_vars() for e in idx)}
            missing = [v for v in P if v not in appears]
            if appears and not injective(tuple(idx), appears):
                raise ScheduleError(f"reorder{tuple(new_order)}: cannot prove distinct iterations write distinct elements of '{buf}' (index {[str(e) for e in idx]})")
            if isinstance(lst[0][1], ir.Window):
                w = lst[0][1]
                for v in appears:
                    if not any(w.lo[d].coef(v) and w.size[d].is_const and abs(w.lo[d].coef(v)) >= w.size[d].const for d in range(len(w.lo))):
                        raise ScheduleError(f"reorder{tuple(new_order)}: windows written to '{buf}' by different iterations of '{v}' may overlap")
            if missing:
                bad = [m for m, _ in lst if m not in ("rw",)]
                if bad:
                    raise ScheduleError(
                        f"reorder{tuple(new_order)}: loop(s) {missing} do not index '{buf}', so they carry a dependence through it, "
                        f"and '{buf}' is not purely accumulated (+=) inside the nest. Reordering is only legal for associative reductions; "
                        f"separate the initialisation/readers of '{buf}' from this nest first (compute_at / fission)."
                    )
                self.notes.append(f"reorder{tuple(new_order)} reassociates the floating-point reduction into '{buf}'")

    # ------------------------------------------------------------------ compute_at
    def compute_at(self, stage: str, at: str):
        """Move the producer stage's loop nests (and its allocation) inside loop `at`, restricted to
        the region the consumers inside `at` actually read (interval analysis)."""
        at_loop = self._find(at)
        root = list(self.proc.body)
        prod_idx = [i for i, s in enumerate(root) if isinstance(s, ir.For) and s.stage in (stage, stage + ".init")]
        if not prod_idx:
            raise ScheduleError(f"stage '{stage}' has no loop nests at the top level (already computed_at somewhere?)")
        prods = [root[i] for i in prod_idx]
        buf = self._buf(stage)
        # consumer accesses
        reads = []
        for s in ir.walk(at_loop.body):
            if isinstance(s, (ir.Assign, ir.Reduce)):
                reads += [r.idx for r in ir.expr_reads(s.rhs) if r.buf == stage]
        if not reads:
            raise ScheduleError(f"compute_at({stage}, at={at}): no reads of '{stage}' inside loop '{at}'")
        try:
            lo, size = region(reads, loop_ranges(at_loop.body))
        except AnalysisError as e:
            raise ScheduleError(f"compute_at({stage}, at={at}): {e}") from None
        for z in size:
            self._const(z, f"compute_at({stage}, at={at}): the region of '{stage}' read inside '{at}'")

        new_prods = []
        for nest in prods:
            writes = [loc for s in ir.walk((nest,)) for m, b, loc in ir.accesses(s) if b == stage and "w" in m]
            widx = {tuple(w) for w in writes}
            if len(widx) != 1:
                raise ScheduleError(f"compute_at: producer nest writes '{stage}' at several index tuples")
            widx = next(iter(widx))
            mapping, ext_of = {}, {}
            for d, e in enumerate(widx):
                if len(e.terms) != 1 or e.const != 0 or e.terms[0][1] != 1 or not isinstance(e.terms[0][0], Var):
                    raise ScheduleError(f"compute_at: producer write index {[str(x) for x in widx]} must be a plain loop variable per dim")
                v = e.terms[0][0].name
                mapping[v] = lo[d] + var(v)
                ext_of[v] = size[d]
            nest2 = ir.subst_vars(nest, mapping)

            def setext(s):
                if isinstance(s, ir.For):
                    return replace(s, extent=ext_of.get(s.var, s.extent), body=tuple(setext(b) for b in s.body))
                return s

            nest2 = setext(nest2)
            new_prods.append(nest2)
        new_prods = [ir.remap_buf(n, stage, stage, lambda idx: tuple(i - l for i, l in zip(idx, lo))) for n in new_prods]
        new_body_stmts = tuple(ir.remap_buf(b, stage, stage, lambda idx: tuple(i - l for i, l in zip(idx, lo))) for b in at_loop.body)
        newbuf = buf.with_(shape=tuple(size))
        new_at = replace(at_loop, body=(ir.Alloc(newbuf),) + tuple(new_prods) + new_body_stmts)

        def rec(stmts):
            out = []
            for s in stmts:
                if isinstance(s, ir.For):
                    s = new_at if s.var == at else replace(s, body=tuple(rec(s.body)))
                out.append(s)
            return out

        rest = [s for i, s in enumerate(root) if i not in prod_idx and not (isinstance(s, ir.Alloc) and s.buf.name == stage)]
        p = self.proc.with_(body=tuple(rec(rest)))
        self._commit(f"compute_at({stage}, at={at})", p)

    # ------------------------------------------------------------------ memory
    def set_memory(self, buf: str, mem: str):
        if mem not in ir.MEMS:
            raise ScheduleError(f"unknown memory {mem}")
        b = self._buf(buf)
        if b.role != "temp":
            raise ScheduleError(f"set_memory: '{buf}' is a kernel argument/output and always lives in HBM")

        def f(s):
            if isinstance(s, ir.Alloc) and s.buf.name == buf:
                return ir.Alloc(s.buf.with_(mem=mem))
            return None

        p = self.proc.with_(body=tuple(ir.map_stmts(self.proc.body, f)))
        self._commit(f"set_memory({buf}, {mem})", p)

    def fold(self, buf: str, tile: Optional[int] = None):
        self._commit(f"fold({buf}, {self.hw.pmax if tile is None else tile})", self._fold_proc(self.proc, buf, tile))

    def _fold_proc(self, proc: ir.Proc, buf: str, tile: Optional[int] = None) -> ir.Proc:
        """Re-layout an on-chip buffer whose axis 0 is a multiple of `tile` (default `hw.pmax`) as
        `[tile, T, ...]`: row `tile*t + r` moves to `[r, t, ...]`. This is how a block of several
        partition tiles (e.g. a 512-row PSUM accumulator or output tile) is held on a 128-partition
        memory. Every access must split cleanly: its axis-0 offset is `tile*q + r` with `q` free of
        the loops that sweep within a tile and `r` provably inside `[0, tile)`; windows must not
        straddle a tile boundary."""
        b = ir.buffers_of(proc)[buf]
        if b.role != "temp":
            raise ScheduleError(f"fold({buf}): only temporaries can be re-laid out")
        tile = self.hw.pmax if tile is None else tile
        rows = self._const(b.shape[0], f"fold({buf}): axis 0 of '{buf}'")
        if rows % tile or rows == tile:
            raise ScheduleError(f"fold({buf}): axis 0 has {rows} rows, which must be a multiple (> 1) of the tile size {tile}")
        ranges = loop_ranges(proc.body)

        def divide(e: Aff, size: Aff):
            q, r = Aff(e.const // tile), Aff(e.const % tile)
            for atom, c in e.terms:
                if c % tile == 0:
                    q = q + Aff.of(atom) * (c // tile)
                else:
                    r = r + Aff.of(atom) * c
            try:
                lo, span = dim_interval(r, ranges)
            except AnalysisError as ex:
                raise ScheduleError(f"fold({buf}): {ex}") from None
            if not (lo.is_const and span.is_const and size.is_const) or lo.const < 0 or lo.const + span.const + size.const - 1 > tile:
                raise ScheduleError(
                    f"fold({buf}): axis-0 access '{e}' (window size {size}) is not confined to one {tile}-row tile; "
                    f"split the loop that indexes axis 0 by {tile} first")
            return q, r

        def fidx(idx):
            q, r = divide(idx[0], Aff(1))
            return (r, q) + tuple(idx[1:])

        def fwin(w):
            q, r = divide(w.lo[0], w.size[0])
            pts = tuple(sorted([p + 1 if p >= 1 else p for p in w.points] + [1]))
            return ir.Window(w.buf, (r, q) + tuple(w.lo[1:]), (w.size[0], Aff(1)) + tuple(w.size[1:]), pts)

        def remap(s):
            if isinstance(s, ir.For):
                return replace(s, body=tuple(remap(x) for x in s.body))
            if isinstance(s, ir.Alloc) and s.buf.name == buf:
                return ir.Alloc(b.with_(shape=(Aff(tile), Aff(rows // tile)) + tuple(b.shape[1:])))
            if isinstance(s, (ir.Assign, ir.Reduce)):
                rhs = ir._remap_expr(s.rhs, buf, buf, fidx)
                return type(s)(s.buf, fidx(s.idx) if s.buf == buf else s.idx, rhs)
            if isinstance(s, ir.Call):
                return ir.Call(s.instr, tuple((rl, fwin(w) if w.buf == buf else w) for rl, w in s.args), s.attrs)
            return s

        return proc.with_(body=tuple(remap(x) for x in proc.body))

    def _staging(self, tensor, at, mem, name, direction, fold=False):
        at_loop = self._find(at)
        self._fresh(name)
        tb = self._buf(tensor)
        if direction == "in":
            idxs = [r.idx for s in ir.walk(at_loop.body) if isinstance(s, (ir.Assign, ir.Reduce)) for r in ir.expr_reads(s.rhs) if r.buf == tensor]
            if any(m != "r" for s in ir.walk(at_loop.body) for m, b, _ in ir.accesses(s) if b == tensor):
                raise ScheduleError(f"stage_in({tensor}): '{tensor}' is also written inside '{at}'; use stage_out or restructure")
        else:
            acc = [(m, loc) for s in ir.walk(at_loop.body) for m, b, loc in ir.accesses(s) if b == tensor]
            if any(m != "w" for m, _ in acc):
                raise ScheduleError(f"stage_out({tensor}): '{tensor}' is also read inside '{at}' (needs a load as well; not supported yet)")
            idxs = [tuple(loc) for _, loc in acc]
        if not idxs:
            raise ScheduleError(f"stage_{direction}({tensor}, at={at}): no accesses to '{tensor}' inside '{at}'")
        try:
            lo, size = region(idxs, loop_ranges(at_loop.body))
        except AnalysisError as e:
            raise ScheduleError(f"stage_{direction}({tensor}): {e}") from None
        for z in size:
            self._const(z, f"stage_{direction}({tensor}, at={at}): the window of '{tensor}' accessed inside '{at}'")
        nb = ir.Buffer(name, tuple(size), tb.dtype, mem, "temp")
        full = ir.Window(name, tuple(Aff(0) for _ in size), tuple(size))
        win = ir.Window(tensor, tuple(lo), tuple(size))
        rebase = lambda idx: tuple(i - l for i, l in zip(idx, lo))
        body = tuple(ir.remap_buf(b, tensor, name, rebase) for b in at_loop.body)
        if direction == "in":
            src_mem, dst_mem = tb.mem, mem
            cp = self._copy_call(full, win, src_mem, dst_mem)
            new_body = (ir.Alloc(nb), cp) + body
        else:
            cp = self._copy_call(win, full, mem, tb.mem)
            new_body = (ir.Alloc(nb),) + body + (cp,)
        new_at = replace(at_loop, body=new_body)
        p = self._rewrite_loop(at, lambda f: (new_at,))
        if fold:
            p = self._fold_proc(p, name)
        self._commit(f"stage_{direction}({tensor}, at={at}, {mem}, {name}{', fold=True' if fold else ''})", p)

    @staticmethod
    def _copy_call(dst: ir.Window, src: ir.Window, src_mem: str, dst_mem: str):
        if ir.HBM in (src_mem, dst_mem):
            if src_mem == dst_mem:
                raise ScheduleError("HBM -> HBM staging copies are not supported")
            return ir.Call("ns.sync.dma_copy", (("dst", dst), ("src", src)))
        return ir.Call("ns.vector.tensor_copy", (("dst", dst), ("src", src)))

    def stage_in(self, tensor: str, at: str, mem: str, name: str, fold: bool = False):
        """cache_read: copy the window of `tensor` read inside loop `at` into a new buffer in
        `mem` at the start of `at`'s body; accesses are redirected. The copy is created directly as
        the right ns instruction (HBM->SBUF: ns.sync.dma_copy)."""
        self._staging(tensor, at, mem, name, "in", fold)
        return name

    def stage_out(self, tensor: str, at: str, mem: str, name: str, fold: bool = False):
        """cache_write: writes to `tensor` inside `at` go to a new buffer in `mem`; the window is
        copied out after `at`'s body. With `fold=True` the new buffer is also folded (see `fold`), for
        windows taller than the partition count."""
        self._staging(tensor, at, mem, name, "out", fold)
        return name

    # ------------------------------------------------------------------ instruction selection
    def replace(self, loop: str, instr: str):
        """Tensorize: replace the loop nest rooted at `loop` by an ns instruction call when the
        nest matches the instruction's semantics (structural unification)."""
        if instr not in INSTRS:
            raise ScheduleError(f"unknown instruction {instr}; known: {list(INSTRS)}")
        l = self._find(loop)
        bufs = ir.buffers_of(self.proc)
        if instr == "ns.tensor.matmul":
            call = self._match_matmul(l)
        else:
            call = copy_nest_to_call(l, bufs)
            if call is None or call.instr != instr:
                raise ScheduleError(f"replace({loop}, {instr}): the nest is not a pure elementwise copy between compatible memories")
        p = self._rewrite_loop(loop, lambda f: (call,))
        self._commit(f"replace({loop}, {instr})", p)

    def _match_matmul(self, l: ir.For) -> ir.Call:
        chain = _chain(l)
        if len(chain) != 3 or len(chain[-1].body) != 1 or not isinstance(chain[-1].body[0], ir.Reduce):
            raise ScheduleError(f"replace(ns.tensor.matmul): '{l.var}' must root a perfect 3-loop nest around a single `+=`")
        red = chain[-1].body[0]
        if not (isinstance(red.rhs, ir.Mul) and isinstance(red.rhs.a, ir.Read) and isinstance(red.rhs.b, ir.Read)):
            raise ScheduleError("replace(ns.tensor.matmul): the update must be dst[m,n] += a[k,m] * b[k,n]")
        a, b = red.rhs.a, red.rhs.b
        ext = {c.var: c.extent for c in chain}
        vs = set(ext)

        def vars_in(idx):
            return {v for v in vs if any(v in e.free_vars() for e in idx)}

        dv, av, bv = vars_in(red.idx), vars_in(a.idx), vars_in(b.idx)
        ks = av & bv - dv
        ms = dv & av - bv
        ns_ = dv & bv - av
        if len(ks) != 1 or len(ms) != 1 or len(ns_) != 1:
            raise ScheduleError("replace(ns.tensor.matmul): cannot identify (k, m, n) loops from the access pattern")
        k, m, n = next(iter(ks)), next(iter(ms)), next(iter(ns_))
        wd, vd = nest_window(red.buf, red.idx, {m: ext[m], n: ext[n]})
        wa, va = nest_window(a.buf, a.idx, {k: ext[k], m: ext[m]})
        wb, vb = nest_window(b.buf, b.idx, {k: ext[k], n: ext[n]})
        if wd is None or wa is None or wb is None:
            raise ScheduleError("replace(ns.tensor.matmul): operand indices must be plain loop variables (plus offsets)")
        if [x for x in vd if x] != [m, n]:
            raise ScheduleError(f"replace(ns.tensor.matmul): dst must be indexed [m, n], found order {vd}")
        if [x for x in va if x] != [k, m]:
            raise ScheduleError(f"replace(ns.tensor.matmul): stationary operand must be laid out [k, m] (contraction on the partition axis); found {va}. "
                                f"Stage a transposed copy of '{a.buf}' first.")
        if [x for x in vb if x] != [k, n]:
            raise ScheduleError(f"replace(ns.tensor.matmul): moving operand must be laid out [k, n]; found {vb}")
        for v in (k, m, n):
            const_int(ext[v], f"extent of {v}")
        return ir.Call("ns.tensor.matmul", (("dst", wd), ("stationary", wa), ("moving", wb)), (("accumulate", True),))

    def fold_init(self, buf: str):
        """Drop the zero-initialisation nest of an accumulator: the first matmul into a freshly
        allocated PSUM tile overwrites (accumulate=None)."""
        b = self._buf(buf)
        if b.mem != ir.PSUM:
            raise ScheduleError(f"fold_init({buf}): only valid for PSUM accumulators (it is in {b.mem})")
        bufs = ir.buffers_of(self.proc)
        found = {}

        def is_init(s):
            if not (isinstance(s, ir.For) and s.stage == buf + ".init"):
                return False
            ch = _chain(s)
            inner = ch[-1].body
            if len(inner) != 1 or not isinstance(inner[0], ir.Assign) or inner[0].buf != buf or inner[0].rhs != ir.Lit(0.0):
                return False
            w, _ = nest_window(buf, inner[0].idx, {c.var: c.extent for c in ch})
            return w is not None and w.is_full(bufs)

        def rec(stmts):
            out = []
            for s in stmts:
                if is_init(s):
                    found["init"] = found.get("init", 0) + 1
                    continue
                if isinstance(s, ir.For):
                    s = replace(s, body=tuple(rec(s.body)))
                out.append(s)
            return out

        body = tuple(rec(self.proc.body))
        if found.get("init") != 1:
            raise ScheduleError(f"fold_init({buf}): expected exactly one full zero-initialisation nest, found {found.get('init', 0)}")
        n_mm = []
        bufs = ir.buffers_of(self.proc)

        def rec2(stmts, stack):
            out = []
            for s in stmts:
                if isinstance(s, ir.For):
                    s = replace(s, body=tuple(rec2(s.body, stack + [s])))
                elif isinstance(s, ir.Call) and s.instr == "ns.tensor.matmul" and s.arg("dst").buf == buf:
                    if not self._tiles_buffer(s.arg("dst"), bufs[buf], {l.var: l.extent for l in stack}):
                        raise ScheduleError(
                            f"fold_init({buf}): the matmuls do not write every element of '{buf}' as non-overlapping whole tiles; "
                            f"first-write-overwrite would leave stale data")
                    n_mm.append(1)
                    s = replace(s, attrs=tuple((k, v) for k, v in s.attrs if k != "accumulate") + (("accumulate", None),))
                out.append(s)
            return out

        body = tuple(rec2(body, []))
        if not n_mm:
            raise ScheduleError(f"fold_init({buf}): no ns.tensor.matmul accumulates into '{buf}' (replace the update nest first)")
        # every other access to buf must be a read after the matmuls (checked by the oracle step too)
        self._commit(f"fold_init({buf})", self.proc.with_(body=body))

    @staticmethod
    def _tiles_buffer(w: ir.Window, b: ir.Buffer, loops: dict) -> bool:
        """Do the windows `w` takes as the enclosing `loops` run tile `b` exactly (every element
        once per sweep of the loops that move it)? Per dimension the moving loop coefficients must
        form a mixed-radix system starting at the window size and ending at the buffer extent."""
        for d, (lo, z) in enumerate(zip(w.lo, w.size)):
            full = b.shape[d]
            if not (full.is_const and z.is_const and lo.const == 0):
                return False
            terms = []
            for atom, c in lo.terms:
                if not (isinstance(atom, Var) and atom.name in loops and loops[atom.name].is_const):
                    return False
                terms.append((c, loops[atom.name].const))
            span = z.const
            for c, ext in sorted(terms):
                if c != span:
                    return False
                span *= ext
            if span != full.const:
                return False
        return True

    def hoist(self, buf: str, to: Optional[str]):
        """Re-stage `buf` (created by stage_in) at the outer loop `to`: its fill moves out of every
        loop between `to` and the original site. Loops whose variable strides the source window by
        exactly one tile (e.g. ko over K) *grow* the buffer: along the partition axis the tiles are
        folded into a new axis 1 ([128, T, n]); along a free axis the buffer is enlarged. Loops that
        do not index the source (e.g. mo for an rhs tile) simply reuse the data. `to=None` hoists to
        the top of the kernel."""
        to_loop = self._find(to) if to is not None else None
        container = to_loop.body if to_loop is not None else self.proc.body

        def search(stmts, path):
            for s in stmts:
                if isinstance(s, ir.Alloc) and s.buf.name == buf:
                    return path, stmts
                if isinstance(s, ir.For):
                    r = search(s.body, path + [s])
                    if r:
                        return r
            return None

        r = search(container, [])
        if r is None:
            raise ScheduleError(f"hoist({buf}, to={to}): '{buf}' is not allocated inside loop '{to}'")
        path, block = r
        if not path:
            raise ScheduleError(f"hoist({buf}, to={to}): '{buf}' is already allocated directly in '{to}'")
        ob = self._buf(buf)
        writers = [s for s in ir.walk(self.proc.body) if not isinstance(s, ir.For) for m, b, _ in ir.accesses(s) if b == buf and "w" in m]
        if len(writers) != 1 or not isinstance(writers[0], ir.Call) or writers[0] not in block:
            raise ScheduleError(f"hoist({buf}): expected a single fill (copy instruction) next to the allocation; found {len(writers)} writers")
        fill = writers[0]
        if not fill.arg("dst").is_full(ir.buffers_of(self.proc)):
            raise ScheduleError(f"hoist({buf}): the fill must write the whole buffer")
        src = fill.arg("src")
        for s in ir.walk(container):
            if any(b == src.buf and "w" in m for m, b, _ in ir.accesses(s)):
                raise ScheduleError(f"hoist({buf}): source '{src.buf}' is written inside '{to}'")
        hv = {l.var: l.extent for l in path}
        growth = {}  # var -> dim
        for d, lo in enumerate(src.lo):
            vs = [v for v in hv if lo.coef(v)]
            if len(vs) > 1:
                raise ScheduleError(f"hoist({buf}): several hoisted loops {vs} index dim {d} of the source")
            if vs:
                v = vs[0]
                if not src.size[d].is_const or lo.coef(v) != src.size[d].const:
                    raise ScheduleError(f"hoist({buf}): loop '{v}' strides the source window by {lo.coef(v)} but the tile is {src.size[d]} — not a whole-tile stride")
                growth[v] = d
        for v in hv:
            if v not in growth and any(v in e.free_vars() for e in src.lo):
                raise ScheduleError(f"hoist({buf}): source offsets depend on '{v}' non-affinely")
        if sum(1 for d in growth.values() if d == 0) > 1:
            raise ScheduleError(f"hoist({buf}): more than one hoisted loop grows the partition axis")
        fold_v = next((v for v, d in growth.items() if d == 0), None)
        sizes = list(ob.shape)
        for v, d in growth.items():
            if d != 0:
                sizes[d] = sizes[d] * hv[v]
        if fold_v is not None:
            sizes.insert(1, hv[fold_v])
        newbuf = ob.with_(shape=tuple(sizes))

        def fidx(idx):
            base = list(idx)
            for v, d in growth.items():
                if d != 0:
                    base[d] = base[d] + var(v) * ob.shape[d].const
            if fold_v is not None:
                base.insert(1, var(fold_v))
            return tuple(base)

        def fwin(w):
            lo = list(w.lo)
            size = list(w.size)
            pts = list(w.points)
            for v, d in growth.items():
                if d != 0:
                    lo[d] = lo[d] + var(v) * ob.shape[d].const
            if fold_v is not None:
                lo.insert(1, var(fold_v))
                size.insert(1, Aff(1))
                pts = [p + 1 if p >= 1 else p for p in pts] + [1]
            return ir.Window(w.buf, tuple(lo), tuple(size), tuple(sorted(pts)))

        def remap(s):
            if isinstance(s, ir.For):
                return replace(s, body=tuple(remap(x) for x in s.body if not (x is fill or (isinstance(x, ir.Alloc) and x.buf.name == buf))))
            if isinstance(s, (ir.Assign, ir.Reduce)):
                rhs = ir._remap_expr(s.rhs, buf, buf, fidx)
                if s.buf == buf:
                    raise ScheduleError(f"hoist({buf}): scalar writes to the buffer are not supported")
                return type(s)(s.buf, s.idx, rhs)
            if isinstance(s, ir.Call):
                return ir.Call(s.instr, tuple((rl, fwin(w) if w.buf == buf else w) for rl, w in s.args), s.attrs)
            return s

        # the new fill: one copy per tile of every growing loop, executed once at `to`
        new_vars = {v: f"{buf}_{v}_ld" for v in growth}
        for nv in new_vars.values():
            self._fresh(nv)
        subs = {v: var(nv) for v, nv in new_vars.items()}
        dst_lo = [Aff(0)] * len(ob.shape)
        dst = ir.Window(buf, tuple(dst_lo), tuple(ob.shape))
        dst = fwin(dst)
        dst = ir.Window(dst.buf, tuple(l.subs(subs) for l in dst.lo), dst.size, dst.points)
        srcw = ir.Window(src.buf, tuple(l.subs(subs) for l in src.lo), src.size, src.points)
        stmt = ir.Call(fill.instr, (("dst", dst), ("src", srcw)), fill.attrs)
        for v in reversed(list(growth)):
            stmt = ir.For(new_vars[v], hv[v], (stmt,), "", "auto")
        new_body = (ir.Alloc(newbuf), stmt) + tuple(remap(x) for x in container)
        if to_loop is None:
            p = self.proc.with_(body=new_body)
        else:
            new_to = replace(to_loop, body=new_body)
            p = self._rewrite_loop(to, lambda f: (new_to,))
        self._commit(f"hoist({buf}, to={to})", p)

    def mark(self, loop: str, kind: str):
        """Assert (or force a more conservative) loop kind: affine | sequential | static."""
        if kind not in ("affine", "sequential", "static"):
            raise ScheduleError("kind must be affine|sequential|static")
        self._find(loop)
        if kind == "affine":
            inferred = infer_kinds(self.proc)
            got = ir.find_loop(inferred.body, loop).kind
            if got != "affine":
                raise ScheduleError(f"mark({loop}, affine): dependence analysis says this loop carries a dependence (inferred {got})")

        def fn(f):
            return (replace(f, kind=kind),)

        self._commit(f"mark({loop}, {kind})", self._rewrite_loop(loop, fn))
