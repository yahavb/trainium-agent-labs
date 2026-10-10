"""INDEX: the NKI names a prompt offers, each with one line on what it does, and fix_name() for a real
name written under the wrong prefix.

Measured on seat-198 (Qwen3-32B, 2026-10-10 ~22:25 UTC; 8 single plans per arm at levels 1 and 4,
LOOKUP off; script /workspace/scratch/koitu/slate/index_test.py). Against the names-only api_map:
  - names that exist, in the right module: L1 48% -> 100%, L4 55% -> 91%;
  - plans with any wrong name: L1 6/8 -> 0/8, L4 8/8 -> 2/8;
  - nisa.tensor_partition_reduce, which reduces ACROSS partitions (the wrong axis for pooling):
    L1 5/8 -> 1/8, L4 4/8 -> 0/8;
  - the index is 436-490 tokens instead of 784.
The names-only map is still there: `agent2.py --index names`, and the LOOKUP topic 'all'.

fix_name() is the safety net. In the names-only plans, 31% of all names were real functions under the
wrong prefix (nl.tensor_reduce for nisa.tensor_reduce, nl.reshape for the tile method t.reshape), and
the planner dropped them and planned again.
"""

# (group, entry, what it does, names that must exist for the line to show, first level, withheld at)
DESCRIBED = [
    ("Allocate", "nl.ndarray(shape, dtype, buffer=nl.sbuf | nl.psum | nl.shared_hbm)",
     "a new on-chip tile, or the output tensor in HBM", ["nl.ndarray"], 1, ()),
    ("Move", "nisa.dma_copy(dst, src)",
     "copies between HBM and SBUF; both sides hold the same number of elements", ["nisa.dma_copy"], 1, ()),
    ("Views", "t.reshape(shape)", "the same elements regrouped into a new shape, no copy",
     ["tile.reshape"], 1, ()),
    ("", "t.permute(dims)", "reorders axes, no copy; keep axis 0 (partition) first", ["tile.permute"], 1,
     (2,)),
    ("", "t.ap([[stride, count], ...])", "any strided view, one pair per axis, no copy", ["tile.ap"], 1, ()),
    ("", "x[a:b, c:d], nl.ds(start, size)", "a slice of a tensor or tile", ["nl.ds"], 1, ()),
    ("Elementwise", "nisa.tensor_scalar(dst, data, op0, operand0)",
     "data <op0> a constant, or a per-row [P,1] tile", ["nisa.tensor_scalar"], 1, ()),
    ("", "nisa.tensor_tensor(dst, data1, data2, op)", "two tiles of the same shape",
     ["nisa.tensor_tensor"], 1, ()),
    ("", "nisa.activation(dst, op, data)", "exp, rsqrt, ... applied to every element",
     ["nisa.activation"], 1, ()),
    ("", "nisa.reciprocal(dst, data)", "1 / x for every element", ["nisa.reciprocal"], 1, ()),
    ("Reduce", "nl.sum(x, axis, keepdims), nl.mean(...), nl.max(...)",
     "over free (trailing) axes; returns a new tile", ["nl.sum"], 1, ()),
    ("", "nisa.tensor_reduce(dst, op, data, axis, keepdims)", "over free axes, into dst",
     ["nisa.tensor_reduce"], 1, ()),
    ("", "nisa.tensor_partition_reduce(dst, op, data)", "ACROSS partitions (axis 0) only",
     ["nisa.tensor_partition_reduce"], 1, ()),
    ("Copy", "nisa.tensor_copy(dst, src)", "on-chip copy or cast, e.g. PSUM to SBUF", ["nisa.tensor_copy"],
     1, ()),
    ("Matmul", "nisa.nc_matmul(dst, stationary, moving)", "dst (a PSUM tile) = stationary.T @ moving",
     ["nisa.nc_matmul"], 3, ()),
    ("Transpose", "nisa.nc_transpose(dst, data)",
     "dst (a PSUM tile) = data with partition and free axes swapped", ["nisa.nc_transpose"], 3, ()),
    ("Ops (op=)", "nl.add, nl.subtract, nl.multiply, nl.divide, nl.maximum, nl.exp",
     "passed to the nisa functions above", [], 1, ()),
    ("Loops", "for i in range(n)", "loops unroll when the kernel is traced", [], 1, ()),
]

HEADER = ("NKI names (write each exactly as shown: nisa = nki.isa, nl = nki.language, t = a tile):")


def described(retriever, level):
    """The index for this level. A line is left out before its first level, at a level that withholds
    it (here or on its card), and when its name doesn't exist in the installed nki."""
    lines = [HEADER]
    for group, entry, about, names, first, withheld in DESCRIBED:
        if level is not None and (level < first or level in withheld):
            continue
        if not all(retriever.exists(n) and retriever.shown(n, level) for n in names):
            continue
        lines.append(f"{group:<12} {entry}: {about}")
    return "\n".join(lines)


def fix_name(retriever, name, level=None):
    """The real name for one written under the wrong prefix: nl.X -> nisa.X (or back) when only the
    other module has a function X, and nl.X / nisa.X -> tile.X for a tile method. Never to a name whose
    card is withheld at this level (nl.permute stays unknown at level 2 rather than becoming
    t.permute). Anything else comes back unchanged, so a name that exists nowhere still reaches the
    planner's re-plan."""
    if retriever.exists(name):
        return name
    alias, _, attr = retriever.short(name).rpartition(".")
    if alias in ("nl", "nisa"):
        other = "nisa" if alias == "nl" else "nl"
        if callable(retriever.get(f"{other}.{attr}")) and retriever.shown(f"{other}.{attr}", level):
            return f"{other}.{attr}"
    if retriever.exists(f"tile.{attr}") and retriever.shown(f"tile.{attr}", level):
        return f"tile.{attr}"
    return name
