"""Single source of truth for the kernel rules: what is banned, and the repair instruction for each.

Static (AST) and runtime (guard) checks both report violations as (kind, detail); `FIX` turns a kind
into an instruction the model can act on. A verdict ("line 16 calls np.max") never fixed anything in
the worked example; an instruction ("replace with a column loop that ...") did.
"""

TILE_R, TILE_C = 128, 512

# numpy names, grouped so each group maps to one repair instruction
SUM_LIKE = {
    "sum", "nansum", "cumsum", "nancumsum", "mean", "nanmean", "average", "prod", "nanprod",
    "cumprod", "nancumprod", "std", "var", "nanstd", "nanvar", "median", "nanmedian", "trace",
    "count_nonzero", "any", "all", "percentile", "quantile",
}
MAX_LIKE = {
    "max", "amax", "nanmax", "min", "amin", "nanmin", "ptp", "argmax", "argmin", "nanargmax",
    "nanargmin", "sort", "argsort", "partition", "argpartition",
}
LINALG = {
    "dot", "matmul", "inner", "outer", "vdot", "tensordot", "einsum", "einsum_path", "kron",
    "linalg", "convolve", "correlate", "fft", "cross",
}
LAYOUT = {
    "transpose", "swapaxes", "moveaxis", "rollaxis", "permute_dims", "matrix_transpose",
    "reshape", "ravel", "squeeze", "expand_dims", "broadcast_to", "broadcast_arrays",
    "atleast_1d", "atleast_2d", "atleast_3d", "tile", "repeat", "flip", "fliplr", "flipud",
    "rot90", "roll", "newaxis", "lib",
}
INDEXING = {
    "take", "put", "take_along_axis", "put_along_axis", "choose", "compress", "extract",
    "nonzero", "argwhere", "flatnonzero", "select", "piecewise", "place", "putmask", "ix_",
    "indices", "meshgrid", "mgrid", "ogrid", "fromfunction", "diag", "diagonal", "tril", "triu",
    "tri", "vectorize", "apply_along_axis", "apply_over_axes", "frompyfunc",
}
ESCAPE = {
    "asarray", "array", "asanyarray", "ascontiguousarray", "asfortranarray", "copy",
    "frombuffer", "fromiter", "ctypeslib", "random", "load", "save", "memmap", "testing",
}
BANNED_NP = SUM_LIKE | MAX_LIKE | LINALG | LAYOUT | INDEXING | ESCAPE

# ndarray methods / attributes
BANNED_METHODS = (SUM_LIKE | MAX_LIKE | {
    "dot", "transpose", "swapaxes", "reshape", "ravel", "flatten", "squeeze", "repeat",
    "nonzero", "compress", "take", "put", "choose", "diagonal", "tolist", "view",
}) - {"median", "nanmedian", "percentile", "quantile"}
BANNED_ATTRS = {"T", "mT", "flat"}

ALLOWED_IMPORTS = {"numpy", "math"}
BANNED_BUILTINS = {"exec", "eval", "compile", "open", "__import__", "globals", "locals", "vars",
                   "input", "breakpoint", "memoryview"}
REDUCING_BUILTINS = {"sum", "max", "min", "sorted"}


def np_kind(name):
    if name in SUM_LIKE:
        return "sum-like"
    if name in MAX_LIKE:
        return "max-like"
    if name in LINALG:
        return "linalg"
    if name in LAYOUT:
        return "layout"
    if name in INDEXING:
        return "indexing-fn"
    return "escape"


_TILE_LOOP = (f"loop i0 over rows in steps of {TILE_R} and j0 over columns in steps of {TILE_C}, "
              f"with i1 = min(i0 + {TILE_R}, M) and j1 = min(j0 + {TILE_C}, N), and operate only on "
              f"x[i0:i1, j0:j1]")

FIX = {
    "sum-like": "Replace `{what}` with an explicit python loop over the columns of the tile that "
                "accumulates into a 1-D array of length (i1 - i0): acc = np.zeros(i1 - i0); "
                "for j in range(j0, j1): acc += x[i0:i1, j]. For a mean, divide the full-row sum by N.",
    "max-like": "Replace `{what}` with an explicit python loop over the columns of the tile: "
                "start from acc = np.full(i1 - i0, -np.inf) (use +np.inf for a min) and do "
                "acc = np.maximum(acc, x[i0:i1, j]) for each column j.",
    "linalg": "Replace `{what}` with explicit python loops: accumulate products one index at a "
              "time, e.g. for k in range(K): acc += a[i0:i1, k] * b[k, j] (a column times a scalar).",
    "layout": "Remove `{what}`. Do not reshape, transpose or add axes; move data with explicit "
              "loops, e.g. for r in range(i0, i1): out[j0:j1, r] = x[r, j0:j1].",
    "indexing-fn": "Replace `{what}` with plain slices and explicit python loops.",
    "escape": "Remove `{what}`; work directly on slices of the input arrays.",
    "whole-array": "Do not touch more than one tile at a time ({what}). " + _TILE_LOOP + ".",
    "fancy-index": "Index only with integers and slices ({what}). Replace the index array or mask "
                   "with a python loop, or with elementwise math such as np.maximum(t, 0.0) or "
                   "np.where(cond, a, b).",
    "newaxis": "Remove `{what}`; do not add axes. Loop over columns and combine 1-D column "
               "vectors of equal length, e.g. out[i0:i1, j] = (x[i0:i1, j] - m) * s.",
    "broadcast": "Elementwise operands must have identical shapes; scalars are fine ({what}). "
                 "Loop over columns: out[i0:i1, j] = (x[i0:i1, j] - m) * s with m, s of length "
                 "i1 - i0, or use a scalar such as g[j].",
    "reduction": "Replace `{what}` with an explicit python loop over columns that accumulates "
                 "into a 1-D array.",
    "builtin-reduce": "Replace builtin `{what}` over an iterable with an explicit loop "
                      "(acc = acc + v or acc = np.maximum(acc, v)).",
    "matmul-op": "Remove the `@` operator; accumulate products with explicit loops: "
                 "for k in range(K): acc += a[i0:i1, k] * b[k, j].",
    "import": "Only `import numpy as np` and `import math` are allowed; remove `{what}`.",
    "builtin": "Remove the call to `{what}`.",
    "keepdims": "Remove `keepdims`; reductions must be explicit loops.",
    "syntax": "Fix the syntax error: {what}.",
    "no-kernel": "Define a top-level function `{what}`.",
}


def fix_for(kind, what):
    return FIX[kind].format(what=what)
