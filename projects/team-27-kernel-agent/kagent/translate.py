"""Verdict -> instruction. Turns a harness Report into ONE change for the model to make.

The worked example in the challenge: feeding back `line 16: calls banned max` reproduced the same
violation; "replace the np.max and np.sum calls with explicit python loops over the columns ..."
fixed it in one round. So every directive names an action, and only one, chosen in this order:
the kernel must load, then run, then return the right shape, then obey the rules, then be right.

`key` is a coarse failure class. The loop uses it to notice repeats (the numbers in a message
change every round; the class doesn't) and the taxonomy groups by it.
"""
import re
from dataclasses import dataclass

from .harness import ROUNDING_CEILING, common_facts, fact_support, pattern_tag
from .rules import TILE_C, TILE_R

# a kernel can break several rules; fix the ones that change its structure first
RULE_ORDER = ["syntax", "no-kernel", "import", "builtin", "escape", "whole-array", "linalg",
              "matmul-op", "sum-like", "max-like", "reduction", "builtin-reduce", "layout",
              "fancy-index", "newaxis", "broadcast", "keepdims", "indexing-fn"]

TAG_MEANING = {
    "partial-col-tile": f"N is not a multiple of {TILE_C}, so the last column tile is partial",
    "partial-row-tile": f"M is not a multiple of {TILE_R}, so the last row tile is partial",
    "multi-col-tile": f"N > {TILE_C}, so there is more than one column tile",
    "multi-row-tile": f"M > {TILE_R}, so there is more than one row tile",
    "dim-1": "M or N is exactly 1",
    "prime-dim": "a dimension is prime",
    "all-negative": "every value is negative",
    "huge-negative": "values are around -1e30",
    "large-magnitude": "values are around +-1e4",
    "zeros": "every value is 0",
    "constant-rows": "every row holds one repeated value",
    "large-mean": "values are around 1e4 with spread 1",
    "cancelling": "values of +-1e6 cancel to a small result",
    "negative-scale": "the scale a is negative",
    "huge-magnitude": "values are around +-1e18",
    "w-ge-S": "the window w is at least the sequence length",
    "w-0": "the window w is 0",
    "receptive-field-eq-L": "the receptive field equals L (one output)",
    "dilation-gt1": "dilation > 1",
    "stride-gt1": "stride > 1",
    "index-coded": "values are 1, 2, 3, ... (they encode their position)",
    "partial-k-tile": f"K is not a multiple of {TILE_R}",
    "multi-k-tile": f"K > {TILE_R}",
    "k-1": "K is exactly 1",
}


# when the failures are exactly the cases with one tag, some tags have a known fix
TAG_FIX = {
    "huge-magnitude": "Values around 1e18 overflow float32 when squared (float32 max is 3.4e38). "
                      "Accumulate the squares in float64: cast each column with "
                      ".astype(np.float64) before squaring it.",
    "partial-k-tile": "Results are wrong exactly when K is not a multiple of the K tile. Loop "
                      "`for k0 in range(0, K, step)` with `k1 = min(k0 + step, K)` so the last, "
                      "partial K tile is included, and use its real size k1 - k0.",
    "multi-row-tile": "Results are wrong exactly when there is more than one row tile. Every tile "
                      "must use its own origin: read and write global row i0 + r, never the local "
                      "r, so later tiles do not overwrite earlier ones.",
    "multi-col-tile": "Results are wrong exactly when there is more than one column tile. Every "
                      "tile must use its own origin: read and write global column j0 + c, never "
                      "the local c, so later tiles do not overwrite earlier ones.",
    "w-ge-S": "Results are wrong exactly when the window w is at least the sequence length. Clip "
              "the window to j in [max(0, i - w), min(S - 1, i + w)].",
    "w-0": "Results are wrong exactly when w = 0. Then the window is just j = i and out[i] = v[i].",
    "receptive-field-eq-L": "Results are wrong exactly when the receptive field dilation*(K-1)+1 "
                            "equals L. Then L_out = 1: compute L_out = (L - dilation*(K-1) - 1) "
                            "// stride + 1 and let every loop run over range(L_out).",
    "dilation-gt1": "Results are wrong exactly when dilation > 1. Tap k of output t reads "
                    "x[c, t*stride + k*dilation].",
    "stride-gt1": "Results are wrong exactly when stride > 1. Output t starts at input position "
                  "t*stride.",
    "k-1": "Results are wrong exactly when K = 1. Make every K loop and slice work for K = 1.",
    "dim-1": "Results are wrong exactly when M or N is 1. Make every loop and slice work for a "
             "dimension of exactly 1.",
}


# the same tag can mean different bugs at different levels (large-mean: exp overflow in softmax,
# cancellation in layernorm), so these take precedence over the generic table and over nonfinite
LEVEL_TAG_FIX = {
    ("layernorm", "large-mean"):
        "Rows with a large mean (around 1e4) and spread 1 lose all precision: computing the "
        "variance as mean(x**2) - mean(x)**2 cancels catastrophically (and can go negative). "
        "Compute the mean first, then the variance as the mean of (x - mean)**2 in a second pass, "
        "accumulating in float64.",
}


# E17: "change only that and keep everything else identical" blocks a fix that needs the loops
# rebuilt (upstream NKI L4 sat on one error 72 of 80 attempts). Only these keys are local edits;
# every other directive tells the model it may restructure.
LOCAL_KEYS = {"rule:escape", "rule:import", "rule:builtin", "rule:keepdims", "rule:fancy-index",
              "numeric:identity", "numeric:precision", "numeric:dilation", "syntax", "no-kernel",
              "shape", "load-crash"}


@dataclass
class Directive:
    key: str            # failure class, e.g. "rule:sum-like", "numeric:partial-col"
    instruction: str    # the one change to make
    evidence: str = ""  # short distilled evidence shown with it

    @property
    def structural(self):
        return self.key not in LOCAL_KEYS

    def short(self, n=110):
        s = self.instruction.split(". ")[0]
        return s if len(s) <= n else s[: n - 3] + "..."


def _first_sentence_of(msg):
    return msg.split(";")[0]


# Row-wise normalisations scale a (rows, cols) tile by a per-row vector, which needs broadcasting,
# which the rules ban. v2 L4: 13 of 37 failures were this (a broadcast crash, then .reshape to fix it,
# a layout violation, then the crash again), and the generic layout fix showed a *transpose* loop.
# The only legal form is one column at a time, with the per-row vector and a scalar per column.
ROWNORM_COLUMN = {
    "rmsnorm": "out[i0:i1, j] = x[i0:i1, j] * rinv",
    "softmax": "out[i0:i1, j] = np.exp(x[i0:i1, j] - m) / s",
    "layernorm": "out[i0:i1, j] = (x[i0:i1, j] - mu) * rinv",
}


def rownorm_fix(level_name):
    line = ROWNORM_COLUMN[level_name]
    return (f"Do not reshape, add axes or broadcast: a (rows, cols) tile cannot be combined with a "
            f"per-row vector. Write the output one column at a time, so every operand is a vector of "
            f"length i1 - i0 or a scalar: for j in range(j0, j1): `{line}` (the per-row values are "
            f"vectors of length i1 - i0, computed over the whole row first).")


def empty_value_fix(slot):
    # live v2/v3 L6: y[j:end_j, r] = tile[r - i, j:end_j] -- a tile sliced with global columns
    return (f"The right-hand side is EMPTY (shape (0,)) while the slot is ({slot}): a slice with global "
            f"positions was taken from a tile that only has local ones. A tile t = x[i0:i1, j0:j1] is "
            f"indexed t[r - i0, 0:j1 - j0]; the full array x is indexed x[r, j0:j1]. Index x directly "
            f"with global positions and drop the tile variable.")


def crash_fix(msg, level_name=None):
    """The fix for the crashes the model actually produces (live L2, L4, L5, L6, L10); None if the
    message is not one we know."""
    m = re.search(r"index (-?\d+) is out of bounds for axis (\d+) with size (\d+)", msg)
    if m:
        i, ax, size = m.groups()
        return (f"Index {i} on axis {ax} is past the end of an array of size {size}. A partial tile "
                f"is smaller than 128 rows / 512 columns. Inside a tile index the tile slice with "
                f"local positions 0 .. end - start - 1, or index the full array with global positions "
                f"start .. end - 1; never mix the two, and never assume a full-size tile.")
    if level_name in ROWNORM_COLUMN and "broadcast" in msg:
        return rownorm_fix(level_name)
    m = re.search(r"could not broadcast input array from shape \(0,?\) into shape \(([^)]*)\)", msg)
    if m:
        return empty_value_fix(m.group(1))
    m = re.search(r"could not broadcast input array from shape \(([^)]*)\) into shape \(([^)]*)\)", msg)
    if m:
        return (f"An assignment puts a value of shape ({m.group(1)}) into a slot of shape "
                f"({m.group(2)}). The slice on the left and the value on the right must have the same "
                f"shape; at a partial tile both must use the real size end - start.")
    m = re.search(r"operands could not be broadcast together with shapes (.*?)(?: at line|$)", msg)
    if m:
        return (f"Two operands of one operation have different shapes ({m.group(1).strip()}). Combine "
                f"only arrays of the same length, e.g. column vectors x[i0:i1, j] with accumulators of "
                f"length i1 - i0; at a partial tile every length is end - start.")
    if "setting an array element with a sequence" in msg:
        return ("A whole array is assigned to a single element (e.g. out[r] = vector). Reduce it to one "
                "value per row first, or assign it to a slice of the same length.")
    if "negative dimensions are not allowed" in msg:
        return ("A size computed for an allocation is negative: the range end is before its start. Check "
                "the window or tile bounds so that start <= end, and handle an empty range without "
                "allocating.")
    return None


def judge_directive(level, message):
    """The organizers' checker failed a kernel our harness accepted: one instruction for why."""
    first = message.splitlines()[0]
    if "RAISED" in message and ("positional argument" in message or "TypeError" in message):
        return Directive("judge:signature",
                         f"The kernel must be callable exactly as `{level.signature}`, i.e. with the "
                         f"input arrays only (keyword arguments keep their defaults). Use that "
                         f"signature.", first)
    if "NON-FINITE" in message:
        return Directive("judge:nonfinite", "Some outputs are nan or inf on the final check: a step "
                         "overflows. Subtract the row maximum before exp, and accumulate in float64.",
                         first)
    if "WRONG SHAPE" in message:
        return Directive("judge:shape", f"Wrong output shape on the final check: {first}", first)
    if "RULES" in message:
        return Directive("judge:rules", f"The final check found a banned call: {first}. Replace it "
                         f"with an explicit loop over columns.", first)
    return Directive("judge:precision",
                     "Correct logic, but the final check needs every output within 1e-4 relative "
                     "error, including outputs close to zero. Compute in float64: cast each slice "
                     "with .astype(np.float64) before multiplying or adding, keep accumulators in "
                     "float64, and convert only the final result.", first)


# A first draft written as whole-array numpy breaks many rules at once (live L8: 5-15 violations per
# attempt, L9: 5-13). One directive per kind with "N other problems will be handled later" cannot
# converge in the attempt budget, and each local fix tends to break something else (E13). Past this
# many violations the kernel is not a near miss, so ask for a rewrite in the one legal loop shape
# for the op: the structure of the loops and the per-column lines, never the reference's formula.
REWRITE_AT = 5

SKELETON = {
    "layernorm": (
        "for each row tile i0..i1 (128 rows): "
        "pass 1, s = np.zeros(i1 - i0, dtype=np.float64), then for every column tile j0..j1 and "
        "every j in it: s += x[i0:i1, j]; mu = s / N. "
        "Pass 2, the same loops: d = x[i0:i1, j] - mu; ss += d * d; rinv = 1.0 / np.sqrt(ss / N + eps). "
        "Pass 3, the same loops: out[i0:i1, j] = (x[i0:i1, j] - mu) * rinv. "
        "Every statistic covers the whole row (all column tiles) before it is used; every operand is a "
        "column vector of length i1 - i0 or a scalar."),
    "softmax": (
        "for each row tile i0..i1 (128 rows): "
        "pass 1, m = np.full(i1 - i0, -np.inf), then for every column tile j0..j1 and every j in it: "
        "m = np.maximum(m, x[i0:i1, j]). "
        "Pass 2, the same loops: s += np.exp((x[i0:i1, j] - m).astype(np.float64)) with "
        "s = np.zeros(i1 - i0, dtype=np.float64). "
        "Pass 3, the same loops: out[i0:i1, j] = np.exp((x[i0:i1, j] - m).astype(np.float64)) / s. "
        "The max and the sum cover the whole row (all column tiles) before they are used; every "
        "operand is a column vector of length i1 - i0 or a scalar."),
    "transpose": (
        "out = np.empty((N, M), dtype=x.dtype); for each row tile i0..i1 (128 rows) and column tile "
        "j0..j1 (512 columns) of x, and each row i in i0..i1: out[j0:j1, i] = x[i, j0:j1]. "
        "Both sides use GLOBAL positions j0:j1 and i; no reshape, .T, transpose or 2-D slice of out."),
    "matmul": (
        "for each row tile i0..i1 (128 rows of a) and column tile j0..j1 (512 columns of b): "
        "acc = np.zeros((i1 - i0, j1 - j0), dtype=np.float64); for each K tile k0..k1 (128): "
        "at = a[i0:i1, k0:k1].astype(np.float64); bt = b[k0:k1, j0:j1].astype(np.float64); for k in "
        "range(k1 - k0): for r in range(i1 - i0): acc[r, :] += at[r, k] * bt[k, :]. After the K "
        "loop: out[i0:i1, j0:j1] = acc. No @, np.dot, matmul or einsum, and no slice of a or b that "
        "is longer than one tile in any dimension (K included)."),
    "band_attention": (
        "for each query row i: lo, hi = max(0, i - w), min(S - 1, i + w); n = hi - lo + 1. "
        "Scores: s = np.zeros(n, dtype=np.float64); for each chunk c0..c1 of at most 128 keys and "
        "each feature c in range(d): s[c0:c1] += q[i, c] * k[lo + c0:lo + c1, c].astype(np.float64). "
        "Row max with a scalar loop: m = -np.inf; for t in range(n): m = max(m, s[t]). "
        "p[c0:c1] = np.exp((s[c0:c1] - m) / np.sqrt(d)) per chunk. "
        "Output: total = 0.0; acc = np.zeros(d, dtype=np.float64); for t in range(n): total += p[t]; "
        "acc += p[t] * v[lo + t, :].astype(np.float64); then out[i, :] = acc / total. "
        "No np.dot, @, einsum, masks or full S x S score matrix: only keys inside the band are touched."),
}


def rewrite_directive(rep, cycling=False):
    """A whole-kernel rewrite in the op's legal loop shape, or None (op without a skeleton, or a
    near miss with few violations). cycling: the run has already failed on several different
    rules, so patching is going round in circles (live v2/v3 L5: sum-like -> max-like -> layout ->
    crash -> sum-like ... until the budget ran out)."""
    if rep.level.name not in SKELETON or not (rep.violations or cycling):
        return None
    if len(rep.violations) < REWRITE_AT and not cycling:
        return None
    kinds = sorted({v.kind for v in rep.violations})
    why = (f"({len(rep.violations)} violations: {', '.join(kinds)}), so patching it one line at a "
           f"time will not finish" if not cycling else
           "and the previous repairs have cycled through several different rule problems, so "
           "patching one line at a time is going round in circles")
    return Directive(
        "rule:rewrite",
        f"This draft computes the operation with array operations the rules ban {why}. Rewrite the "
        f"function body from scratch in this loop shape, keeping the signature: "
        f"{SKELETON[rep.level.name]}",
        f"{len(rep.violations)} rule violations of {len(kinds)} kinds.")


# After a rewrite the draft is close (live L9 run 1 attempt 4: 9/9 cases right, 2 violations), but the
# generic fixes talk about x[i0:i1, j0:j1] tiles, which a band-attention kernel does not have; the
# model kept `np.dot(q_slice, k_window[t])` on a (257, d) window. For an op with a skeleton, a
# structural rule fix names the skeleton line that replaces the banned one.
SKELETON_LINE = {
    "transpose": (
        "In a transpose write one output COLUMN segment per input row, 1-D on both sides: "
        "out[j0:j1, i] = x[i, j0:j1] (global positions, at most 512 columns of x per segment)."),
    "matmul": (
        "In a matmul every operand slice is one tile: at = a[i0:i1, k0:k1], bt = b[k0:k1, j0:j1] "
        "(K tiled in steps of 128 too), cast to float64, and accumulate acc[r, :] += at[r, k] * "
        "bt[k, :] into a (i1 - i0, j1 - j0) float64 accumulator; write out[i0:i1, j0:j1] = acc once "
        "after the K loop."),
    "band_attention": (
        "In band attention the only legal way to form the scores is one feature column at a time "
        "over a chunk of at most 128 keys: for c0 in range(0, n, 128): c1 = min(c0 + 128, n); for c "
        "in range(d): s[c0:c1] += q[i, c] * k[lo + c0:lo + c1, c].astype(np.float64). No np.dot, no slice of k with "
        "more than 128 rows, no per-key dot product."),
}
SKELETON_KINDS = {"whole-array", "linalg", "broadcast", "sum-like", "layout", "newaxis"}


def _crash_line(msg):
    m = re.search(r" at line (\d+)", msg)
    return int(m.group(1)) if m else None


def translate(rep):
    if rep.load_error:
        kinds = {v.kind for v in rep.violations}
        if "syntax" in kinds:
            return Directive("syntax", f"The code does not parse. {rep.load_error} Send the "
                                       f"complete function.")
        if "no-kernel" in kinds:
            return Directive("no-kernel", f"Start the function with exactly this line: "
                                          f"def {rep.level.signature}:")
        return Directive("load-crash", f"The module fails when it is loaded: {rep.load_error}. "
                                       f"Fix that line.")

    rw = rewrite_directive(rep)
    if rw:
        return rw

    # A crash raised by a call the rules ban (live L9: `np.dot(scores, v_window.T)` shape errors, 3
    # attempts in a row on one run) is not worth fixing: the fix is to remove the call, which the
    # rule directive says. Crashes on legal lines still come first.
    banned_lines = {v.line for v in rep.violations if v.line}
    crashes = [r for r in rep.results if r.status in ("error", "timeout")
               and not (r.status == "error" and _crash_line(r.msg) in banned_lines)]
    if crashes:
        r = crashes[0]
        if r.status == "timeout":
            return Directive("timeout", f"The kernel is too slow on '{r.case.name}'. Loop over "
                                        f"tiles and, inside a tile, over columns only, operating "
                                        f"on whole column slices; do not loop over single elements.")
        exc = r.msg.split(":")[0]
        names = ", ".join(f"'{c.case.name}'" for c in crashes[:3])
        where = f"It crashes on {len(crashes)} of {len(rep.results)} cases, e.g. {names}"
        fix = crash_fix(r.msg, rep.level.name)
        if fix:
            return Directive(f"crash:{exc}", fix, f"The kernel raises {r.msg}. {where}.")
        return Directive(f"crash:{exc}",
                         f"The kernel raises {r.msg}. {where}. Fix it so it works for every "
                         f"shape, including dimensions of 1 and sizes that are not multiples of "
                         f"the tile size.")

    shape = [r for r in rep.results if r.status == "shape"]
    if shape:
        return Directive("shape", f"Wrong output shape: {shape[0].msg} on '{shape[0].case.name}'. "
                                  f"Return exactly that shape.")

    if rep.violations:
        by_kind = {}
        for v in rep.violations:
            by_kind.setdefault(v.kind, []).append(v)
        kind = min(by_kind, key=lambda k: RULE_ORDER.index(k) if k in RULE_ORDER else 99)
        vs = by_kind[kind]
        where = ", ".join(f"line {v.line} `{v.source_line}`" for v in vs if v.line)
        rest = len(rep.violations) - len(vs)
        later = f" ({rest} other rule problem(s) will be handled after this one.)" if rest else ""
        m0 = re.search(r"assigning shape \(0,?\) into a region of shape \(([^)]*)\)", vs[0].what)
        if kind == "broadcast" and m0:
            fix = empty_value_fix(m0.group(1))
            if rep.level.name in SKELETON_LINE:
                fix += " " + SKELETON_LINE[rep.level.name]
        elif kind in ("layout", "newaxis", "broadcast") and rep.level.name in ROWNORM_COLUMN:
            fix = rownorm_fix(rep.level.name)
        elif kind in SKELETON_KINDS and rep.level.name in SKELETON_LINE:
            fix = vs[0].fix() + " " + SKELETON_LINE[rep.level.name]
        else:
            fix = vs[0].fix()
        return Directive(f"rule:{kind}", fix + later, f"Rule broken at {where}." if where else "")

    return _numeric(rep)


# Level-specific hypotheses (explain hooks) are checked numerically against the reference, so
# they are not coincidences: one that explains half the wrong cases is worth fixing first even
# when a second bug causes the rest (live L4: per-tile statistic AND float32 squares overflowing).
HYPOTHESES = [
    ("mean is large compared to their spread", "numeric:cancellation",
     "Only rows whose mean is large next to their spread are wrong: the variance computed as "
     "mean(x**2) - mean(x)**2 cancels catastrophically (and can go negative). Compute the mean "
     "first, then the variance as the mean of (x - mean)**2 in a second pass, accumulating in "
     "float64."),
    ("column tile only", "numeric:tile-overwrite",
     "Each wrong row's result is the result over a single column tile only: the per-row result is "
     "overwritten for every column tile instead of combined across tiles. Keep one running value "
     "per row across ALL column tiles: initialize it before the column-tile loop, combine each tile "
     "into it inside the loop, and write it to the output only after the loop."),
    ("normalised down each column", "numeric:wrong-axis",
     "Each column was normalised with the mean and variance of that column over the rows of a tile. "
     "Layernorm normalises each ROW: mean_i and var_i are over all N columns of row i. Keep a vector "
     "of length i1 - i0 (one value per row), accumulate it over every column j of every column tile "
     "with s += x[i0:i1, j], then write out[i0:i1, j] = (x[i0:i1, j] - mu) * rinv."),
    ("tile alone, not of the whole row", "numeric:per-tile-stat",
     "Each column tile is normalised with a statistic (max, sum or mean of squares) of that tile "
     "alone. The statistic must cover the WHOLE row: first loop over all column tiles to accumulate "
     "it, then loop over the tiles again to write the output."),
    ("of K only", "numeric:partial-k",
     "Part of K is left out of the sum. Loop `for k0 in range(0, K, step)` with "
     "`k1 = min(k0 + step, K)` and use its real size k1 - k0, so every k in 0..K-1 contributes."),
    ("last partial K tile is missing", "numeric:partial-k",
     "The last, partial K tile is left out of the sum. Loop `for k0 in range(0, K, step)` with "
     "`k1 = min(k0 + step, K)` and use its real size k1 - k0, so every k in 0..K-1 contributes."),
    ("result for window w", "numeric:band-edge",
     "The window has the wrong width: it must include both edges, j from max(0, i - w) to "
     "min(S - 1, i + w) inclusive (2w + 1 keys away from the sequence ends)."),
    ("one-sided (causal) window", "numeric:one-sided",
     "The window is two-sided: include keys after the query (j up to min(S - 1, i + w)) as well "
     "as before it (j down to max(0, i - w))."),
    ("the dilation is ignored", "numeric:dilation",
     "The dilation is ignored. Tap k of output t reads x[c, t*stride + k*dilation], so the slice "
     "for tap k starts at t0*stride + k*dilation."),
    ("copied without transposing", "numeric:not-transposed",
     "Wrong outputs hold x[r, c] at out[r, c]: the data was copied, not transposed. Write row i of "
     "each input tile into column i of the output: out[j0:j1, i] = x[i, j0:j1]."),
    ("instead of x[c, r]", "numeric:tile-origin",
     "Wrong outputs are read from the wrong place (an offset in the tile origin). Inside a tile, "
     "add the tile origin to every index: the input row is i0 + r and the output row is j0 + c."),
]


def _hypothesis(fails, evidence):
    best = None
    for needle, key, instruction in HYPOTHESES:
        k, n = fact_support(fails, needle)
        if n and k >= 0.5 * n and (best is None or k > best[0]):
            best = (k, n, key, instruction)
    if best is None:
        return None
    k, n, key, instruction = best
    if k < n:
        instruction += (f" (This explains {k} of the {n} wrong cases; fix it first, the others may "
                        f"have a second cause.)")
    return Directive(key, instruction, evidence)


def _numeric(rep):
    fails = [r for r in rep.results if r.status != "pass"]
    evidence = rep.numeric_feedback(max_cases=1)
    facts, _ = common_facts(fails)
    hyp = _hypothesis(fails, evidence)
    if hyp:
        return hyp
    pt = pattern_tag(rep.results)
    if pt and (rep.level.name, pt[0]) in LEVEL_TAG_FIX:
        return Directive(f"numeric:tag:{pt[0]}", LEVEL_TAG_FIX[rep.level.name, pt[0]], evidence)
    if any(r.status == "nonfinite" for r in fails):
        # one remedy, chosen by the op (E13: two remedies let the model pick one that breaks a rule)
        remedy = ("Subtract the row maximum before exp: first loop over the columns to find the "
                  "running maximum m of each row, then use exp(x - m)."
                  if rep.level.name in ("softmax", "band_attention") else
                  "Accumulate in float64: cast each column with .astype(np.float64) before squaring "
                  "or summing it.")
        return Directive("numeric:nonfinite",
                         "Some outputs are nan or inf although the true answer is finite: a step "
                         "overflows for these inputs. " + remedy, evidence)
    if all(r.worst < ROUNDING_CEILING for r in fails):
        return Directive("numeric:precision",
                         "The logic is right but float32 rounding is too large: every output must be "
                         "within 1e-4 relative error, including outputs close to zero. Compute in "
                         "float64: cast each slice with .astype(np.float64) before multiplying or "
                         "adding, keep accumulators in float64, and convert only the final result.",
                         evidence)

    fact = lambda s: any(s in f for f in facts)
    const = next((f.rsplit(" ", 1)[1] for f in facts if f.startswith("every wrong output is exactly")), None)
    if const is not None and fact("negative expected value"):
        return Directive("numeric:identity",
                         f"Outputs whose true value is negative come out as exactly {const}: the "
                         f"running result starts from {const}. Start it from -np.inf instead "
                         f"(np.full(n, -np.inf)), the identity of max.", evidence)
    if fact("only the last row is wrong") or fact("only the last col is wrong"):
        axis = "row" if fact("only the last row is wrong") else "column"
        return Directive(f"numeric:last-{axis}",
                         f"Only the last {axis} is wrong: an off-by-one in the {axis} bounds. "
                         f"The end of each tile must be min(start + tile, size), and loops must "
                         f"run up to the size itself, not size - 1.", evidence)
    lv = rep.level
    for ax, name in enumerate(lv.out_axes):
        T = lv.out_tile[ax]
        if fact(f"only in the last partial {name} tile"):
            return Directive(f"numeric:partial-{name}",
                             f"Only the last, partial {T}-wide tile along the output's {name} "
                             f"axis is wrong (missing or mis-sized). Loop the tile start up to "
                             f"the full size with end = min(start + {T}, size), and use the "
                             f"tile's real size end - start (never a fixed {T}) in every inner "
                             f"loop and slice.", evidence)
    if pt and pt[0] in TAG_FIX:
        return Directive(f"numeric:tag:{pt[0]}", TAG_FIX[pt[0]], evidence)
    if const is not None:
        what = "never computed or written" if float(const) == 0 else "left at an initial value"
        return Directive("numeric:constant",
                         f"Wrong outputs are all exactly {const}, i.e. {what}. Make sure every "
                         f"tile, including partial ones, is computed and written, and that "
                         f"accumulators start from the identity of the operation.", evidence)

    if pt:
        tag = pt[0]
        meaning = TAG_MEANING.get(tag, tag)
        return Directive(f"numeric:tag:{tag}",
                         f"The kernel is right except when {meaning}. Find the assumption in the "
                         f"code that these inputs break and remove it.", evidence)
    return Directive("numeric:other",
                     "The results are wrong. Re-derive the formula from the task statement and "
                     "check each step against it.", evidence)
