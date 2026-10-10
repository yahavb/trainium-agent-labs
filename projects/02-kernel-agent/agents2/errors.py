"""Error kinds: what a failed check means, and, where a rule knows, the one change that fixes it.

The kinds come from the failures recorded on seat-35 on 2026-10-10 (levels 1-4, ~800 attempts). The
most common: `nki.isa has no attribute 'multiply'` (64), dma_copy element-count mismatches (~150 across
levels), out-of-bound access (52), broadcast mismatches (29), 1-D tiles (23), invented nc_matmul
keywords (24), PSUM/SBUF placement (44).
"""

import re

# Checked in order; the first match wins.
RUNTIME_KINDS = [
    ("TILE_SIZE", r"_TileSize"),            # before NAME: "'_TileSize' object has no attribute 'value'"
    ("NAME", r"has no attribute|No module named|is not defined"),
    ("KWARG", r"unexpected keyword argument|missing \d+ required|got multiple values|"
              r"takes \d+ positional|too many positional|missing a required argument"),
    ("MATMUL_SHAPE", r"contraction dimension mismatch|Matmul .*mismatch"),
    ("PARTITION", r"partition dimension \d+ exceeds|exceeds pmax|contraction dimension \d+ exceeds"),
    ("DMA_SHAPE", r"requires src and dst to have the same number of elements|could not be broadcast|"
                  r"cannot reshape|shape mismatch"),
    ("BOUNDS", r"Out-of-bound access"),
    ("TILE_RANK", r"at least \d+ dimensions"),
    ("MEMSPACE", r"must be in \[|MemoryRegion' object is not callable"),
    ("REDUCE_AXIS", r"axis must be the last|reduce axis|reduction axis"),
    ("TRANSPOSE", r"dma_transpose|nc_transpose"),
    ("UNSUPPORTED", r"only supported for NeuronCore|not supported on"),
    ("OPERATOR", r"unsupported operand type"),
]

# nkibench.describe_mismatch and the checks after it start their messages with these.
MISMATCH_KINDS = [
    ("WRONG_SHAPE", r"^WRONG SHAPE"),
    ("NONFINITE", r"^NON-FINITE"),
    ("ZEROS", r"^OUTPUT IS \d+% ZEROS"),
    ("PARTIAL", r"^\d+% of the output is zero"),
    ("INPUT_MODIFIED", r"^THE KERNEL MODIFIED ITS INPUT"),
    ("HW_HAZARD", r"^CORRECT ON CPU BUT WRONG ON HARDWARE"),
    ("TRAFFIC", r"^CORRECT, BUT TOO MUCH HBM TRAFFIC|^NO HBM TRAFFIC COUNTED"),
    ("VALUES", r"^NUMERICAL MISMATCH"),
]


def classify_runtime(text):
    for kind, pat in RUNTIME_KINDS:
        if re.search(pat, text or ""):
            return kind
    return "OTHER"


def classify_mismatch(text):
    for kind, pat in MISMATCH_KINDS:
        if re.search(pat, (text or "").strip()):
            return kind
    return "VALUES"


def classify_feedback(feedback):
    """Kind for one of agent.py's recorded feedback strings, for analysing old runs."""
    fb = feedback or ""
    if fb.startswith("No code came back"):
        return "EMPTY"
    if fb.startswith("The code does not parse"):
        return "PARSE"
    if fb.startswith("Rule violations"):
        return "RULES"
    if "There is no module named" in fb or "could not be loaded" in fb:
        return "LOAD"
    m = re.search(r"shapes passed\. On [^:]+: (.*)", fb, re.S)
    body = m.group(1) if m else fb
    if body.startswith("raised "):
        return classify_runtime(body)
    if body.startswith("Correct on every shape"):
        return "CORRECT"
    return classify_mismatch(body)


# ---------------------------------------------------------------- rule templates

REDUCE_AXIS_CHANGE = (
    "Reductions (nl.sum, nl.max, nisa.tensor_reduce) work only over the LAST axes of a tile, never over "
    "the partition axis (the first). Make the axes you reduce over come last{view}, then pass those "
    "axes, e.g. axis=[2, 3].")
REDUCE_AXIS_VIEW = ", with a view rather than a copy (see the cards below)"

BOUNDS_CHANGE = (
    "A slice reaches past the end of a tile or tensor. Derive every slice bound from the tensor's own "
    "shape (e.g. a.shape[1], or min(limit, size) for the last chunk) instead of a fixed number, and make "
    "each slice exactly the size of the tile it is copied into. The partition dimension (the first) "
    "holds at most 128; the others may be larger.")

MATMUL_SHAPE_CHANGE = (
    "nisa.nc_matmul contracts over the partition axis (the first dimension) of BOTH operands: "
    "stationary is [K, M] and moving is [K, N] with the same K (at most 128), and dst is a PSUM tile "
    "[M, N]. Slice both operands with the same K range.")

TRANSPOSE_CHANGE = (
    "nisa.dma_transpose only supports these axes: (1, 0) for 2-D, (2, 1, 0) for 3-D and (3, 1, 2, 0) for "
    "4-D tiles, and dst must have exactly the transposed shape. Trying other axes will not work. To "
    "reorder the free axes of a tile, use a view instead of a transpose (see the cards below), and keep "
    "the partition axis first.")

UNSUPPORTED_CHANGE = (
    "This argument or mode is not available on this chip (Trainium2, NeuronCore-v3). Remove it and use "
    "the plain form of the call, as in the signature below.")

TIMEOUT_CHANGE = (
    "The check timed out. Kernel code runs once per loop iteration while it is traced, so a loop over "
    "every element is far too slow. Move whole tiles with one call each instead of single elements.")


def rule_change(check, retriever, level, enrich, fragment_note):
    """The change a rule can name for this failed check, or None when only the model can.

    Returns dict(cause, change, names, example). `names` are functions whose cards the coder should see;
    `example` is a checked example kernel (agent.FRAGMENTS) when one matches. A name whose card is
    withheld at this level is never passed on (Retriever.shown), and no rule restates a withheld card.
    """
    rule = _rule_change(check, retriever, level, enrich, fragment_note)
    if rule:
        rule["names"] = [n for n in rule["names"] if retriever.shown(n, level)]
    return rule


def _rule_change(check, retriever, level, enrich, fragment_note):
    kind, err = check.get("kind"), check.get("error") or ""
    example = fragment_note(err, level) if fragment_note else ""
    if kind == "EMPTY":
        return dict(cause="no code came back", names=[], example="",
                    change="Reply with one python code block containing the imports and the kernel.")
    if kind == "PARSE":
        return dict(cause=err, names=[], example="",
                    change="Fix the syntax error so the file parses; send one complete code block.")
    if kind == "RULES":
        return dict(cause=err, names=[], example="", change=err)
    if kind == "ALIAS":
        # Levels 5-7, found by lint before the kernel runs; its message names the change.
        return dict(cause=err, names=["nisa.dma_copy"], example="", change=err)
    if kind == "NAME":
        m = re.search(r"module '([\w.]+)' has no attribute '(\w+)'", err)
        if m:
            dotted = f"{m.group(1)}.{m.group(2)}"
            close = retriever.close(dotted)
            if close:
                return dict(cause=f"`{retriever.short(dotted)}` does not exist in nki "
                                  f"{retriever.version}", names=close[:2], example=example,
                            change=f"Replace `{retriever.short(dotted)}` with the real function that does "
                                   f"this. The closest real names are: {', '.join(close)}. Their "
                                   f"signatures are below.")
            return None                     # nothing close: the model has to rethink the call
        if "No module named" in err:
            return dict(cause=err, names=[], example="",
                        change="Use exactly these imports: import nki; import nki.language as nl; "
                               "import nki.isa as nisa.")
        return None
    if kind == "KWARG":
        m = re.search(r"(?:(nl|nisa|language|isa|NkiTensor)\.)?(\w+)\(\)", err)
        fn = m.group(2) if m else None
        names = ([f"tile.{fn}"] if m and m.group(1) == "NkiTensor" else retriever.find(fn) if fn else [])
        change = "Call it with exactly the arguments of its real signature, shown below, each one once."
        kw = re.search(r"unexpected keyword argument '(\w+)'", err)
        if kw:
            change = f"Remove the `{kw.group(1)}=` argument; the real signature is below."
        if kw and kw.group(1) in ("dst", "src") and any(n.startswith("nl.") for n in names):
            # Measured on seat-35: nl.copy(dst=..., src=...) for nisa.tensor_copy, 5 checks running.
            # The nl functions return a new tile; the nisa ones write into dst=.
            alt = [f"nisa.{x}" for x in retriever.public("nisa") if fn and fn in x][:3]
            change = (f"nl.{fn} returns a new tile and has no dst= or src=. To write into a tile you "
                      f"allocated, use the nisa function that takes dst=" +
                      (f": {', '.join(alt)}" if alt else "") + ". Signatures below.")
            names = names + alt
        return dict(cause=err, names=names[:3], example=example, change=change)
    if kind == "REDUCE_AXIS":
        view = retriever.shown("tile.permute", level)
        return dict(cause=err, names=["nl.sum", "tile.permute", "tile.reshape"], example=example,
                    change=REDUCE_AXIS_CHANGE.format(view=REDUCE_AXIS_VIEW if view else ""))
    if kind == "TIMEOUT":
        return dict(cause=err, names=[], example="", change=TIMEOUT_CHANGE)
    if kind == "TILE_SIZE":
        # Measured on seat-35: nl.tile_size called, subscripted and used as an int, 3 checks running.
        # Its attributes are properties that need a kernel being traced, so list names, not values.
        ts = retriever.get("nl.tile_size")
        attrs = sorted(n for n in dir(type(ts)) if not n.startswith("_")) if ts is not None else []
        return dict(cause=err, names=[], example="",
                    change="nl.tile_size is not a function, list or number: it is an object whose "
                           "attributes are the limits, read inside the kernel, e.g. "
                           "nl.tile_size.pmax (128, the partition maximum)"
                           + (". Its attributes: " + ", ".join(attrs) if attrs else "") + ".")
    if kind == "TRANSPOSE":
        if not retriever.shown("nisa.dma_transpose", level):
            # Its card is withheld here (level 2 is a transpose), and this text restates it: the
            # model path reads the error instead.
            return None
        return dict(cause=err, names=["nisa.dma_transpose", "tile.permute"], example="",
                    change=TRANSPOSE_CHANGE)
    if kind == "UNSUPPORTED":
        m = re.search(r"(\w+) (\w+) is only supported", err)
        names = retriever.find(m.group(1)) if m else []
        change = (f"Remove the `{m.group(2)}=` argument from {m.group(1)}: it is not available on this "
                  f"chip (Trainium2, NeuronCore-v3)." if m else UNSUPPORTED_CHANGE)
        return dict(cause=err, names=names[:1], example="", change=change)
    if kind in ("MATMUL_SHAPE", "PARTITION", "DMA_SHAPE", "BOUNDS", "TILE_RANK", "MEMSPACE",
                "OPERATOR"):
        # enrich()'s advice for the broadcast error is wrong (it says to assign into out[...]; NKI writes
        # with dma_copy), and so is "never reshape" for a matmul dst that is too small. When a checked
        # example matched, its note is the change (nki_fix_examples.md, "mmshape").
        if example and re.search(r"could not be broadcast|cannot reshape array", err):
            note = example.strip().split("\n```")[0].strip()
            return dict(cause=err, names=["nisa.nc_matmul"] if level >= 3 else [], example=example,
                        change=note)
        if kind == "MEMSPACE" and re.search(r"got (?:private_hbm|hbm)\b", err):
            # Reproduced on seat-35: nl.mean straight on the kernel's input, which lives in HBM.
            return dict(cause=err, names=["nisa.dma_copy"], example=example,
                        change="That data is still in HBM. nl.* and nisa compute functions work on on-chip "
                               "tiles: copy the input into an SBUF tile with nisa.dma_copy first, and "
                               "compute on the tile.")
        advice = enrich(err)[len(err):].strip()
        if not advice:
            advice = {"BOUNDS": BOUNDS_CHANGE, "MATMUL_SHAPE": MATMUL_SHAPE_CHANGE}.get(kind, "")
        if advice:
            return dict(cause=err, names=["nisa.nc_matmul"] if kind == "MATMUL_SHAPE" else [],
                        example=example, change=advice)
        return None
    if kind in ("WRONG_SHAPE", "NONFINITE", "ZEROS", "PARTIAL", "INPUT_MODIFIED", "HW_HAZARD"):
        # describe_mismatch already says what to do for these; it was written as an instruction.
        return dict(cause=err.split(". ")[0], names=[], example="", change=err)
    return None                             # VALUES, OTHER, LOAD: the model looks at the kernel
