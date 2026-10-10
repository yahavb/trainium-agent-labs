"""Per-tensor HBM traffic attribution, turned into one instruction.

nkibench.simulate_and_count reports one total. This wraps nisa.dma_copy one level OUTSIDE it and
delegates the whole simulation to it, so nkibench's own counts are untouched and the per-tensor
buckets sum to exactly nkibench's number.
"""
import numpy as np

ONCHIP = "_on_chip"


def _is_hbm(t):
    return "hbm" in str(getattr(t, "buffer", "")).lower()


def _handle(t):
    return getattr(t, "_storage", None)


def _nbytes(src):
    import nkibench
    n = getattr(src, "nbytes", None)
    if not isinstance(n, int) or n <= 0:
        n = int(np.prod(src.shape)) * nkibench.itemsize_of(src)
    return int(n)


class _Registry:
    """Maps a simulator storage handle to an argument name (or 'output')."""

    def __init__(self, args, arg_names):
        self.args = list(args)
        self.names = list(arg_names)
        self.by_handle = {}

    def name_of(self, t):
        h = _handle(t)
        key = id(h) if h is not None else id(t)
        if key in self.by_handle:
            return self.by_handle[key]
        name = self._match(t, h)
        self.by_handle[key] = name
        return name

    def _match(self, t, h):
        data = getattr(h, "data", None)
        # 1. identity or shared memory with a caller array
        for a, n in zip(self.args, self.names):
            if isinstance(a, np.ndarray) and (a is t or (isinstance(data, np.ndarray) and (
                    data is a or np.shares_memory(data, a)))):
                return n
        # 2. same contents (the simulator copies inputs into its own handles)
        if isinstance(data, np.ndarray):
            for a, n in zip(self.args, self.names):
                if (isinstance(a, np.ndarray) and a.shape == data.shape and a.dtype == data.dtype
                        and np.array_equal(a, data)):
                    return n
        # 3. a shared_hbm tensor that is not an input is the kernel's output
        if "shared" in str(getattr(t, "buffer", "")).lower():
            return "output"
        # 4. creation order: the simulator numbers input tensors 0..n-1 in argument order
        tid = getattr(h, "tensor_id", None)
        if isinstance(tid, int) and 0 <= tid < len(self.names):
            return self.names[tid]
        return "hbm_other"


def simulate_and_attribute(kernel, args, arg_names):
    """Returns (output, counted, attrib).

    counted is exactly nkibench.simulate_and_count's dict. attrib maps tensor name ->
    dict(read_bytes, write_bytes, size_bytes, reads_x); attrib[ONCHIP] holds sbuf/psum-only
    copies and attrib['_total'] the sum of every bucket (== counted['bytes']).
    """
    import nkibench
    import nki.isa as nisa

    reg = _Registry(args, arg_names)
    buckets = {}

    def bucket(name):
        return buckets.setdefault(name, dict(read_bytes=0, write_bytes=0))

    original = nisa.dma_copy

    def attributing_dma_copy(dst=None, src=None, **kw):
        try:
            nb = _nbytes(src)
            if _is_hbm(src):
                bucket(reg.name_of(src))["read_bytes"] += nb
            elif _is_hbm(dst):
                bucket(reg.name_of(dst))["write_bytes"] += nb
            else:
                bucket(ONCHIP)["read_bytes"] += nb
        except Exception:
            bucket("_unmeasured")["read_bytes"] += 0
        return original(dst=dst, src=src, **kw)

    nisa.dma_copy = attributing_dma_copy
    try:
        out, counted = nkibench.simulate_and_count(kernel, args)
    finally:
        nisa.dma_copy = original

    sizes = {n: int(a.nbytes) for a, n in zip(args, arg_names) if isinstance(a, np.ndarray)}
    if isinstance(out, np.ndarray):
        sizes["output"] = int(out.nbytes)
    attrib = {}
    for name, b in buckets.items():
        size = sizes.get(name, 0)
        moved = b["read_bytes"] if name != "output" else b["write_bytes"]
        attrib[name] = dict(b, size_bytes=size, reads_x=(moved / size) if size else None)
    attrib["_total"] = sum(b["read_bytes"] + b["write_bytes"] for b in buckets.values())
    return out, counted, attrib


def explain(attrib, args=None, threshold=1.05):
    """ONE instruction sentence about the worst-offending tensor, or '' if none is over."""
    worst, wx = None, threshold
    for name, b in attrib.items():
        if name.startswith("_"):
            continue
        x = b.get("reads_x")
        if x is not None and x > wx:
            worst, wx = name, x
    if worst is None:
        return ""
    if worst == "output":
        return (f"`output` was written {wx:.1f}x its size: partial results are being written to "
                f"HBM more than once; keep the running sum on chip and write each output tile "
                f"once, after its last contribution.")
    return (f"`{worst}` was read {wx:.1f}x its size: each {worst} tile is reloaded inside an "
            f"inner loop; load it once, outside the loop that does not use it.")
