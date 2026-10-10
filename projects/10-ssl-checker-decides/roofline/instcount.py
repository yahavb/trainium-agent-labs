"""Count the instructions a kernel issues in the simulator, and turn the biggest excess into ONE named change.

Why this exists: the roofline floor says how far a kernel is from the hardware's best, not what to change.
Measured on 2026-10-10: given only the roofline diagnosis ("compute bound on TensorE ... loaded more than
once ... busiest DMA 15%"), Qwen3-8B returned the slow kernel unchanged in 24 of 24 attempts. The slow
kernel's real problem shows up as a count: it issues 4x more nc_matmul instructions than needed because
its moving tiles are 128 wide where the hardware allows 512. This module measures that and says so.

    from instcount import simulate_and_count_all, specific_feedback
    out, counts = simulate_and_count_all(kernel, args)
    msg = specific_feedback(counts, M=..., K=..., N=...)

Needs the Neuron SDK (runs nki.simulate). The compiler's latency model charges about 231 ns per
nc_matmul on top of its columns (peaks_probe2.py), so instruction count matters, not only FLOPs.
"""
import collections
import sys

sys.path.insert(1, "/workspace/projects/02-kernel-agent")
import nkibench  # noqa: E402

PMAX = nkibench.PMAX
STAT_FMAX = nkibench.GEMM_STATIONARY_FMAX
MOV_FMAX = nkibench.GEMM_MOVING_FMAX
MATMUL_FIXED_NS = 231  # compiler model, per nc_matmul instruction (peaks_probe2.py)

_COUNTED = ("tensor_copy", "tensor_scalar", "tensor_tensor", "tensor_reduce", "activation", "reciprocal")


def _shape(x):
    s = getattr(x, "shape", None)
    return tuple(int(v) for v in s) if s is not None else None


def simulate_and_count_all(kernel, args):
    """nkibench.simulate_and_count, plus every nc_matmul's operand shapes and a count of engine ops."""
    import nki.isa as nisa
    counts = dict(matmuls=[], engine=collections.Counter())
    originals = {}

    def mm(*a, **kw):
        names = ("dst", "stationary", "moving")
        got = dict(zip(names, a))
        got.update({k: v for k, v in kw.items() if k in names})
        counts["matmuls"].append((_shape(got.get("stationary")), _shape(got.get("moving"))))
        return originals["nc_matmul"](*a, **kw)

    def wrap(name):
        def f(*a, **kw):
            counts["engine"][name] += 1
            return originals[name](*a, **kw)
        return f

    originals["nc_matmul"] = nisa.nc_matmul
    nisa.nc_matmul = mm
    for n in _COUNTED:
        if hasattr(nisa, n):
            originals[n] = getattr(nisa, n)
            setattr(nisa, n, wrap(n))
    try:
        out, dma = nkibench.simulate_and_count(kernel, list(args))
    finally:
        for n, f in originals.items():
            setattr(nisa, n, f)
    counts["dma_bytes"], counts["dma_transfers"] = dma["bytes"], dma["transfers"]
    return out, counts


def _ceil(a, b):
    return -(-a // b)


def matmul_counts(counts, M, K, N):
    """Measured vs minimum nc_matmul count, and how full each operand's tiles are on average."""
    mms = [m for m in counts["matmuls"] if m[0] and m[1]]
    minimum = _ceil(K, PMAX) * _ceil(M, STAT_FMAX) * _ceil(N, MOV_FMAX)
    if not mms:
        return dict(n=0, minimum=minimum)
    n = len(mms)
    avg = lambda vals: sum(vals) / len(vals)  # noqa: E731
    return dict(n=n, minimum=minimum,
                partition=avg([s[0] for s, _ in mms]),   # K rows per instruction, limit 128
                stationary_free=avg([s[1] for s, _ in mms]),  # M columns, limit 128
                moving_free=avg([m[1] for _, m in mms]))  # N columns, limit 512


def specific_feedback(counts, M, K, N, shape_label=""):
    """ONE named change, biggest first, from the instruction counts. Returns a short string, or ''."""
    c = matmul_counts(counts, M, K, N)
    where = f" on {shape_label}" if shape_label else ""
    if c["n"] and c["n"] > 1.25 * c["minimum"]:
        excess = (c["n"] - c["minimum"]) * MATMUL_FIXED_NS / 1000
        narrow = []
        if c["moving_free"] < 0.8 * MOV_FMAX:
            narrow.append(f"the tile you pass as moving= is only {c['moving_free']:.0f} columns wide; "
                          f"nc_matmul accepts up to {MOV_FMAX}")
        if c["stationary_free"] < 0.8 * STAT_FMAX:
            narrow.append(f"the tile you pass as stationary= is only {c['stationary_free']:.0f} columns "
                          f"wide; the limit is {STAT_FMAX}")
        if c["partition"] < 0.8 * PMAX:
            narrow.append(f"each matmul contracts only {c['partition']:.0f} rows of K; the partition "
                          f"limit is {PMAX}")
        why = "; ".join(narrow) or "some matmuls are repeated"
        return (f"The biggest cost{where}: the kernel issues {c['n']} nc_matmul instructions where "
                f"{c['minimum']} would do, because {why}. Each instruction has a fixed cost (about "
                f"{MATMUL_FIXED_NS} ns), so that is roughly {excess:.1f} us of overhead. Make the tiles as "
                f"wide as the limit allows, and keep every loop bound and slice consistent with the new "
                f"width.")
    min_bytes = (M * K + K * N + M * N) * 4
    if counts["dma_bytes"] > 1.25 * min_bytes:
        return (f"The biggest cost{where}: the kernel moves {counts['dma_bytes'] / 1024:.0f} KiB between "
                f"HBM and SBUF, {counts['dma_bytes'] / min_bytes:.1f}x the minimum of "
                f"{min_bytes / 1024:.0f} KiB, in {counts['dma_transfers']} transfers. Some input tile is "
                f"loaded more than once: load each tile once and reuse it across the loop that does not "
                f"change it.")
    return ""
