#!/usr/bin/env python3
"""
profile_steps.py -- run my_attention.py in the simulator and show every instruction it issues,
then where the work piles up.

For each instruction, as it executes:
    which source line issued it, which engine runs it, which memory the data moves between,
    the shape, how much work it is, what the values look like afterwards, and which earlier
    step it had to wait for.

Then a summary: work per engine, bytes per memory path, the dependency chain, how much of the
kernel only rearranges data, and a verdict.

WHAT THIS IS NOT: a timing. Everything here is counted from the simulator, which runs on the
CPU. It says how much work lands where and what must wait for what -- the STRUCTURE of the
bottleneck. Which step is actually slowest on a NeuronCore needs a device profile.

    python profile_steps.py                    # seq=128 dim=64, full trace
    python profile_steps.py --seq 96 --dim 32
    python profile_steps.py --summary-only     # the three level-8 shapes, summaries only
"""

import argparse
import ast
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
KERNEL_FILE = os.path.join(HERE, "my_attention.py")

# Which engine usually executes each instruction, per the NKI ISA docs. The compiler can place
# some elsewhere, so treat this column as the default, not a guarantee.
ENGINE = {
    "dma_copy": "DMA",
    "nc_matmul": "Tensor",
    "nc_transpose": "Tensor",
    "activation": "Scalar",
    "activation_reduce": "Scalar",
    "tensor_reduce": "Vector",
    "tensor_scalar": "Vector",
    "tensor_copy": "Vector",
}

# Steps that compute nothing and exist only to put data in the layout the next step needs.
LAYOUT_ONLY = {"nc_transpose", "tensor_copy"}

DST_KEYS = ("dst", "reduce_res")     # activation_reduce writes its row sum through reduce_res
SRC_KEYS = ("src", "data", "stationary", "moving", "bias", "operand0")


# ---------------------------------------------------------------- what the source says

def source_map():
    """Static facts from the kernel's source: where each variable lives, and what each
    instruction line reads and writes, by variable name."""
    tree = ast.parse(open(KERNEL_FILE).read())
    lines = open(KERNEL_FILE).read().splitlines()
    buffers, calls = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for a in node.args.args:
                buffers[a.arg] = "HBM"
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and isinstance(node.targets[0], ast.Name):
            for kw in node.value.keywords:
                if kw.arg == "buffer" and isinstance(kw.value, ast.Attribute):
                    buffers[node.targets[0].id] = {"shared_hbm": "HBM", "sbuf": "SBUF",
                                                   "psum": "PSUM"}.get(kw.value.attr, kw.value.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "nisa":
            def names(keys):
                out = []
                for kw in node.keywords:
                    if kw.arg in keys:
                        base = kw.value.value if isinstance(kw.value, ast.Subscript) else kw.value
                        if isinstance(base, ast.Name):
                            out.append(base.id)
                return out
            call = dict(op=node.func.attr, dst=names(DST_KEYS),
                        src=names(SRC_KEYS), text=lines[node.lineno - 1].strip())
            # A call spread over several lines can report any of them at run time, depending on
            # the Python version, so every line of it maps to the call.
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                calls[ln] = call
    return buffers, calls


# ---------------------------------------------------------------- what the simulator does

def numel(x):
    shape = getattr(x, "shape", None)
    return int(np.prod(shape)) if shape is not None else 0


def nbytes(x):
    import nkibench
    return numel(x) * nkibench.itemsize_of(x) if hasattr(x, "shape") else 0


def values_of(x):
    """min / max / mean of a tile after the step wrote it, if the simulator lets us read it."""
    try:
        a = np.asarray(x, dtype=np.float64)
        if a.size == 0 or a.dtype == object:
            return ""
        return f"[{a.min():+.3g} .. {a.max():+.3g}] mean {a.mean():+.3g}"
    except Exception:
        return "(values not readable)"


def work_of(op, kw):
    """How much each instruction does: FLOPs for arithmetic, elements for data movement."""
    if op == "nc_matmul":
        K, M = kw["stationary"].shape
        N = kw["moving"].shape[1]
        return 2 * K * M * N, "FLOP"
    if op == "activation":
        return 3 * numel(kw["data"]), "FLOP"      # scale, add bias, exp -- one fused pass
    if op == "activation_reduce":
        return 4 * numel(kw["data"]), "FLOP"      # the same, plus the running sum
    if op in ("tensor_reduce", "tensor_scalar"):
        return numel(kw["data"]), "FLOP"
    if op in ("nc_transpose", "tensor_copy"):
        return numel(kw.get("data", kw.get("src"))), "elem moved"
    return nbytes(kw.get("src")), "byte"


def trace(kernel, args, buffers, calls):
    """Run the kernel under the simulator with every nisa instruction wrapped."""
    import nki
    import nki.isa as nisa
    import nkibench

    steps, originals = [], {}

    def wrap(op, fn):
        def recorded(*a, **kw):
            frame = sys._getframe(1)
            while frame and os.path.abspath(frame.f_code.co_filename) != KERNEL_FILE:
                frame = frame.f_back
            line = frame.f_lineno if frame else None
            out = fn(*a, **kw)
            dst = kw.get("dst", a[0] if a else None)
            info = calls.get(line, dict(dst=[], src=[], text="?"))
            src_bufs = sorted({buffers.get(s, "?") for s in info["src"]
                               if buffers.get(s) in ("HBM", "SBUF", "PSUM")})
            dst_buf = buffers.get(info["dst"][0], "?") if info["dst"] else "?"
            work, unit = work_of(op, kw)
            moved = sum(nbytes(kw[k]) for k in SRC_KEYS if k in kw and hasattr(kw[k], "shape"))
            first_in = next((kw[k] for k in ("data", "src", "stationary") if k in kw), None)
            in_shape = "x".join(map(str, getattr(first_in, "shape", ())))
            steps.append(dict(n=len(steps) + 1, line=line, op=op, engine=ENGINE.get(op, "?"),
                              path=f"{'+'.join(src_bufs) or '?'} -> {dst_buf}",
                              shape="x".join(map(str, getattr(dst, "shape", ()))), in_shape=in_shape,
                              work=work, unit=unit, read_bytes=moved, write_bytes=nbytes(dst),
                              values=values_of(dst), dst=info["dst"], src=info["src"],
                              text=info["text"]))
            return out
        return recorded

    for op in {c["op"] for c in calls.values()}:
        if hasattr(nisa, op):
            originals[op] = getattr(nisa, op)
            setattr(nisa, op, wrap(op, originals[op]))
    try:
        run, _ = nkibench._simulator(nki, kernel)
        out = run(*args)
    finally:
        for op, fn in originals.items():
            setattr(nisa, op, fn)
    return steps, out


# ---------------------------------------------------------------- dependencies

def dependencies(steps):
    """Each step waits for the most recent earlier step that wrote something it reads.
    The longest chain of waits is the critical path: the minimum number of steps that must run
    one after another however many engines are free."""
    last_writer, depth = {}, {}
    for s in steps:
        waits = sorted({last_writer[v] for v in s["src"] if v in last_writer})
        s["waits_for"] = waits
        depth[s["n"]] = 1 + max((depth[w] for w in waits), default=0)
        for v in s["dst"]:
            last_writer[v] = s["n"]
    longest = max(depth.values())
    end = max(depth, key=depth.get)
    chain = [end]
    while True:
        prev = [w for w in next(s for s in steps if s["n"] == chain[-1])["waits_for"]]
        if not prev:
            break
        chain.append(max(prev, key=lambda w: depth[w]))
    return longest, list(reversed(chain)), depth


# ---------------------------------------------------------------- report

def print_trace(steps):
    dependencies(steps)          # fills in each step's waits_for
    print(f"\n{'#':>2} {'line':>4}  {'instruction':<14} {'engine':<7} {'memory path':<17} "
          f"{'shape':<9} {'work':>16}  {'waits for':<9}  values after the step")
    for s in steps:
        print(f"{s['n']:>2} {s['line'] or '?':>4}  {s['op']:<14} {s['engine']:<7} {s['path']:<17} "
              f"{s['shape']:<9} {s['work']:>9,} {s['unit']:<6}  "
              f"{','.join(map(str, s['waits_for'])) or '-':<9}  {s['values']}")
        print(f"{'':>9}{s['text']}")


def summarize(steps, seq, dim, out_ok):
    longest, chain, depth = dependencies(steps)
    print(f"\n========== SUMMARY  seq={seq} dim={dim}  ({len(steps)} instructions, "
          f"output {'correct' if out_ok else 'WRONG'}) ==========")

    print("\nWORK PER ENGINE")
    engines = {}
    for s in steps:
        e = engines.setdefault(s["engine"], dict(n=0, flop=0, moved=0, byte=0))
        e["n"] += 1
        e[{"FLOP": "flop", "elem moved": "moved", "byte": "byte"}[s["unit"]]] += s["work"]
    for name, e in sorted(engines.items(), key=lambda kv: -kv[1]["n"]):
        parts = [f"{e['flop']:,} FLOP" if e["flop"] else "",
                 f"{e['moved']:,} elements rearranged" if e["moved"] else "",
                 f"{e['byte']:,} bytes over DMA" if e["byte"] else ""]
        print(f"  {name:<7} {e['n']:>2} instructions   " + ", ".join(p for p in parts if p))

    print("\nBYTES PER MEMORY PATH (read + written)")
    paths = {}
    for s in steps:
        paths[s["path"]] = paths.get(s["path"], 0) + s["read_bytes"] + s["write_bytes"]
    for p, b in sorted(paths.items(), key=lambda kv: -kv[1]):
        print(f"  {p:<17} {b:>10,} bytes")

    hbm = sum(s["work"] for s in steps if s["op"] == "dma_copy")
    flops = sum(s["work"] for s in steps if s["unit"] == "FLOP")
    mm = sum(s["work"] for s in steps if s["op"] == "nc_matmul")
    print(f"\nARITHMETIC INTENSITY  {flops:,} FLOP / {hbm:,} HBM bytes = {flops / hbm:.1f} FLOP/byte "
          f"(nkibench's bf16 ridge is 222)")

    layout = [s for s in steps if s["op"] in LAYOUT_ONLY]
    print(f"\nLAYOUT-ONLY STEPS  {len(layout)} of {len(steps)} instructions compute nothing; they "
          f"transpose or copy data so the next step can read it:")
    for s in layout:
        print(f"    step {s['n']:>2}  {s['op']:<13} {s['shape']:<9} {s['text'][:70]}")

    print(f"\nDEPENDENCY CHAIN  {longest} of {len(steps)} steps must run strictly one after another:")
    print("    " + " -> ".join(f"{n}:{next(s['op'] for s in steps if s['n'] == n)}" for n in chain))
    free = [s["n"] for s in steps if depth[s["n"]] < longest and s["n"] not in chain]
    print(f"    off the critical path (could overlap with it): steps {free or 'none'}")

    sq = f"{seq}x{seq}"
    nn = [s for s in steps if sq in (s["shape"], s["in_shape"]) and s["op"] != "nc_matmul"]
    print(f"\nTHE n x n MATRIX  {len(nn)} non-matmul steps read or write a {seq}x{seq} tile "
          f"({seq * seq:,} elements each): " + ", ".join(f"{s['n']}:{s['op']}" for s in nn))

    print("\nVERDICT")
    print(f"  * {flops / hbm:.0f} FLOP/byte is under the ridge, and the whole kernel is "
          f"{flops / 1e6:.1f} MFLOP. At this size the chip finishes the math almost instantly, so")
    print(f"    the cost is the {longest}-step chain plus launch overhead, not any single engine.")
    print(f"  * {len(layout)} of {len(steps)} instructions only rearrange data. Accepting Q and K "
          f"already transposed removes {sum(1 for s in layout if s['shape'] == f'{dim}x{seq}')} "
          f"of them and shortens the chain.")
    print(f"  * Matmuls are {mm / flops:.0%} of the FLOPs, yet {len(nn)} other steps each walk the "
          f"whole n x n tile on the Vector/Scalar engines rather than the Tensor engine.")
    print("    FLOP share is not time share: those steps grow as seq^2 too, and only a device "
          "profile shows whether softmax or the matmuls set the pace.")
    print("  * Every step here waits on the previous one. Several heads per launch would let one "
          "head's softmax overlap the next head's matmul.")
    print("\n  Counted in the simulator, not timed. A device profile is what ranks the steps by time.")


def reference64(q, k, v):
    q, k, v = (a.astype(np.float64) for a in (q, k, v))
    s = q @ k.T / np.sqrt(q.shape[1])
    s -= s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(axis=-1, keepdims=True)) @ v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--summary-only", action="store_true",
                    help="summaries for the three level-8 shapes, no per-step trace")
    a = ap.parse_args()

    try:
        import nkibench
        kernel = nkibench.load_kernel(KERNEL_FILE, "nki_attention_")
    except ImportError as e:
        sys.exit(f"cannot load the NKI kernel here ({e}). Run this inside the seat pod.")

    buffers, calls = source_map()
    shapes = [(128, 64), (64, 128), (96, 32)] if a.summary_only else [(a.seq, a.dim)]
    for seq, dim in shapes:
        r = np.random.default_rng(0)
        args = tuple(r.standard_normal((seq, dim)).astype(np.float32) for _ in range(3))
        steps, out = trace(kernel, args, buffers, calls)
        want = reference64(*args)
        got = np.asarray(out, np.float64)
        ok = bool(np.all(np.isfinite(got)) and np.abs(got - want).max() <= 2e-2 * np.sqrt((want ** 2).mean()))
        if not a.summary_only:
            print(f"EVERY INSTRUCTION, IN THE ORDER THE SIMULATOR RAN IT  (seq={seq} dim={dim})")
            print_trace(steps)
        summarize(steps, seq, dim, ok)


if __name__ == "__main__":
    main()
