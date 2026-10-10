#!/usr/bin/env python3
"""
speedcheck.py -- the CHIPBOOST referee. One candidate kernel in, one verdict and ONE named change out.

In order, stopping at the first failure:

  1. rules      static scan (nkibench.check_rules)                              -> "rules"
  2. sim        CPU simulation vs NumPy on small shapes; bytes counted over EVERY
                DMA (dma_copy, dma_transpose, dma_compute); inputs must be untouched -> "wrong"
  3. chip       on the device at the timing shapes, fresh random inputs each call,
                outputs poisoned first, inputs read back afterwards               -> "wrong"
  4. held-out   on the device at shapes the loop never sees, plus hostile values  -> "heldout_fail"
  5. timing     device clock, interleaved A B A B against the baseline, and a
                speedup only counts above the measured noise                     -> "faster" / "slower"
  6. why slow   bytes vs the floor, transfer count, intensity vs ceiling -> ONE instruction

A wrong kernel scores zero however fast it is: correctness is checked on the chip, not only in the
simulator, because a harness can return zeros with exit code 0 and "measure" a kernel that computes
nothing.

    python speedcheck.py --op matmul --check cand.py                 # verdict, human readable
    python speedcheck.py --op matmul --check cand.py --json          # one schema record on stdout
    python speedcheck.py --op matmul --check cand.py --log attempts.jsonl
    python speedcheck.py --op matmul --check kernels/matmul_start.py --heldout   # include held-out

Ops come from shapes.py (P2) when it exists; otherwise the built-in matmul spec below.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "02-kernel-agent"))
sys.path.insert(0, HERE)

import nkibench  # noqa: E402
import schema    # noqa: E402

# ---------------------------------------------------------------- ops


def _bf16():
    import ml_dtypes
    return ml_dtypes.bfloat16


def _matmul_inputs(shape, seed, hostile=False):
    K, M, N = shape
    r = np.random.default_rng(seed)
    a = r.standard_normal((K, M)).astype(np.float32)
    b = r.standard_normal((K, N)).astype(np.float32)
    if hostile:                       # large magnitudes, exact zeros, a sign-flipped block
        a[: K // 8] *= 64.0
        b[:, : N // 16] = 0.0
        a[:, M // 2:] *= -1.0
    return {"lhsT": a.astype(_bf16()), "rhs": b.astype(_bf16())}


def _matmul_ref(inp):
    return inp["lhsT"].astype(np.float32).T @ inp["rhs"].astype(np.float32)


# Qwen3-8B per-core shapes under tensor parallel 2, as (K, M, N) with M = prompt tokens.
# Matmul shapes are tile multiples on purpose: reference_level4 asserts K, M % 128 and N % 512.
MATMUL = dict(
    level=4,                                   # rules (banned calls, entry name) come from nkibench level 4
    entry="nki_matmul_tiled_",
    make_inputs=_matmul_inputs,
    ref=_matmul_ref,
    flops=lambda s: 2 * s[0] * s[1] * s[2],
    sim_shapes=[(256, 512, 1024), (512, 256, 2048)],          # small: CPU simulator, byte counting
    time_shapes=[(4096, 256, 2048), (4096, 256, 6144)],       # q_proj, gate/up at 256 tokens
    heldout_shapes=[(6144, 256, 4096), (2048, 256, 4096),     # down_proj, o_proj
                    (4096, 512, 2048), (4096, 128, 6144)],    # other token buckets
    tol=2e-2,
)

OPS = {"matmul": MATMUL}
try:                                            # P2's shapes.py extends or overrides
    import shapes as _shapes
    OPS.update(getattr(_shapes, "OPS", {}))
except ImportError:
    pass


# ---------------------------------------------------------------- stage 2: simulate, counting every DMA

def simulate_count_all(kernel, args):
    """nkibench.simulate_and_count only hooks nisa.dma_copy, so a kernel moving data with
    dma_transpose or dma_compute looks like it moves fewer bytes than it does. Hook all three."""
    import nki
    import nki.isa as nisa
    run, api = nkibench._simulator(nki, kernel)
    counter = dict(bytes=0, transfers=0, api=api, dtypes=set(), by_op={})
    originals = {}

    def size(t):
        n = getattr(t, "nbytes", None)
        return int(n) if isinstance(n, int) and n > 0 else int(np.prod(t.shape)) * nkibench.itemsize_of(t)

    def wrap(name, fn):
        def counted(*a, **kw):
            srcs = kw.get("srcs") or ([kw["src"]] if "src" in kw else list(a[1:2]))
            if not isinstance(srcs, (list, tuple)):
                srcs = [srcs]
            for s in srcs:
                try:
                    counter["bytes"] += size(s)
                    counter["dtypes"].add(str(getattr(s, "dtype", "?")))
                except Exception:
                    counter["unmeasured"] = counter.get("unmeasured", 0) + 1
            counter["transfers"] += 1
            counter["by_op"][name] = counter["by_op"].get(name, 0) + 1
            return fn(*a, **kw)
        return counted

    for name in ("dma_copy", "dma_transpose", "dma_compute"):
        if hasattr(nisa, name):
            originals[name] = getattr(nisa, name)
            setattr(nisa, name, wrap(name, originals[name]))
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = run(*args)
    finally:
        for name, fn in originals.items():
            setattr(nisa, name, fn)
    return out, counter


# ---------------------------------------------------------------- stage 6: one instruction

def one_instruction(counter, args, want, flops):
    """Turn the measurements into ONE change to make. Never just 'too slow'."""
    elements = int(np.prod(np.shape(want)))
    floor = nkibench.minimum_hbm_bytes(list(args), want)
    waste = counter["bytes"] / floor if floor else 1.0
    if counter["transfers"] > max(8, elements // 64):
        return (f"One transfer per few elements ({counter['transfers']:,} transfers for {elements:,} outputs): "
                f"move whole 128-row tiles per DMA, not elements.")
    if waste > 1.15:
        return (f"Same tiles reloaded every pass ({waste:.2f}x the byte floor): move the operand loads out of "
                f"the innermost loop so each tile is loaded once and reused across it.")
    ceiling = flops / floor if floor else float("inf")
    ridge = nkibench.RIDGE_FLOPS_PER_BYTE["bfloat16"]
    if ceiling >= ridge:
        return ("Bytes are already near the floor and this shape can be compute bound: keep the Tensor "
                "Engine busy -- block K so one PSUM tile accumulates across the whole contraction, and "
                "overlap the next tile's load with the current matmul.")
    return ("At the byte floor on a memory-bound shape: the remaining cost is transfer efficiency -- "
            "issue fewer, larger DMAs and overlap them with compute.")


# ---------------------------------------------------------------- the referee

def _hash(src):
    return hashlib.sha1(src.encode()).hexdigest()[:12]


def _record(**kw):
    rec = {k: None for k in schema.ATTEMPT_FIELDS}
    rec.update(seat=int(os.environ.get("CHIPBOOST_SEAT", "100")), timestamp=time.time())
    rec.update({k: v for k, v in kw.items() if k in schema.ATTEMPT_FIELDS})
    return rec


def bf16_ulps(got, want):
    """Worst error in bf16 units-in-the-last-place of the fp32 reference.

    A relative-to-RMS tolerance is the wrong yardstick for a bf16 output: the reference kernel's own
    output rounding already reaches 1.6e-2 of RMS on large-magnitude inputs, uncomfortably close to a
    2e-2 bar, while a kernel that accumulates in bf16 instead of fp32 can still hide under it. In ulps
    the honest rounding is ~1 and lower-precision accumulation over K=4096 is far above."""
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    mag = np.maximum(np.abs(want), 1e-30)
    ulp = np.exp2(np.floor(np.log2(mag)) - 7)                       # bf16 has 8 significant bits
    floor = 1e-3 * (np.sqrt((want ** 2).mean()) or 1.0)             # cancellation near zero is not a bug
    return float((np.abs(got - want) / np.maximum(ulp, floor)).max())


MAX_ULPS = 4.0


def _mismatch(got, want, tol, label):
    got32 = np.asarray(got, np.float32)
    m = nkibench.describe_mismatch(got32, np.asarray(want, np.float32), max(tol, 5e-2))
    if m:
        return f"at {label}: {m}"
    u = bf16_ulps(got32, want)
    if u > MAX_ULPS:
        return (f"at {label}: PRECISION LOSS: errors up to {u:.1f} bf16 ulps against an fp32 reference "
                f"(the limit is {MAX_ULPS:g}; honest bf16 output rounding is ~1). Accumulate in an fp32 PSUM "
                f"tile across the whole contraction and round to bf16 once, at the end.")
    return None


def check(path, op="matmul", baseline=None, heldout=True, rounds=3, verbose=False):
    """Referee one candidate. Returns a dict with the schema fields the referee owns."""
    spec = OPS[op]
    src = open(path).read()
    base = dict(kernel=op, code_hash=_hash(src), sim_ok=False, chip_ok=False)
    say = print if verbose else (lambda *a, **k: None)

    # 1. rules
    rules = nkibench.check_rules(src, spec["level"])
    if rules:
        msg = "RULE VIOLATIONS (scores zero):\n" + "\n".join(f"  {v}" for v in rules)
        return _record(**base, verdict="rules", referee_message=msg,
                       instruction_given=rules[0].split(": ", 1)[-1])
    say("  rules      clean")

    try:
        kernel = nkibench.load_kernel(path, spec["entry"])
    except Exception as e:
        msg = f"failed to import: {type(e).__name__}: {e}"
        return _record(**base, verdict="rules", referee_message=msg, instruction_given=msg)

    # 2. simulator, small shapes, every DMA counted, inputs untouched
    seed = int.from_bytes(os.urandom(4), "little")       # fresh inputs: a cached answer cannot pass
    diag = None
    for shape in spec["sim_shapes"]:
        inp = spec["make_inputs"](shape, seed)
        args = list(inp.values())
        before = [a.copy() for a in args]
        want = spec["ref"](inp)
        try:
            got, counter = simulate_count_all(kernel, args)
        except Exception as e:
            msg = f"raised in the simulator at {shape}: {type(e).__name__}: {e}"
            return _record(**base, verdict="wrong", referee_message=msg, instruction_given=msg)
        bad = nkibench.check_inputs_untouched(before, args) or _mismatch(got, want, spec["tol"], f"sim {shape}")
        if bad:
            return _record(**base, verdict="wrong", referee_message=bad, instruction_given=bad.split("\n")[0])
        diag = (counter, args, want, spec["flops"](shape))
    base["sim_ok"] = True
    say(f"  simulator  {len(spec['sim_shapes'])}/{len(spec['sim_shapes'])} shapes correct; "
        f"{diag[0]['bytes']:,} bytes in {diag[0]['transfers']} transfers {diag[0]['by_op']}")

    # 3 + 4. on the chip: timing shapes, then held-out shapes with hostile values
    import timing
    loaded = {}
    chip_cases = [(s, False, "dev") for s in spec["time_shapes"]]
    if heldout:
        chip_cases += [(s, True, "heldout") for s in spec["heldout_shapes"]]
    worst = 0.0
    for shape, hostile, kind in chip_cases:
        inp = spec["make_inputs"](shape, seed + 1, hostile=hostile)
        want = spec["ref"](inp)
        try:
            L = timing.Loaded(timing.compile_kernel(kernel, inp), inp)
            got = L.run()[0]
            after = [t.numpy() for t in L.inputs.values()]
        except Exception as e:
            msg = f"failed on the chip at {shape}: {type(e).__name__}: {str(e)[:400]}"
            v = "heldout_fail" if kind == "heldout" else "wrong"
            return _record(**base, verdict=v, referee_message=msg, instruction_given=msg)
        bad = nkibench.check_inputs_untouched(list(inp.values()), after) or \
            _mismatch(got, want, spec["tol"], f"chip {kind} {shape}{' hostile' if hostile else ''}")
        if bad:
            v = "heldout_fail" if kind == "heldout" else "wrong"
            if kind == "heldout":
                bad = ("Correct on the development shapes but WRONG on a shape the loop never saw -- "
                       "the kernel must not depend on the shapes it was tuned on.\n" + bad)
            return _record(**base, verdict=v, referee_message=bad, instruction_given=bad.split("\n")[0])
        worst = max(worst, bf16_ulps(np.asarray(got, np.float32), want))
        if kind == "dev":
            loaded[shape] = L
    base["chip_ok"] = True
    say(f"  chip       correct on {len(chip_cases)} shapes (worst error {worst:.2f} bf16 ulps)")

    # 5. timing vs the baseline, interleaved, summed over the timing shapes
    baseline = baseline or os.path.join(HERE, "kernels", f"{op}_start.py")
    if not os.path.exists(baseline):
        baseline = os.path.join(HERE, "..", "02-kernel-agent", "reference_level4.py")
    bkern = nkibench.load_kernel(baseline, spec["entry"])
    t_cand = t_base = iqr = 0.0
    for shape, L in loaded.items():
        inp = spec["make_inputs"](shape, seed + 1)
        B = timing.Loaded(timing.compile_kernel(bkern, inp), inp)
        ab = timing.time_ab(B, L, rounds=rounds)
        t_base += ab["a"]["median_us"]
        t_cand += ab["b"]["median_us"]
        iqr = max(iqr, ab["a"]["iqr_us"] / ab["a"]["median_us"], ab["b"]["iqr_us"] / ab["b"]["median_us"])
        say(f"  timing     {shape}: baseline {ab['a']['median_us']:.1f} us, candidate "
            f"{ab['b']['median_us']:.1f} us -> {ab['speedup']:.3f}x")
    speedup = t_base / t_cand
    threshold = 1.0 + max(0.05, 2.0 * iqr)
    verdict = "faster" if speedup >= threshold else "slower"

    # 6. one instruction
    instr = one_instruction(*diag)
    msg = (f"correct everywhere; {t_cand:.1f} us vs baseline {t_base:.1f} us = {speedup:.3f}x "
           f"({'beats' if verdict == 'faster' else 'does not beat'} the noise threshold {threshold:.3f}).")
    return _record(**base, verdict=verdict, referee_message=msg, instruction_given=instr,
                   time_us_median=t_cand, time_us_iqr=t_cand * iqr, baseline_us_same_session=t_base,
                   speedup=speedup, source="chip")


def check_isolated(path, op="matmul", timeout=900, **kw):
    """Run the referee in a fresh process, so a hung or crashing kernel cannot take the agent with
    it and device memory from hundreds of candidates is released each time."""
    cmd = [sys.executable, os.path.abspath(__file__), "--op", op, "--check", path, "--json"]
    if kw.get("baseline"):
        cmd += ["--baseline", kw["baseline"]]
    if not kw.get("heldout", True):
        cmd += ["--no-heldout"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return _record(kernel=op, verdict="wrong", referee_message=f"timed out after {timeout}s",
                       instruction_given="The kernel did not finish; check for an unbounded loop.")
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if not lines:
        return _record(kernel=op, verdict="wrong", referee_message=f"referee crashed:\n{p.stderr[-1500:]}",
                       instruction_given="The kernel crashed the runtime; simplify it.")
    return json.loads(lines[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="matmul", choices=sorted(OPS))
    ap.add_argument("--check", required=True, metavar="KERNEL.py")
    ap.add_argument("--baseline")
    ap.add_argument("--no-heldout", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="print one schema record, nothing else")
    ap.add_argument("--log", metavar="JSONL", help="append the record to this file")
    a = ap.parse_args()

    rec = check(a.check, a.op, baseline=a.baseline, heldout=not a.no_heldout, rounds=a.rounds,
                verbose=not a.json)
    problems = schema.validate(rec)
    if problems:
        raise SystemExit(f"referee produced an invalid record: {problems}")
    if a.log:
        with open(a.log, "a") as f:
            f.write(json.dumps(rec) + "\n")
    if a.json:
        print(json.dumps(rec))
    else:
        print(f"\nVERDICT: {rec['verdict'].upper()}")
        print(rec["referee_message"])
        print(f"\nONE CHANGE: {rec['instruction_given']}")
    sys.exit(0 if rec["verdict"] in ("faster", "slower") else 1)


if __name__ == "__main__":
    main()
