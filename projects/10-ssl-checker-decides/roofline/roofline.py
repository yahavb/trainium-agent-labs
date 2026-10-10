"""Theoretical roofline for one Trainium2 NeuronCore-v3 (what `compile_kernel_to_nir(lnc=1)` models).

Pure Python, no Neuron SDK needed. For an operation and a shape it counts the work each engine class
must do at minimum, the minimum HBM bytes, the lower-bound time on each resource, and the binding one.

Two peak sets, kept apart on purpose:

  HW     hardware peaks from the docs / SDK source shipped in the pod (cited per number below).
         Floor = max over resources, perfect overlap, no fixed costs. A true lower bound.
  MODEL  the rates the compiler's latency model actually charges, measured by differencing predicted
         times (peaks_probe.py, peaks_probe2.py). Model floor = max over resources + the model's fixed
         ~2.6 us load->store latency. This is what a perfect kernel would be PREDICTED at, so
         predicted / model_floor is the part of the gap the kernel author can still remove.

Pod doc paths (relative to .../site-packages/neuron_agentic_development/artifacts/skills/neuron-nki-docs/
references/): architecture/trainium2_arch.md, programming/tutorials/matrix_multiplication.md.
SDK: .../site-packages/nki/compiler/ncc_driver.py, nki/_backends/mlir_tracer/target_info.py.

    python roofline.py                 # table for the default shapes
"""
import math
from itertools import product

GHZ = 1e9
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}

HW = dict(
    # trainium2_arch.md:132 "79 BF16/FP16/TF32 and 20 FP32 dense TFLOPS"; the bf16 figure is
    # also ncc_driver.py:490-495 _PEAK_OPS_PER_SEC["trn2"] = 128*128*2*2.4e9 = 78.6e12.
    tensor_flops={"bfloat16": 128 * 128 * 2 * 2.4 * GHZ, "float16": 128 * 128 * 2 * 2.4 * GHZ,
                  "float32": 20e12},
    # trainium2_arch.md:124 Vector "512 BF16/FP16 input/output; 256 for other data types", 0.96 GHz.
    vector_elems={"bfloat16": 512 * 0.96 * GHZ, "float16": 512 * 0.96 * GHZ, "float32": 256 * 0.96 * GHZ},
    # trainium2_arch.md:125 Scalar "128 input/output", 1.2 GHz (any dtype; math is fp32).
    scalar_elems={d: 128 * 1.2 * GHZ for d in DTYPE_BYTES},
    # trainium2_arch.md:8,10: 8 NeuronCores share 4 HBM stacks of 3 TB/s total -> 375 GB/s per core.
    # (ncc_driver.py:497-503 says 716e9 "per NeuronCore"; see the report: that matches one LNC=2
    # logical core, i.e. two physical cores, not the single core lnc=1 compiles for.)
    hbm_Bps=3e12 / 8,
    fixed_ns=0.0,
)
# GpSimd: trainium2_arch.md:126 gives 1.2 GHz but leaves the width blank, so it is left out.

MODEL = dict(
    # peaks_probe2.py: matmul cost = 231 ns + N/2.4 GHz per instruction, identical for fp32 and bf16
    # and independent of K, M <= 128 -> 1 PE column per cycle at 2.4 GHz, dtype-blind.
    tensor_flops={d: 128 * 128 * 2 * 2.4 * GHZ for d in DTYPE_BYTES},
    # peaks_probe2.py: tensor_scalar cost ~430 ns + F/0.96 GHz per instruction -> 128 elem/cycle,
    # dtype-blind (half the doc's fp32 rate, a quarter of its bf16 rate).
    vector_elems={d: 128 * 0.96 * GHZ for d in DTYPE_BYTES},
    # peaks_probe2.py: activation(exp) cost ~320 ns + F/1.2 GHz -> 128 elem/cycle, matches the doc.
    scalar_elems={d: 128 * 1.2 * GHZ for d in DTYPE_BYTES},
    # peaks_probe.py: slope of predicted time over 8 -> 32 MiB of load+store = 368 GB/s.
    hbm_Bps=368e9,
    # peaks_probe2.py: a 128x1 load+store kernel is predicted at 2.6 us.
    fixed_ns=2600.0,
)

PMAX = 128  # nl.tile_size.pmax; target_info.py:27 sbuf/psum partitions = 128


def _ceil(a, b):
    return -(-a // b)


class Op:
    """name, tensor work as PE columns (128-deep x up-to-128-wide), elementwise passes with allowed
    engines, and the minimum HBM bytes (every input read once, the output written once)."""

    def __init__(self, name, dtype, flops=0, pe_columns=0, pe_padded_flops=0, passes=(), hbm_bytes=0,
                 notes=()):
        self.name, self.dtype, self.flops, self.pe_columns = name, dtype, flops, pe_columns
        self.pe_padded_flops = pe_padded_flops or flops
        self.passes, self.hbm_bytes, self.notes = list(passes), hbm_bytes, list(notes)


def softmax(R, C, dtype="float32"):
    E = R * C
    return Op(f"softmax over last axis {R}x{C} {dtype}", dtype,
              passes=[("row max", E, ("vector",)),            # tensor_reduce: Vector only
                      ("exp(x-max) + fused row sum", E, ("scalar",)),  # exp is ScalarE on trn2
                      ("x * 1/sum", E, ("vector", "scalar"))],
              hbm_bytes=2 * E * DTYPE_BYTES[dtype],
              notes=["exp on VectorE (nisa.exponential) is NeuronCore-v4+ only (isa/__init__.pyi:2272)"])


def matmul(M, K, N, dtype="float32"):
    b = DTYPE_BYTES[dtype]
    kt, mt = _ceil(K, PMAX), _ceil(M, PMAX)
    return Op(f"matmul M={M} K={K} N={N} {dtype}", dtype, flops=2 * M * K * N,
              pe_columns=kt * mt * N, pe_padded_flops=2 * kt * PMAX * mt * PMAX * N,
              passes=[("PSUM -> SBUF eviction", M * N, ("vector", "scalar"))],
              hbm_bytes=(M * K + K * N + M * N) * b)


def conv_block(Cin, H, W, Cout, k, stride, pad, pool_k=3, pool_s=2, dtype="float32"):
    """conv (im2col view on TensorE) -> bias -> relu (one fused activation that also evicts PSUM)
    -> maxpool (separable window max on VectorE). Batch 1. Intermediates never touch HBM."""
    b = DTYPE_BYTES[dtype]
    Ho, Wo = (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1
    Hp, Wp = (Ho - pool_k) // pool_s + 1, (Wo - pool_k) // pool_s + 1
    Kc = Cin * k * k
    kt, mt = _ceil(Kc, PMAX), _ceil(Cout, PMAX)
    return Op(f"conv{k}x{k}/s{stride} {Cin}->{Cout} on {H}x{W} +bias+relu+maxpool{pool_k}/{pool_s} "
              f"-> {Cout}x{Hp}x{Wp} {dtype}", dtype,
              flops=2 * Cout * Ho * Wo * Kc, pe_columns=kt * mt * Ho * Wo,
              pe_padded_flops=2 * kt * PMAX * mt * PMAX * Ho * Wo,
              passes=[("bias+relu (evicts PSUM)", Cout * Ho * Wo, ("scalar", "vector")),
                      ("maxpool along W", pool_k * Cout * Ho * Wp, ("vector",)),
                      ("maxpool along H", pool_k * Cout * Hp * Wp, ("vector",))],
              hbm_bytes=(Cin * H * W + Cout * Kc + Cout + Cout * Hp * Wp) * b,
              notes=[f"conv output {Cout}x{Ho}x{Wo}; contraction {Kc} pads to {kt * PMAX}, "
                     f"Cout {Cout} pads to {mt * PMAX}: PE array at most "
                     f"{(Kc * Cout) / (kt * PMAX * mt * PMAX):.0%} occupied",
                     "assumes the im2col patches are formed on-chip (input read from HBM once)"])


def avgpool(C, H, W, p, dtype="float32"):
    Ho, Wo = H // p, W // p
    return Op(f"avgpool C,H,W={C},{H},{W} pool={p} {dtype}", dtype,
              passes=[("window sum", C * Ho * Wo * p * p, ("vector",)),
                      ("x 1/p^2", C * Ho * Wo, ("vector", "scalar"))],
              hbm_bytes=(C * H * W + C * Ho * Wo) * DTYPE_BYTES[dtype])


def transpose2d(P, F, dtype="float32"):
    return Op(f"per-row transpose {P}x{F} {dtype}", dtype, passes=[],
              hbm_bytes=2 * P * F * DTYPE_BYTES[dtype],
              notes=["a DMA access pattern can do the permutation, so no engine work is required"])


def _engine_times(op, peaks):
    """Best assignment of flexible passes to engines (brute force; passes are few)."""
    rate = dict(vector=peaks["vector_elems"][op.dtype], scalar=peaks["scalar_elems"][op.dtype])
    best = None
    for choice in product(*[p[2] for p in op.passes]):
        t = dict(vector=0.0, scalar=0.0)
        for (_, n, _), eng in zip(op.passes, choice):
            t[eng] += n / rate[eng] * 1e9
        if best is None or max(t.values()) < max(best[0].values()):
            best = (t, choice)
    return best if best else (dict(vector=0.0, scalar=0.0), ())


def analyze(op, peaks=HW):
    eng, choice = _engine_times(op, peaks)
    # Tensor: padded PE flops (K and M tiles of 128 occupy the whole array whether full or not).
    t_tensor = op.pe_padded_flops / peaks["tensor_flops"][op.dtype] * 1e9 if op.flops else 0.0
    t = dict(tensor=t_tensor, vector=eng["vector"], scalar=eng["scalar"],
             hbm=op.hbm_bytes / peaks["hbm_Bps"] * 1e9)
    bound = max(t, key=t.get)
    work = dict(tensor_flops=op.flops, vector_elems=0, scalar_elems=0)
    for (_, n, _), e in zip(op.passes, choice):
        work[f"{e}_elems"] += n
    return dict(op=op.name, times_ns=t, bound=bound, floor_ns=t[bound] + peaks["fixed_ns"],
                hbm_bytes=op.hbm_bytes, work=work,
                assignment=[(p[0], e) for p, e in zip(op.passes, choice)], notes=op.notes)


BOUND_WORDS = dict(tensor="compute bound on TensorE", vector="compute bound on VectorE",
                   scalar="compute bound on ScalarE", hbm="memory (HBM bandwidth) bound")


def us(ns):
    return f"{ns / 1000:.2f} us"


def feedback(op, predicted_ns=None, measured_bytes=None, utilization=None):
    """The message an agent gets. predicted_ns from compile_kernel_to_nir(...).total_time_ns,
    measured_bytes from counting dma_copy traffic in nki.simulate (nkibench.simulate_and_count)."""
    hw, md = analyze(op, HW), analyze(op, MODEL)
    t = hw["times_ns"]
    second = sorted((v, k) for k, v in t.items() if k != hw["bound"])[-1]
    msg = [f"{op.name}: {BOUND_WORDS[hw['bound']]}. Hardware floor {us(hw['floor_ns'])} "
           f"(next: {second[1]} {us(second[0])}). Minimum HBM traffic {op.hbm_bytes / 1024:.0f} KiB."]
    if predicted_ns is not None:
        msg.append(f"Your kernel is predicted at {us(predicted_ns)}: {predicted_ns / hw['floor_ns']:.1f}x the "
                   f"hardware floor. A perfect kernel would be predicted at about {us(md['floor_ns'])}, so "
                   f"{predicted_ns / md['floor_ns']:.1f}x is yours to remove.")
    if measured_bytes is not None:
        w = measured_bytes / op.hbm_bytes
        msg.append(f"It moves {measured_bytes / 1024:.0f} KiB, {w:.2f}x the minimum"
                   + (" -- some input is loaded more than once; keep it in SBUF." if w > 1.05 else "."))
    if utilization:
        busy = max(utilization, key=utilization.get)
        msg.append(f"Busiest resource in the prediction: {busy} {utilization[busy]:.0f}%.")
    wasted_bytes = measured_bytes is not None and measured_bytes > 1.05 * op.hbm_bytes
    if op.passes and (measured_bytes is None or not wasted_bytes):
        plan = "; ".join(f"{name} on {e}" for name, e in hw["assignment"])
        msg.append(f"The minimum full-size engine work is {len(op.passes)} pass(es): {plan}. Each extra "
                   f"full-size pass, and any step that waits for the previous one instead of overlapping "
                   f"with the next tile's DMA, adds time.")
    if hw["bound"] == "hbm":
        msg.append(f"Floor is set by HBM: aim to keep DMA busy the whole time ({us(t['hbm'])}).")
    else:
        msg.append(f"Floor is set by the {hw['bound']} engine: it needs {us(t[hw['bound']])} even with "
                   "perfect overlap, so cut or densify that work first.")
    return " ".join(msg)


DEFAULT_OPS = [
    softmax(128, 4096), softmax(512, 1024), softmax(128, 4096, "bfloat16"),
    conv_block(3, 63, 63, 64, k=11, stride=4, pad=2),     # AlexNet conv1 geometry on a 63x63 image
    conv_block(64, 27, 27, 192, k=5, stride=1, pad=2),    # AlexNet conv2 at its real 27x27 input
    conv_block(64, 27, 27, 192, k=5, stride=1, pad=2, dtype="bfloat16"),
    matmul(128, 128, 512), matmul(128, 512, 512), matmul(512, 256, 1024),
    matmul(512, 256, 1024, "bfloat16"), matmul(2048, 2048, 2048, "bfloat16"),
]


def table(ops=DEFAULT_OPS):
    for op in ops:
        for label, peaks in (("HW   ", HW), ("MODEL", MODEL)):
            r = analyze(op, peaks)
            t = r["times_ns"]
            if label == "HW   ":
                w = r["work"]
                print(f"\n{op.name}\n  work: tensor {w['tensor_flops'] / 1e6:.1f} MFLOP, vector "
                      f"{w['vector_elems'] / 1e6:.2f} M elem, scalar {w['scalar_elems'] / 1e6:.2f} M elem, "
                      f"HBM {r['hbm_bytes'] / 1024:.0f} KiB   assignment {r['assignment']}")
                for n in r["notes"]:
                    print(f"  note: {n}")
            print(f"  {label} tensor {us(t['tensor']):>9} vector {us(t['vector']):>9} scalar "
                  f"{us(t['scalar']):>9} hbm {us(t['hbm']):>9} -> {r['bound']:6} floor {us(r['floor_ns'])}")


if __name__ == "__main__":
    print("Trainium2, one NeuronCore-v3. HW = doc peaks (floor, no fixed cost); MODEL = rates the "
          "compiler's latency model charges (+2.6 us fixed).")
    for d in ("bfloat16", "float32"):
        print(f"  ridge (TensorE flops / HBM byte), {d}: HW {HW['tensor_flops'][d] / HW['hbm_Bps']:.0f}, "
              f"MODEL {MODEL['tensor_flops'][d] / MODEL['hbm_Bps']:.0f}")
    table()
