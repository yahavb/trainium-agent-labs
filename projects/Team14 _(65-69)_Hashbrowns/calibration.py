"""
calibration.py -- tiny kernels that price the things an attention kernel this small is made of.

At seq=96 dim=32 every instruction moves at most 96 values per partition, so the hypothesis is that
the kernel's time goes to fixed costs -- launching it, each DMA, each instruction, each hand-off
between engines -- not to the work. These measure those costs directly, on the device:

    calib_dma    q -> SBUF -> out: two DMAs and nothing else. The floor: launch plus a load and a store.
    calib_vec8   the same, plus 8 dependent [n, d] multiplies on the Vector engine
    calib_act8   the same, plus 8 dependent [n, d] copies on the Scalar engine
    calib_alt8   the same 8 steps alternating Vector, Scalar, Vector, ... -- every step a hand-off

    (vec8 - dma) / 8                      cost of one more instruction on one engine
    (alt8 - (vec8 + act8) / 2) / 8        extra cost of a hand-off between engines

Every kernel returns q unchanged (multiplying by 1.0 and copying are exact), so the benchmark can
check the device's answer like any other. Run them through bench_device.py:

    python bench_device.py calibration.py:calib_dma calibration.py:calib_vec8 \\
                           calibration.py:calib_act8 calibration.py:calib_alt8

The chains are written out by hand rather than looped, so nothing depends on how the frontend
unrolls a Python loop.
"""

import nki
import nki.isa as nisa
import nki.language as nl

REFERENCES = {name: (lambda q, k, v: q)
              for name in ("calib_dma", "calib_vec8", "calib_act8", "calib_alt8")}


@nki.jit
def calib_dma(q, k, v):
    n, d = q.shape
    out = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.shared_hbm)
    x = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=q)
    nisa.dma_copy(dst=out, src=x)
    return out


@nki.jit
def calib_vec8(q, k, v):
    n, d = q.shape
    out = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.shared_hbm)
    x0 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x1 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x2 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x3 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x4 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x5 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x6 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x7 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x8 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x0, src=q)
    nisa.tensor_scalar(dst=x1, data=x0, op0=nl.multiply, operand0=1.0)
    nisa.tensor_scalar(dst=x2, data=x1, op0=nl.multiply, operand0=1.0)
    nisa.tensor_scalar(dst=x3, data=x2, op0=nl.multiply, operand0=1.0)
    nisa.tensor_scalar(dst=x4, data=x3, op0=nl.multiply, operand0=1.0)
    nisa.tensor_scalar(dst=x5, data=x4, op0=nl.multiply, operand0=1.0)
    nisa.tensor_scalar(dst=x6, data=x5, op0=nl.multiply, operand0=1.0)
    nisa.tensor_scalar(dst=x7, data=x6, op0=nl.multiply, operand0=1.0)
    nisa.tensor_scalar(dst=x8, data=x7, op0=nl.multiply, operand0=1.0)
    nisa.dma_copy(dst=out, src=x8)
    return out


@nki.jit
def calib_act8(q, k, v):
    n, d = q.shape
    out = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.shared_hbm)
    x0 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x1 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x2 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x3 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x4 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x5 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x6 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x7 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x8 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x0, src=q)
    nisa.activation(dst=x1, op=nl.copy, data=x0, scale=1.0)
    nisa.activation(dst=x2, op=nl.copy, data=x1, scale=1.0)
    nisa.activation(dst=x3, op=nl.copy, data=x2, scale=1.0)
    nisa.activation(dst=x4, op=nl.copy, data=x3, scale=1.0)
    nisa.activation(dst=x5, op=nl.copy, data=x4, scale=1.0)
    nisa.activation(dst=x6, op=nl.copy, data=x5, scale=1.0)
    nisa.activation(dst=x7, op=nl.copy, data=x6, scale=1.0)
    nisa.activation(dst=x8, op=nl.copy, data=x7, scale=1.0)
    nisa.dma_copy(dst=out, src=x8)
    return out


@nki.jit
def calib_alt8(q, k, v):
    n, d = q.shape
    out = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.shared_hbm)
    x0 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x1 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x2 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x3 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x4 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x5 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x6 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x7 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    x8 = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x0, src=q)
    nisa.tensor_scalar(dst=x1, data=x0, op0=nl.multiply, operand0=1.0)
    nisa.activation(dst=x2, op=nl.copy, data=x1, scale=1.0)
    nisa.tensor_scalar(dst=x3, data=x2, op0=nl.multiply, operand0=1.0)
    nisa.activation(dst=x4, op=nl.copy, data=x3, scale=1.0)
    nisa.tensor_scalar(dst=x5, data=x4, op0=nl.multiply, operand0=1.0)
    nisa.activation(dst=x6, op=nl.copy, data=x5, scale=1.0)
    nisa.tensor_scalar(dst=x7, data=x6, op0=nl.multiply, operand0=1.0)
    nisa.activation(dst=x8, op=nl.copy, data=x7, scale=1.0)
    nisa.dma_copy(dst=out, src=x8)
    return out
