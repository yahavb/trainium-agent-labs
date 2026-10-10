"""Softmax over the last axis, x [R, C] fp32, R a multiple of 128, one 128-row tile per iteration.

softmax_fused     : 1 load, Vector max (negated), Scalar exp(x - max) with the row sum fused into the
                    same instruction, Vector reciprocal of [128,1], Vector multiply, 1 store.
softmax_reload    : same math but loads the input tile from HBM twice (once for max, once for exp):
                    correct, 1.5x the minimum HBM bytes.
softmax_extrapass : sums with a separate Vector tensor_reduce instead of fusing it into exp, and multiplies
                    by 1/sum with a Vector tensor_tensor against a broadcast copy (nl.divide simulates
                    but does not compile): 2 extra full-size Vector passes.
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def softmax_fused(x):
    R, C = x.shape
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    for i in nl.affine_range(R // 128):
        t = nl.ndarray((128, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[i * 128:(i + 1) * 128, :])
        nmx = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=nmx, op=nl.maximum, data=t, axis=1, negate=True)
        e = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
        s = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=e, op=nl.exp, data=t, bias=nmx, reduce_op=nl.add, reduce_res=s,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        rs = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rs, data=s)
        o = nl.ndarray((128, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o, data=e, op0=nl.multiply, operand0=rs)
        nisa.dma_copy(dst=out[i * 128:(i + 1) * 128, :], src=o)
    return out


@nki.jit
def softmax_reload(x):
    R, C = x.shape
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    for i in nl.affine_range(R // 128):
        t = nl.ndarray((128, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[i * 128:(i + 1) * 128, :])
        nmx = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=nmx, op=nl.maximum, data=t, axis=1, negate=True)
        t2 = nl.ndarray((128, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t2, src=x[i * 128:(i + 1) * 128, :])
        e = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
        s = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=e, op=nl.exp, data=t2, bias=nmx, reduce_op=nl.add, reduce_res=s,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        rs = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rs, data=s)
        o = nl.ndarray((128, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o, data=e, op0=nl.multiply, operand0=rs)
        nisa.dma_copy(dst=out[i * 128:(i + 1) * 128, :], src=o)
    return out


@nki.jit
def softmax_extrapass(x):
    R, C = x.shape
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    for i in nl.affine_range(R // 128):
        t = nl.ndarray((128, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[i * 128:(i + 1) * 128, :])
        nmx = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=nmx, op=nl.maximum, data=t, axis=1, negate=True)
        e = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=e, op=nl.exp, data=t, bias=nmx)
        s = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=s, op=nl.add, data=e, axis=1)
        rs = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rs, data=s)
        sb = nl.ndarray((128, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sb, data=e, op0=nl.multiply, operand0=0.0, op1=nl.add, operand1=rs)
        o = nl.ndarray((128, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o, data1=e, data2=sb, op=nl.multiply)
        nisa.dma_copy(dst=out[i * 128:(i + 1) * 128, :], src=o)
    return out
