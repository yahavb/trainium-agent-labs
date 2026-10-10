"""Level 10, HAND-OPTIMIZED by team 20 from the agent's solved/level10_row_softmax.py (not agent output).

Change: the row max is reduced straight into a negated (seq, 1) tile, and one activation computes
exp(x - max) AND its row sum (bias + reduce), so the shifted matrix and two reduction copies are gone.
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_row_softmax_(x):
    seq, dim = x.shape
    out = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.shared_hbm)
    x_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x)
    neg_max = nl.ndarray((seq, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=x_sb, axis=1, negate=True, keepdims=True)
    e_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    row_sum = nl.ndarray((seq, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=e_sb, op=nl.exp, data=x_sb, reduce_op=nl.add, reduce_res=row_sum, bias=neg_max)
    inv = nl.ndarray((seq, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv, data=row_sum)
    y_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=y_sb, data=e_sb, op0=nl.multiply, operand0=inv)
    nisa.dma_copy(dst=out, src=y_sb)
    return out
