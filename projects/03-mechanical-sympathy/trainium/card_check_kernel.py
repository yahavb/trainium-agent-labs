# A correct kernel written ONLY with the techniques on agent_loop.API_CARD (prompt v2).
# Purpose: prove (a) every fact on the card works in NKI 0.6, (b) the task is solvable.
# NEVER shown to the model: agent_loop.py does not read this file.
#   python -c "import agent_loop as al; print(al.grade(open('card_check_kernel.py').read(), 2e-2))"
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def instnorm_gelu_kernel(x):
    C, N = x.shape
    y = nl.ndarray((C, N), dtype=x.dtype, buffer=nl.shared_hbm)
    for i in range((C + 127) // 128):              # blocks of at most 128 rows
        r0 = 128 * i
        rows = min(128, C - r0)                    # the last block may be smaller
        t = nl.ndarray((rows, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[r0:r0 + rows, :])
        # mean = sum / N, as a (rows, 1) tile
        s = nl.sum(t, axis=1, keepdims=True)
        mean = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mean, data=s, op0=nl.multiply, operand0=1.0 / N)
        # d = x - mean (one value per row, applied across the row)
        d = nl.ndarray((rows, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=d, data=t, op0=nl.subtract, operand0=mean)
        # var = sum(d*d) / N ; rstd = 1 / sqrt(var + eps)
        d2 = nl.ndarray((rows, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=d2, data1=d, data2=d, op=nl.multiply)
        v = nl.sum(d2, axis=1, keepdims=True)
        ve = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ve, data=v, op0=nl.multiply, operand0=1.0 / N, op1=nl.add, operand1=1e-5)
        rstd = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=ve)
        # xn = d * rstd ; g = gelu(xn) ; o = minimum(g, 10)
        xn = nl.ndarray((rows, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=xn, data=d, op0=nl.multiply, operand0=rstd)
        g = nl.ndarray((rows, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=g, op=nl.gelu, data=xn)
        o = nl.ndarray((rows, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o, data=g, op0=nl.minimum, operand0=10.0)
        nisa.dma_copy(dst=y[r0:r0 + rows, :], src=o)
    return y
