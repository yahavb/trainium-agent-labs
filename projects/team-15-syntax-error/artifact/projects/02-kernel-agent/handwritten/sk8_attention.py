import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_attention_(q, k, v):
    S, D = q.shape                                   # S, D <= 128: one tile each
    out = nl.ndarray((S, D), dtype=q.dtype, buffer=nl.shared_hbm)
    q_t = nl.ndarray((S, D), dtype=q.dtype, buffer=nl.sbuf)
    k_t = nl.ndarray((S, D), dtype=k.dtype, buffer=nl.sbuf)
    v_t = nl.ndarray((S, D), dtype=v.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_t, src=q)
    nisa.dma_copy(dst=k_t, src=k)
    nisa.dma_copy(dst=v_t, src=v)
    # transposes on chip: qT, kT are [D, S] so that D is the contraction (partition) axis
    qT_p = nl.ndarray((D, S), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=qT_p, data=q_t)
    qT = nl.ndarray((D, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qT, src=qT_p)
    kT_p = nl.ndarray((D, S), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=kT_p, data=k_t)
    kT = nl.ndarray((D, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kT, src=kT_p)
    # scores[i, j] = q_i . k_j   -> [S, S]
    sc_p = nl.ndarray((S, S), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=sc_p, stationary=qT, moving=kT)
    sc = nl.ndarray((S, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sc, data=sc_p, op0=nl.multiply, operand0=1.0 / (D ** 0.5))
    # numerically stable softmax over the free axis
    mx = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mx, data=sc, op=nl.maximum, axis=(1,))
    sh = nl.ndarray((S, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sh, data=sc, op0=nl.subtract, operand0=mx)
    e = nl.ndarray((S, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=e, op=nl.exp, data=sh)
    sm = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=sm, data=e, op=nl.add, axis=(1,))
    rs = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rs, data=sm)
    p = nl.ndarray((S, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=p, data=e, op0=nl.multiply, operand0=rs)
    # out = P @ V: contraction over the key axis -> stationary must be P^T [S_k, S_q]
    pT_p = nl.ndarray((S, S), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=pT_p, data=p)
    pT = nl.ndarray((S, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pT, src=pT_p)
    o_p = nl.ndarray((S, D), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=o_p, stationary=pT, moving=v_t)
    o = nl.ndarray((S, D), dtype=q.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o, src=o_p)
    nisa.dma_copy(dst=out, src=o)
    return out
