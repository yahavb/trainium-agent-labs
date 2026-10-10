import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_attention_(q, k, v):
    seq, dim = q.shape
    q_sbuf = nl.ndarray(shape=(seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    k_sbuf = nl.ndarray(shape=(seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sbuf, src=q)
    nisa.dma_copy(dst=k_sbuf, src=k)
    q_psum = nl.ndarray(shape=(dim, seq), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=q_psum, data=q_sbuf)
    k_psum = nl.ndarray(shape=(dim, seq), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=k_psum, data=k_sbuf)
    q_sbuf_trans = nl.ndarray(shape=(dim, seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=q_sbuf_trans, src=q_psum)
    k_sbuf_trans = nl.ndarray(shape=(dim, seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=k_sbuf_trans, src=k_psum)
    scores = nl.ndarray(shape=(seq, seq), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=scores, stationary=q_sbuf_trans, moving=k_sbuf_trans)
    scaled_scores = nl.ndarray(shape=(seq, seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=scaled_scores, data=scores, op0=nl.multiply, operand0=1.0 / (dim ** 0.5))
    row_max = nl.ndarray(shape=(seq, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=scaled_scores, axis=1, negate=True)
    exp_scores = nl.ndarray(shape=(seq, seq), dtype=nl.float32, buffer=nl.sbuf)
    exp_sum = nl.ndarray(shape=(seq, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=exp_scores, op=nl.exp, data=scaled_scores, bias=row_max, reduce_op=nl.add, reduce_res=exp_sum, reduce_cmd=nisa.reduce_cmd.reset_reduce)
    row_sums = nl.ndarray(shape=(seq, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=row_sums, data=exp_sum)
    probs = nl.ndarray(shape=(seq, seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=probs, data=exp_scores, op0=nl.multiply, operand0=row_sums)
    probs_psum = nl.ndarray(shape=(seq, seq), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=probs_psum, data=probs)
    probs_sbuf = nl.ndarray(shape=(seq, seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=probs_sbuf, src=probs_psum)
    v_sbuf = nl.ndarray(shape=(seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_sbuf, src=v)
    result_psum = nl.ndarray(shape=(seq, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=result_psum, stationary=probs_sbuf, moving=v_sbuf)
    result_sbuf = nl.ndarray(shape=(seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=result_sbuf, src=result_psum)
    result = nl.ndarray(shape=(seq, dim), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=result, src=result_sbuf)
    return result
