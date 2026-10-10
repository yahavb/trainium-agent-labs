import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_basic_(lhsT, rhs):
    lhs_sbuf = nl.ndarray(lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
    rhs_sbuf = nl.ndarray(rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=lhs_sbuf, src=lhsT)
    nisa.dma_copy(dst=rhs_sbuf, src=rhs)
    acc_psum = nl.ndarray((lhsT.shape[1], rhs.shape[1]), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=acc_psum, stationary=lhs_sbuf, moving=rhs_sbuf)
    acc_psum_sbuf = nl.ndarray(acc_psum.shape, dtype=acc_psum.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=acc_psum_sbuf, src=acc_psum)
    result = nl.ndarray((lhsT.shape[1], rhs.shape[1]), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=result, src=acc_psum_sbuf)
    return result