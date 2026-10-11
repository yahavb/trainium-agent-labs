import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_basic_(lhsT, rhs):
    K, M = lhsT.shape
    _, N = rhs.shape
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    TM, TK, TN = 128, 128, 512
    for m in nl.affine_range((M + TM - 1) // TM):
        m0 = m * TM
        m_sz = min(TM, M - m0)
        for n in nl.affine_range((N + TN - 1) // TN):
            n0 = n * TN
            n_sz = min(TN, N - n0)
            acc = nl.ndarray((m_sz, n_sz), dtype=nl.float32, buffer=nl.psum)
            for k in nl.affine_range((K + TK - 1) // TK):
                k0 = k * TK
                k_sz = min(TK, K - k0)
                a_t = nl.ndarray((k_sz, m_sz), dtype=lhsT.dtype, buffer=nl.sbuf)
                b_t = nl.ndarray((k_sz, n_sz), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=a_t, src=lhsT[k0:k0 + k_sz, m0:m0 + m_sz])
                nisa.dma_copy(dst=b_t, src=rhs[k0:k0 + k_sz, n0:n0 + n_sz])
                nisa.nc_matmul(dst=acc, stationary=a_t, moving=b_t)
            out_t = nl.ndarray((m_sz, n_sz), dtype=lhsT.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_t, src=acc)
            nisa.dma_copy(dst=result[m0:m0 + m_sz, n0:n0 + n_sz], src=out_t)
    return result
