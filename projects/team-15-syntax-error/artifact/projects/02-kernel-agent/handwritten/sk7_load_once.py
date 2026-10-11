import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_fully_optimized_(lhsT, rhs):
    K, M = lhsT.shape
    _, N = rhs.shape
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    TM, TK, TN = 128, 128, 512
    NM, NK, NN = (M + TM - 1) // TM, (K + TK - 1) // TK, (N + TN - 1) // TN
    # every operand tile is loaded from HBM exactly ONCE and stays resident in SBUF
    a = []
    for k in range(NK):
        k0 = k * TK
        k_sz = min(TK, K - k0)
        row = []
        for m in range(NM):
            m0 = m * TM
            m_sz = min(TM, M - m0)
            t = nl.ndarray((k_sz, m_sz), dtype=lhsT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=t, src=lhsT[k0:k0 + k_sz, m0:m0 + m_sz])
            row.append(t)
        a.append(row)
    b = []
    for k in range(NK):
        k0 = k * TK
        k_sz = min(TK, K - k0)
        row = []
        for n in range(NN):
            n0 = n * TN
            n_sz = min(TN, N - n0)
            t = nl.ndarray((k_sz, n_sz), dtype=rhs.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=t, src=rhs[k0:k0 + k_sz, n0:n0 + n_sz])
            row.append(t)
        b.append(row)
    for m in range(NM):
        m0 = m * TM
        m_sz = min(TM, M - m0)
        for n in range(NN):
            n0 = n * TN
            n_sz = min(TN, N - n0)
            acc = nl.ndarray((m_sz, n_sz), dtype=nl.float32, buffer=nl.psum)
            for k in range(NK):
                nisa.nc_matmul(dst=acc, stationary=a[k][m], moving=b[k][n])
            out_t = nl.ndarray((m_sz, n_sz), dtype=lhsT.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_t, src=acc)
            nisa.dma_copy(dst=result[m0:m0 + m_sz, n0:n0 + n_sz], src=out_t)
    return result
