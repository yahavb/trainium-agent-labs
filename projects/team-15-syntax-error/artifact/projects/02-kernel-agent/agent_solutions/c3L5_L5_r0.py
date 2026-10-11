# Written by Qwen3-8B (local, seat 73) in run c3L5_L5_r0, round 2, under --skeleton --v2/v3/v4-verdicts --directional.
# Verified [sim]: level 5 incl. the 1.90x traffic bar, and all held-out cases. Not edited by hand.
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_hoist_load_(lhsT, rhs):
    K, M = lhsT.shape                         # left operand arrives TRANSPOSED: K is the partition dim
    _, N = rhs.shape
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    TM, TK, TN = 128, 128, 512                # hardware MAXIMA, not targets
    for m in nl.affine_range((M + TM - 1) // TM):
        m0 = m * TM
        m_sz = min(TM, M - m0)
        lhs_tiles = []                        # Hoist lhsT tiles out of the n loop
        for k in nl.affine_range((K + TK - 1) // TK):
            k0 = k * TK
            k_sz = min(TK, K - k0)
            lhs_tile = nl.ndarray((k_sz, m_sz), dtype=lhsT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=lhs_tile, src=lhsT[k0:k0 + k_sz, m0:m0 + m_sz])
            lhs_tiles.append(lhs_tile)
        for n in nl.affine_range((N + TN - 1) // TN):
            n0 = n * TN
            n_sz = min(TN, N - n0)
            acc = nl.ndarray((m_sz, n_sz), dtype=nl.float32, buffer=nl.psum)   # ONE accumulator
            for k in nl.affine_range((K + TK - 1) // TK):
                k0 = k * TK
                k_sz = min(TK, K - k0)
                rhs_tile = nl.ndarray((k_sz, n_sz), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=rhs_tile, src=rhs[k0:k0 + k_sz, n0:n0 + n_sz])
                nisa.nc_matmul(dst=acc, stationary=lhs_tiles[k], moving=rhs_tile)
            # TODO: copy acc PSUM -> an SBUF tile, then dma_copy it to result[m0:m0 + m_sz, n0:n0 + n_sz]
            out_tile = nl.ndarray((m_sz, n_sz), dtype=acc.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_tile, src=acc)
            nisa.dma_copy(dst=result[m0:m0 + m_sz, n0:n0 + n_sz], src=out_tile)
    return result
