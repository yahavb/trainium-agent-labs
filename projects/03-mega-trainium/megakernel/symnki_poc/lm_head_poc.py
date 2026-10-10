"""Proof of concept: our LM-head inner loop checked with nkiequiv (github.com/forthoney/symnki).

The kernels are the matmul core of kernels/qwen3_decode_step.py::lm_head_tiled_argmax, unchanged in
structure: host-pre-tiled weights, one weight block per DMA, weights stationary, the normed hidden as
the moving operand, 32 k-tiles accumulated in PSUM per 128-wide vocab tile. Sizes come from
nl.tile_size, so nkiequiv can prove them at its small profile and test numerics at full Trainium2 size.

  lm_head_core              the version in the final code: one PSUM tile per vocab tile
  lm_head_core_shared_psum  attempt 17's first version: four accumulation groups in one [P, 4] PSUM tile.
                            On the device it lost one k-contribution in tiles 0..2 of every block
                            (14.5% rel-L2) and was caught by the device head test (kernels/test_head.py --tiled).
  lm_head_core_skip_k       a deliberately wrong variant (one k-tile skipped), as a negative control

Run (from anywhere, symnki on PYTHONPATH):
  python -m nkiequiv check numpy:lm_head_poc.py::ref_lm_head lm_head_poc.py::lm_head_core --spec lm_head_poc.py
"""
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

H1 = 4          # k-tiles of the hidden dim (32 at Qwen3-8B: H = 32 x 128)
NT = 4          # vocab tiles per weight block (LMH_NB = 512 = 4 x 128 in the real kernel)
NBLK = 2        # weight blocks per core in this check (149 at Qwen3-8B)


def ref_lm_head(x, W_tiled):
    """NumPy spec, in the kernel's SBUF output layout.

    x [P, H1]: the normed hidden as rmsnorm_tkg leaves it (element [k, f] is hidden row f*P + k).
    W_tiled [NBLK, P, H1*NT*P]: W_tiled[b, k, f*NB + t*P + p] = W[f*P + k, b*NB + t*P + p], NB = NT*P.
    out [P, NBLK*NT]: out[p, b*NT + t] = logit of vocab column b*NB + t*P + p.
    """
    n_blk, P, blk = W_tiled.shape
    nt = blk // (H1 * P)
    Wr = W_tiled.astype(np.float32).reshape(n_blk, P, H1, nt, P)        # [b, k, f, t, p]
    prod = Wr * x.astype(np.float32).reshape(1, P, H1, 1, 1)
    s = prod.sum(axis=1).sum(axis=1)                                      # [b, t, p]
    return s.transpose(2, 0, 1).reshape(P, n_blk * nt)


def args(ts):
    P = ts.pmax
    return [("x", (P, H1), "bfloat16"), ("W_tiled", (NBLK, P, H1 * NT * P), "bfloat16")]


@nki.jit
def lm_head_core(x, W_tiled):
    P = nl.tile_size.pmax
    n_blk, _, blk = W_tiled.shape
    nb = blk // H1
    nt = nb // P
    out = nl.ndarray((P, n_blk * nt), dtype=nl.float32, buffer=nl.shared_hbm)
    x_sb = nl.ndarray((P, H1), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x)
    logit_sb = nl.ndarray((P, n_blk * nt), dtype=nl.float32, buffer=nl.sbuf)
    for b in range(n_blk):
        w_sb = nl.ndarray((P, blk), dtype=W_tiled.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_sb, src=W_tiled[b])
        for t in range(nt):
            ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
            for f in range(H1):
                nisa.nc_matmul(dst=ps, stationary=w_sb[:, f * nb + t * P:f * nb + (t + 1) * P],
                               moving=x_sb[:, f:f + 1], accumulate=(f > 0))
            nisa.tensor_copy(dst=logit_sb[:, b * nt + t:b * nt + t + 1], src=ps)
    nisa.dma_copy(dst=out, src=logit_sb)
    return out


@nki.jit
def lm_head_core_shared_psum(x, W_tiled):
    P = nl.tile_size.pmax
    n_blk, _, blk = W_tiled.shape
    nb = blk // H1
    nt = nb // P
    out = nl.ndarray((P, n_blk * nt), dtype=nl.float32, buffer=nl.shared_hbm)
    x_sb = nl.ndarray((P, H1), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x)
    logit_sb = nl.ndarray((P, n_blk * nt), dtype=nl.float32, buffer=nl.sbuf)
    for b in range(n_blk):
        w_sb = nl.ndarray((P, blk), dtype=W_tiled.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_sb, src=W_tiled[b])
        ps = nl.ndarray((P, nt), dtype=nl.float32, buffer=nl.psum)
        for t in range(nt):
            for f in range(H1):
                nisa.nc_matmul(dst=ps[:, t:t + 1], stationary=w_sb[:, f * nb + t * P:f * nb + (t + 1) * P],
                               moving=x_sb[:, f:f + 1], accumulate=(f > 0))
        nisa.tensor_copy(dst=logit_sb[:, b * nt:(b + 1) * nt], src=ps)
    nisa.dma_copy(dst=out, src=logit_sb)
    return out


@nki.jit
def lm_head_core_skip_k(x, W_tiled):
    P = nl.tile_size.pmax
    n_blk, _, blk = W_tiled.shape
    nb = blk // H1
    nt = nb // P
    out = nl.ndarray((P, n_blk * nt), dtype=nl.float32, buffer=nl.shared_hbm)
    x_sb = nl.ndarray((P, H1), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x)
    logit_sb = nl.ndarray((P, n_blk * nt), dtype=nl.float32, buffer=nl.sbuf)
    for b in range(n_blk):
        w_sb = nl.ndarray((P, blk), dtype=W_tiled.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_sb, src=W_tiled[b])
        for t in range(nt):
            ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
            for f in range(H1 - 1):                     # BUG: last k-tile never accumulated
                nisa.nc_matmul(dst=ps, stationary=w_sb[:, f * nb + t * P:f * nb + (t + 1) * P],
                               moving=x_sb[:, f:f + 1], accumulate=(f > 0))
            nisa.tensor_copy(dst=logit_sb[:, b * nt + t:b * nt + t + 1], src=ps)
    nisa.dma_copy(dst=out, src=logit_sb)
    return out
