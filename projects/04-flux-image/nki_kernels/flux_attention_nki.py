"""Fused QK-RMSNorm + RoPE + flash attention for FLUX joint / single-stream attention (NKI 0.6).

One kernel call replaces this NxDI glue (see NeuronFluxAttention.forward in
neuronx_distributed_inference/models/diffusers/flux/modeling_flux.py):

    q/k/v.view(B,S,H,D).transpose(1,2)          # XLA transposes
    norm_q / norm_k / norm_added_q / norm_added_k  # CustomRMSNorm, fp32 round trip
    torch.cat([txt, img], dim=2)                # joint blocks
    apply_rotary_emb(q), apply_rotary_emb(k)    # fp32 elementwise + stack/flatten
    attention_cte(q, k, v)                      # nkilib flash kernel
    out.transpose(1,2).reshape(B,S,H*D)         # XLA transpose

Inputs are taken directly in the projection layout [B, S, H*D] (token-major, heads packed in
the free dim), and the output is written in the same layout, so no XLA transposes remain.

Math (per batch b, head h, token s):
    qn = rmsnorm(q[s], w_q)      (w = w_txt for s < n_txt, else w_img; fp32, eps)
    qr = rope_interleaved(qn, cos[s], sin[s])
    same for k
    out = softmax(qr @ kr^T * D^-0.5) @ v       (online / flash softmax, no S x S matrix)

Layout trick: q.k is invariant to any permutation of the head dim applied identically to q and
k, so RoPE (interleaved pairs, FLUX/diffusers `use_real_unbind_dim=-1`) is applied with stride-2
free-dim views and the result is stored de-interleaved ([even | odd]). That avoids any relayout.
V and the output keep the natural order.

Tiling (per (b, h)):
    Q^T, K^T  : [D=128 partitions, S] bf16 in SBUF (built per 128-token tile via TensorE transpose)
    V         : [128 tokens, S/128, D] bf16 in SBUF
    per 128-query tile, per 512-key chunk:
        S_chunk = Q^T_tile^T @ K^T_chunk      -> PSUM [128, 512] fp32 (one bank)
        online max / exp (ScalarE, fused row-sum) / rescale
        P^T via 4 TensorE transposes, O += P @ V (PSUM accumulate over 4 key sub-tiles)
    SBUF per partition ~ 3 x S x 2B (27 KB at S=4608) + small working tiles; no spills by design.

Constraints: head_dim == 128, S % 128 == 0, n_txt % 128 == 0, bf16 q/k/v.
cos/sin: [S, D//2] fp32 = FluxPosEmbed cos/sin with the repeat-interleave removed ([:, 0::2]).
Norm weights: [1, D] fp32.
"""

import nki
import nki.isa as nisa
import nki.language as nl

_P = 128          # partition tile (tokens per q tile / k sub-tile)
_K_CHUNK = 512    # keys per score chunk = one fp32 PSUM bank / gemm moving fmax


def _kassert(cond, msg):
    # NKI's device frontend rejects `raise` inside kernel code; `assert` is the supported fatal check.
    assert cond, f"[flux_attention_nki] {msg}"


def _load_bcast_row(w_hbm, D):
    """Load a [1, D] fp32 HBM row and broadcast it over all 128 partitions -> [128, D] SBUF."""
    w_sb = nl.ndarray((_P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_sb, src=w_hbm.ap(pattern=[[0, _P], [1, D]], offset=0))
    return w_sb


def _norm_rope_tile(x_tile, w_sb, cos_t, sin_t, dst_tile, D, eps, apply_norm, apply_rope):
    """x_tile [128 tok, D] bf16 -> dst_tile [128 tok, D] bf16 (RMSNorm*w, then RoPE, de-interleaved)."""
    Dh = D // 2
    xf = nl.ndarray((_P, D), dtype=nl.float32, buffer=nl.sbuf)
    if apply_norm:
        sq = nl.ndarray((_P, D), dtype=nl.float32, buffer=nl.sbuf)
        ss = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=sq, op=nl.square, data=x_tile, reduce_op=nl.add, reduce_res=ss)
        rstd = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        # rstd = 1 / sqrt(sum(x^2) / D + eps)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=ss, scale=1.0 / D, bias=eps)
        # xf = (x * rstd) * w
        nisa.scalar_tensor_tensor(dst=xf, data=x_tile, op0=nl.multiply, operand0=rstd,
                                  op1=nl.multiply, operand1=w_sb)
    else:
        nisa.tensor_copy(dst=xf, src=x_tile)

    if apply_rope:
        x_even = xf.slice(1, 0, D, 2)   # [128, Dh] stride-2 view
        x_odd = xf.slice(1, 1, D, 2)
        t_a = nl.ndarray((_P, Dh), dtype=nl.float32, buffer=nl.sbuf)
        t_b = nl.ndarray((_P, Dh), dtype=nl.float32, buffer=nl.sbuf)
        # even' = x_e*cos - x_o*sin
        nisa.tensor_tensor(dst=t_a, data1=x_even, data2=cos_t, op=nl.multiply)
        nisa.tensor_tensor(dst=t_b, data1=x_odd, data2=sin_t, op=nl.multiply)
        nisa.tensor_tensor(dst=dst_tile[:, 0:Dh], data1=t_a, data2=t_b, op=nl.subtract)
        # odd' = x_o*cos + x_e*sin
        t_c = nl.ndarray((_P, Dh), dtype=nl.float32, buffer=nl.sbuf)
        t_d = nl.ndarray((_P, Dh), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t_c, data1=x_odd, data2=cos_t, op=nl.multiply)
        nisa.tensor_tensor(dst=t_d, data1=x_even, data2=sin_t, op=nl.multiply)
        nisa.tensor_tensor(dst=dst_tile[:, Dh:D], data1=t_c, data2=t_d, op=nl.add)
    else:
        nisa.tensor_copy(dst=dst_tile, src=xf)


def _build_transposed(x_hbm, b, h, S, HD, D, n_txt, w_img, w_txt, cos_sb, sin_sb, eps,
                      apply_norm, apply_rope, xT):
    """Load head h of x_hbm[b] ([S, H*D]), norm+rope per 128-token tile, write transpose into xT [D, S]."""
    n_tiles = S // _P
    x_sb = nl.ndarray((_P, n_tiles, D), dtype=x_hbm.dtype, buffer=nl.sbuf)
    # token s = t*128 + p ; row stride HD ; 256 B contiguous per (p, t)
    nisa.dma_copy(dst=x_sb, src=x_hbm.ap(pattern=[[HD, _P], [_P * HD, n_tiles], [1, D]],
                                         offset=b * S * HD + h * D))
    for t in range(n_tiles):
        w_sb = w_txt if (t * _P) < n_txt else w_img
        y = nl.ndarray((_P, D), dtype=x_hbm.dtype, buffer=nl.sbuf)
        _norm_rope_tile(x_sb[:, t, :], w_sb, cos_sb[:, t, :], sin_sb[:, t, :], y, D, eps,
                        apply_norm, apply_rope)
        yT_ps = nl.ndarray((D, _P), dtype=x_hbm.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=yT_ps, data=y)
        nisa.tensor_copy(dst=xT[:, t * _P:(t + 1) * _P], src=yT_ps)


@nki.jit
def flux_qknorm_rope_attention(q, k, v, cos, sin, q_w_img, k_w_img, q_w_txt, k_w_txt,
                               n_txt=0, eps=1e-6, apply_norm=True, apply_rope=True):
    """Fused QK-RMSNorm + RoPE + non-causal flash attention.

    Args:
        q, k, v: [B, S, H*D] bf16 @ HBM (projection outputs, per-core heads).
        cos, sin: [S, D//2] fp32 @ HBM (de-duplicated FLUX rope tables).
        q_w_img, k_w_img: [1, D] fp32 RMSNorm weights for tokens >= n_txt (norm_q / norm_k).
        q_w_txt, k_w_txt: [1, D] fp32 RMSNorm weights for tokens < n_txt (norm_added_q / _k).
            For single-stream blocks pass the img weights again and n_txt=0.
        n_txt: number of leading text tokens (multiple of 128).
        eps: RMSNorm epsilon (FLUX: 1e-6).
        apply_norm / apply_rope: compile-time switches (for staged validation).

    Returns:
        out: [B, S, H*D] bf16 @ HBM, softmax(q k^T / sqrt(D)) v per head.
    """
    B, S, HD = q.shape
    D = 2 * cos.shape[1]
    H = HD // D
    Dh = D // 2
    _kassert(D == _P, f"head_dim must be 128, got {D}")
    _kassert(H * D == HD, f"H*D mismatch: {HD} vs D={D}")
    _kassert(S % _P == 0, f"S must be a multiple of 128, got {S}")
    _kassert(n_txt % _P == 0 and n_txt <= S, f"n_txt must be a multiple of 128 and <= S, got {n_txt}")
    _kassert(tuple(k.shape) == (B, S, HD) and tuple(v.shape) == (B, S, HD), "q/k/v shape mismatch")
    _kassert(tuple(cos.shape) == (S, Dh) and tuple(sin.shape) == (S, Dh), "cos/sin must be [S, D//2]")

    n_tiles = S // _P
    n_chunks = (S + _K_CHUNK - 1) // _K_CHUNK
    scale = float(D) ** -0.5

    out = nl.ndarray((B, S, HD), dtype=q.dtype, buffer=nl.shared_hbm)

    # ---- constants: norm weights (partition-broadcast) and rope tables, loaded once ----
    qw_img = _load_bcast_row(q_w_img, D)
    kw_img = _load_bcast_row(k_w_img, D)
    qw_txt = _load_bcast_row(q_w_txt, D)
    kw_txt = _load_bcast_row(k_w_txt, D)
    cos_sb = nl.ndarray((_P, n_tiles, Dh), dtype=nl.float32, buffer=nl.sbuf)
    sin_sb = nl.ndarray((_P, n_tiles, Dh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos_sb, src=cos.ap(pattern=[[Dh, _P], [_P * Dh, n_tiles], [1, Dh]], offset=0))
    nisa.dma_copy(dst=sin_sb, src=sin.ap(pattern=[[Dh, _P], [_P * Dh, n_tiles], [1, Dh]], offset=0))

    # LNC sharding: launched as kernel[2] on an LNC=2 logical core, each physical core takes H/2 heads
    # (FLUX TP=4: 6 heads/rank -> 3 per physical core). Without a grid, one program does all heads.
    n_prgs, prg_id = 1, 0
    if nl.program_ndim() != 0:
        n_prgs = nl.num_programs(axes=0)
        prg_id = nl.program_id(axis=0)
    _kassert(H % n_prgs == 0, f"heads per rank ({H}) must be divisible by the LNC grid ({n_prgs})")
    H_local = H // n_prgs

    for b in range(B):
        for h_local in range(H_local):
            h = prg_id * H_local + h_local
            # ---- prologue: Q^T, K^T (normed, roped, de-interleaved) and V, all SBUF-resident ----
            qT = nl.ndarray((D, S), dtype=q.dtype, buffer=nl.sbuf)
            kT = nl.ndarray((D, S), dtype=k.dtype, buffer=nl.sbuf)
            _build_transposed(q, b, h, S, HD, D, n_txt, qw_img, qw_txt, cos_sb, sin_sb, eps,
                              apply_norm, apply_rope, qT)
            _build_transposed(k, b, h, S, HD, D, n_txt, kw_img, kw_txt, cos_sb, sin_sb, eps,
                              apply_norm, apply_rope, kT)
            v_sb = nl.ndarray((_P, n_tiles, D), dtype=v.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_sb, src=v.ap(pattern=[[HD, _P], [_P * HD, n_tiles], [1, D]],
                                             offset=b * S * HD + h * D))

            # ---- flash attention over 128-query tiles ----
            for qi in range(n_tiles):
                neg_m = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)   # -running max (unscaled)
                l_run = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)   # running row sum
                acc = nl.ndarray((_P, D), dtype=nl.float32, buffer=nl.sbuf)     # running P@V

                for c in range(n_chunks):
                    k0 = c * _K_CHUNK
                    kc = min(_K_CHUNK, S - k0)
                    n_sub = kc // _P

                    s_ps = nl.ndarray((_P, kc), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=s_ps, stationary=qT[:, qi * _P:(qi + 1) * _P],
                                   moving=kT[:, k0:k0 + kc])

                    # chunk max (negated)
                    neg_cmax = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_reduce(dst=neg_cmax, op=nl.maximum, data=s_ps, axis=1, negate=True)

                    alpha = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    if c == 0:
                        nisa.tensor_copy(dst=neg_m, src=neg_cmax)
                    else:
                        neg_m_new = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_tensor(dst=neg_m_new, data1=neg_m, data2=neg_cmax, op=nl.minimum)
                        bias_new = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_scalar(dst=bias_new, data=neg_m_new, op0=nl.multiply, operand0=scale)
                        # alpha = exp(scale*(m_old - m_new)) = exp(-scale*neg_m_old + scale*neg_m_new)
                        nisa.activation(dst=alpha, op=nl.exp, data=neg_m, scale=-scale, bias=bias_new)
                        nisa.tensor_copy(dst=neg_m, src=neg_m_new)

                    bias = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=bias, data=neg_m, op0=nl.multiply, operand0=scale)

                    # p = exp(scale*s - scale*m), row sum fused on ScalarE
                    p_sb = nl.ndarray((_P, kc), dtype=q.dtype, buffer=nl.sbuf)
                    csum = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation_reduce(dst=p_sb, op=nl.exp, data=s_ps, reduce_op=nl.add,
                                           reduce_res=csum, bias=bias, scale=scale)

                    # P^T sub-tiles [keys, queries]
                    pT_ps = nl.ndarray((_P, n_sub, _P), dtype=q.dtype, buffer=nl.psum)
                    for j in range(n_sub):
                        nisa.nc_transpose(dst=pT_ps[:, j, :], data=p_sb[:, j * _P:(j + 1) * _P])
                    pT_sb = nl.ndarray((_P, n_sub, _P), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=pT_sb, src=pT_ps)

                    # O_chunk = P @ V  [queries, D]
                    pv_ps = nl.ndarray((_P, D), dtype=nl.float32, buffer=nl.psum)
                    for j in range(n_sub):
                        nisa.nc_matmul(dst=pv_ps, stationary=pT_sb[:, j, :],
                                       moving=v_sb[:, k0 // _P + j, :], accumulate=(j > 0))

                    if c == 0:
                        nisa.tensor_copy(dst=l_run, src=csum)
                        nisa.tensor_copy(dst=acc, src=pv_ps)
                    else:
                        nisa.scalar_tensor_tensor(dst=l_run, data=l_run, op0=nl.multiply, operand0=alpha,
                                                  op1=nl.add, operand1=csum)
                        nisa.scalar_tensor_tensor(dst=acc, data=acc, op0=nl.multiply, operand0=alpha,
                                                  op1=nl.add, operand1=pv_ps)

                # normalize and store [128 queries, D] into out[b, q-tile, h*D:(h+1)*D]
                rl = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.reciprocal(dst=rl, data=l_run)
                o_sb = nl.ndarray((_P, D), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=o_sb, data=acc, op0=nl.multiply, operand0=rl)
                nisa.dma_copy(dst=out[b, qi * _P:(qi + 1) * _P, h * D:(h + 1) * D], src=o_sb)

    return out
