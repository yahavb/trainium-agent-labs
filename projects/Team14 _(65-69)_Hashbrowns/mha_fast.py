"""
Multi-head attention, fast version: the V6 algorithm, restructured to use the whole NeuronCore pair.
nkibench level 9.

    out[:, h, :] = softmax(q[:, h, :] @ k[:, h, :].T / sqrt(d)) @ v[:, h, :]    q, k, v: [S, H, d]

MEASURED on seat-68 (bench_suites.py mha, then named bench_device.py; 2,000 timed iterations, four runs): correct on all four
level-9 shapes on the device and in the simulator, and 2.187x-2.193x faster than V0 run once per head
(mha_v0_loop.py) at LNC 2, 1.491x-1.493x at LNC 1. Run 1, per shape, two cores:
    seq 128, 16 heads, dim 64     74.50 -> 33.72 us   2.21x
    seq 128, 16 heads, dim 128    76.08 -> 34.40 us   2.21x
    seq  96,  8 heads, dim 32     38.38 -> 21.61 us   1.78x
    seq 128, 32 heads, dim 128   125.32 -> 47.19 us   2.66x
The profile (32 heads, dim 128) shows where it came from: per-head DMAs made GpSimd the bottleneck in
V0's loop (86 us busy building descriptors in software) and Sync in V4's (82 us issuing hardware
descriptors at ~0.6 us each); chunked loads cut the hardware DMA packets 18,432 -> 4,608 and Sync to
15-22 us. Two cores give 1.47x over one, not 2x: about 10 us per call is fixed launch cost that does not
split. Tensor and Scalar now lead at ~34 us each.

Compared with running V6 once per head (mha_v6_loop.py), it changes how the work is laid out, not the
arithmetic. Each change follows a pattern from nkilib's production attention kernel
(nkilib/core/attention/attention_cte.py):

  1. Both physical cores. Launched at LNC 2, the kernel runs as two programs; each takes a contiguous
     half of the heads, using nl.program_ndim / num_programs / program_id the way nkilib's
     get_program_sharding_info does. At LNC 1 the same code runs every head on one core.
  2. SBUF chunks sized for DMA. Q, K, V and the output move one chunk of heads per DMA, not one head:
     in [S, H, d] a token's row holds every head contiguously, so a chunk of hc heads is hc*d
     contiguous elements per partition. hc is the smallest number of heads that reaches 2 KB per
     partition (where DMA throughput approaches its plateau), capped so a chunk stays well inside
     SBUF, which leaves several chunks per core: the next chunk's loads overlap this chunk's compute.
  3. Fewer instructions. Per group of G heads, whose G score tiles share one 2 KB PSUM bank: one
     tensor_reduce for all G row maxima (a (S, G, S) tile reduced on its last axis) and one reciprocal
     for all G row sums, instead of G of each. Every head's output lands in one SBUF tile per chunk,
     so there is one store per chunk.
  4. Overlap between heads. Every per-head tile is allocated inside the head loop, so the compiler
     can rotate buffers (as nkilib does with num_free_tiles=2) and run one head's transposes and
     matmuls on the Tensor engine while the previous head's exp is on Scalar.
  Kept from V6: hardware DMA descriptors with Q and K triggered from different engines, the softmax
  scale folded into the Q^T copy out of PSUM, -max from the reduce itself, and the row sum from the
  exp instruction. NOT kept: bf16 for P @ V. At the 16-head seq-128 shapes it misses nkibench's
  tolerance -- worst error 0.0245 of the output RMS against 0.02, measured identically in
  mha_v6_loop.py, so it is V6's precision, not this layout. P @ V here is float32, as in V5.

ENTRY POINTS. nki_mha_ is the kernel above. The other three are the experiments tried on top of it,
kept as switches on the same body instead of near-copies of this file:
  nki_mha_balanced_   the scaled Q^T copy out of PSUM on Vector instead of Scalar, so Scalar runs only
                      the exp. NO GAIN: 2.198x over the V0 loop against nki_mha_'s 2.193x in the same
                      run, within 1% on every shape. Tried first on GpSimd, which hardware DMA
                      descriptors leave idle: rejected, "tensor_scalar data must be in ['sbuf'], got
                      psum" -- GpSimd cannot read PSUM, and every per-head step reads a PSUM result.
  nki_mha_tf32_pv_    P @ V in TF32 (P^T leaves PSUM as TF32, V converted once per chunk on GpSimd)
  nki_mha_tf32_all_   Q @ K.T in TF32 as well (Q^T and K^T leave PSUM as TF32)
                      Both TF32 entries pass nkibench's CPU simulator 4/4 but DO NOT COMPILE for the
                      device with nki 0.6.0 / neuronx-cc 2.27: "'nisa.matmul' op 'stationary' must be
                      a float type, got '!nisa.float32r'". TF32 would run the Tensor engine at 79
                      TFLOPS against FP32's 20.

    python nkibench.py --level 9 --check mha_fast.py      # checks nki_mha_
    python bench_suites.py mha                            # nki_mha_ at LNC 1 and 2, and nki_mha_balanced_
"""

import nki
import nki.isa as nisa
import nki.language as nl

HW = nisa.dge_mode.hwdge
DMA_MIN_BYTES = 2048          # per partition: DMA throughput is near its plateau from 2 KB
CHUNK_MAX_BYTES = 16384       # per partition per tensor; Q, K, V and out stay under 64 KB
PSUM_BANK_BYTES = 2048        # per partition


def _divisor(n, lo, hi):
    """The smallest divisor of n that is >= lo, if one is <= hi; else the largest divisor <= hi."""
    smallest = 0
    largest = 1
    for c in range(1, n + 1):
        if n % c == 0:
            if c <= hi:
                largest = c
            if c >= lo and c <= hi and smallest == 0:
                smallest = c
    if smallest == 0:
        return largest
    return smallest


def _mha(q, k, v, qT_on_vector=False, pv_tf32=False, qk_tf32=False):
    S, H, d = q.shape
    assert S <= 128 and d <= 128, f"single-tile heads: needs seq <= 128 and dim <= 128, got {S}, {d}"
    scale = 1.0 / (d ** 0.5)
    qT_dtype = nl.tfloat32 if qk_tf32 else q.dtype
    kT_dtype = nl.tfloat32 if qk_tf32 else k.dtype
    pT_dtype = nl.tfloat32 if pv_tf32 else nl.float32

    out = nl.ndarray((S, H, d), dtype=q.dtype, buffer=nl.shared_hbm)

    # 1. Split the heads across the programs of an LNC 2 launch; (1, 0) when not launched on a grid.
    n_prg = 1
    prg = 0
    if nl.program_ndim() != 0:
        n_prg = nl.num_programs(axes=0)
        prg = nl.program_id(axis=0)
    assert H % n_prg == 0, f"heads ({H}) must split evenly across {n_prg} cores"
    heads_per_core = H // n_prg
    head0 = prg * heads_per_core

    # 2. Chunk: fewest heads per DMA that reach DMA_MIN_BYTES per partition, within CHUNK_MAX_BYTES.
    hc = _divisor(heads_per_core, (DMA_MIN_BYTES + d * 4 - 1) // (d * 4), CHUNK_MAX_BYTES // (d * 4))
    # 3. Group: heads whose scores (S floats) and outputs (d floats) each fit one PSUM bank together.
    G = _divisor(hc, hc, min(PSUM_BANK_BYTES // (S * 4), PSUM_BANK_BYTES // (d * 4)))
    W = hc * d

    for c in nl.affine_range(heads_per_core // hc):
        off = (head0 + c * hc) * d
        q_sb = nl.ndarray((S, W), dtype=q.dtype, buffer=nl.sbuf)
        k_sb = nl.ndarray((S, W), dtype=k.dtype, buffer=nl.sbuf)
        v_sb = nl.ndarray((S, W), dtype=v.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_sb, src=q.ap(pattern=[[H * d, S], [1, W]], offset=off),
                      dge_mode=HW, engine=nisa.engine.sync)
        nisa.dma_copy(dst=k_sb, src=k.ap(pattern=[[H * d, S], [1, W]], offset=off),
                      dge_mode=HW, engine=nisa.engine.scalar)
        nisa.dma_copy(dst=v_sb, src=v.ap(pattern=[[H * d, S], [1, W]], offset=off),
                      dge_mode=HW, engine=nisa.engine.sync)

        v_mm = v_sb
        if pv_tf32:
            # V to TF32 once per chunk, on GpSimd, which hardware DMA descriptors leave idle.
            v_mm = nl.ndarray((S, W), dtype=nl.tfloat32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=v_mm, src=v_sb, engine=nisa.engine.gpsimd)

        o_sb = nl.ndarray((S, W), dtype=out.dtype, buffer=nl.sbuf)
        row_sum = nl.ndarray((S, hc), dtype=nl.float32, buffer=nl.sbuf)
        inv_sum = nl.ndarray((S, hc), dtype=nl.float32, buffer=nl.sbuf)

        for g in nl.affine_range(hc // G):
            s_grp = nl.ndarray((S, G, S), dtype=nl.float32, buffer=nl.psum)
            for j in nl.affine_range(G):
                h = g * G + j
                # Q^T and K^T share one PSUM tile: d partitions, 2*S floats.
                qk_ps = nl.ndarray((d, 2 * S), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=qk_ps[0:d, 0:S], data=q_sb[0:S, h * d:(h + 1) * d])
                nisa.nc_transpose(dst=qk_ps[0:d, S:2 * S], data=k_sb[0:S, h * d:(h + 1) * d])
                qT = nl.ndarray((d, S), dtype=qT_dtype, buffer=nl.sbuf)
                kT = nl.ndarray((d, S), dtype=kT_dtype, buffer=nl.sbuf)
                if qT_on_vector:
                    nisa.tensor_scalar(dst=qT, data=qk_ps[0:d, 0:S], op0=nl.multiply, operand0=scale,
                                       engine=nisa.engine.vector)
                else:
                    nisa.activation(dst=qT, op=nl.copy, data=qk_ps[0:d, 0:S], scale=scale)
                nisa.tensor_copy(dst=kT, src=qk_ps[0:d, S:2 * S], engine=nisa.engine.vector)
                nisa.nc_matmul(dst=s_grp[0:S, j, 0:S], stationary=qT, moving=kT)

            # One reduce gives -max for all G heads: (S, G, S) -> (S, G).
            neg_max = nl.ndarray((S, G), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=s_grp, axis=(2,), negate=True)

            o_grp = nl.ndarray((S, G, d), dtype=nl.float32, buffer=nl.psum)
            for j in nl.affine_range(G):
                h = g * G + j
                p = nl.ndarray((S, S), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=p, op=nl.exp, data=s_grp[0:S, j, 0:S], bias=neg_max[0:S, j:j + 1],
                                reduce_op=nl.add, reduce_res=row_sum[0:S, h:h + 1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
                pT_ps = nl.ndarray((S, S), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=pT_ps, data=p)
                pT = nl.ndarray((S, S), dtype=pT_dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pT, src=pT_ps, engine=nisa.engine.vector)
                nisa.nc_matmul(dst=o_grp[0:S, j, 0:d], stationary=pT, moving=v_mm[0:S, h * d:(h + 1) * d])

            # One reciprocal for all G row sums, then normalise each head into the chunk's output tile.
            nisa.reciprocal(dst=inv_sum[0:S, g * G:(g + 1) * G], data=row_sum[0:S, g * G:(g + 1) * G])
            for j in nl.affine_range(G):
                h = g * G + j
                nisa.tensor_scalar(dst=o_sb[0:S, h * d:(h + 1) * d], data=o_grp[0:S, j, 0:d],
                                   op0=nl.multiply, operand0=inv_sum[0:S, h:h + 1])

        nisa.dma_copy(dst=out.ap(pattern=[[H * d, S], [1, W]], offset=off), src=o_sb,
                      dge_mode=HW, engine=nisa.engine.sync)
    return out


@nki.jit
def nki_mha_(q, k, v):
    return _mha(q, k, v)


@nki.jit
def nki_mha_balanced_(q, k, v):
    return _mha(q, k, v, qT_on_vector=True)


@nki.jit
def nki_mha_tf32_pv_(q, k, v):
    return _mha(q, k, v, pv_tf32=True)


@nki.jit
def nki_mha_tf32_all_(q, k, v):
    return _mha(q, k, v, pv_tf32=True, qk_tf32=True)
