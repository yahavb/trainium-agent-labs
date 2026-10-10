"""Forward token scoring in NKI: stable FP32 streaming state, BF16 input.

Fusion extends the online log-sum-exp idea used in AWS nkilib cross entropy
with a shifted weighted sum for entropy. See NOTICE.md for prior work.
"""

import nki
import nki.language as nl
import nki.isa as nisa
from nkilib.core.utils.allocator import create_auto_alloc_manager
from nkilib.core.utils.kernel_helpers import div_ceil
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.tiled_range import TiledRange


def _score(logits, indices, row_tile, vocab_tile, need_logprob, need_entropy):
    rows, vocab = logits.shape
    kernel_assert(len(indices.shape) == 1 and indices.shape[0] == rows, "indices shape must be [T]")
    kernel_assert(row_tile > 0 and row_tile <= nl.tile_size.pmax, "invalid row tile")
    kernel_assert(vocab_tile > 0 and vocab_tile <= 16384, "vocab tile must be <= 16384")
    logprobs = nl.ndarray((rows,), dtype=nl.float32, buffer=nl.shared_hbm)
    entropy = nl.ndarray((rows,), dtype=nl.float32, buffer=nl.shared_hbm)
    core = nl.program_id(0)
    cores = nl.num_programs(0)
    shard_size = rows // cores
    shard_start = core * shard_size
    if core == cores - 1:
        shard_size = rows - shard_start
    sbm = create_auto_alloc_manager()
    for rt in TiledRange(shard_size, row_tile):
        start = shard_start + rt.start_offset
        end = start + rt.size
        r = rt.size
        sbm.open_scope()
        m = sbm.alloc((r, 1), nl.float32)
        s = sbm.alloc((r, 1), nl.float32)
        u = sbm.alloc((r, 1), nl.float32)
        new_m = sbm.alloc((r, 1), nl.float32)
        delta = sbm.alloc((r, 1), nl.float32)
        scale = sbm.alloc((r, 1), nl.float32)
        tile_sum = sbm.alloc((r, 1), nl.float32)
        tile_weighted = sbm.alloc((r, 1), nl.float32)
        selected = sbm.alloc((r, 1), nl.float32)
        # Scalar operands use FP32 in the installed ISA. Integers through 2**24
        # are represented exactly; the host contract checks the vocabulary bound.
        target = sbm.alloc((r, 1), nl.float32)
        tile_selected = sbm.alloc((r, 1), nl.float32)
        relative = sbm.alloc((r, 1), nl.float32)
        valid_low = sbm.alloc((r, 1), nl.float32)
        valid_high = sbm.alloc((r, 1), nl.float32)
        gather_index = sbm.alloc((r, 1), nl.uint32)
        if need_logprob:
            nisa.dma_copy(dst=target[:, 0], src=indices[start:end])
            nisa.memset(dst=selected, value=0.0)
        for vt in TiledRange(vocab, vocab_tile):
            sbm.open_scope()
            x = sbm.alloc((r, vt.size), nl.float32)
            e = sbm.alloc((r, vt.size), nl.float32)
            if need_entropy:
                tmp = sbm.alloc((r, vt.size), nl.float32)
            nisa.dma_copy(dst=x, src=logits[start:end, vt.start_offset:vt.end_offset])
            if need_logprob:
                # Gather just one value per partition. Clamp the index BEFORE
                # gathering, then zero it if the target is outside this tile.
                nisa.tensor_scalar(dst=relative, data=target, op0=nl.subtract, operand0=vt.start_offset)
                nisa.tensor_scalar(dst=valid_low, data=relative, op0=nl.greater_equal, operand0=0.0)
                nisa.tensor_scalar(dst=valid_high, data=relative, op0=nl.less, operand0=vt.size)
                nisa.tensor_tensor(dst=valid_low, data1=valid_low, data2=valid_high, op=nl.multiply)
                nisa.tensor_scalar(dst=relative, data=relative, op0=nl.maximum, operand0=0.0,
                                   op1=nl.minimum, operand1=vt.size - 1)
                nisa.tensor_copy(dst=gather_index, src=relative)
                nisa.nc_n_gather(dst=tile_selected, data=x, indices=gather_index)
                nisa.tensor_tensor(dst=tile_selected, data1=tile_selected, data2=valid_low, op=nl.multiply)
                nisa.tensor_tensor(dst=selected, data1=selected, data2=tile_selected, op=nl.add)
            nisa.tensor_reduce(dst=new_m, data=x, op=nl.maximum, axis=1)
            if vt.index > 0:
                nisa.tensor_tensor(dst=new_m, data1=m, data2=new_m, op=nl.maximum)
                nisa.tensor_tensor(dst=delta, data1=m, data2=new_m, op=nl.subtract)
                nisa.activation(dst=scale, data=delta, op=nl.exp)
                if need_entropy:
                    # u' = exp(delta) * (u + delta*s), using the OLD s.
                    nisa.tensor_tensor(dst=tile_weighted, data1=delta, data2=s, op=nl.multiply)
                    nisa.tensor_tensor(dst=u, data1=u, data2=tile_weighted, op=nl.add)
                    nisa.tensor_tensor(dst=u, data1=u, data2=scale, op=nl.multiply)
                nisa.tensor_tensor(dst=s, data1=s, data2=scale, op=nl.multiply)
            # Keep x read-only so the gather and normalization can overlap.
            # ScalarE applies the shift and sums exp while producing e.
            nisa.tensor_scalar(dst=delta, data=new_m, op0=nl.multiply, operand0=-1.0)
            nisa.activation_reduce(dst=e, data=x, op=nl.exp, bias=delta,
                                   reduce_op=nl.add, reduce_res=tile_sum)
            if need_entropy:
                nisa.tensor_scalar(dst=tmp, data=x, op0=nl.subtract, operand0=new_m)
                nisa.tensor_tensor(dst=tmp, data1=e, data2=tmp, op=nl.multiply)
                nisa.tensor_reduce(dst=tile_weighted, data=tmp, op=nl.add, axis=1)
            if vt.index == 0:
                nisa.tensor_copy(dst=s, src=tile_sum)
                if need_entropy:
                    nisa.tensor_copy(dst=u, src=tile_weighted)
            else:
                nisa.tensor_tensor(dst=s, data1=s, data2=tile_sum, op=nl.add)
                if need_entropy:
                    nisa.tensor_tensor(dst=u, data1=u, data2=tile_weighted, op=nl.add)
            nisa.tensor_copy(dst=m, src=new_m)
            sbm.close_scope()
        nisa.activation(dst=tile_sum, data=s, op=nl.log)
        if need_logprob:
            nisa.tensor_tensor(dst=selected, data1=selected, data2=m, op=nl.subtract)
            nisa.tensor_tensor(dst=selected, data1=selected, data2=tile_sum, op=nl.subtract)
            nisa.dma_copy(dst=logprobs[start:end], src=selected[:, 0])
        if need_entropy:
            nisa.reciprocal(dst=scale, data=s)
            nisa.tensor_tensor(dst=u, data1=u, data2=scale, op=nl.multiply)
            nisa.tensor_tensor(dst=u, data1=tile_sum, data2=u, op=nl.subtract)
            nisa.dma_copy(dst=entropy[start:end], src=u[:, 0])
        sbm.close_scope()
    if need_logprob and need_entropy:
        return logprobs, entropy
    if need_logprob:
        return logprobs
    return entropy


@nki.jit
def fused_score(logits, indices, row_tile=128, vocab_tile=8192):
    return _score(logits, indices, row_tile, vocab_tile, True, True)


@nki.jit
def logprob_only(logits, indices, row_tile=128, vocab_tile=8192):
    return _score(logits, indices, row_tile, vocab_tile, True, False)


@nki.jit
def entropy_only(logits, indices, row_tile=128, vocab_tile=8192):
    return _score(logits, indices, row_tile, vocab_tile, False, True)


@nki.jit
def separate_score(logits, indices, row_tile=128, vocab_tile=8192):
    # One dispatch for a strong baseline: normalization/exp and logits loads
    # happen independently for the two outputs, with identical FP32 arithmetic.
    lp = _score(logits, indices, row_tile, vocab_tile, True, False)
    h = _score(logits, indices, row_tile, vocab_tile, False, True)
    return lp, h


@nki.jit
def smoke_kernel(values):
    output = nl.ndarray(values.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    x = nl.ndarray(values.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=values)
    nisa.tensor_scalar(dst=x, data=x, op0=nl.add, operand0=1.0)
    nisa.dma_copy(dst=output, src=x)
    return output
