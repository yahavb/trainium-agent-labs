"""A second full-output baseline using the installed AWS NKI cross entropy."""

import nki
import nki.language as nl
import nki.isa as nisa
from nkilib.experimental.loss.cross_entropy import cross_entropy_forward
from nkilib.core.utils.tiled_range import TiledRange
from kernel import _score


@nki.jit
def aws_separate_score(logits, indices, row_tile=128, vocab_tile=8192):
    loss, lse = cross_entropy_forward(logits, indices, positions_per_batch=row_tile,
                                     chunk_size=min(vocab_tile, logits.shape[1]), dtype=nl.float32)
    entropy = _score(logits, indices, row_tile, vocab_tile, False, True)
    logprobs = nl.ndarray((logits.shape[0],), dtype=nl.float32, buffer=nl.shared_hbm)
    core = nl.program_id(0)
    cores = nl.num_programs(0)
    count = logits.shape[0] // cores
    start = core * count
    if core == cores - 1:
        count = logits.shape[0] - start
    for tile in TiledRange(count, row_tile):
        lo = start + tile.start_offset
        hi = lo + tile.size
        temp = nl.ndarray((tile.size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=temp[:, 0], src=loss[lo:hi])
        nisa.tensor_scalar(dst=temp, data=temp, op0=nl.multiply, operand0=-1.0)
        nisa.dma_copy(dst=logprobs[lo:hi], src=temp[:, 0])
    return logprobs, entropy
