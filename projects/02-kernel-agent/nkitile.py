"""nkitile.py -- a tiny tiling helper the agent may use, offered as a TOOL rather than a hint.

The level-4 wall is "one tile for the whole tensor". A prompt EXAMPLE of chunking was measured to
make every level worse (it pushed the model toward dma_transpose, which it got wrong five ways).
A callable helper is different in kind: it turns tiling into one function call and gets the ragged
final tile right by construction, so the model still has to choose the loops, the layout, the PSUM
accumulation and the copy-out -- it is given a utility, not the answer.

`tiles(n, size)` yields (start, length) pairs covering 0..n, the last one partial when n is not a
multiple of size. It is plain Python over ints, evaluated at trace time, so it composes with
nl.affine_range / range and with slicing.

    for m0, mm in tiles(M, 128):
        for n0, nn in tiles(N, 512):
            acc = nl.ndarray((mm, nn), dtype=nl.float32, buffer=nl.psum)
            for k0, kk in tiles(K, 128):
                a = nl.ndarray((kk, mm), dtype=lhsT.dtype, buffer=nl.sbuf)
                b = nl.ndarray((kk, nn), dtype=rhs.dtype,  buffer=nl.sbuf)
                nisa.dma_copy(dst=a, src=lhsT[k0:k0+kk, m0:m0+mm])
                nisa.dma_copy(dst=b, src=rhs[k0:k0+kk, n0:n0+nn])
                nisa.nc_matmul(dst=acc, stationary=a, moving=b)   # accumulates into acc across k
            out = nl.ndarray((mm, nn), dtype=lhsT.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out, src=acc)
            nisa.dma_copy(dst=result[m0:m0+mm, n0:n0+nn], src=out)

VERIFY in the simulator before trusting it: if the trace-time plain-Python form does not compile
on your SDK, inline the list comprehension inside the kernel body instead.
"""


def tiles(n, size):
    """[(start, length), ...] covering range(n) in chunks of `size`; the last is partial."""
    if size <= 0:
        raise ValueError(f"tile size must be positive, got {size}")
    return [(s, min(size, n - s)) for s in range(0, n, size)]


def ntiles(n, size):
    """How many tiles `tiles(n, size)` yields -- for allocating or reasoning about counts."""
    return (n + size - 1) // size if size > 0 else 0


if __name__ == "__main__":
    # Self-check the ragged behaviour that is the whole point.
    assert tiles(256, 128) == [(0, 128), (128, 128)]
    assert tiles(300, 128) == [(0, 128), (128, 128), (256, 44)]
    assert tiles(100, 128) == [(0, 100)]
    assert sum(l for _, l in tiles(700, 512)) == 700
    assert ntiles(300, 128) == 3 and ntiles(100, 128) == 1
    print("nkitile ok:", tiles(300, 128))
