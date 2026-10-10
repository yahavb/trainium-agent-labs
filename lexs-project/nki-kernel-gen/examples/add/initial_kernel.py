import nki.language as nl
import nki.isa as nisa


# EVOLVE-BLOCK-START
def kernel(a, b):
    rows, cols = a.shape
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    for i in nl.affine_range((rows + 127) // 128):
        for j in nl.affine_range((cols + 511) // 512):
            r, c = i * 128, j * 512
            height, width = min(128, rows - r), min(512, cols - c)
            x = nl.ndarray((height, width), dtype=a.dtype, buffer=nl.sbuf)
            y = nl.ndarray((height, width), dtype=a.dtype, buffer=nl.sbuf)
            z = nl.ndarray((height, width), dtype=a.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x, src=a[r:r + height, c:c + width])
            nisa.dma_copy(dst=y, src=b[r:r + height, c:c + width])
            nisa.tensor_tensor(dst=z, data1=x, data2=y, op=nl.add)
            nisa.dma_copy(dst=out[r:r + height, c:c + width], src=z)
    return out
# EVOLVE-BLOCK-END
