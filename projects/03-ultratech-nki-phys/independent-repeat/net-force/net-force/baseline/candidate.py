import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def calculate(forces):
    rows, count = forces.shape
    values = nl.ndarray((rows, count), dtype=nl.float32, buffer=nl.sbuf)
    total = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=values, src=forces)
    nisa.tensor_copy(dst=total, src=values[:, 0:1])
    for i in nl.static_range(1, count):
        nisa.tensor_tensor(dst=total, data1=total, data2=values[:, i:i + 1], op=nl.add)
    result = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=result, src=total)
    return result
