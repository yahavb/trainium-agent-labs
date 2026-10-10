import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def calculate(forces):
    input_forces = nl.ndarray((64, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=input_forces, src=forces)
    v0 = nl.ndarray((64, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=v0, data=input_forces, op=nl.add, axis=(1,), keepdims=True)
    output = nl.ndarray((64, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=output, src=v0)
    return output
