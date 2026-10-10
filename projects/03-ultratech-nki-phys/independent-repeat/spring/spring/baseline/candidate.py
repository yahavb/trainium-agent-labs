import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def calculate(displacement, velocity, stiffness, damping):
    rows, worlds = displacement.shape
    x = nl.ndarray((rows, worlds), dtype=nl.float32, buffer=nl.sbuf)
    v = nl.ndarray((rows, worlds), dtype=nl.float32, buffer=nl.sbuf)
    k = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    c = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=displacement)
    nisa.dma_copy(dst=v, src=velocity)
    nisa.dma_copy(dst=k, src=stiffness)
    nisa.dma_copy(dst=c, src=damping)
    spring = nl.ndarray((rows, worlds), dtype=nl.float32, buffer=nl.sbuf)
    damped = nl.ndarray((rows, worlds), dtype=nl.float32, buffer=nl.sbuf)
    summed = nl.ndarray((rows, worlds), dtype=nl.float32, buffer=nl.sbuf)
    force = nl.ndarray((rows, worlds), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=spring, data=x, op0=nl.multiply, operand0=k)
    nisa.tensor_scalar(dst=damped, data=v, op0=nl.multiply, operand0=c)
    nisa.tensor_tensor(dst=summed, data1=spring, data2=damped, op=nl.add)
    nisa.tensor_scalar(dst=force, data=summed, op0=nl.multiply, operand0=-1.0)
    result = nl.ndarray((rows, worlds), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=result, src=force)
    return result
