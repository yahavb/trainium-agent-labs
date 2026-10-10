import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def calculate(displacement, velocity, stiffness, damping):
    input_displacement = nl.ndarray((64, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=input_displacement, src=displacement)
    input_velocity = nl.ndarray((64, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=input_velocity, src=velocity)
    input_stiffness = nl.ndarray((64, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=input_stiffness, src=stiffness)
    input_damping = nl.ndarray((64, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=input_damping, src=damping)
    v0 = nl.ndarray((64, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=v0, data=input_displacement, op0=nl.multiply, operand0=input_stiffness, engine=nisa.engine.vector)
    v1 = nl.ndarray((64, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=v1, data=input_velocity, op0=nl.multiply, operand0=input_damping, engine=nisa.engine.vector)
    v2 = nl.ndarray((64, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=v2, data1=v0, data2=v1, op=nl.add, engine=nisa.engine.vector)
    v3 = nl.ndarray((64, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=v3, data=v2, op0=nl.multiply, operand0=-1.0, engine=nisa.engine.vector)
    output = nl.ndarray((64, 8), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=output, src=v3)
    return output
