"""One-world fixed-step baseline; no convergence or parallel-world claims yet."""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def contact_steps(a_transposed, bias, initial, alpha, steps):
    contacts, columns = a_transposed.shape
    assert contacts == columns and contacts <= 128
    assert bias.shape == (contacts, 1) and initial.shape == (contacts, 1)
    result = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    matrix = nl.ndarray((contacts, contacts), dtype=nl.float32, buffer=nl.sbuf)
    b = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
    impulses = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=matrix, src=a_transposed)
    nisa.dma_copy(dst=b, src=bias)
    nisa.dma_copy(dst=impulses, src=initial)
    for _ in range(steps):
        product = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=product, stationary=matrix, moving=impulses)
        gradient = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gradient, src=product)
        summed = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=summed, data1=gradient, data2=b, op=nl.add)
        scaled = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=scaled, data=summed, op0=nl.multiply, operand0=alpha)
        updated = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=updated, data1=impulses, data2=scaled, op=nl.subtract)
        nisa.tensor_scalar(dst=impulses, data=updated, op0=nl.maximum, operand0=0.0)
    nisa.dma_copy(dst=result, src=impulses)
    return result
