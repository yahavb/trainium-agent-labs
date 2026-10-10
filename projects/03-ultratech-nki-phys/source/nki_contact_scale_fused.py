"""Human-reviewed scaling/update fusion; pending installed-SDK validation."""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def contact_batch(a_transposed, bias, initial, alpha, steps):
    contacts, worlds = bias.shape
    assert contacts <= 128 and worlds <= 16
    assert a_transposed.shape == (contacts, contacts * worlds)
    assert initial.shape == bias.shape and alpha.shape == bias.shape
    result = nl.ndarray((contacts, worlds), dtype=nl.float32, buffer=nl.shared_hbm)
    for world in nl.static_range(worlds):
        matrix = nl.ndarray((contacts, contacts), dtype=nl.float32, buffer=nl.sbuf)
        b = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
        impulses = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
        rate = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
        negative_rate = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=matrix, src=a_transposed[:, world * contacts:(world + 1) * contacts])
        nisa.dma_copy(dst=b, src=bias[:, world:world + 1])
        nisa.dma_copy(dst=impulses, src=initial[:, world:world + 1])
        nisa.dma_copy(dst=rate, src=alpha[:, world:world + 1])
        nisa.tensor_scalar(dst=negative_rate, data=rate, op0=nl.multiply, operand0=-1.0)
        for _ in range(steps):
            product = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=product, stationary=matrix, moving=impulses)
            summed = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=summed, data1=product, data2=b, op=nl.add)
            updated = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=updated, data=summed, op0=nl.multiply,
                               operand0=negative_rate, op1=nl.add, operand1=impulses)
            nisa.tensor_scalar(dst=impulses, data=updated, op0=nl.maximum, operand0=0.0)
        nisa.dma_copy(dst=result[:, world:world + 1], src=impulses)
    return result
