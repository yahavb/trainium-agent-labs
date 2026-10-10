"""Independent worlds in one launch; serial-world baseline, not multi-core."""

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
        nisa.dma_copy(dst=matrix, src=a_transposed[:, world * contacts:(world + 1) * contacts])
        nisa.dma_copy(dst=b, src=bias[:, world:world + 1])
        nisa.dma_copy(dst=impulses, src=initial[:, world:world + 1])
        nisa.dma_copy(dst=rate, src=alpha[:, world:world + 1])
        for _ in range(steps):
            product = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=product, stationary=matrix, moving=impulses)
            gradient = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gradient, src=product)
            summed = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=summed, data1=gradient, data2=b, op=nl.add)
            scaled = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=scaled, data1=summed, data2=rate, op=nl.multiply)
            updated = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=updated, data1=impulses, data2=scaled, op=nl.subtract)
            nisa.tensor_scalar(dst=impulses, data=updated, op0=nl.maximum, operand0=0.0)
        nisa.dma_copy(dst=result[:, world:world + 1], src=impulses)
    return result
