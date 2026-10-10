"""Per-world FISTA schedule and gradient restart; pending installed-SDK validation."""

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
        x = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
        y = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
        rate = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
        t = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=matrix, src=a_transposed[:, world * contacts:(world + 1) * contacts])
        nisa.dma_copy(dst=b, src=bias[:, world:world + 1])
        nisa.dma_copy(dst=x, src=initial[:, world:world + 1])
        nisa.dma_copy(dst=y, src=initial[:, world:world + 1])
        nisa.dma_copy(dst=rate, src=alpha[:, world:world + 1])
        nisa.tensor_scalar(dst=t, data=rate, op0=nl.multiply, operand0=0.0, op1=nl.add, operand1=1.0)
        for _ in range(steps):
            product = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.psum)
            gradient = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            scaled = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            candidate = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            new = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            delta = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            direction = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.nc_matmul(dst=product, stationary=matrix, moving=y)
            nisa.tensor_tensor(dst=gradient, data1=product, data2=b, op=nl.add)
            nisa.tensor_tensor(dst=scaled, data1=gradient, data2=rate, op=nl.multiply)
            nisa.tensor_tensor(dst=candidate, data1=y, data2=scaled, op=nl.subtract)
            nisa.tensor_scalar(dst=new, data=candidate, op0=nl.maximum, operand0=0.0)
            nisa.tensor_tensor(dst=delta, data1=new, data2=x, op=nl.subtract)
            nisa.tensor_tensor(dst=direction, data1=y, data2=new, op=nl.subtract)

            # Contract over contacts, then broadcast the world-local restart decision.
            dot = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
            keep_scalar = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.nc_matmul(dst=dot, stationary=direction, moving=delta)
            nisa.tensor_scalar(dst=keep_scalar, data=dot, op0=nl.less_equal, operand0=0.0)
            keep = nl.broadcast_to(keep_scalar, shape=(contacts, 1))

            squared = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            radicand = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            root = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            next_t = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            inverse = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            numerator = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            beta = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            active_beta = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            momentum = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            next_y = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            t_offset = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            active_offset = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=squared, data1=t, data2=t, op=nl.multiply)
            nisa.tensor_scalar(dst=radicand, data=squared, op0=nl.multiply, operand0=4.0, op1=nl.add, operand1=1.0)
            nisa.activation(dst=root, op=nl.sqrt, data=radicand)
            nisa.tensor_scalar(dst=next_t, data=root, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=.5)
            nisa.reciprocal(dst=inverse, data=next_t)
            nisa.tensor_scalar(dst=numerator, data=t, op0=nl.subtract, operand0=1.0)
            nisa.tensor_tensor(dst=beta, data1=numerator, data2=inverse, op=nl.multiply)
            nisa.tensor_tensor(dst=active_beta, data1=beta, data2=keep, op=nl.multiply)
            nisa.tensor_tensor(dst=momentum, data1=delta, data2=active_beta, op=nl.multiply)
            nisa.tensor_tensor(dst=next_y, data1=new, data2=momentum, op=nl.add)
            nisa.tensor_scalar(dst=t_offset, data=next_t, op0=nl.subtract, operand0=1.0)
            nisa.tensor_tensor(dst=active_offset, data1=t_offset, data2=keep, op=nl.multiply)
            nisa.tensor_scalar(dst=t, data=active_offset, op0=nl.add, operand0=1.0)
            nisa.tensor_copy(dst=x, src=new)
            nisa.tensor_copy(dst=y, src=next_y)
        nisa.dma_copy(dst=result[:, world:world + 1], src=x)
    return result
