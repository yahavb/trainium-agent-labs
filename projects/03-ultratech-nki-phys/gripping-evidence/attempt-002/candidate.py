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
    
    # Momentum coefficient (tunable parameter between 0 and 1)
    beta = 0.98
    
    for world in nl.static_range(worlds):
        matrix = nl.ndarray((contacts, contacts), dtype=nl.float32, buffer=nl.sbuf)
        b = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
        x = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
        y = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
        rate = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
        
        nisa.dma_copy(dst=matrix, src=a_transposed[:, world * contacts:(world + 1) * contacts])
        nisa.dma_copy(dst=b, src=bias[:, world:world + 1])
        nisa.dma_copy(dst=x, src=initial[:, world:world + 1])
        nisa.dma_copy(dst=y, src=initial[:, world:world + 1])
        nisa.dma_copy(dst=rate, src=alpha[:, world:world + 1])
        
        for _ in range(steps):
            # Compute gradient at y
            product = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=product, stationary=matrix, moving=y)
            
            gradient = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=gradient, data1=product, data2=b, op=nl.add)
            
            # Scale gradient by rate
            scaled = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=scaled, data1=gradient, data2=rate, op=nl.multiply)
            
            # Compute candidate
            candidate = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=candidate, data1=y, data2=scaled, op=nl.subtract)
            
            # Project candidate to nonnegative
            x_new = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=x_new, data=candidate, op0=nl.maximum, operand0=0.0)
            
            # Compute delta
            delta = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=delta, data1=x_new, data2=x, op=nl.subtract)
            
            # Compute momentum delta
            momentum_delta = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=momentum_delta, data=delta, op0=nl.multiply, operand0=beta)
            
            # Compute y_new
            y_new = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=y_new, data1=x_new, data2=momentum_delta, op=nl.add)
            
            # Update x and y
            nisa.tensor_copy(dst=x, src=x_new)
            nisa.tensor_copy(dst=y, src=y_new)
        
        nisa.dma_copy(dst=result[:, world:world + 1], src=x)
    
    return result
