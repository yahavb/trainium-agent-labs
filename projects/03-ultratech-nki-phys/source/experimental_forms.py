"""Bounded optimization hypotheses; installed SDK and hardware validation required."""


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Baseline changed; experiment needs renewed review")
    return source.replace(old, new, 1)


def experimental_forms(baseline):
    hoisted = replace_once(baseline, "        for _ in range(steps):", """        scaled_bias = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=scaled_bias, data1=b, data2=rate, op=nl.multiply)
        for _ in range(steps):""")
    hoisted = replace_once(hoisted, """            summed = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=summed, data1=gradient, data2=b, op=nl.add)
""", "")
    hoisted = replace_once(hoisted, "data1=summed, data2=rate", "data1=gradient, data2=rate")
    hoisted = replace_once(hoisted,
        "            nisa.tensor_tensor(dst=updated, data1=impulses, data2=scaled, op=nl.subtract)",
        """            adjusted = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=adjusted, data1=impulses, data2=scaled, op=nl.subtract)
            nisa.tensor_tensor(dst=updated, data1=adjusted, data2=scaled_bias, op=nl.subtract)""")

    engine = baseline
    for ending in ("data2=b, op=nl.add)", "data2=rate, op=nl.multiply)",
                   "data2=scaled, op=nl.subtract)", "op0=nl.maximum, operand0=0.0)"):
        engine = replace_once(engine, ending, ending[:-1] + ", engine=nisa.engine.vector)")

    tiled = replace_once(baseline, "    contacts, worlds = bias.shape",
                         "    contacts, worlds = bias.shape\n    assert contacts % 32 == 0")
    tiled = replace_once(tiled, """            product = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=product, stationary=matrix, moving=impulses)
            gradient = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gradient, src=product)""", """            gradient = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            for tile in nl.static_range(contacts // 32):
                partial = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=partial, stationary=matrix[tile * 32:(tile + 1) * 32, :],
                               moving=impulses[tile * 32:(tile + 1) * 32, :])
                chunk = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=chunk, src=partial)
                if tile == 0:
                    nisa.tensor_copy(dst=gradient, src=chunk)
                else:
                    nisa.tensor_tensor(dst=gradient, data1=gradient, data2=chunk, op=nl.add)""")
    return {"hoist-bias": hoisted, "engine-vector": engine,
            "interleave": INTERLEAVED, "tile32": tiled}


INTERLEAVED = '''"""Round-robin independent worlds; engine overlap is a hypothesis, not guaranteed."""

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
    matrices = nl.ndarray((contacts, contacts * worlds), dtype=nl.float32, buffer=nl.sbuf)
    biases = nl.ndarray((contacts, worlds), dtype=nl.float32, buffer=nl.sbuf)
    states = nl.ndarray((contacts, worlds), dtype=nl.float32, buffer=nl.sbuf)
    rates = nl.ndarray((contacts, worlds), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=matrices, src=a_transposed)
    nisa.dma_copy(dst=biases, src=bias)
    nisa.dma_copy(dst=states, src=initial)
    nisa.dma_copy(dst=rates, src=alpha)
    for _ in range(steps):
        for world in nl.static_range(worlds):
            product = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=product,
                           stationary=matrices[:, world * contacts:(world + 1) * contacts],
                           moving=states[:, world:world + 1])
            gradient = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gradient, src=product)
            summed = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=summed, data1=gradient, data2=biases[:, world:world + 1], op=nl.add)
            scaled = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=scaled, data1=summed, data2=rates[:, world:world + 1], op=nl.multiply)
            updated = nl.ndarray((contacts, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=updated, data1=states[:, world:world + 1], data2=scaled, op=nl.subtract)
            nisa.tensor_scalar(dst=states[:, world:world + 1], data=updated, op0=nl.maximum, operand0=0.0)
    nisa.dma_copy(dst=result, src=states)
    return result
'''
