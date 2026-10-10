"""Distinct physics calculations with frozen-input FP64 reference answers."""

import numpy as np


TASKS = {
    "spring": {"equation": "force = -(stiffness * displacement + damping * velocity)",
               "structure": "elementwise", "technique": "sign fusion",
               "input_names": ["displacement", "velocity", "stiffness", "damping"]},
    "net-force": {"equation": "net_force[p, 0] = sum(forces[p, :])",
                  "structure": "reduction", "technique": "native reduction instead of serial adds",
                  "input_names": ["forces"]},
}


def fixture(task, seed):
    rng = np.random.default_rng(seed)
    if task == "spring":
        x = rng.normal(size=(64, 8)).astype(np.float32)
        v = rng.normal(size=(64, 8)).astype(np.float32)
        k = rng.uniform(0, 100, size=(64, 1)).astype(np.float32)
        c = rng.uniform(0, 10, size=(64, 1)).astype(np.float32)
        if seed % 4 == 0:
            x.fill(0)
        elif seed % 4 == 1:
            v.fill(0)
        elif seed % 4 == 2:
            c.fill(0)
        return x, v, k, c
    if task == "net-force":
        forces = rng.normal(size=(64, 8)).astype(np.float32)
        if seed % 4 == 0:
            forces.fill(0)
        elif seed % 4 == 1:
            forces[:, 4:] = -forces[:, :4]
        elif seed % 4 == 2:
            forces[:, 0] *= 1000
        return (forces,)
    raise ValueError(task)


def reference(task, inputs):
    values = [x.astype(np.float64) for x in inputs]
    if task == "spring":
        x, v, k, c = values
        return -(k * x + c * v), np.abs(k * x) + np.abs(c * v)
    forces, = values
    return np.sum(forces, axis=1, keepdims=True), np.sum(np.abs(forces), axis=1, keepdims=True)


def check(task, actual, inputs, originals):
    expected, magnitude = reference(task, originals)
    actual = np.asarray(actual)
    unchanged = all(np.array_equal(a, b) for a, b in zip(inputs, originals))
    if actual.shape != expected.shape or actual.dtype != np.float32 or not np.isfinite(actual).all():
        return dict(score=0, reason="Wrong shape/dtype or nonfinite output", inputs_unchanged=unchanged)
    # Bound cancellation error by contributing terms, not by a near-zero result.
    limit = 1e-6 + 2e-6 * magnitude
    error = np.abs(actual.astype(np.float64) - expected)
    passed = bool(np.all(error <= limit) and unchanged)
    return dict(score=int(passed), inputs_unchanged=unchanged,
                max_absolute_error=float(error.max()), max_normalized_error=float((error / limit).max()),
                reason="All equation and input-preservation gates passed" if passed else
                       "Output differs from FP64 equation beyond the fixed FP32 error budget, or inputs changed")


def cpu(task, inputs, optimized=False):
    if task == "spring":
        x, v, k, c = inputs
        return -(k * x + c * v)
    forces, = inputs
    if optimized:
        return np.sum(forces, axis=1, keepdims=True, dtype=np.float32)
    output = forces[:, :1].copy()
    for i in range(1, forces.shape[1]):
        output += forces[:, i:i + 1]
    return output


SPRING = '''import nki
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
'''

REDUCTION = '''import nki
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
'''


def sources(task):
    if task == "spring":
        # Full-width damping is not a legal scalar operand. Fuse each term's
        # negative sign instead, retaining the full-width tensor addition.
        optimized = SPRING.replace(
            "nisa.tensor_scalar(dst=spring, data=x, op0=nl.multiply, operand0=k)",
            "nisa.tensor_scalar(dst=spring, data=x, op0=nl.multiply, operand0=k, op1=nl.multiply, operand1=-1.0)").replace(
            "nisa.tensor_scalar(dst=damped, data=v, op0=nl.multiply, operand0=c)",
            "nisa.tensor_scalar(dst=damped, data=v, op0=nl.multiply, operand0=c, op1=nl.multiply, operand1=-1.0)").replace(
            "    nisa.tensor_tensor(dst=summed, data1=spring, data2=damped, op=nl.add)\n"
            "    nisa.tensor_scalar(dst=force, data=summed, op0=nl.multiply, operand0=-1.0)",
            "    nisa.tensor_tensor(dst=force, data1=spring, data2=damped, op=nl.add)")
        return {"baseline": SPRING, "sign-fusion": optimized}
    optimized = REDUCTION.replace(
        "    nisa.tensor_copy(dst=total, src=values[:, 0:1])\n"
        "    for i in nl.static_range(1, count):\n"
        "        nisa.tensor_tensor(dst=total, data1=total, data2=values[:, i:i + 1], op=nl.add)",
        "    nisa.tensor_reduce(dst=total, data=values, op=nl.add, axis=(1,), keepdims=True)")
    return {"baseline": REDUCTION, "native-reduction": optimized}
