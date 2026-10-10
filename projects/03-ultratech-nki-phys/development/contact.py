"""Normalized, frictionless sphere-contact fixtures and an independent QP oracle."""

import numpy as np
from scipy.optimize import nnls


def build_fixture(scene, contacts, seed):
    if contacts < 1 or scene not in ("plane", "pairs", "stack"):
        raise ValueError("Choose plane, pairs or stack with a positive contact count")
    rng = np.random.default_rng(seed)
    bodies = 2 * contacts if scene == "pairs" else contacts
    mass = rng.uniform(0.5, 2.0, bodies)
    velocity = rng.uniform(-1.0, 1.0, (bodies, 3))
    jacobian = np.zeros((contacts, bodies * 3))
    if scene == "pairs":
        for c in range(contacts):
            normal = rng.normal(size=3)
            normal /= np.linalg.norm(normal)
            jacobian[c, 6 * c:6 * c + 3] = -normal
            jacobian[c, 6 * c + 3:6 * c + 6] = normal
    else:
        for c in range(contacts):
            jacobian[c, 3 * c + 2] = 1.0
            if scene == "stack" and c:
                jacobian[c, 3 * (c - 1) + 2] = -1.0
    inverse_mass = np.repeat(1.0 / mass, 3)
    epsilon = 1e-3
    a = (jacobian * inverse_mass) @ jacobian.T + epsilon * np.eye(contacts)
    b = jacobian @ velocity.ravel()
    return dict(A=a, b=b, J=jacobian, inverse_mass=inverse_mass,
                v_free=velocity.ravel(), active=np.ones(contacts, dtype=bool))


def oracle(a, b):
    # A=R.T R transforms the constrained QP into nonnegative least squares.
    r = np.linalg.cholesky(a).T
    target = np.linalg.solve(r.T, -b)
    impulses, _ = nnls(r, target, maxiter=max(1000, 10 * len(b)))
    return impulses


def diagnostics(a, b, impulses, reference, fixture=None):
    impulses = np.asarray(impulses)
    if impulses.shape != b.shape or not np.isfinite(impulses).all():
        return dict(passed=False, reason="wrong shape or nonfinite impulses")
    lipschitz = float(np.linalg.eigvalsh(a)[-1])
    gradient = a @ impulses + b
    mapping = lipschitz * (impulses - np.maximum(0, impulses - gradient / lipschitz))
    residual = float(np.max(np.abs(mapping)) / max(1.0, np.max(np.abs(b))))
    objective = lambda x: float(0.5 * x @ a @ x + b @ x)
    gap = abs(objective(impulses) - objective(reference)) / max(1.0, abs(objective(reference)))
    feasible = float(impulses.min()) >= -1e-6
    velocity_error = 0.0
    if fixture is not None:
        update = lambda x: fixture["v_free"] + fixture["inverse_mass"] * (fixture["J"].T @ x)
        expected = update(reference)
        velocity_error = float(np.max(np.abs(update(impulses) - expected)) /
                               max(1.0, np.max(np.abs(expected))))
    return dict(passed=bool(feasible and residual <= 1e-4 and gap <= 1e-5 and velocity_error <= 1e-4),
                residual=residual, objective_error=gap, velocity_error=velocity_error,
                feasible=bool(feasible))


def velocity_error_bound(a, b, impulses, fixture):
    eigenvalues = np.linalg.eigvalsh(a)
    lower, upper = float(eigenvalues[0]), float(eigenvalues[-1])
    if lower <= 0:
        raise ValueError("Velocity bound requires a positive-definite contact matrix")
    mapping = upper * (impulses - np.maximum(0, impulses - (a @ impulses + b) / upper))
    response = fixture["inverse_mass"][:, None] * fixture["J"].T
    gain = float(np.max(np.linalg.norm(response, axis=1)))
    return gain * float(np.linalg.norm(mapping)) / lower


def projected_gradient(a, b, tolerance=1e-4, max_iterations=4096, fixture=None):
    original_a, original_b = a.astype(np.float64), b.astype(np.float64)
    eigenvalues = np.linalg.eigvalsh(original_a)
    lower, lipschitz = float(eigenvalues[0]), float(eigenvalues[-1])
    if lower <= 0:
        raise ValueError("Projected gradient requires a positive-definite contact matrix")
    if fixture is not None:
        response = fixture["inverse_mass"][:, None] * fixture["J"].T
        gain = float(np.max(np.linalg.norm(response, axis=1)))
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    impulses = np.zeros_like(b)
    for iteration in range(1, max_iterations + 1):
        impulses = np.maximum(0, impulses - (a @ impulses + b) / lipschitz)
        if iteration % 8 == 0:
            # Independent FP64 stopping checks avoid FP32 residual rounding at the gate.
            candidate = impulses.astype(np.float64)
            gradient = original_a @ candidate + original_b
            mapping = lipschitz * (candidate - np.maximum(0, candidate - gradient / lipschitz))
            residual_ok = np.max(np.abs(mapping)) / max(1.0, np.max(np.abs(original_b))) <= tolerance
            # The projected map contracts by 1-mu/L, so ||error||2 <= ||G||2/mu.
            physical_ok = fixture is None or gain * np.linalg.norm(mapping) / lower <= 1e-4
            if residual_ok and physical_ok:
                return impulses, iteration
    return impulses, max_iterations
