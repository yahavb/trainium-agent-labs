"""Two-pad gripping snapshots with MuJoCo pyramidal sliding friction."""

import xml.etree.ElementTree as ET

import numpy as np

from contact import diagnostics, oracle
from engine_validation import load_engine
from force_checker import check_forces

CONTRACT = dict(version=1, output="pyramidal_edge_forces", contact_dim=3,
                cone="pyramidal", force_error_limit=1e-4,
                acceleration_error_limit=1e-4, wrench_error_limit=1e-4,
                cone_violation_limit=1e-6, projected_residual_limit=1e-4,
                objective_error_limit=1e-5, nonnegative_tolerance=1e-6,
                scope="snapshots only; no sustained-hold or rollout claim")


def scene_xml(mass, friction, penetration):
    if not (np.isfinite([mass, friction, penetration]).all() and
            mass > 0 and friction > 0 and 0 < penetration < 0.01):
        raise ValueError("Positive mass/friction and penetration in (0, .01) required")
    root = ET.Element("mujoco", model="two_pad_grip")
    option = ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81",
                           cone="pyramidal", solver="Newton", jacobian="dense",
                           iterations="200", tolerance="1e-12")
    ET.SubElement(option, "flag", warmstart="disable")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", condim="3", friction=f"{friction:.17g} 0 0",
                  solref="0.02 1", solimp="0.9 0.95 0.001")
    world = ET.SubElement(root, "worldbody")
    for side, sign in (("left", -1), ("right", 1)):
        ET.SubElement(world, "geom", name=side, type="box", size="0.02 0.08 0.08",
                      pos=f"{sign * (0.07 - penetration):.17g} 0 0.3")
    body = ET.SubElement(world, "body", name="object", pos="0 0 0.3")
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "geom", name="object_geom", type="sphere", size="0.05",
                  mass=f"{mass:.17g}")
    return ET.tostring(root, encoding="unicode")


def extract(xml, qpos, qvel):
    mj = load_engine()
    model = mj.MjModel.from_xml_string(xml)
    if model.opt.cone != int(mj.mjtCone.mjCONE_PYRAMIDAL):
        raise ValueError("Only pyramidal friction is supported")
    data = mj.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mj.mj_forward(model, data)
    if (data.ncon != 2 or data.nefc != 8 or
            not np.all(data.efc_type == int(mj.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL))):
        raise ValueError("Expected exactly two active condim=3 frictional contacts")
    mass = np.zeros((model.nv, model.nv))
    mj.mj_fullM(model, data, mass)
    jacobian = np.asarray(data.efc_J).reshape(data.nefc, model.nv).copy()
    response = np.linalg.solve(mass, jacobian.T)
    regularizer = np.asarray(data.efc_R).copy()
    a = jacobian @ response + np.diag(regularizer)
    b = np.asarray(data.efc_b).copy()
    np.testing.assert_allclose(b, jacobian @ data.qacc_smooth - data.efc_aref,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(data.qfrc_constraint, jacobian.T @ data.efc_force,
                               rtol=1e-10, atol=1e-10)
    if not np.all(regularizer > 0) or np.linalg.eigvalsh(a)[0] <= 0:
        raise ValueError("Force QP must be positive definite")
    # Four edge coefficients map to normal and two tangential contact forces.
    decode = np.zeros((2, 6, data.nefc))
    engine_wrenches = np.zeros((2, 6))
    friction = []
    for i, contact in enumerate(data.contact):
        if contact.dim != 3 or contact.efc_address < 0:
            raise ValueError("Unsupported contact dimensionality or inactive contact")
        start = contact.efc_address
        mu = np.asarray(contact.friction[:2]).copy()
        if not np.all(mu > 0):
            raise ValueError("Two positive sliding-friction coefficients required")
        decode[i, 0, start:start + 4] = 1
        decode[i, 1, start:start + 2] = [mu[0], -mu[0]]
        decode[i, 2, start + 2:start + 4] = [mu[1], -mu[1]]
        mj.mj_contactForce(model, data, i, engine_wrenches[i])
        friction.append(mu)
    np.testing.assert_allclose(decode @ data.efc_force, engine_wrenches,
                               rtol=1e-10, atol=1e-10)
    return dict(A=a, b=b, M=mass, J=jacobian, R=regularizer, response=response,
                qpos=np.asarray(qpos).copy(), qvel=np.asarray(qvel).copy(),
                qacc_smooth=np.asarray(data.qacc_smooth).copy(),
                engine_qacc=np.asarray(data.qacc).copy(),
                engine_forces=np.asarray(data.efc_force).copy(),
                decode=decode, friction=np.array(friction), engine_wrenches=engine_wrenches,
                contact_position=np.array([c.pos for c in data.contact]),
                contact_frame=np.array([c.frame for c in data.contact]))


def build_fixture(mass, friction, penetration, slip):
    if not np.isfinite(slip):
        raise ValueError("Finite slip speed required")
    xml = scene_xml(mass, friction, penetration)
    model = load_engine().MjModel.from_xml_string(xml)
    qvel = np.zeros(model.nv)
    qvel[2] = -slip
    return xml, extract(xml, model.qpos0.copy(), qvel)


def check_grip(fixture, candidate, reference):
    base = check_forces(fixture, candidate, reference)
    candidate = np.asarray(candidate)
    if candidate.shape != fixture["b"].shape or not np.isfinite(candidate).all():
        return dict(passed=False, force_check=base)
    wrench = fixture["decode"] @ candidate.astype(np.float64)
    expected = fixture["decode"] @ reference
    wrench_error = float(np.max(np.abs(wrench - expected)) / max(1, np.max(np.abs(expected))))
    # Pyramidal feasibility: |t1|/mu1 + |t2|/mu2 <= normal force.
    violation = np.maximum(0, np.sum(np.abs(wrench[:, 1:3]) / fixture["friction"], axis=1) - wrench[:, 0])
    cone_error = float(np.max(violation) / max(1, np.max(np.abs(wrench[:, 0]))))
    return dict(passed=bool(base["passed"] and wrench_error <= CONTRACT["wrench_error_limit"] and
                           cone_error <= CONTRACT["cone_violation_limit"]),
                force_check=base, wrench_error=wrench_error, cone_violation=cone_error)


def certify(fixture):
    reference = oracle(fixture["A"], fixture["b"])
    engine = fixture["engine_forces"]
    oracle_check = diagnostics(fixture["A"], fixture["b"], reference, reference)
    force_error = float(np.max(np.abs(engine - reference)) / max(1, np.max(np.abs(reference))))
    acceleration = fixture["qacc_smooth"] + fixture["response"] @ reference
    acceleration_error = float(np.max(np.abs(acceleration - fixture["engine_qacc"])) /
                               max(1, np.max(np.abs(fixture["engine_qacc"]))))
    wrenches = fixture["decode"] @ reference
    wrench_error = float(np.max(np.abs(wrenches - fixture["engine_wrenches"])) /
                         max(1, np.max(np.abs(fixture["engine_wrenches"]))))
    check = check_grip(fixture, engine, reference)
    passed = bool(oracle_check["passed"] and oracle_check["residual"] <= 1e-9 and
                  check["passed"] and max(force_error, acceleration_error, wrench_error) <= 1e-7)
    return reference, dict(passed=passed, force_error=force_error,
                           acceleration_error=acceleration_error, wrench_error=wrench_error,
                           oracle_residual=oracle_check["residual"])
