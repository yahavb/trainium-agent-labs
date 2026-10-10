"""Gripping extraction, friction feasibility, and checker rejection tests."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from generate_grip import generate, replay
from grip_physics import build_fixture, certify, check_grip, extract


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "Install requirements-engine.txt")
class GripTests(unittest.TestCase):
    def test_all_parameter_combinations_certified(self):
        import itertools
        for parameters in itertools.product((.5, 2), (.2, .8), (.0002, .001), (0, .2)):
            with self.subTest(parameters=parameters):
                _, fixture = build_fixture(*parameters)
                reference, certification = certify(fixture)
                self.assertTrue(certification["passed"])
                self.assertEqual(fixture["A"].shape, (8, 8))
                self.assertTrue(check_grip(fixture, reference.astype(np.float32), reference)["passed"])

    def test_missing_friction_and_wrong_outputs_rejected(self):
        _, f = build_fixture(1, .8, .001, 0)
        reference, _ = certify(f)
        wrong = reference.copy()
        for c in range(2):
            start = c * 4
            wrong[start:start + 4] = np.mean(reference[start:start + 4])
        self.assertFalse(check_grip(f, wrong, reference)["passed"])
        for wrong in (np.zeros(8), -np.ones(8), np.full(8, np.nan), np.zeros(7)):
            self.assertFalse(check_grip(f, wrong, reference)["passed"])

    def test_contact_wrench_decoding_matches_engine(self):
        _, f = build_fixture(1, .8, .001, .2)
        np.testing.assert_allclose(f["decode"] @ f["engine_forces"], f["engine_wrenches"], atol=1e-10)
        self.assertGreater(np.max(np.abs(f["engine_wrenches"][:, 1:3])), 0)

    def test_elliptic_contacts_rejected(self):
        xml, f = build_fixture(1, .8, .001, 0)
        root = ET.fromstring(xml)
        root.find("option").set("cone", "elliptic")
        with self.assertRaises(ValueError):
            extract(ET.tostring(root, encoding="unicode"), f["qpos"], f["qvel"])

    def test_bad_parameters_rejected(self):
        for parameters in ((0, .8, .001, 0), (1, 0, .001, 0), (1, .8, -.001, 0), (1, .8, .001, np.nan)):
            with self.assertRaises(ValueError):
                build_fixture(*parameters)

    def test_saved_scores_replay_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "suite"
            generate(out, 1)
            replay(out)
            with self.assertRaises(FileExistsError):
                generate(out, 1)
            with (out / "public/grip-000.npz").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaises(ValueError):
                replay(out)


if __name__ == "__main__":
    unittest.main()
