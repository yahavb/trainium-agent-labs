"""Independent analytic checks and replay properties for engine extraction."""

import importlib.util
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from engine_validation import extract, initial_state, load_engine, problem_digest, scene_xml, verify


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "Install requirements-engine.txt for engine tests")
class EngineTests(unittest.TestCase):
    def fixture(self, scene, count, seed=0):
        xml = scene_xml(scene, count, seed)
        model = load_engine().MjModel.from_xml_string(xml)
        return extract(xml, *initial_state(model, scene, seed))

    def test_single_contact_matches_closed_form(self):
        f = self.fixture("plane", 1)
        expected = max(0, -f["b"][0] / f["A"][0, 0])
        np.testing.assert_allclose(f["engine_forces"], [expected], rtol=1e-9, atol=1e-9)

    def test_contact_acceleration_conserves_pair_linear_momentum(self):
        f = self.fixture("pairs", 1)
        momentum_rate = np.zeros(3)
        for i in range(2):
            start = 6 * i
            momentum_rate += f["M"][start, start] * f["engine_qacc"][start:start + 3]
        np.testing.assert_allclose(momentum_rate, np.zeros(3), atol=1e-10)

    def test_engine_and_independent_solver_agree(self):
        for scene in ("plane", "pairs", "stack"):
            self.assertTrue(verify(self.fixture(scene, 8))[1]["passed"])

    def test_frictional_constraints_rejected(self):
        root = ET.fromstring(scene_xml("plane", 1, 0))
        root.find("default/geom").set("condim", "3")
        root.find("default/geom").set("friction", "0.5 0.1 0.1")
        xml = ET.tostring(root, encoding="unicode")
        model = load_engine().MjModel.from_xml_string(xml)
        with self.assertRaises(ValueError):
            extract(xml, *initial_state(model, "plane", 0))

    def test_wrong_engine_answer_rejected(self):
        f = self.fixture("stack", 8)
        f["engine_forces"] = np.zeros_like(f["engine_forces"])
        self.assertFalse(verify(f)[1]["passed"])

    def test_duplicate_physics_detected_across_scene_labels(self):
        self.assertEqual(problem_digest(self.fixture("plane", 1)),
                         problem_digest(self.fixture("stack", 1)))


if __name__ == "__main__":
    unittest.main()
