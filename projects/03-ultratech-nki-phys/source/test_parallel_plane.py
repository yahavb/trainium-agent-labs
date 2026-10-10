import unittest

from parallel_plane_runs import ensure_disjoint


class SeatIsolationTests(unittest.TestCase):
    def test_same_chip_rejected(self):
        device = dict(instance_id="instance-a", bdf="0000:85:00.0")
        with self.assertRaisesRegex(ValueError, "same chip"):
            ensure_disjoint({"seat-260": [device], "seat-261": [device]})

    def test_separate_chips_allowed(self):
        ensure_disjoint({"seat-260": [dict(instance_id="instance-a", bdf="0000:85:00.0")],
                         "seat-261": [dict(instance_id="instance-a", bdf="0000:86:00.0")]})

    def test_unknown_identity_rejected(self):
        with self.assertRaises(ValueError):
            ensure_disjoint({"seat-260": [dict(neuron_device=0)]})


if __name__ == "__main__":
    unittest.main()
