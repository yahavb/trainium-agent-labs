import unittest

from optimization_harness import plan, select
from plane_loop import reviewed_forms


class HarnessTests(unittest.TestCase):
    def test_both_available(self):
        report = plan(reviewed_forms()["baseline"])
        self.assertEqual(report["candidate_forms"],
                         ["copyfree", "scale-fused", "hoist-bias", "engine-vector", "interleave", "tile32"])
        self.assertTrue(all(t["applicable"] for t in report["techniques"]))

    def test_no_copy_but_scaling_available(self):
        report = plan(reviewed_forms()["copyfree"])
        self.assertEqual(report["candidate_forms"], ["scale-fused"])
        self.assertFalse(report["techniques"][0]["applicable"])
        self.assertTrue(report["techniques"][1]["applicable"])

    def test_neither_available(self):
        report = plan(reviewed_forms()["scale-fused"])
        self.assertTrue(report["supported"])
        self.assertEqual(report["candidate_forms"], [])

    def test_unknown_fails_closed(self):
        report = plan(reviewed_forms()["baseline"].replace("op=nl.add", "op=nl.subtract"))
        self.assertFalse(report["supported"])
        self.assertEqual(report["candidate_forms"], [])

    def measurement(self, report, form, ratio, **updates):
        row = dict(input_sha256=report["input_sha256"], form=form,
                   throughput_ratio=ratio, physics_score=1.0,
                   stable_baseline=True, evidence_path="mock-test-only")
        row.update(updates)
        return row

    def test_fastest_correct_candidate_not_first_applicable(self):
        report = plan(reviewed_forms()["baseline"])
        rows = [self.measurement(report, "copyfree", 1.02),
                self.measurement(report, "scale-fused", 1.05)]
        self.assertEqual(select(report, rows)["winner"], "scale-fused")

    def test_scaling_available_but_slower_keep_original(self):
        report = plan(reviewed_forms()["copyfree"])
        self.assertEqual(select(report, [self.measurement(report, "scale-fused", .997)])
                         ["winner"], "original")

    def test_reject_invalid_evidence(self):
        report = plan(reviewed_forms()["baseline"])
        for change in ({"physics_score": .9}, {"stable_baseline": False},
                       {"input_sha256": "wrong"}, {"evidence_path": None},
                       {"throughput_ratio": float("nan")}, {"throughput_ratio": True}):
            with self.subTest(change=change):
                result = select(report, [self.measurement(report, "copyfree", 2, **change)])
                self.assertEqual(result["winner"], "original")
                self.assertFalse(result["decisions"][0]["eligible"])


if __name__ == "__main__":
    unittest.main()
