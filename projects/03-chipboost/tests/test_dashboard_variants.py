"""Static rendering and version grouping checks; no browser or device required."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dashboard", ROOT / "dashboard/build.py")
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


class VariantTests(unittest.TestCase):
    def records(self):
        base = next(iter(dashboard.schema.fake_attempts()))
        records = []
        for arm, rid, speedup in (("referee", "matmul-referee-r0", 2.0),
                                  ("referee", "matmul-referee-v2-r0", 4.0),
                                  ("referee", "matmul-referee-v2-p1fix-r0", 3.0),
                                  ("referee", "matmul-referee-v2-p1fix-r1", 3.5),
                                  ("model_alone", "matmul-model_alone-r0", 1.5),
                                  ("random_search", "random_search-r0", 8.0)):
            rec = dict(base)
            for key in dashboard.schema.ATTEMPT_FIELDS:
                rec.setdefault(key, None)
            rec.update(kernel="matmul", arm=arm, run_id=rid, seat=100, attempt_no=1,
                       verdict="faster", speedup=speedup, time_us_median=100 / speedup,
                       baseline_us_same_session=100, source="chip", timestamp=1)
            records.append(rec)
        return records

    def test_versions_have_separate_runs_and_curves_without_mutation(self):
        records = self.records()
        original = copy.deepcopy(records)
        arms = dashboard.summarize(records)["matmul"]["arms"]
        self.assertEqual([r["best_x"] for r in arms["referee"]], [2.0])
        self.assertEqual([r["best_x"] for r in arms["referee_v2"]], [4.0])
        self.assertEqual(arms["referee_v2"][0]["curve"], [4.0])
        self.assertEqual([r["best_x"] for r in arms["referee_v2_p1fix"]], [3.0, 3.5])
        self.assertEqual(records, original)

    def test_render_separates_versions_and_qwen_headlines(self):
        records = self.records()
        results = dashboard.merge_results([])
        page = dashboard.build(records, results, [], False, ["explicit-qwen-inputs"])
        for label in ("Referee v1", "Referee v2", "Qwen + P1 fixes", "Model alone v1", "expert prior"):
            self.assertIn(label, page)
        headline = dashboard.kpis(dashboard.summarize(records), results, records, None)
        self.assertIn("Referee v1", headline)
        self.assertIn("Referee v2", headline)
        self.assertIn("Qwen + P1 fixes", headline)
        self.assertNotIn("Template search", headline)
        self.assertNotEqual(dashboard.ARM_COLOR["referee"], dashboard.ARM_COLOR["referee_v2"])


if __name__ == "__main__":
    unittest.main()
