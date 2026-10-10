import json
from pathlib import Path
import tempfile
import unittest

import summarize_comparison as report


class ComparisonTests(unittest.TestCase):
    def row(self, verdict="no_gain", speedup=1.0):
        return dict(arm="referee", kernel="matmul", seat=100, run_id="test", attempt_no=0,
                    verdict=verdict, sim_ok=True, chip_ok=True, source="chip",
                    referee_message="real", speedup=speedup, time_us_median=100 / speedup,
                    baseline_us_same_session=100)

    def test_failed_runs_remain_at_baseline(self):
        rows = [self.row("wrong", 3), self.row("no_gain", 1.005)]
        result = report.aggregate(rows)
        self.assertEqual(result["best_verified_speedup"], 1)
        self.assertEqual(result["failure_rate"], 0.5)
        self.assertEqual(result["correct"], 1)
        self.assertIsNone(result["heldout_failure_rate"])

    def test_heldout_denominator_and_verified_best(self):
        rows = [self.row("faster", 1.2), self.row("heldout_fail", 3), self.row("slower", .8)]
        result = report.aggregate(rows)
        self.assertEqual(result["best_verified_speedup"], 1.2)
        self.assertEqual(result["heldout_failure_rate"], 0.5)
        self.assertAlmostEqual(result["failure_rate"], 1 / 3)

    def test_timing_mismatch_is_rejected(self):
        row = self.row("faster", 2)
        row["time_us_median"] = 75
        with self.assertRaisesRegex(ValueError, "timing ratio"):
            report.check_record(row, lambda _: [], "test")

    def batch(self, path):
        (path / "state.json").write_text(json.dumps(dict(phase="complete", acceptance="passed", repeat=3, budget=8,
            core=3, referee_commit="434e5f9", p2_commit="919c6be", p3_commit="2ce9416",
            completed=[f"{arm}-r{r}" for arm in report.ARMS for r in range(3)])))
        (path / "acceptance.json").write_text(json.dumps(self.row()))
        for arm in report.ARMS:
            for repeat in range(3):
                tag = f"{arm}-r{repeat}"
                rows = []
                for i in range(8):
                    row = self.row("faster" if repeat else "no_gain", 1 + repeat / 10)
                    row.update(arm=arm, run_id=tag, attempt_no=i if arm == "random_search" else i + 1)
                    rows.append(json.dumps(row))
                (path / f"{tag}.jsonl").write_text("\n".join(rows) + "\n")
        (path / "infrastructure.jsonl").write_text(json.dumps(dict(kind="prior_interruption")) + "\n")

    def test_complete_and_infrastructure_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.batch(path)
            result = report.summarize(path, lambda _: [])
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["attempts"], 72)
            self.assertEqual(result["infrastructure"]["events"], 1)
            for arm in report.ARMS:
                self.assertEqual(result["arms"][arm]["attempts"], 24)
                self.assertEqual(result["arms"][arm]["median_best_speedup"], 1.1)

    def test_missing_run_is_partial_not_zero_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.batch(path)
            (path / "referee-r0.jsonl").unlink()
            state = json.loads((path / "state.json").read_text())
            state["phase"] = "referee-r0"
            state["completed"].remove("referee-r0")
            (path / "state.json").write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "Incomplete batch"):
                report.summarize(path, lambda _: [])
            result = report.summarize(path, lambda _: [], allow_partial=True)
            self.assertEqual(result["status"], "partial")
            self.assertAlmostEqual(result["arms"]["referee"]["median_best_speedup"], 1.15)
            self.assertEqual(result["arms"]["referee"]["started_runs"], 2)

    def test_duplicate_attempt_and_schema_errors_never_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.batch(path)
            with self.assertRaisesRegex(ValueError, "bad schema"):
                report.summarize(path, lambda _: ["bad schema"], allow_partial=True)
            log = path / "referee-r0.jsonl"
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            rows[1]["attempt_no"] = 0
            log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "contiguous"):
                report.summarize(path, lambda _: [], allow_partial=True)

    def test_truncated_line_allowed_only_in_live_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live.jsonl"
            path.write_text('{}\n{"partial":')
            rows, notes = report.read_jsonl(path, True)
            self.assertEqual(rows, [{}])
            self.assertTrue(notes)
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                report.read_jsonl(path, False)

    def test_false_completion_and_changed_snapshot_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.batch(path)
            state_path = path / "state.json"
            state = json.loads(state_path.read_text())
            state["p1_unused"] = "ignored metadata"
            state["p2_commit"] = "different"
            state_path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "pinned snapshot"):
                report.summarize(path, lambda _: [], allow_partial=True)
            state["p2_commit"] = "919c6be"
            state["completed"].pop()
            state_path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "expected run tags"):
                report.summarize(path, lambda _: [], allow_partial=True)


if __name__ == "__main__":
    unittest.main()
