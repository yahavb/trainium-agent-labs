import copy
import argparse
import json
from pathlib import Path
import tempfile
import unittest

from full_plane_loop import FullLoop, comparison, measured
from optimization_knowledge import retrieve
from plane_loop import reviewed_forms


class FullLoopTests(unittest.TestCase):
    def report(self):
        check = dict(score=1, physics_passes=8)
        return dict(context=dict(backend="device", initial_check=check), results=[dict(
            score=1, physics_passes=8, timing_ok=True, post_benchmark_check=check,
            device_stats=dict(durations_ms=[.1, .2])) for _ in range(5)])

    def test_aggregate_uses_total_work_over_total_time(self):
        actual = measured(self.report(), 5, 2)
        self.assertAlmostEqual(actual["worlds_per_second"], 8000 / .15)

    def test_partial_physics_never_receives_throughput(self):
        report = copy.deepcopy(self.report())
        report["results"][0]["post_benchmark_check"]["physics_passes"] = 7
        with self.assertRaises(ValueError):
            measured(report, 5, 2)

    def test_simulation_is_not_device_timing(self):
        report = self.report()
        report["context"]["backend"] = "simulate"
        with self.assertRaises(ValueError):
            measured(report, 5, 2)

    def test_paired_ratio_and_drift(self):
        baseline = measured(self.report(), 5, 2)
        report = self.report()
        for row in report["results"]:
            row["device_stats"]["durations_ms"] = [.05, .1]
        candidate = measured(report, 5, 2)
        base = dict(plane=baseline, pairs=baseline)
        values = dict(plane=candidate, pairs=candidate)
        result = comparison(base, values, base)
        self.assertAlmostEqual(result["speedup"], 2)
        self.assertTrue(result["stable_baseline"])
        drift = copy.deepcopy(base)
        drift["plane"]["mean_ms"] *= 1.2
        self.assertFalse(comparison(base, values, drift)["stable_baseline"])

    def test_retrieval_is_source_linked_and_keyword_ranked(self):
        text, metadata = retrieve("matmul transpose stationary", count=1)
        self.assertEqual(metadata["cards"][0]["id"], "matmul-layout")
        self.assertIn("https://", text)
        self.assertFalse(metadata["live_search"])

    def test_correct_candidate_is_timed_and_kept_with_feedback(self):
        events = []
        base = measured(self.report(), 5, 2)
        faster = copy.deepcopy(base)
        faster["samples_ms"] = [v / 2 for v in base["samples_ms"]]
        faster["mean_ms"] /= 2
        faster["worlds_per_second"] *= 2

        class FakeLoop(FullLoop):
            def command(self, argv, log):
                pass

            def upload(self, local, remote, log):
                pass

            def stop_model(self, log):
                events.append("stop-model")

            def suite(self, candidate, digest, backend, directory):
                events.append((backend, bool(candidate)))
                return dict(plane=faster, pairs=faster) if candidate else dict(plane=base, pairs=base)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.py"
            candidate.write_text(reviewed_forms()["fused"])
            args = argparse.Namespace(out=root / "loop", minutes=1, seat="test", cores="0,1",
                attempts=1, runs=5, iters=2, initial_candidate=candidate)
            loop = FakeLoop(args)
            loop.run()
            self.assertEqual(events, [("simulate", True), "stop-model", ("device", False),
                                      ("device", True), ("device", False)])
            self.assertEqual(loop.rows[0]["status"], "benchmarked")
            self.assertAlmostEqual(loop.rows[0]["throughput_score"], 2)
            self.assertTrue((args.out / "best.json").exists())
            self.assertIn("Measured Trainium", (args.out / "attempt-000/next-feedback.txt").read_text())

            events.clear()
            candidate.write_text(reviewed_forms()["scale-fused"])
            args.out = root / "scale-loop"
            loop = FakeLoop(args)
            loop.run()
            self.assertEqual(events, [("simulate", True), "stop-model", ("device", False),
                                      ("device", True), ("device", True), ("device", True), ("device", False)])
            self.assertIn("versus_copyfree", loop.rows[0]["comparison"])
            self.assertAlmostEqual(loop.rows[0]["comparison"]["versus_copyfree"]["speedup"], 1)


if __name__ == "__main__":
    unittest.main()
