from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from frozen_math_eval import freeze, manifest
from submission_math import independent_repeat
import test_frozen_math as frozen_tests


class SubmissionTests(unittest.TestCase):
    def test_repeat_uses_frozen_proposals_and_never_generates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = freeze(frozen_tests.FrozenMathTests().create_run(root), root / "frozen")
            pinned = manifest(bundle)
            calls = []

            class FakeController:
                def __init__(self, args): self.remote = "/mock-repeat"
                def stop_model(self, log): calls.append("stop")
                def command(self, argv, log): calls.append(argv)
                def upload(self, local, remote, log): calls.append([str(local), remote])
                def remote_command(self, argv, log): calls.append(argv)
                def download(self, remote, local, log):
                    task = local.name
                    local.mkdir()
                    (local / "context.json").write_text(json.dumps(dict(backend="device", cores="0,1")))
                    (local / task).mkdir()
                    row = dict(status="benchmarked", correctness_score=1.0, throughput_ratio=1.02,
                               source_sha256=pinned["tasks"][task]["kernel_sha256"])
                    (local / task / "results.json").write_text(json.dumps(dict(attempts=[{}, row])))
                    (local / task / "timing.json").write_text(json.dumps(dict(
                        baseline_drift_fraction=.001, candidate=[dict(timing=dict(mean_ms=.01))] * 5)))

            args = Namespace(bundle=bundle, out=root / "repeat", seat="mock-seat", cores="0,1", minutes=1)
            with patch("full_plane_loop.FullLoop", FakeController): independent_repeat(args)
            summary = json.loads((args.out / "summary.json").read_text())
            self.assertEqual(set(summary["tasks"]), {"spring", "net-force"})
            self.assertNotIn("qwen_math.py", str(calls))
            self.assertIn("--program", str(calls))
            self.assertFalse((bundle / "assessment-started.json").exists())


if __name__ == "__main__":
    unittest.main()
