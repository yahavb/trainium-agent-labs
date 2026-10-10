from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "runners" / "trainium_runner.py"
SUMMARY_FIXTURE = Path(__file__).parent / "neuron_summary_samudra_1deg_fp32.json"

sys.path.insert(0, str(PROJECT_ROOT / "runners"))
import profiling  # noqa: E402


class NeuronSummaryTest(unittest.TestCase):
    """Uses a real neuron-explorer summary: Samudra 2, 1 degree, fp32, seat-211."""

    def setUp(self) -> None:
        self.metrics = profiling.key_metrics(profiling.load_neuron_summary(SUMMARY_FIXTURE))

    def test_key_metrics(self) -> None:
        m = self.metrics
        self.assertAlmostEqual(m["pass_ms"], 227.33, places=1)
        self.assertEqual(m["bound"], "memory")
        self.assertAlmostEqual(m["spill_save_gb"], 23.38, places=1)
        self.assertAlmostEqual(m["spill_reload_gb"], 31.61, places=1)
        self.assertAlmostEqual(m["inputs_and_weights_gb"], 0.58, places=2)
        self.assertGreater(m["spill_fraction_of_hbm"], 0.9)
        dma = m["engines"]["DMA (memory transfers)"]
        self.assertAlmostEqual(dma["active_ms"], 148.3, places=1)
        self.assertAlmostEqual(dma["fraction_of_pass"], 0.65, places=2)

    def test_markdown_report(self) -> None:
        text = profiling.render_markdown(self.metrics)
        self.assertIn("| Bound by | memory |", text)
        self.assertIn("| DMA (memory transfers) | 148.3 | 65% |", text)
        self.assertIn("| Compute utilization (MFU) | 20% |", text)

    def test_summarize_command(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = profiling.main(["summarize", str(SUMMARY_FIXTURE)])
        self.assertEqual(status, 0)
        self.assertIn("One pass on the device: 227.3 ms", output.getvalue())


class FindNeffsTest(unittest.TestCase):
    def test_filters_by_time_and_sorts_by_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "MODULE_old" / "model.neff"
            small = root / "MODULE_small" / "model.neff"
            large = root / "MODULE_large" / "model.neff"
            for path, size, mtime in [(old, 900, 1000.0), (small, 10, 2000.0), (large, 500, 2000.0)]:
                path.parent.mkdir()
                path.write_bytes(b"x" * size)
                os.utime(path, (mtime, mtime))
            self.assertEqual(profiling.find_neffs(root, newer_than=1500.0), [large, small])
            self.assertEqual(profiling.find_neffs(root)[0], old)


class StageTimerTest(unittest.TestCase):
    def test_records_stages_and_syncs_only_when_asked(self) -> None:
        calls = []
        timer = profiling.StageTimer(synchronize=lambda: calls.append(1))
        for _ in range(3):
            with timer.time("model", sync=True):
                pass
        with timer.time("read"):
            pass
        self.assertEqual(len(calls), 3)
        summary = timer.summary()
        self.assertEqual(summary["model"]["count"], 3)
        self.assertEqual(summary["read"]["count"], 1)
        self.assertIsNone(summary["read"]["median_excl_first_s"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timings.json"
            timer.write_json(path)
            self.assertEqual(json.loads(path.read_text())["summary"]["model"]["count"], 3)


class RunnerProfilingTest(unittest.TestCase):
    def run_runner(self, root: Path, extra: list[str]) -> dict:
        fixture = root / "reference.npz"
        manifest = root / "manifest.json"
        adapter = root / "adapter.py"
        metrics = root / "metrics.json"
        prediction = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4)
        np.savez_compressed(
            fixture,
            prognostic=prediction.copy(),
            prediction=prediction,
            time=np.asarray(["2014-10-20"], dtype="U"),
            variables=np.asarray(["t+1:thetao_0"], dtype="U"),
        )
        manifest.write_text(
            json.dumps({"prediction_key": "prediction", "coordinate_keys": ["time", "variables"]}),
            encoding="utf-8",
        )
        adapter.write_text(
            textwrap.dedent(
                """
                class Prepared:
                    def __init__(self, inputs):
                        self.inputs = inputs

                    def run(self):
                        return self.inputs["prognostic"] * 1.0

                def prepare(inputs, context):
                    return Prepared(inputs)
                """
            ),
            encoding="utf-8",
        )
        result = subprocess.run(
            [sys.executable, str(RUNNER), "--adapter", str(adapter), "--fixture", str(fixture),
             "--manifest", str(manifest), "--candidate-output", str(root / "candidate.npz"),
             "--metrics-json", str(metrics), "--precision", "float32",
             "--warmup", "1", "--repeats", "2", *extra],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(metrics.read_text(encoding="utf-8"))

    def test_records_compile_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = self.run_runner(Path(directory), [])
            self.assertLessEqual(metrics["compile_started_unix"], metrics["compile_finished_unix"])
            self.assertNotIn("torch_profile_dir", metrics)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_torch_profile_is_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "torch-profile"
            metrics = self.run_runner(root, ["--torch-profile", str(out), "--torch-profile-runs", "2"])
            self.assertEqual(metrics["torch_profile_dir"], str(out))
            for name in ("torch_ops.txt", "torch_trace.json", "torch_stacks.txt"):
                self.assertTrue((out / name).exists(), name)
            self.assertEqual(metrics["timed_runs"], 2)


if __name__ == "__main__":
    unittest.main()
