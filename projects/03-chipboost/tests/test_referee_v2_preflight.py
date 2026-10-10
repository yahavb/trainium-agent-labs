"""The treatment may not start while the original experiment is incomplete."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("run_referee_v2", Path(__file__).parents[1] / "run_referee_v2.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PredecessorGate(unittest.TestCase):
    def rejected(self, phase="complete", core=3, omit_tag=False):
        state = dict(phase=phase, core=core, budget=8, repeat=3,
                     referee_commit="434e5f9", p2_commit="919c6be", p3_commit="2ce9416",
                     acceptance="passed", referee_sha256=runner.ORIGINAL_REFEREE,
                     baseline_sha256=runner.BASELINE, candidate_sha256=runner.CANDIDATE,
                     completed=[f"{arm}-r{r}" for arm in ("referee", "model_alone", "random_search")
                                for r in range(3)])
        if omit_tag:
            state["completed"].pop()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "state.json").write_text(json.dumps(state))
            # Missing source arguments deliberately establish rejection before source imports or hardware.
            with self.assertRaises(AssertionError):
                runner.preflight(SimpleNamespace(predecessor=tmp, core=3))

    def test_running_original_is_rejected(self):
        self.rejected(phase="model_alone-r2")

    def test_failed_original_is_rejected(self):
        self.rejected(phase="failed")

    def test_false_completion_is_rejected(self):
        self.rejected(omit_tag=True)

    def test_different_core_is_rejected(self):
        self.rejected(core=2)

    def test_dma_only_delta_is_allowed(self):
        runner.check_pure_delta("def _child_failure(e): return e\ndef unchanged(): return 1\n",
                                "_DMA_4X_ERROR = 'error'\n_DMA_4X_INSTR = 'repair'\n"
                                "def _child_failure(e): return 'mapped'\ndef unchanged(): return 1\n")

    def test_outer_loop_change_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "more than"):
            runner.check_pure_delta("def one_instruction(): return 'old'\n",
                                    "def one_instruction(): return 'new'\n")


if __name__ == "__main__":
    unittest.main()
