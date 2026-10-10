"""CPU regression: the launchers' pinned hashes hold for an LF git checkout and the CRLF seat snapshots.

Every pin (referee, baseline, P3 agent, P2 search, DMA treatment) was taken from CRLF copies on seat-100;
git stores LF. A content change must still fail. Needs this repo's git history for 434e5f9/2ce9416/919c6be.
Run: python projects/03-chipboost/tests/test_pinned_hashes.py
"""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_referee_v2", ROOT / "run_referee_v2.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
LAUNCH = __import__("json").loads((ROOT / "comparison_launch.json").read_text())
MANIFEST = __import__("json").loads((ROOT / "experiments/dma-feedback-v2/source_manifest.json").read_text())
PATCH = ROOT / "experiments/dma-feedback-v2/dma-only.patch"


def show(rev, path):
    done = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True)
    if done.returncode:
        raise unittest.SkipTest(f"no git object {rev}:{path}")
    return done.stdout


class PinnedHashes(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def both_endings(self, data, name="f.py"):
        lf, crlf = self.tmp / ("lf_" + name), self.tmp / ("crlf_" + name)
        lf.write_bytes(data)
        crlf.write_bytes(data.replace(b"\n", b"\r\n"))
        return lf, crlf

    def test_pins_match_lf_checkout_and_crlf_snapshot(self):
        for rev, path, pin in (("434e5f9", "projects/03-chipboost/speedcheck.py", runner.ORIGINAL_REFEREE),
                               ("HEAD", "projects/02-kernel-agent/reference_level4.py", runner.BASELINE),
                               ("2ce9416", "projects/03-chipboost/agent.py", LAUNCH["agent_sha256"]),
                               ("919c6be", "projects/03-chipboost/search.py", LAUNCH["search_sha256"])):
            for copy in self.both_endings(show(rev, path)):
                self.assertEqual(runner.pinned_digest(copy), pin, f"{rev}:{path} as {copy.name}")

    def test_changed_content_still_fails(self):
        for copy in self.both_endings(show("434e5f9", "projects/03-chipboost/speedcheck.py") + b"# x\n"):
            self.assertNotEqual(runner.pinned_digest(copy), runner.ORIGINAL_REFEREE)

    def test_dma_patch_rebuilds_the_pinned_treatment(self):
        self.assertNotIn(b"\r", PATCH.read_bytes(), "the patch must be plain LF to apply")
        work = self.tmp / "v2"
        work.mkdir()
        (work / "speedcheck.py").write_bytes(show("434e5f9", "projects/03-chipboost/speedcheck.py"))
        done = subprocess.run(["patch", "-p1", "-s", "-i", str(PATCH)], cwd=work, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(runner.pinned_digest(work / "speedcheck.py"), MANIFEST["treatment_sha256"])
        original = show("434e5f9", "projects/03-chipboost/speedcheck.py").decode()
        runner.check_pure_delta(original, (work / "speedcheck.py").read_text())


if __name__ == "__main__":
    unittest.main()
