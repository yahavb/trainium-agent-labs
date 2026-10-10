"""CPU-only regression of feedback selection; fixture source is parsed, never imported.

Run: python projects/03-chipboost/tests/test_feedback.py
The byte counts below are analytical expectations, not claimed hardware measurements.
"""
from pathlib import Path
import ast
import sys
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "02-kernel-agent"))
import nkibench

# Import just the real feedback definitions: speedcheck's module initialization needs
# ml_dtypes even though these CPU-only functions do not. Never execute fixture source.
definitions = {"_outer_reuse", "one_instruction", "WASTE_HINT", "FULL_MATMUL_FLOPS"}
tree = ast.parse((ROOT / "speedcheck.py").read_text())
nodes = [node for node in tree.body if
         (isinstance(node, ast.FunctionDef) and node.name in definitions) or
         (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in definitions
                                               for t in node.targets))]
sc = types.ModuleType("feedback_under_test")
sc.__dict__.update(ast=ast, np=np, nkibench=nkibench)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / "speedcheck.py"), "exec"), sc.__dict__)


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.args = [np.empty((256, 512), np.float16), np.empty((256, 1024), np.float16)]
        self.want = np.empty((512, 1024), np.float16)
        self.sources = {
            "h1": (ROOT.parent / "02-kernel-agent/reference_level4.py").read_text(),
            **{key: (ROOT / "tests/fixtures" / name).read_text() for key, name in
               (("h2", "h2_nm_order.py"), ("h3", "h3_hoist.py"), ("h4", "h4_rhs_hoist.py"))},
        }

    def instruction(self, key, *, chip=None, unmeasured=0, op="matmul", byte_count=None, src=None):
        # lhs bytes * N tiles + rhs bytes * M tiles + output, adjusted for each hoist.
        lhs, rhs, output = *(a.nbytes for a in self.args), self.want.nbytes
        counts = {"h1": lhs * 2 + rhs * 4 + output, "h2": lhs * 2 + rhs * 4 + output,
                  "h3": lhs + rhs * 4 + output, "h4": lhs * 2 + rhs + output}
        counter = dict(bytes=counts[key] if byte_count is None else byte_count, transfers=40,
                       by_op={"nc_matmul": 16}, unmeasured=unmeasured)
        return sc.one_instruction(counter, self.args, self.want, 2 * 256 * 512 * 1024,
                                  chip=chip, src=self.sources[key] if src is None else src, op=op)

    def test_reference_and_reordered_both_reload_rhs(self):
        for key in ("h1", "h2"):
            for speedup in (None, 0.3, 1.0, 2.0):
                chip = None if speedup is None else dict(speedup=speedup, threshold=1.01)
                text = self.instruction(key, chip=chip)
                self.assertIn("rhs slice depends on k and n", text)
                self.assertIn("distinct SBUF slots", text)
                self.assertIn("Keep the k contraction loop", text)
                self.assertNotIn("innermost loop", text)
                self.assertNotIn("existing lhsT reuse", text)

    def test_single_n_tile_is_not_evidence_of_hoisted_lhs(self):
        args = [np.empty((256, 512), np.float16), np.empty((256, 512), np.float16)]
        for key in ("h1", "h2"):
            text = sc._outer_reuse(self.sources[key], args, "matmul")
            self.assertIn("Make n the outer loop", text)
            self.assertNotIn("existing lhsT reuse", text)
        # Actual source evidence of an outside-n load still preserves the hoist.
        self.assertIn("existing lhsT reuse", sc._outer_reuse(self.sources["h3"], args, "matmul"))

    def test_lhs_hoist_preserves_reuse_and_targets_rhs(self):
        text = self.instruction("h3", chip=dict(speedup=2, threshold=1.01))
        self.assertIn("rhs slice depends on k and n", text)
        self.assertIn("existing lhsT reuse", text)
        self.assertNotIn("lhsT slice depends", text)

    def test_rhs_hoist_targets_remaining_lhs_even_at_114_percent(self):
        text = self.instruction("h4", chip=dict(speedup=1, threshold=1.01))
        self.assertIn("1.14x", text)
        self.assertIn("lhsT slice depends on k and m", text)
        self.assertIn("existing rhs reuse", text)

    def test_no_specific_claim_without_evidence(self):
        floor = sum(a.nbytes for a in self.args) + self.want.nbytes
        for kwargs in (dict(op="rmsnorm"), dict(unmeasured=1), dict(byte_count=floor),
                       dict(src="not valid python !"), dict(src="")):
            self.assertNotIn("slice depends", self.instruction("h1", **kwargs))
        # Hidden data dependencies and conditional loads are intentionally not diagnosed.
        source = self.sources["h2"].replace("n * TILE_N", "index * TILE_N").replace("m * TILE_M", "index * TILE_M")
        self.assertNotIn("slice depends", self.instruction("h2", src=source))
        body = "\n".join("    " + line for line in self.sources["h2"].splitlines())
        self.assertNotIn("slice depends", self.instruction("h2", src="def nki_matmul_tiled_(lhsT, rhs):\n" + body))
        body = "\n".join("    " + line for line in self.sources["h2"].split("def nki_matmul_tiled_(lhsT, rhs):\n", 1)[1].splitlines())
        conditional = "def nki_matmul_tiled_(lhsT, rhs):\n    if unknown:\n" + body
        self.assertNotIn("slice depends", self.instruction("h2", src=conditional))
        for changed in (self.sources["h2"].replace("K, M = lhsT.shape", "k = m\n  K, M = lhsT.shape"),
                        self.sources["h2"].replace("nl.tile_size.gemm_stationary_fmax", "512"),
                        self.sources["h2"].replace("nl.affine_range(M // TILE_M)", "nl.affine_range(1)")):
            self.assertNotIn("slice depends", self.instruction("h2", src=changed))

    def test_deep_expression_gives_no_claim_instead_of_crashing(self):
        # Legal Python that compiles, but is too deep for a recursive AST walk: no diagnosis, no crash.
        deep = "z = " + " + ".join(["0"] * 3000)
        source = self.sources["h2"].replace("K, M = lhsT.shape", f"K, M = lhsT.shape\n  {deep}")
        self.assertIn(deep, source)
        self.assertNotIn("slice depends", self.instruction("h2", src=source))

    def test_source_is_never_executed(self):
        source = "raise RuntimeError('MUST NOT EXECUTE')\n" + self.sources["h2"]
        self.assertIn("rhs slice depends", self.instruction("h2", src=source))


if __name__ == "__main__":
    unittest.main()
