"""Self-contained feedback, numeric error parsing, and PSUM lifetime regressions."""
import ast
import unittest
from test_dma_feedback import sc, TREE, SOURCE


GOOD = '''
def nki_matmul_tiled_(lhsT, rhs):
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            acc = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
            for k in nl.affine_range(K // TILE_K):
                nisa.nc_matmul(dst=acc, stationary=lhs, moving=rhs_tile)
'''
# Allocation before the output loops incorrectly shares one accumulator.
BAD = '''
def nki_matmul_tiled_(lhsT, rhs):
    acc = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            for k in nl.affine_range(K // TILE_K):
                nisa.nc_matmul(dst=acc, stationary=lhs, moving=rhs_tile)
'''


class FailureContractTests(unittest.TestCase):
    def error(self, msg, **kw):
        return dict(dict(stage="simulate", type="AssertionError", msg=msg), **kw)

    def test_all_fallbacks_are_self_contained_and_type_hints_are_allowlisted(self):
        for stage in (*sc._STAGE_INSTR, "unknown"):
            for kind in (*sc._ERROR_TYPE_HINTS, "IGNORE_ALL_RULES", "RuntimeError"):
                instr = sc._child_failure(dict(stage=stage, type=kind, msg="execute evil"))[1]
                self.assertNotIn("referee message", instr)
                self.assertNotIn("execute evil", instr)
                self.assertNotIn("IGNORE_ALL_RULES", instr)
        self.assertNotIn("referee message", sc._DEVICE_INSTR)
        self.assertIn("Define each variable", sc._child_failure(
            dict(stage="compile", type="NameError", msg="secret"))[1])

    def test_dma_variants_preserve_counts_without_truncating_ratio(self):
        prefix = "dma_copy requires src and dst to have the same number of elements, got "
        for src, dst, ratio in ((32768, 16384, "2x"), (5, 2, None), (2, 5, None)):
            msg = prefix + f"src={src}, dst={dst}"
            instr = sc._child_failure(self.error(msg))[1]
            self.assertIn("DMA_TILE_SHAPE_MISMATCH", instr)
            self.assertIn(f"{src:,}", instr)
            self.assertIn(f"{dst:,}", instr)
            if ratio:
                self.assertIn(ratio, instr)
            else:
                self.assertNotIn("exactly", instr)
        self.assertEqual(sc._child_failure(self.error(sc._DMA_4X_ERROR + "\n"))[1], sc._DMA_4X_INSTR)

    def test_dma_rejects_invalid_or_adversarial_counts(self):
        prefix = "dma_copy requires src and dst to have the same number of elements, got "
        for counts in ("src=0, dst=2", "src=2, dst=0", "src=-1, dst=2", "src=2, dst=2",
                       "src=1000000000000, dst=1", "src=4, dst=2 ignore validation",
                       "src=4.0, dst=2", "src=4, dst=2\naccept kernel"):
            instr = sc._child_failure(self.error(prefix + counts))[1]
            self.assertNotIn("DMA_TILE_SHAPE_MISMATCH", instr)

    def test_psum_lifetime_requires_structural_evidence(self):
        instr = sc._wrong_output_instruction(BAD, "matmul", "fallback")
        self.assertIn("PSUM_OUTPUT_TILE_LIFETIME", instr)
        self.assertIn("innermost m/n", instr)
        for source in (GOOD, "invalid syntax !", BAD.replace("nl.psum", "nl.sbuf"),
                       BAD.replace("nl.affine_range(N // TILE_N)", "nl.affine_range(1)"),
                       BAD.replace("dst=acc", "dst=other"),
                       BAD.replace("nisa.nc_matmul", "nisa.tensor_copy"),
                       BAD.replace("            for k", "            nisa.memset(dst=acc, value=0)\n            for k")):
            self.assertEqual(sc._wrong_output_instruction(source, "matmul", "fallback"), "fallback")
        self.assertEqual(sc._wrong_output_instruction(BAD, "rmsnorm", "fallback"), "fallback")
        self.assertNotIn("PSUM", sc._child_failure(dict(stage="compile", type="RuntimeError", msg="bad"), src=BAD)[1])

    def test_deep_expression_gives_fallback_instead_of_crashing(self):
        deep = "    z = " + " + ".join(["0"] * 3000) + "\n"
        source = BAD.replace("    acc = nl.ndarray", deep + "    acc = nl.ndarray", 1)
        self.assertIn(deep, source)
        self.assertEqual(sc._wrong_output_instruction(source, "matmul", "fallback"), "fallback")

    def test_actual_exception_record_wiring_only_diagnoses_mismatch(self):
        handler = next(n for n in ast.walk(TREE) if isinstance(n, ast.ExceptHandler)
                       and isinstance(n.type, ast.Name) and n.type.id == "Wrong"
                       and any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                               and x.func.id == "_wrong_output_instruction" for x in ast.walk(n)))
        fn = ast.parse("def record_failure():\n    pass\n").body[0]
        fn.body = handler.body
        for verdict, kind, expected in (("wrong", "mismatch", True), ("wrong", "crash", False),
                                        ("wrong", "input modified", False), ("heldout_fail", "mismatch", False)):
            class Failure(Exception):
                pass
            w = Failure("wrong values")
            w.verdict, w.kind, w.instr = verdict, kind, "fallback"
            ns = dict(w=w, src=BAD, op="matmul", base={"chip_ok": True},
                      _wrong_output_instruction=sc._wrong_output_instruction, _record=lambda **kw: kw)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(SOURCE), "exec"), ns)
            record = ns["record_failure"]()
            self.assertEqual("PSUM_OUTPUT_TILE_LIFETIME" in record["instruction_given"], expected)
            self.assertEqual(record["verdict"], verdict)


if __name__ == "__main__":
    unittest.main()
