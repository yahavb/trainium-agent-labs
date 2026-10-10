"""Harness self-test: hand-written kernels must pass; each deliberate bug must be caught with a
message that names its cause. If the harness can't catch our bugs it won't catch the model's."""
from pathlib import Path

import pytest

from kagent.harness import verify
from kagent.levels import LEVELS

K = Path(__file__).parent.parent / "kernels"


def run(level, path):
    return verify(LEVELS[level], (K / path).read_text())


def level_of(path):
    return int(Path(path).name[1:].split("_")[0])


@pytest.mark.parametrize("level,path", [(1, "hand/l1_relu_affine.py"), (2, "hand/l2_row_sum.py"),
                                        (3, "hand/l3_row_max.py"), (4, "hand/l4_rmsnorm.py"),
                                        (5, "hand/l5_softmax.py"), (6, "hand/l6_transpose.py"),
                                        (7, "hand/l7_matmul.py"), (8, "hand/l8_layernorm.py"),
                                        (8, "hand/l8_layernorm_f32_two_pass.py"),
                                        (9, "hand/l9_band_attention.py"), (10, "hand/l10_conv1d.py")])
def test_correct_kernels_pass(level, path):
    rep = run(level, path)
    assert rep.passed, rep.text()
    assert rep.bound_use() < 0.5  # comfortably inside the proven float32 bound
    assert verify(LEVELS[level], (K / path).read_text(), holdout=True).passed


def test_float32_accumulation_is_within_the_proven_bound():
    # sum of N float32 terms: |s_hat - s| <= gamma_(N-1) sum|x|. A float32 accumulator is a
    # correct float32 algorithm, so it must pass; it just spends more of its bound than float64.
    f32, f64 = run(2, "hand/l2_row_sum_f32acc.py"), run(2, "hand/l2_row_sum.py")
    assert f32.passed and f64.passed
    assert 0.05 < f32.bound_use() <= 1 and f64.bound_use() < 0.05


# (level, file, expected violation kinds, substring the feedback must contain)
BROKEN = [
    (1, "l1_whole_array.py", {"whole-array"}, None),
    (1, "l1_mask.py", {"fancy-index"}, None),
    (1, "l1_skip_partial_col.py", set(), "'partial-col-tile'"),
    (1, "l1_off_by_one_rows.py", set(), "only the last row"),
    (2, "l2_np_sum.py", {"sum-like"}, None),
    (2, "l2_tile_width_assumed.py", set(), "'partial-col-tile'"),
    (3, "l3_zero_init.py", set(), "every wrong output is exactly 0"),
    (3, "l3_sentinel_init.py", set(), "every failing case is 'huge-negative'"),
    # found in the first live run: the per-row max is overwritten for each column tile
    (3, "l3_overwrite_per_tile.py", set(), "equals the result over the last column tile only"),
    (4, "l4_float32_squares.py", set(), "'huge-magnitude'"),
    (4, "l4_per_tile_rms.py", set(), "statistic of that tile alone"),
    (5, "l5_no_max_subtraction.py", set(), "non-finite"),
    (5, "l5_per_tile_softmax.py", set(), "statistic of that tile alone"),
    (6, "l6_skip_partial.py", set(), "never written"),
    (6, "l6_tile_origin.py", set(), "'multi-row-tile'"),
    (7, "l7_drop_k_tail.py", set(), "'partial-k-tile'"),
    (7, "l7_matmul_operator.py", {"whole-array", "matmul-op"}, None),
    (8, "l8_naive_variance.py", set(), "mean is large compared to their spread"),
    (9, "l9_band_edge.py", set(), "negative dimensions"),
    (9, "l9_causal.py", set(), "one-sided (causal) window"),
    (9, "l9_no_max_subtraction.py", set(), "non-finite"),
    (10, "l10_ignore_dilation.py", set(), "the dilation is ignored"),
    # one case's slice comes out length 1 and numpy broadcasts it: a real broadcast violation
    (10, "l10_lout_off_by_one.py", {"broadcast"}, "ValueError"),
    (10, "l10_drop_tail.py", set(), "last partial col tile"),
]


@pytest.mark.parametrize("level,path,kinds,needle", BROKEN)
def test_broken_kernels_caught(level, path, kinds, needle):
    rep = run(level, "broken/" + path)
    assert not rep.passed
    assert {v.kind for v in rep.violations} == kinds, rep.text()
    if needle:
        assert needle in rep.text(), rep.text()
    for v in rep.violations:
        assert v.line is not None, v  # every violation must point at a line


def test_static_catches_unreached_code():
    src = ("import numpy as np\n\ndef kernel(x):\n    if x.shape[0] > 10**9:\n"
           "        return np.sum(x, axis=1)\n    return x[:, 0]\n")
    rep = verify(LEVELS[2], src)
    assert any(v.kind == "sum-like" and v.line == 5 for v in rep.violations)


def test_banned_imports_and_syntax():
    assert verify(LEVELS[1], "import scipy\ndef kernel(x, a, b):\n    return x\n").violations[0].kind == "import"
    rep = verify(LEVELS[1], "def kernel(x, a, b)\n    return x\n")
    assert rep.load_error and "syntax" in rep.failure_classes()


def test_cost_counts_rereads():
    good = run(1, "hand/l1_relu_affine.py").cost
    assert good["flops"] == good["min_flops"] and good["hbm_read"] == good["min_read"]
    # loads every tile twice: same answer, double the HBM traffic
    src = (K / "hand/l1_relu_affine.py").read_text().replace(
        "out[i0:i1, j0:j1] = np.maximum(x[i0:i1, j0:j1] * a + b, 0.0)",
        "t = x[i0:i1, j0:j1] * a\n            out[i0:i1, j0:j1] = np.maximum(t + b + 0.0 * x[i0:i1, j0:j1], 0.0)")
    c = verify(LEVELS[1], src).cost
    assert c["hbm_read"] == 2 * c["min_read"] and c["flops"] > c["min_flops"]


@pytest.mark.parametrize("level,path,key", [
    (3, "l3_overwrite_per_tile.py", "numeric:tile-overwrite"),
    (4, "l4_float32_squares.py", "numeric:tag:huge-magnitude"),
    (4, "l4_per_tile_rms.py", "numeric:per-tile-stat"),
    (5, "l5_no_max_subtraction.py", "numeric:nonfinite"),
    (5, "l5_per_tile_softmax.py", "numeric:per-tile-stat"),
    (6, "l6_skip_partial.py", "numeric:partial-row"),
    (7, "l7_drop_k_tail.py", "numeric:tag:partial-k-tile"),
    (8, "l8_naive_variance.py", "numeric:cancellation"),
    # live L4: two bugs at once; the verified hypothesis covers 6 of 9 wrong cases and goes first
    (4, "l4_live_two_bugs.py", "numeric:per-tile-stat"),
    (9, "l9_causal.py", "numeric:one-sided"),
    (9, "l9_no_max_subtraction.py", "numeric:nonfinite"),
    (10, "l10_ignore_dilation.py", "numeric:dilation"),
    (10, "l10_drop_tail.py", "numeric:partial-col"),
])
def test_translator_names_the_cause(level, path, key):
    from kagent.translate import translate
    assert translate(run(level, "broken/" + path)).key == key


def test_numpy_alias_on_an_unreached_branch_is_caught():
    # red team, PR #2: `xp = np; xp.max` on a branch the tests never take passed with 0 violations
    rep = run(3, "redteam/l3_unreached_alias_max.py")
    assert not rep.passed and any(v.kind == "max-like" for v in rep.violations), rep.text()


# the broadcast crash is raised on the banned line itself, so it now comes back as the rule (whose fix
# is the same column loop) rather than as a crash
@pytest.mark.parametrize("path,key", [("l4_live_broadcast_crash.py", "rule:broadcast"),
                                      ("l4_live_reshape.py", "rule:layout")])
def test_rownorm_broadcast_gets_the_column_loop(path, key):
    # v2 L4: 13 of 37 failures cycled between a broadcast crash and a .reshape violation, and the
    # layout fix showed a transpose loop; both must now point at the per-column form
    from kagent.translate import translate
    d = translate(run(4, "broken/" + path))
    assert d.key == key and "out[i0:i1, j] = x[i0:i1, j] * rinv" in d.instruction, d.instruction


def test_judge_directive_maps_the_organizers_failures():
    from kagent.translate import judge_directive
    lv = LEVELS[4]
    assert judge_directive(lv, "RAISED: TypeError: kernel() missing 1 required positional argument: 'g'").key == "judge:signature"
    assert judge_directive(lv, "large x: NUMERICAL MISMATCH: worst relative error 0.0008").key == "judge:precision"
    assert "kernel(x, eps=1e-6)" in judge_directive(lv, "RAISED: TypeError: x").instruction

@pytest.mark.parametrize("lv,path", [(8, "l8_live_whole_array.py"), (9, "l9_live_dot_crash.py")])
def test_many_violations_get_a_rewrite_in_the_legal_loop_shape(lv, path):
    # live L8/L9 drafts break 5-15 rules at once; one directive per kind never finished in budget
    from kagent.translate import translate, REWRITE_AT
    rep = run(lv, "broken/" + path)
    assert len(rep.violations) >= REWRITE_AT
    d = translate(rep)
    assert d.key == "rule:rewrite" and d.structural, d.instruction
    assert ("rinv" in d.instruction) if lv == 8 else ("lo + t" in d.instruction)


def test_crash_on_a_banned_line_is_reported_as_the_rule():
    # live v2 L9 run 1: `np.dot(...)` raised a shape error; v2 asked to fix the crash three times
    from kagent.translate import translate, _crash_line
    import kagent.translate as t
    rep = run(9, "broken/l9_live_dot_crash.py")
    crash = next(r for r in rep.results if r.status == "error")
    assert _crash_line(crash.msg) in {v.line for v in rep.violations}
    old, t.REWRITE_AT = t.REWRITE_AT, 99  # isolate this rule from the rewrite threshold
    try:
        assert translate(rep).key.startswith("rule:")
    finally:
        t.REWRITE_AT = old


def test_few_violations_are_still_repaired_one_at_a_time():
    from kagent.translate import translate
    d = translate(run(2, "broken/l2_np_sum.py"))
    assert d.key == "rule:sum-like"


def test_near_miss_after_rewrite_gets_the_skeleton_line():
    # live v3-rewrite L9: right numbers, but np.dot over a (257, d) window; name the legal score line
    from kagent.translate import translate, REWRITE_AT
    rep = run(9, "broken/l9_live_after_rewrite.py")
    assert 0 < len(rep.violations) < REWRITE_AT
    d = translate(rep)
    assert d.key.startswith("rule:") and "s[c0:c1] += q[i, c] * k[lo + c0:lo + c1, c]" in d.instruction, d.instruction


def test_layernorm_down_the_columns_is_named():
    # live v3 L8: rule-clean, every output wrong, translator could only say numeric:other (15x)
    from kagent.translate import translate
    d = translate(run(8, "broken/l8_live_column_stats.py"))
    assert d.key == "numeric:wrong-axis" and "normalises each ROW" in d.instruction, d.instruction


def test_column_slice_longer_than_a_tile_is_a_violation():
    # red team on live v3-rewrite L9: k[lo:hi+1, c] read 259 rows and x[:, j] read 131 rows, 0 violations
    from kagent.harness import verify
    probe = ("import numpy as np\n\ndef kernel(x):\n    M, N = x.shape\n    out = np.zeros(M, dtype=np.float32)\n"
             "    for j in range(N):\n        out += x[:, j]\n    return out\n")
    rep = verify(LEVELS[2], probe)
    assert not rep.passed and any(v.kind == "whole-array" and "column slice" in v.what for v in rep.violations), rep.text()


def test_empty_value_crash_names_global_vs_local_columns():
    # live v3 L6: y[j:end_j, r] = tile[r - i, j:end_j] crashed 3 runs in a row as a generic ValueError
    from kagent.translate import translate
    d = translate(run(6, "broken/l6_live_global_in_tile.py"))
    # the crash is on the line the broadcast rule flags, so it is reported as that rule
    assert d.key == "rule:broadcast" and "EMPTY" in d.instruction and "out[j0:j1, i] = x[i, j0:j1]" in d.instruction, d.instruction


def test_matmul_whole_array_near_miss_names_the_tiled_operands():
    from kagent.translate import translate
    d = translate(run(7, "broken/l7_live_whole_array.py"))
    assert d.key.startswith("rule:") and ("b[k0:k1, j0:j1]" in d.instruction), d.instruction


@pytest.mark.parametrize("lv", [5, 6, 7])
def test_cycling_rewrite_is_available_for_l5_to_l7(lv):
    from kagent.translate import rewrite_directive
    rep = run(lv, {5: "broken/l5_per_tile_softmax.py", 6: "broken/l6_live_global_in_tile.py",
                   7: "broken/l7_live_whole_array.py"}[lv])
    d = rewrite_directive(rep, cycling=True)
    assert d is not None and d.key == "rule:rewrite" and "going round in circles" in d.instruction
