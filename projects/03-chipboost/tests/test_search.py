#!/usr/bin/env python3
"""
tests/test_search.py -- search.py's logic, no chip and no NKI needed. Plain asserts.

    python tests/test_search.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import schema  # noqa: E402
import search  # noqa: E402
import shapes  # noqa: E402


def test_space_at_the_primary_shape():
    case = shapes.cases("matmul", "timing")[0]
    tiles = search.tile_counts(case)
    assert tiles == (2, 12, 32), tiles
    space = search.candidate_space(tiles)
    assert len(space) == 72, len(space)
    fit = [t for t in space if search.sbuf_bytes_per_partition(*t, case["M"]) <= search.SBUF_LIMIT]
    assert len(fit) == 62, len(fit)     # REVIEW.md counted 62 independently
    assert search.effective((16, 2, 8), tiles) == (2, 2, 8)


def test_plan_is_seeded_and_never_shortened():
    pool = [(1, n, k) for n in (1, 2, 3) for k in (1, 2, 4)]
    a = search.plan(pool, (16, 2, 8), 5, seed=0)
    assert a == search.plan(pool, (16, 2, 8), 5, seed=0), "same seed, same order"
    assert a[0] == (16, 2, 8) and len(set(a[1:])) == 4, a
    assert a != search.plan(pool, (16, 2, 8), 5, seed=1), "another seed, another order"
    try:
        search.plan(pool, (16, 2, 8), len(pool) + 2, seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("a budget larger than the pool was silently accepted")


def test_cap_lines_rewrite_and_fail_loudly():
    src = open(os.path.join(ROOT, search.TEMPLATE)).read()
    assert search.read_caps(src) == (16, 2, 8)
    assert search.read_caps(search.rewrite_caps(src, (2, 3, 4))) == (2, 3, 4)
    for broken in (src.replace("TILES_IN_BLOCK_N = 2", ""),
                   src + "\nTILES_IN_BLOCK_K = 4\n"):
        try:
            search.rewrite_caps(broken, (1, 1, 1))
        except ValueError:
            continue
        raise AssertionError("a template without exactly one line per cap was accepted")


def test_referee_failures_are_retried_never_logged():
    calls = []

    class FakeReferee:
        def __init__(self, answers):
            self.answers = list(answers)

        def check_isolated(self, path, op, baseline):
            calls.append(path)
            return self.answers.pop(0)

    rec = {"verdict": "slower"}
    assert search.call_referee(FakeReferee([None, None, rec]), "c.py", wait=0) is rec
    assert search.call_referee(FakeReferee([None, None, None]), "c.py", wait=0) is None
    assert len(calls) == 6, calls


def test_stub_run_writes_valid_fenced_records():
    run_id = "test-stub-run"
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "attempts.jsonl")
        p = subprocess.run([sys.executable, os.path.join(ROOT, "search.py"), "--stub", "--budget", "5",
                            "--seed", "3", "--run-id", run_id, "--out", out],
                           capture_output=True, text=True)
        shutil.rmtree(os.path.join(search.RUNS_ROOT, run_id), ignore_errors=True)
        assert not os.path.exists(os.path.join(ROOT, "search_runs", run_id)), "candidates inside the referee tree"
        assert p.returncode == 0, p.stdout + p.stderr
        assert "STUB REFEREE" in p.stdout, "stub output must be fenced"
        lines = open(out).read().splitlines()
        assert len(lines) == 5, len(lines)
        for line in lines:
            rec = json.loads(line)
            assert schema.validate(rec) == [], schema.validate(rec)
            assert rec["arm"] == "random_search" and rec["source"] == "sim" and rec["run_id"] == run_id
            assert rec["code"] and rec["code_hash"] and rec["prompt"] is None
        assert [json.loads(l)["attempt_no"] for l in lines] == [0, 1, 2, 3, 4]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
