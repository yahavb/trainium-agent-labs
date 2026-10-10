"""P1 hardware acceptance check. Run as root in a Trainium seat, outside active jobs.

python validate_p1.py --core 2 --out /tmp/p1-validation.json
Uses the shipped reference, an intentionally narrow baseline, and temporary candidates.
Does not replace the historical adversarial regression suite in results_p1.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import schema
import speedcheck as sc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--core", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["CHIPBOOST_CORE"] = str(a.core)
    here = Path(__file__).resolve().parent
    report = {"source_sha256": hashlib.sha256((here / "speedcheck.py").read_bytes()).hexdigest(),
              "core": a.core, "checks": []}

    def save():
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n")

    def grade(name, call, expected):
        start = time.monotonic()
        rec = call()
        item = {"name": name, "seconds": time.monotonic() - start, "record": rec}
        report["checks"].append(item)
        save()
        assert rec is not None, name + ": referee failure"
        assert not schema.validate(rec), (name, schema.validate(rec))
        assert rec["verdict"] == expected, (name, rec)
        print(name, expected, round(item["seconds"], 2), rec.get("speedup"), flush=True)
        return rec

    # Exercise the threaded input path and its reproducibility across calls.
    x = sc._normal(np.random.SeedSequence(123), (1024, 1024))
    y = sc._normal(np.random.SeedSequence(123), (1024, 1024))
    assert np.array_equal(x, y) and x.dtype == np.float32
    assert abs(float(x.mean())) < .01 and abs(float(x.std()) - 1) < .01
    report["threaded_inputs"] = "pass"
    with tempfile.TemporaryDirectory(prefix="p1_candidates_") as d:
        d = Path(d)
        src = (here.parent / "02-kernel-agent/reference_level4.py").read_text()
        candidate = d / "honest.py"
        candidate.write_text(src)
        narrow = d / "narrow.py"
        narrow.write_text(src.replace("TILE_N = nl.tile_size.gemm_moving_fmax", "TILE_N = 128"))
        forbidden = d / "forbidden.py"
        forbidden.write_text("import os\n" + src)
        wrong = d / "wrong.py"
        wrong.write_text(src.replace("return result", "return lhsT"))
        grade("isolated_AA", lambda: sc.check_isolated(str(candidate)), "no_gain")
        with sc.RefereeWorker(core=a.core, max_checks=3) as w:
            grade("worker_AA_cold", lambda: w.check(str(candidate)), "no_gain")
            grade("worker_AA_warm", lambda: w.check(str(candidate)), "no_gain")
            assert w.starts == 1
            grade("rules_rejection", lambda: w.check(str(forbidden)), "rules")
            grade("recycle_and_wrong_output", lambda: w.check(str(wrong)), "wrong")
            assert w.starts == 2
            w.timeout = .001
            rec = grade("watchdog", lambda: w.check(str(candidate)), "wrong")
            assert "timed out" in rec["referee_message"]
            w.timeout = 1800
            grade("watchdog_recovery", lambda: w.check(str(candidate)), "no_gain")
            assert w.starts == 3
        grade("faster_with_heldout", lambda: sc.check_isolated(str(candidate), baseline=str(narrow)), "faster")
        grade("slower", lambda: sc.check_isolated(str(narrow)), "slower")
    report["passed"] = True
    save()
    print("P1 ACCEPTANCE PASSED", flush=True)


if __name__ == "__main__":
    main()
