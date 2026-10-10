"""Replay the historical P1 kernel suite through the public, sandboxed referee API.

Fixtures are never imported by this driver. Root + setpriv are required by the referee.
This does not execute the separate resource-exhaustion / sandbox-bypass probes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--referee-root", required=True)
    ap.add_argument("--fixtures", nargs="+", required=True)
    ap.add_argument("--core", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.referee_root).resolve()
    out = Path(a.out).resolve()
    out.mkdir(exist_ok=False)
    os.environ.update(CHIPBOOST_CORE=str(a.core), CHIPBOOST_SEAT="100")
    os.environ.pop("CHIPBOOST_ALLOW_UNSANDBOXED", None)
    sys.path.insert(0, str(root))
    import speedcheck as sc
    import schema
    old = json.loads((root / "results_p1.json").read_text())["redteam"]
    assert len(old) == 33, "Unexpected historical manifest"
    sources = {}
    for directory in a.fixtures:
        for p in Path(directory).glob("*.py"):
            if p.stem in {r["cheat"] for r in old}:
                assert p.stem not in sources, p.stem
                sources[p.stem] = p
    assert set(sources) == {r["cheat"] for r in old}, "Missing historical fixture"
    report = {"referee_version": "P1 throughput referee; source SHA256 recorded", "seat": 100,
              "core": a.core, "source_sha256": hashlib.sha256((root / "speedcheck.py").read_bytes()).hexdigest(),
              "phase": "running", "redteam": [], "fixture_sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k,v in sources.items()},
              "note": "33 historical fixtures; c3a wrong off timing shapes may be no_gain/slower but must never be faster. Resource-exhaustion probes are not included."}

    def save():
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    with sc.RefereeWorker(core=a.core, max_checks=100,
                          baseline=str(root.parent / "02-kernel-agent/reference_level4.py")) as worker:
        for previous in old:
            name = previous["cheat"]
            start = time.monotonic()
            rec = None
            for retry in range(3):
                rec = worker.check(str(sources[name]))
                if rec is not None:
                    break
                with (out / "infrastructure.jsonl").open("a") as f:
                    f.write(json.dumps({"timestamp": time.time(), "fixture": name, "retry": retry,
                                        "error": worker.last_error, "counted_as_kernel_verdict": False}) + "\n")
                time.sleep(10)
            if rec is None:
                report.update(phase="infrastructure_failure", failed_fixture=name)
                save()
                raise RuntimeError(worker.last_error)
            assert not schema.validate(rec), schema.validate(rec)
            accepted = rec["verdict"] in ("faster", "slower", "no_gain")
            honest = previous["honest"]
            ok = accepted if honest else (not accepted or (name == "c3a_shape_branch" and rec["verdict"] in ("no_gain", "slower")))
            row = dict(cheat=name, description=previous["description"], honest=honest,
                       caught=not accepted if not honest else False, ok=ok, verdict=rec["verdict"],
                       stage="rules" if rec["verdict"]=="rules" else "heldout" if rec["verdict"]=="heldout_fail" else "accepted" if accepted else "simulator" if not rec["sim_ok"] else "chip",
                       message=rec["referee_message"], record=rec, seconds=time.monotonic()-start, run="throughput-regression")
            report["redteam"].append(row)
            save()
            print(name, rec["verdict"], "PASS" if ok else "FAIL", flush=True)
    report.update(phase="complete", passed=all(r["ok"] for r in report["redteam"]))
    save()
    if not report["passed"]:
        raise SystemExit(1)
    print("33-CASE REGRESSION PASSED", flush=True)


if __name__ == "__main__":
    main()
