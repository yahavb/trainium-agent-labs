"""Retry pinned P1 acceptance, then run sequential comparisons on one referee worker.

Source trees are isolated snapshots, not active checkouts. The P1 tree must be 434e5f9.
Run in a Trainium seat with root, the Neuron SDK, and vLLM already serving.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("p1", "p2", "p3", "out"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--core", type=int, required=True)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--base", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "state.json").exists():
        raise RuntimeError("Use a fresh output directory; refusing to duplicate a comparison run")
    roots = {name: Path(getattr(a, name)).resolve() / "projects/03-chipboost" for name in ("p1", "p2", "p3")}
    os.environ.update(CHIPBOOST_CORE=str(a.core), CHIPBOOST_SEAT="100")
    # Prevent another copy of this launcher from targeting the same core. The worker's runtime
    # allocation is the authoritative free-core check; an occupied core never falls back.
    import fcntl
    lock = open(f"/tmp/p1-comparison-core{a.core}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    state = {"phase": "preflight", "core": a.core, "pid": os.getpid(),
             "budget": a.budget, "repeat": a.repeat, "completed": [],
             "referee_commit": "434e5f9", "p2_commit": "919c6be", "p3_commit": "2ce9416"}

    def save():
        (out / "state.json").write_text(json.dumps(state, indent=2) + "\n")

    def event(**fields):
        with (out / "infrastructure.jsonl").open("a") as f:
            f.write(json.dumps(dict(logged_at=time.time(), **fields)) + "\n")

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    save()
    event(kind="prior_interruption", observed_at="2026-10-10T19:01:09Z", core=2,
          reason="NRT reported logical core 2 busy; check_isolated returned None",
          acceptance_case="isolated_AA", counted_as_kernel_attempt=False,
          source="Earlier direct referee diagnostic; no kernel verdict was recorded")
    sys.path.insert(0, str(roots["p1"]))
    import speedcheck as sc
    import schema
    digest = hashlib.sha256((roots["p1"] / "speedcheck.py").read_bytes()).hexdigest()
    assert digest == "5f366558ae933c88262bbde806afbcd4a568fdbfab4fc3395951f1b3799d71be"
    baseline = roots["p1"].parent / "02-kernel-agent/reference_level4.py"
    # The original acceptance script used read_text/write_text; preserve that candidate exactly.
    # Referee code_hash also hashes decoded, newline-normalized source, not raw checkout bytes.
    src = baseline.read_text().encode("utf-8")
    assert hashlib.sha1(src).hexdigest()[:12] == "638cfdf0f2fa"
    candidate = out / "acceptance_candidate.py"
    candidate.write_bytes(src)
    state.update(referee_sha256=digest, candidate_sha256=hashlib.sha256(src).hexdigest(),
                 baseline=str(baseline), baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest())
    with urllib.request.urlopen(a.base.rstrip("/") + "/models", timeout=15) as response:
        models = json.load(response)
    assert a.model in [m["id"] for m in models["data"]], "Requested model is not served"
    sys.path.insert(0, str(roots["p3"]))
    agent = load("comparison_agent", roots["p3"] / "agent.py")
    sys.path.insert(0, str(roots["p2"]))
    # P2 shapes registers extra ops using P2's nkibench. Keep those imports separate from the
    # pinned P1/P3 checker objects; the worker always imports from the untouched P1 tree.
    pinned_nkibench = sys.modules["nkibench"]
    try:
        sys.modules["nkibench"] = load("comparison_p2_nkibench", roots["p2"].parent / "02-kernel-agent/nkibench.py")
        search = load("comparison_search", roots["p2"] / "search.py")
    finally:
        sys.modules["nkibench"] = pinned_nkibench
    state["agent_sha256"] = hashlib.sha256((roots["p3"] / "agent.py").read_bytes()).hexdigest()
    state["search_sha256"] = hashlib.sha256((roots["p2"] / "search.py").read_bytes()).hexdigest()
    state["search_design"] = "P2 expert-template cap search; same timing baseline, different candidate prior from agent arms"

    with sc.RefereeWorker(core=a.core, baseline=str(baseline),
                          max_checks=1 + 3 * a.budget * a.repeat + 100) as worker:
        class Bridge:
            pending_start = False

            @property
            def last_error(self):
                return worker.last_error

            def check(self, path):
                if self.pending_start:
                    assert Path(path).read_bytes() == src, "Unexpected P3 startup candidate"
                    self.pending_start = False
                    return dict(acceptance_record)
                rec = worker.check(path)
                if rec is None:
                    event(kind="referee_failure", phase=state["phase"], core=a.core,
                          reason=worker.last_error, candidate_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                          counted_as_kernel_attempt=False)
                elif "the check timed out" in (rec.get("referee_message") or ""):
                    event(kind="watchdog_timeout", phase=state["phase"], core=a.core,
                          reason=rec["referee_message"], counted_as_kernel_attempt=True,
                          note="Referee classifies candidate timeout as wrong; not automatically an infrastructure failure")
                return rec

            def check_isolated(self, path, **kwargs):
                return self.check(path)

            def close(self):
                pass  # The launcher owns the one worker across all sequential arms.

        bridge = Bridge()
        state["phase"] = "acceptance_retry"
        save()
        rec = bridge.check(str(candidate))
        if rec is None:
            raise RuntimeError("Pinned core unavailable or referee failed; comparison not launched")
        assert not schema.validate(rec), schema.validate(rec)
        (out / "acceptance.json").write_text(json.dumps(rec, indent=2) + "\n")
        assert rec["verdict"] == "no_gain" and rec["sim_ok"] and rec["chip_ok"], rec
        acceptance_record = dict(rec)
        state.update(acceptance="passed", worker_pid=worker._p.pid)
        state["startup_baseline"] = "Reuse acceptance record for P3's unbudgeted startup measurement; all generated candidates are newly graded"
        save()
        print(f"ACCEPTANCE PASSED: core {a.core}, worker {worker._p.pid}, speedup {rec['speedup']:.6f}", flush=True)
        agent.pick_referee = lambda: ("speedcheck (single pinned RefereeWorker)", bridge)
        search.open_referee = lambda _: (bridge, "single pinned RefereeWorker")

        # Interleave the arm order across repeats; never run two graders at once.
        arms = ["referee", "model_alone", "random_search"]
        for repeat in range(a.repeat):
            for arm in arms[repeat % 3:] + arms[:repeat % 3]:
                tag = f"{arm}-r{repeat}"
                state["phase"] = tag
                save()
                logpath = out / f"{tag}.jsonl"
                print("START", tag, flush=True)
                with (out / f"{tag}.log").open("w", buffering=1) as log:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        if arm == "random_search":
                            sys.argv = ["search.py", "--budget", str(a.budget), "--seed", str(repeat),
                                        "--seat", "100", "--out", str(logpath), "--run-id", tag]
                            assert search.main() == 0
                        else:
                            bridge.pending_start = True
                            sys.argv = ["agent.py", "--arm", arm, "--budget", str(a.budget),
                                        "--repeat", "1", "--samples", "1", "--give-up-after", "0",
                                        "--start", str(baseline), "--base", a.base, "--model", a.model,
                                        "--seat", "100", "--log", str(logpath)]
                            agent.main()
                            assert not bridge.pending_start, "P3 did not consume the startup baseline"
                records = [json.loads(line) for line in logpath.read_text().splitlines() if line.strip()]
                assert len(records) == a.budget, f"{tag} incomplete budget; stopping comparison"
                assert all(not schema.validate(r) for r in records)
                state["completed"].append(tag)
                state.update(worker_pid=worker._p.pid if worker._p else None, worker_starts=worker.starts)
                save()
                print("DONE", tag, flush=True)
        state["phase"] = "complete"
        save()


if __name__ == "__main__":
    original_args = sys.argv[:]
    try:
        main()
    except BaseException as exc:
        if "--out" in original_args:
            status = Path(original_args[original_args.index("--out") + 1]) / "state.json"
            if status.exists():
                failed = json.loads(status.read_text())
                if failed.get("pid") == os.getpid():
                    failed.update(failed_phase=failed.get("phase"), phase="failed",
                                  error=f"{type(exc).__name__}: {exc}")
                    status.write_text(json.dumps(failed, indent=2) + "\n")
        raise
