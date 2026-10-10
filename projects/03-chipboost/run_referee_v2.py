"""Run the DMA-feedback-only referee treatment after the pinned comparison completes."""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ORIGINAL_REFEREE = "5f366558ae933c88262bbde806afbcd4a568fdbfab4fc3395951f1b3799d71be"
BASELINE = "0dde78bd65e0a724a4fab05b2b893b215c8335e2086bc6c945ed43250a1f728b"
CANDIDATE = "656515ac59183ba6a9dd6a716e992b386871389e9e4a6447f459522c4ad820e6"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_digest(path):
    """SHA-256 in the CRLF form the pins were taken in (seat-100's snapshots); git checkouts are LF."""
    lf = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()


def check_pure_delta(original, treatment):
    """Only the two DMA constants and _child_failure may differ semantically."""
    allowed = {"_DMA_4X_ERROR", "_DMA_4X_INSTR"}
    def unchanged(source):
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.FunctionDef) and node.name == "_child_failure"
            or isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id in allowed)]
        return ast.dump(tree, include_attributes=False)
    assert unchanged(original) == unchanged(treatment), "Treatment changes more than the DMA mapping"


def preflight(a):
    previous = json.loads((Path(a.predecessor) / "state.json").read_text())
    expected = {f"{arm}-r{r}" for arm in ("referee", "model_alone", "random_search") for r in range(3)}
    assert previous["phase"] == "complete", "The original comparison must complete before v2"
    assert set(previous["completed"]) == expected and len(previous["completed"]) == 9
    for key, value in dict(core=a.core, budget=8, repeat=3, referee_commit="434e5f9",
                           p2_commit="919c6be", p3_commit="2ce9416", acceptance="passed",
                           referee_sha256=ORIGINAL_REFEREE, baseline_sha256=BASELINE,
                           candidate_sha256=CANDIDATE).items():
        assert previous.get(key) == value, f"Predecessor {key} differs"
    p1 = Path(a.p1).resolve() / "projects/03-chipboost"
    p3 = Path(a.p3).resolve() / "projects/03-chipboost"
    baseline = p1.parent / "02-kernel-agent/reference_level4.py"
    assert len(a.expected_referee_sha256) == 64
    assert a.expected_referee_sha256 != ORIGINAL_REFEREE, "Treatment must contain the DMA mapping"
    assert pinned_digest(p1 / "speedcheck.py") == a.expected_referee_sha256
    original_referee = Path(previous["baseline"]).parent.parent / "03-chipboost/speedcheck.py"
    assert pinned_digest(original_referee) == ORIGINAL_REFEREE
    check_pure_delta(original_referee.read_text(), (p1 / "speedcheck.py").read_text())
    assert pinned_digest(p3 / "agent.py") == previous["agent_sha256"], "Keep original P3 agent"
    assert pinned_digest(baseline) == BASELINE
    src = baseline.read_text().encode("utf-8")
    assert hashlib.sha256(src).hexdigest() == CANDIDATE
    # Verify actual records as well as the terminal status marker.
    for tag in sorted(expected):
        records = [json.loads(line) for line in (Path(a.predecessor) / f"{tag}.jsonl").read_text().splitlines() if line.strip()]
        assert len(records) == 8 and all(r["arm"] == tag.rsplit("-r", 1)[0] for r in records)
    return previous, p1, p3, baseline, src


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("p1", "p3", "predecessor", "out", "expected-referee-sha256"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--core", type=int, required=True)
    ap.add_argument("--wait-for-predecessor", action="store_true",
                    help="Queue with a durable status file; poll every 30 seconds until original completes")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError("Use an empty output directory")
    waiting = dict(phase="waiting_for_predecessor", pid=os.getpid(), core=a.core,
                   predecessor=str(Path(a.predecessor).resolve()))
    def waiting_save():
        (out / "state.json").write_text(json.dumps(waiting, indent=2) + "\n")
    try:
        if a.wait_for_predecessor:
            waiting_save()
            while True:
                try:
                    prior = json.loads((Path(a.predecessor) / "state.json").read_text())
                except json.JSONDecodeError:
                    # The original writer truncates then writes rather than using atomic rename.
                    time.sleep(30)
                    continue
                if prior["phase"] == "failed":
                    raise RuntimeError("Predecessor failed; v2 not launched")
                if prior["phase"] == "complete":
                    break
                time.sleep(30)
        # Completion is saved just before the old process closes its worker and lock.
        # Block on that short tail, then recheck the complete experiment under the lock.
        previous, p1, p3, baseline, src = preflight(a)
        import fcntl
        lock = open(f"/tmp/p1-comparison-core{a.core}.lock", "w")
        waiting["phase"] = "waiting_for_core_lock"
        waiting_save()
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous, p1, p3, baseline, src = preflight(a)
    except BaseException as exc:
        waiting.update(failed_phase=waiting["phase"], phase="failed", error=f"{type(exc).__name__}: {exc}")
        waiting_save()
        raise
    os.environ.update(CHIPBOOST_CORE=str(a.core), CHIPBOOST_SEAT="100")
    state = dict(phase="preflight", core=a.core, pid=os.getpid(), budget=8, repeat=3,
                 completed=[], treatment="dma_shape_feedback_only", referee_parent_commit="434e5f9",
                 referee_sha256=a.expected_referee_sha256, original_referee_sha256=ORIGINAL_REFEREE,
                 p3_commit="2ce9416", agent_sha256=previous["agent_sha256"],
                 baseline_sha256=BASELINE, candidate_sha256=CANDIDATE,
                 predecessor=str(Path(a.predecessor).resolve()), model="Qwen/Qwen3-8B",
                 base="http://localhost:8000/v1", temperature=0.6, top_p=0.95,
                 max_tokens=2500, context=8192, think=False, samples=1,
                 seed_policy="Original P3 sends no model seed; independent stochastic repeats, not seed paired")

    def save():
        (out / "state.json").write_text(json.dumps(state, indent=2) + "\n")

    def event(**fields):
        with (out / "infrastructure.jsonl").open("a") as f:
            f.write(json.dumps(dict(logged_at=time.time(), **fields)) + "\n")

    save()
    try:
        sys.path.insert(0, str(p1))
        import speedcheck as sc
        import schema
        sys.path.insert(0, str(p3))
        spec = importlib.util.spec_from_file_location("v2_agent", p3 / "agent.py")
        agent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent)
        with urllib.request.urlopen(state["base"] + "/models", timeout=15) as response:
            assert state["model"] in [m["id"] for m in json.load(response)["data"]]
        candidate = out / "acceptance_candidate.py"
        candidate.write_bytes(src)
        with sc.RefereeWorker(core=a.core, baseline=str(baseline), max_checks=125) as worker:
            class Bridge:
                pending_start = False

                def check_isolated(self, path, **kwargs):
                    if self.pending_start:
                        assert Path(path).read_bytes() == src
                        self.pending_start = False
                        return dict(acceptance)
                    rec = worker.check(path)
                    if rec is None:
                        event(kind="referee_failure", phase=state["phase"], core=a.core,
                              reason=worker.last_error, candidate_sha256=digest(path),
                              counted_as_kernel_attempt=False)
                    elif "the check timed out" in (rec.get("referee_message") or ""):
                        event(kind="watchdog_timeout", phase=state["phase"], core=a.core,
                              reason=rec["referee_message"], counted_as_kernel_attempt=True)
                    return rec

            bridge = Bridge()
            state["phase"] = "acceptance_retry"
            save()
            acceptance = bridge.check_isolated(str(candidate))
            assert acceptance is not None, "Core unavailable or referee failed"
            assert not schema.validate(acceptance), schema.validate(acceptance)
            (out / "acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
            assert acceptance["verdict"] == "no_gain" and acceptance["sim_ok"] and acceptance["chip_ok"]
            state.update(acceptance="passed", worker_pid=worker._p.pid)
            save()
            agent.pick_referee = lambda: ("speedcheck (DMA-only v2, single pinned worker)", bridge)
            for repeat in range(3):
                tag = f"referee-r{repeat}"
                state["phase"] = tag
                save()
                print("START", tag, flush=True)
                logpath = out / f"{tag}.jsonl"
                bridge.pending_start = True
                with (out / f"{tag}.log").open("w", buffering=1) as log:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        sys.argv = ["agent.py", "--arm", "referee", "--budget", "8", "--repeat", "1",
                                    "--samples", "1", "--give-up-after", "0", "--start", str(baseline),
                                    "--base", state["base"], "--model", state["model"], "--max-tokens", "2500",
                                    "--context", "8192", "--seat", "100", "--log", str(logpath)]
                        agent.main()
                assert not bridge.pending_start
                records = [json.loads(line) for line in logpath.read_text().splitlines() if line.strip()]
                assert len(records) == 8 and all(not schema.validate(r) for r in records)
                assert all(r["arm"] == "referee" for r in records)
                state["completed"].append(tag)
                state.update(worker_pid=worker._p.pid if worker._p else None, worker_starts=worker.starts)
                save()
                print("DONE", tag, flush=True)
            state["phase"] = "complete"
            save()
    except BaseException as exc:
        state.update(failed_phase=state["phase"], phase="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
