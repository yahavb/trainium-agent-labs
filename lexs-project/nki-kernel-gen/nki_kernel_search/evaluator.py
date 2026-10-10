import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


def evaluate_local(candidate, task, mode="simulate", baseline=None, timeout=600, seeds="0,17,42"):
    env = dict(os.environ)
    # Do not expose provider credentials to generated Python.
    for key in list(env):
        if any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            env.pop(key)
    with tempfile.TemporaryDirectory(prefix="nki-eval-") as directory:
        result_path = Path(directory) / "result.json"
        cmd = [sys.executable, str(Path(__file__).with_name("worker.py")),
               "--candidate", str(Path(candidate).resolve()), "--task", str(Path(task).resolve()),
               "--mode", mode, "--result", str(result_path), "--seeds", seeds]
        if baseline:
            cmd += ["--baseline", str(Path(baseline).resolve())]
        # A shared host lock prevents overlapping benchmarks from contaminating latency.
        with open("/tmp/nki-kernel-search.lock", "a") as lock, open(Path(directory) / "worker.log", "w+") as log:
            if mode == "hardware":
                fcntl.flock(lock, fcntl.LOCK_EX)
            process = subprocess.Popen(cmd, cwd=directory, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return {"metrics": {"correctness": 0.0, "combined_score": 0.0},
                        "artifacts": {"error": f"Evaluation exceeded {timeout}s"}}
            if process.returncode or not result_path.exists():
                log.seek(0, 2)
                log.seek(max(0, log.tell() - 8000))
                return {"metrics": {"correctness": 0.0, "combined_score": 0.0},
                        "artifacts": {"error": log.read() or f"Worker exited {process.returncode}"}}
            result = json.loads(result_path.read_text())
            if not result["metrics"].get("correctness"):
                log.seek(0, 2)
                log.seek(max(0, log.tell() - 8000))
                result["artifacts"]["compiler_log"] = log.read()
            return result


def evaluate(program_path):
    from openevolve.evaluation_result import EvaluationResult
    result = evaluate_local(program_path, os.environ["NKI_SEARCH_TASK"],
                            os.environ.get("NKI_SEARCH_MODE", "hardware"),
                            os.environ.get("NKI_SEARCH_BASELINE"),
                            float(os.environ.get("NKI_SEARCH_TIMEOUT", "600")))
    return EvaluationResult(**result)
