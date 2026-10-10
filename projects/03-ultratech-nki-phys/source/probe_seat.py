"""Read-only resource probe; optional compilation executes an existing reference kernel."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("seat-capabilities.json"))
    args = parser.parse_args()
    report = dict(hostname=os.uname().nodename, python=sys.version,
                  cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  visible_cores=os.environ.get("NEURON_RT_VISIBLE_CORES"), packages={})
    for name in ("numpy", "scipy", "mujoco", "nki", "torch", "torch-neuronx", "neuronx-distributed"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"):
        try:
            report[path] = Path(path).read_text().strip()
        except OSError as exc:
            report[path] = str(exc)
    try:
        result = subprocess.run(["neuron-ls", "--json-output"], capture_output=True, text=True, timeout=20)
        report["neuron_ls"] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["neuron_ls"] = dict(error=str(exc))
    if args.compile_smoke:
        if os.uname().nodename != "seat-260":
            parser.error("Compilation smoke test is scoped to seat-260")
        if not report["visible_cores"]:
            parser.error("Set NEURON_RT_VISIBLE_CORES to confirmed free cores before compilation")
        script = """import sys, numpy as np
sys.path.insert(0, '../02-kernel-agent')
from reference_level3 import nki_matmul_basic_
rng = np.random.default_rng(7)
a = rng.normal(size=(128,64)).astype(np.float32)
b = rng.normal(size=(128,512)).astype(np.float32)
result = np.asarray(nki_matmul_basic_(a,b))
np.testing.assert_allclose(result, a.T @ b, rtol=1e-3, atol=1e-3)
print('Device compilation, execution and numerical comparison passed')
"""
        start = time.monotonic()
        try:
            result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parent,
                                    capture_output=True, text=True, timeout=600)
            report["compile_smoke"] = dict(returncode=result.returncode, stdout=result.stdout,
                                           stderr=result.stderr, seconds=time.monotonic() - start)
        except subprocess.TimeoutExpired:
            report["compile_smoke"] = dict(error="timed out after 600 seconds")
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if "compile_smoke" in report and report["compile_smoke"].get("returncode") != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
