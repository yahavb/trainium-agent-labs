"""Standalone adapter for the workshop's installed NKI 0.6 / nrtpy APIs.

Compilation is separate from all timing. This SDK snapshot exposes compilation
through CompileKernel; keep the snapshot recorded in results/environment.json.
"""

import hashlib
import json
import os
import inspect
from pathlib import Path
import time

import numpy as np
import nki
from nki.framework.compiled import CompileKernel, compile_kernel_to_nir
from nki.compiler.driver import _compile_bir_to_neff
from nrtpy import SpikeModel, SpikeTensor
from nrtpy._nrtpy import SystemTraceSession

from reference import validate_inputs

ROOT = Path(__file__).resolve().parent


class LoadedKernel:
    def __init__(self, kernel, logits, indices=None, row_tile=128, vocab_tile=8192, extra=None):
        source = (ROOT / "kernel.py").read_bytes() + (ROOT / "runtime.py").read_bytes()
        if (ROOT / "baseline.py").exists():
            source += (ROOT / "baseline.py").read_bytes()
        key = hashlib.sha256(source + str(nki.__version__).encode()).hexdigest()[:12]
        tag = f"{kernel.__name__}-{logits.shape[0]}x{logits.shape[1]}-r{row_tile}-v{vocab_tile}-{key}"
        self.build = ROOT / "build" / tag
        self.build.mkdir(parents=True, exist_ok=True)
        self.neff = self.build / "kernel.neff"
        signature = self.build / "signature.json"
        self.scoring_rows = logits.shape[0] if indices is not None else None
        self.compile_seconds = 0.0
        args = (logits,) if indices is None else (logits, indices, row_tile, vocab_tile)
        if extra:
            args += tuple(extra)
        if not self.neff.exists() or not signature.exists():
            print(f"COMPILE {tag}", flush=True)
            started = time.perf_counter()
            compiler = CompileKernel(func=kernel.func, lnc=2, target="trn2", artifacts_dir=str(self.build))
            # Lower and compile without initializing the runtime. This lets
            # compilation proceed even while another process holds the cores.
            inputs = inspect.signature(kernel.func).bind(*args).arguments
            opts = compiler._compile_opts()
            lowered = compile_kernel_to_nir(kernel, inputs=inputs, compile_opts=opts)
            compiled = _compile_bir_to_neff(lowered, opts, inputs)
            signature.write_text(json.dumps({"output_names": [s.name for s in lowered.descriptor.output_specs]}) + "\n")
            self.compile_seconds = time.perf_counter() - started
            if not self.neff.exists():
                self.neff.write_bytes(Path(compiled.neff_path).read_bytes())
        self.model = SpikeModel.load_from_neff(self.neff, core_id=0)
        self.output_names = json.loads(signature.read_text())["output_names"]
        host_inputs = {"logits": logits, "indices": indices, "values": logits,
                       "logits_hbm": logits, "targets_hbm": indices}
        self.inputs = {}
        for name in self.model.input_tensors_info:
            if name not in host_inputs or host_inputs[name] is None:
                raise RuntimeError(f"Unrecognized compiled input: {name}")
            self.inputs[name] = SpikeTensor.from_numpy(host_inputs[name], name=name, core_id=0)
        self.outputs = self.model._allocate_outputs_with_aliases(self.inputs)

    def run(self, logits=None, indices=None):
        if logits is not None:
            for name, tensor in self.inputs.items():
                value = indices if name in ("indices", "targets_hbm") else logits
                tensor.write_from_numpy(value)
        self.model(inputs=self.inputs, outputs=self.outputs)
        result = tuple(self.outputs[name].numpy().view(np.float32) for name in self.output_names)
        if self.scoring_rows is not None:
            result = tuple(value.reshape(self.scoring_rows) for value in result)
        return result

    def measure(self, mode="device", warmup=10, iterations=100):
        if mode == "device":
            for _ in range(warmup):
                self.model(inputs=self.inputs, outputs=self.outputs)
            with SystemTraceSession(self.model.model_ref.core_id) as trace:
                for _ in range(iterations):
                    self.model(inputs=self.inputs, outputs=self.outputs)
                events = json.loads(trace.fetch_events_json())["events"]
            # LNC=2 emits one interval per physical core. Pair by tracking_id,
            # group by exec_id, then take earliest start -> latest finish using
            # trace-synchronized timestamps. Raw nc_timestamp_ns clocks have
            # per-core offsets and must not be mixed across physical cores.
            starts = {}
            executions = {}
            for event in events:
                if event.get("event_type") != "nc_exec_running":
                    continue
                key = event["tracking_id"]
                data = event["data"]
                if event["phase"] == "start":
                    starts[key] = (data, event["timestamp_ns"])
                elif event["phase"] == "stop" and key in starts:
                    start, timestamp = starts.pop(key)
                    executions.setdefault(start["exec_id"], []).append(
                        (timestamp, event["timestamp_ns"], start["device_core_idx"]))
            if starts or len(executions) != iterations:
                raise RuntimeError(f"Incomplete trace: {len(executions)} executions, {len(starts)} unmatched starts")
            durations = []
            for intervals in executions.values():
                if len(intervals) != 2 or len({i[2] for i in intervals}) != 2:
                    raise RuntimeError(f"Expected two physical-core intervals per LNC=2 execution: {intervals}")
                durations.append((max(i[1] for i in intervals) - min(i[0] for i in intervals)) / 1e6)
            self.last_device_trace = [e for e in events if e.get("event_type") == "nc_exec_running"]
            return np.asarray(durations, dtype=np.float64)
        result = self.model.benchmark(inputs=self.inputs, outputs=self.outputs,
                                      warmup_iter=warmup, benchmark_iter=iterations, mode=mode)
        # Refuse to report subgraph events or missing trace samples as iterations.
        if len(result.durations_ms) != iterations:
            raise RuntimeError(f"Expected {iterations} execution durations, got {len(result.durations_ms)}")
        return np.asarray(result.durations_ms, dtype=np.float64)


def score_tokens(logits, indices, row_tile=128, vocab_tile=8192):
    """Convenience host API; not timed. Reuse LoadedKernel for repeated device calls."""
    validate_inputs(logits, indices)
    from kernel import fused_score
    return LoadedKernel(fused_score, logits, indices, row_tile, vocab_tile).run()


def smoke():
    from kernel import smoke_kernel
    x = np.arange(32, dtype=np.float32).reshape(4, 8)
    loaded = LoadedKernel(smoke_kernel, x)
    actual, = loaded.run()
    np.testing.assert_array_equal(actual, x + 1)
    durations = loaded.measure(iterations=10, warmup=2)
    print(json.dumps({"smoke": "PASS", "shape": list(actual.shape),
                      "device_p50_ms": float(np.median(durations)),
                      "samples": len(durations), "visible_cores": os.environ.get("NEURON_RT_VISIBLE_CORES")}), flush=True)


if __name__ == "__main__":
    smoke()
