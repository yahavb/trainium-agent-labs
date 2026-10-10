# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure one call and a forward-only evaluation rollout on CPU or Trainium."""

import argparse
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

# Select the explicit Samudra checkout before importing its modules.
_root_parser = argparse.ArgumentParser(add_help=False)
_root_parser.add_argument("--samudra-root", type=Path, required=True)
_root_args, _ = _root_parser.parse_known_args()
sys.path.insert(0, str(_root_args.samudra_root.resolve() / "src"))

from forward_timing import summarize_timings, time_forward
from samudra.config import EvalConfig
from samudra.datasets import InferenceDataset
from samudra.utils.data import CanonicalSource


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_case(cfg):
    """Stage only the initial ocean state and forcing windows, never labels."""
    if len(cfg.data.sources) != 1:
        raise ValueError("This benchmark requires exactly one data source")
    source_cfg = cfg.data.sources[0]
    if source_cfg.inference_time is None:
        raise ValueError("An inference_time range is required")
    root = cfg.experiment.resolved_data_root
    raw = root.resolve(source_cfg.data_location).open({})
    means = root.resolve(source_cfg.data_means_location).open({})
    stds = root.resolve(source_cfg.data_stds_location).open({})
    canonical, means, stds, layout = source_cfg.canonicalize_datasets(raw, means, stds)
    source = CanonicalSource.from_datasets(
        canonical,
        means,
        stds,
        data_layout=layout,
        prognostic_var_names=layout.prognostic_var_names,
        boundary_var_names=layout.boundary_var_names,
    ).slice_time(source_cfg.inference_time)
    dataset = InferenceDataset(
        source,
        layout.prognostic_var_names,
        layout.boundary_var_names,
        cfg.data.input_steps,
        cfg.data.output_steps,
        cfg.data.normalize_before_mask,
        cfg.data.masked_fill_value,
        long_rollout=True,
    )
    if not len(dataset):
        raise ValueError("The evaluation range does not contain a complete window")
    initial = dataset.initial_prognostic
    boundaries = tuple(dataset.get_boundary(step) for step in range(len(dataset)))
    if not torch.isfinite(initial).all() or any(
        not torch.isfinite(boundary).all() for boundary in boundaries
    ):
        raise ValueError("Prepared evaluation inputs contain non-finite values")
    return {
        "initial": initial,
        "boundaries": boundaries,
        "ctx": dataset.ctx,
        "prognostic_vars": layout.prognostic_var_names,
        "boundary_vars": layout.boundary_var_names,
        "requested_range": str(source_cfg.inference_time),
        "initial_times": [
            str(value) for value in source.time.values[: cfg.data.input_steps]
        ],
        "target_times": [
            str(value) for value in dataset.get_target_time(0, len(dataset)).values
        ],
        "source_location": str(root.resolve(source_cfg.data_location)),
        "available_data_range": [str(raw.time.values[0]), str(raw.time.values[-1])],
        "model_config": cfg.model.model_dump(mode="json"),
        "input_steps": cfg.data.input_steps,
        "output_steps": cfg.data.output_steps,
        "preparation_note": (
            "Canonicalization and anomaly statistics use the supplied source span, "
            "as in normal evaluation; a sliced source changes that climatology."
        ),
    }


def neuron_backend():
    """Initialize Neuron only when selected; keep CPU installs independent."""
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
    os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
    os.environ.setdefault(
        "NEURON_CC_FLAGS", "--target=trn2 --auto-cast=none --logical-nc-config=2"
    )
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    library = (
        Path(sys.base_prefix)
        / "lib"
        / f"libpython{sys.version_info.major}.{sys.version_info.minor}.so.1.0"
    )
    if library.exists():
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        os.environ["LD_LIBRARY_PATH"] = (
            str(library.parent) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        )
    importlib.import_module("torch_neuronx")
    xm = importlib.import_module("torch_xla.core.xla_model")
    metrics = importlib.import_module("torch_xla.debug.metrics")
    runtime = importlib.import_module("torch_xla.runtime")

    if runtime.device_type() != "NEURON":
        raise RuntimeError(f"Expected Neuron execution, got {runtime.device_type()}")

    device = xm.xla_device()

    def synchronize():
        xm.mark_step(wait=True)
        xm.wait_device_ops()

    def compilation_count():
        data = metrics.metric_data("CompileTime")
        return data[0] if data else 0

    def cpu_fallback_count():
        return sum(
            metrics.counter_value(name) or 0
            for name in metrics.counter_names()
            if name.startswith("aten::")
        )

    return device, synchronize, compilation_count, cpu_fallback_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samudra-root", type=Path, required=True)
    parser.add_argument("config", help="Evaluation YAML or bundled preset")
    parser.add_argument(
        "--data", help="Replacement data YAML, e.g. a locally staged dataset"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backend", choices=("cpu", "neuron"), default="cpu")
    parser.add_argument(
        "--case", type=Path, required=True, help="Reusable prepared-input cache"
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("forward_benchmark.json"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--single-repeats", type=int, default=20)
    parser.add_argument(
        "--max-calls", type=int, help="Explicitly truncate the range for a smoke run"
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.single_repeats < 1 or args.threads < 1:
        parser.error("warmup, single-repeats and threads must be positive")
    if args.max_calls is not None and args.max_calls < 1:
        parser.error("max-calls must be positive")
    torch.set_num_threads(args.threads)
    cfg = EvalConfig.from_yaml(args.config)
    if args.data:
        import yaml
        from samudra.config import DataConfig

        with open(args.data) as stream:
            cfg.data = DataConfig.model_validate(yaml.safe_load(stream))
    data_spec = cfg.data.model_dump(mode="json")
    case_spec = {
        "data": data_spec,
        "root": str(cfg.experiment.resolved_data_root),
        "model": cfg.model.model_dump(mode="json"),
    }
    preparation_start = time.perf_counter()
    if args.case.exists():
        # The cache is a local artifact created by this script, containing BatchGrid.
        case = torch.load(args.case, map_location="cpu", weights_only=False)
        if case["case_spec"] != case_spec:
            raise ValueError(
                "Prepared case configuration differs; choose a new --case path"
            )
    else:
        print("Preparing real evaluation inputs outside all forward timers", flush=True)
        case = prepare_case(cfg)
        case["case_spec"] = case_spec
        args.case.parent.mkdir(parents=True, exist_ok=True)
        torch.save(case, args.case)
    preparation_seconds = time.perf_counter() - preparation_start
    print(
        f"Prepared {len(case['boundaries'])} calls over {case['requested_range']}",
        flush=True,
    )
    if args.prepare_only:
        return
    checkpoint_path = args.checkpoint or (
        Path(cfg.ckpt_path) if cfg.ckpt_path else None
    )
    if checkpoint_path is None:
        parser.error("--checkpoint or config.ckpt_path is required")
    setup_start = time.perf_counter()
    device, synchronize, compilation_count, cpu_fallback_count = (
        neuron_backend()
        if args.backend == "neuron"
        else (torch.device("cpu"), lambda: None, lambda: 0, lambda: 0)
    )
    initial = case["initial"].to(device)
    boundaries = tuple(item.to(device) for item in case["boundaries"][: args.max_calls])
    ctx = case["ctx"].to(device)
    cfg.model.checkpointing = None
    model = cfg.model.build(
        prog_channels=initial.shape[1],
        boundary_channels=boundaries[0].shape[1],
        out_channels=ctx.label_mask.shape[0],
        input_steps=cfg.data.input_steps,
        grid_sizes=[tuple(initial.shape[-2:])],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(
        {k.removeprefix("module."): v for k, v in checkpoint["model"].items()},
        strict=True,
    )
    checkpoint_epoch = checkpoint["epoch"]
    del checkpoint
    model.requires_grad_(False).eval().to(device)
    synchronize()
    setup_seconds = time.perf_counter() - setup_start

    def predict(prognostic, boundary):
        decoding = model.forward_once(prognostic, boundary, ctx)
        return model._assemble_prediction(prognostic, decoding)

    with torch.inference_mode():
        warmup_start = time.perf_counter()
        for _ in range(args.warmup):
            prediction = predict(initial, boundaries[0])
            synchronize()
            del prediction
        warmup_seconds = time.perf_counter() - warmup_start
        print(
            f"Warmup complete in {warmup_seconds:.3f}s; measuring single calls",
            flush=True,
        )
        single = []
        for _ in range(args.single_repeats):
            prediction, timing = time_forward(
                lambda: predict(initial, boundaries[0]),
                synchronize,
                compilation_count,
                cpu_fallback_count,
            )
            single.append(timing)
            del prediction

        rollout = []
        prognostic = initial
        rollout_loop_start = time.perf_counter()
        for step, boundary in enumerate(boundaries):
            prediction, timing = time_forward(
                lambda prognostic=prognostic, boundary=boundary: predict(
                    prognostic, boundary
                ),
                synchronize,
                compilation_count,
                cpu_fallback_count,
            )
            rollout.append(timing)
            # History assembly is outside the forward timer. With 2-in/2-out,
            # the whole history is replaced, so no concatenate/copy is needed.
            prognostic = (
                prediction
                if prediction.shape[1] == prognostic.shape[1]
                else model._advance_prognostic_history(prognostic, prediction)
            )
            synchronize()
            if (step + 1) % 25 == 0:
                print(
                    f"Forward-only rollout: {step + 1}/{len(boundaries)} calls",
                    flush=True,
                )
        rollout_loop_seconds = time.perf_counter() - rollout_loop_start

    report = {
        "backend": args.backend,
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "cpu_threads": torch.get_num_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "input_prognostic_shape": list(initial.shape),
        "input_boundary_shape": list(boundaries[0].shape),
        "output_shape": list(prediction.shape),
        "dtype": str(initial.dtype),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "prepared_case_sha256": sha256(args.case),
        "case_spec": case_spec,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(args.samudra_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "requested_range": case["requested_range"],
        "available_data_range": case["available_data_range"],
        "initial_times": case["initial_times"],
        "source_location": case["source_location"],
        "target_time_start": case["target_times"][0],
        "target_time_end": case["target_times"][
            len(boundaries) * cfg.data.output_steps - 1
        ],
        "complete_eval_range": len(boundaries) == len(case["boundaries"]),
        "forecast_timesteps": len(boundaries) * cfg.data.output_steps,
        "single_call": summarize_timings(single),
        "eval_range": summarize_timings(rollout),
        "preparation_seconds": preparation_seconds,
        "backend_setup_seconds": setup_seconds,
        "warmup_including_compilation_seconds": warmup_seconds,
        "warmup_calls": args.warmup,
        "rollout_loop_wall_seconds": rollout_loop_seconds,
        "timing_contract": (
            "Device-resident weights and preloaded inputs; perf_counter_ns around "
            "forward_once + residual assembly + completion synchronization. "
            "Excludes compilation (guarded), loading, normalization, transfers, "
            "history assembly, forecast metrics, logging, and output writes. "
            "Includes Python dispatch and synchronization; not kernel-only device profiling."
        ),
        "host_cpu_time_contract": "process_time_ns includes host worker threads, excludes Neuron compute",
        "memory_contract": "process lifetime peak RSS on Linux; includes setup and resident case, not forward-only peak",
        "forecast_metrics_computed": False,
        "preparation_note": case["preparation_note"],
    }
    if args.backend == "neuron":
        report["neuron"] = {
            "torch_neuronx_version": importlib.metadata.version("torch-neuronx"),
            "compiler_version": importlib.metadata.version("neuronx-cc"),
            "flags": os.environ["NEURON_CC_FLAGS"],
            "visible_cores": os.environ["NEURON_RT_VISIBLE_CORES"],
            "logical_nc_config": os.environ["NEURON_LOGICAL_NC_CONFIG"],
            "compilation_count": compilation_count(),
            "cpu_fallback_count": cpu_fallback_count(),
            "hardware": json.loads(
                subprocess.check_output(["neuron-ls", "--json-output"], text=True)
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "single_call": {
                    k: v for k, v in report["single_call"].items() if k != "samples"
                },
                "eval_range": {
                    k: v for k, v in report["eval_range"].items() if k != "samples"
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
