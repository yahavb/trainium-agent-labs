#!/usr/bin/env python3
"""Run one real-data Samudra forward pass and an optional short rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samudra-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/samudra/configs/samudra_om4_v2/eval.yaml"),
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=0,
        help="Optional autoregressive rollout length. Keep 0 for one forward only.",
    )
    parser.add_argument("--start", default="2014-10-10")
    parser.add_argument("--end", default="2014-12-04")
    parser.add_argument(
        "--reference-output",
        type=Path,
        help=(
            "Write a compressed CPU reference fixture. The file contains the "
            "exact model inputs, grid context, coordinates, and prediction."
        ),
    )
    parser.add_argument(
        "--profile-ops",
        action="store_true",
        help="Profile one extra CPU forward and report its top operators.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    samudra_root = args.samudra_root.resolve()
    sys.path.insert(0, str(samudra_root / "src"))

    from samudra.config import EvalConfig
    from samudra.eval import Eval
    from samudra.utils.device import get_device

    config_path = args.config
    if not config_path.is_absolute():
        config_path = samudra_root / config_path
    raw = EvalConfig._load_yaml(config_path)
    if len(raw["data"]["sources"]) != 1:
        raise ValueError("This smoke test requires one OM4 data source")

    source = raw["data"]["sources"][0]
    source["train_time"] = {"start": args.start, "end": "2014-10-15"}
    source["val_time"] = {"start": "2014-10-15", "end": "2014-10-20"}
    source["inference_time"] = {"start": args.start, "end": args.end}
    source["data_location"] = "OM4.zarr"
    for kind in ("means", "stds"):
        source[f"data_{kind}_location"] = {
            "type": "s3",
            "endpoint_url": "https://nyu1.osn.mghpcc.org",
            "anon": True,
            "bucket": "m2lines-pubs",
            "path": (
                "Samudra/v2026-09/om4_onedeg/"
                f"OM4_{kind}.zarr"
            ),
        }

    raw["backend"] = "cpu"
    raw["save_zarr"] = False
    raw["ckpt_path"] = str(args.checkpoint.resolve())
    raw["experiment"]["data_root"] = str(args.data_root.resolve())
    raw["experiment"]["name"] = "samudra-data-smoke"
    raw["experiment"]["base_output_dir"] = str(
        (args.data_root.parent / "runs").resolve()
    )
    raw["experiment"]["wandb"]["mode"] = "disabled"
    cfg = EvalConfig(**raw)

    evaluator = Eval(cfg)
    dataset = evaluator.inference_dataset
    if len(dataset) < args.rollout_steps:
        raise ValueError(
            f"Data has {len(dataset)} model steps, but "
            f"{args.rollout_steps} were requested"
        )
    dataset.to(get_device())
    data_read_start = time.perf_counter()
    initial = dataset.initial_prognostic
    boundary = dataset.get_boundary(0)
    target = dataset.inference_target(slice(0, 1))
    sample_read_ms = (time.perf_counter() - data_read_start) * 1000

    evaluator.model.eval()
    with torch.inference_mode():
        start = time.perf_counter()
        prediction = evaluator.model.forward_once(initial, boundary, dataset.ctx)
        prediction = evaluator.model._assemble_prediction(initial, prediction)
        first_forward_ms = (time.perf_counter() - start) * 1000

        if tuple(prediction.shape) != tuple(target.shape):
            raise RuntimeError(
                f"Prediction shape {tuple(prediction.shape)} does not match "
                f"target shape {tuple(target.shape)}"
            )
        if not torch.isfinite(prediction).all().item():
            raise RuntimeError("Forward pass returned NaN or infinite values")

        operator_profile = None
        if args.profile_ops:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=True,
            ) as profiler:
                profiled = evaluator.model.forward_once(
                    initial, boundary, dataset.ctx
                )
                profiled = evaluator.model._assemble_prediction(initial, profiled)
            if tuple(profiled.shape) != tuple(target.shape):
                raise RuntimeError("Profiled forward pass returned an unexpected shape")
            operator_profile = [
                {
                    "operator": item.key,
                    "calls": item.count,
                    "self_cpu_time_ms": round(item.self_cpu_time_total / 1000, 3),
                    "cpu_time_ms": round(item.cpu_time_total / 1000, 3),
                    "self_cpu_memory_bytes": item.self_cpu_memory_usage,
                    "input_shapes": item.input_shapes,
                }
                for item in sorted(
                    profiler.key_averages(group_by_input_shape=True),
                    key=lambda record: record.self_cpu_time_total,
                    reverse=True,
                )[:20]
            ]

        rollout_report = None
        if args.rollout_steps:
            if args.rollout_steps < 1:
                raise ValueError("rollout-steps must be zero or a positive number")
            start = time.perf_counter()
            rollout = evaluator.model.inference(
                dataset,
                initial_prognostic=initial,
                steps_completed=0,
                num_steps=args.rollout_steps,
                epoch=0,
            )
            rollout_ms = (time.perf_counter() - start) * 1000
            rollout_prediction = rollout.prediction
            if not torch.isfinite(rollout_prediction).all().item():
                raise RuntimeError("Rollout returned NaN or infinite values")
            rollout_error = rollout_prediction - rollout.target
            rollout_report = {
                "steps": args.rollout_steps,
                "prediction_shape": list(rollout_prediction.shape),
                "finite": True,
                "latency_ms_including_local_data_reads": rollout_ms,
                "normalized_rmse": float(
                    torch.mean(rollout_error.square()).sqrt().item()
                ),
                "normalized_max_abs_error": float(
                    rollout_error.abs().max().item()
                ),
            }

    error = prediction - target
    reference_sha256 = None
    if args.reference_output is not None:
        reference_path = args.reference_output.resolve()
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        target_time = dataset.get_target_time(0, 1)
        channel_names = np.asarray(
            [
                f"t+{step + 1}:{name}"
                for step in range(evaluator.output_steps)
                for name in evaluator.prognostic_var_names
            ],
            dtype="U",
        )
        np.savez_compressed(
            reference_path,
            prognostic=initial.detach().cpu().numpy(),
            boundary=boundary.detach().cpu().numpy(),
            label_mask=dataset.ctx.label_mask.detach().cpu().numpy(),
            input_lat=dataset.ctx.input_resolution_cpu[0].detach().cpu().numpy(),
            input_lon=dataset.ctx.input_resolution_cpu[1].detach().cpu().numpy(),
            output_lat=dataset.ctx.output_resolution_cpu[0].detach().cpu().numpy(),
            output_lon=dataset.ctx.output_resolution_cpu[1].detach().cpu().numpy(),
            prediction=prediction.detach().cpu().numpy(),
            time=np.asarray([str(value) for value in target_time.values], dtype="U"),
            variables=channel_names,
        )
        reference_sha256 = sha256(reference_path)

    report = {
        "status": "passed",
        "test_kind": "real_om4_forward",
        "config": str(config_path),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "device": str(get_device()),
        "data_root": str(args.data_root.resolve()),
        "inference_time": [args.start, args.end],
        "available_model_steps": len(dataset),
        "forward_output_shape": list(prediction.shape),
        "forward_finite": True,
        "sample_read_ms": sample_read_ms,
        "forward_ms": first_forward_ms,
        "normalized_rmse": float(torch.mean(error.square()).sqrt().item()),
        "normalized_max_abs_error": float(error.abs().max().item()),
        "top_cpu_operators_profiled_in_extra_pass": operator_profile,
        "optional_rollout": rollout_report,
        "reference_output": str(args.reference_output.resolve())
        if args.reference_output is not None
        else None,
        "reference_sha256": reference_sha256,
        "note": "CPU reference only. Forward timing excludes local data reads and is not a Trainium benchmark.",
    }
    print(json.dumps(report, indent=2))
    evaluator.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
