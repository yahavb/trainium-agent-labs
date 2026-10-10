#!/usr/bin/env python3
"""Run an autoregressive Samudra rollout on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


DATASET_PREFIX = "Samudra/v2026-09/om4_onedeg"
OSN_ENDPOINT = "https://nyu1.osn.mghpcc.org"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samudra-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--start", default="2014-10-10")
    parser.add_argument("--end", default="2022-12-24")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--prediction-manifest",
        type=Path,
        help="Output JSON manifest. Defaults beside predictions.zarr.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional small preflight limit. Omit to run the full time window.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def prediction_store_manifest(path: Path) -> dict[str, object]:
    files = [item for item in path.rglob("*") if item.is_file()]
    metadata = []
    for item in files:
        if item.name in {".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"}:
            metadata.append(
                {
                    "path": str(item.relative_to(path)),
                    "bytes": item.stat().st_size,
                    "sha256": sha256(item),
                }
            )
    return {
        "schema_version": 1,
        "store": str(path),
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
        "metadata_files": metadata,
        "note": "Chunk payloads are not hashed. Keep the Zarr store outside Git.",
    }


def main() -> int:
    args = parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")

    samudra_root = args.samudra_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if not (samudra_root / "src").is_dir():
        raise FileNotFoundError(f"Samudra source not found: {samudra_root / 'src'}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    sys.path.insert(0, str(samudra_root / "src"))

    from samudra.config import EvalConfig
    from samudra.data_backend import PythonSourceBackend
    from samudra.datasets import InferenceDataset
    from samudra.utils.location import LocalLocation
    from samudra.backend import init_eval_backend
    from samudra.utils.writer import ZarrWriter

    raw = EvalConfig._load_yaml(
        samudra_root / "src/samudra/configs/samudra_om4_v2/eval.yaml"
    )
    if len(raw["data"]["sources"]) != 1:
        raise ValueError("This runner requires one OM4 data source")

    source = raw["data"]["sources"][0]
    # Only the inference range is long. Short train and validation slices supply
    # the grid metadata needed by Samudra's shared eval configuration.
    source["train_time"] = {"start": args.start, "end": "2014-10-20"}
    source["val_time"] = {"start": "2014-10-20", "end": "2014-10-30"}
    source["inference_time"] = {"start": args.start, "end": args.end}
    for key in ("data_location", "data_means_location", "data_stds_location"):
        suffix = {
            "data_location": "OM4.zarr",
            "data_means_location": "OM4_means.zarr",
            "data_stds_location": "OM4_stds.zarr",
        }[key]
        source[key] = {
            "type": "s3",
            "endpoint_url": OSN_ENDPOINT,
            "anon": True,
            "bucket": "m2lines-pubs",
            "path": f"{DATASET_PREFIX}/{suffix}",
        }

    raw["backend"] = "cpu"
    cfg = EvalConfig(**raw)

    wall_start = time.perf_counter()
    device = init_eval_backend("cpu")
    print("Opening one-degree OM4 inference source from public OSN...", flush=True)
    splits = cfg.data.sources[0].build(
        LocalLocation(path=Path("/tmp")),
        use_dask=True,
        is_primary=True,
        include_inference=True,
        source_backend=PythonSourceBackend(),
    )
    if splits.inference is None:
        raise RuntimeError("No inference source is configured")
    source = splits.inference
    prognostic_names = source.data_layout.prognostic_var_names
    boundary_names = source.data_layout.boundary_var_names
    input_steps = cfg.data.input_steps
    output_steps = cfg.data.output_steps
    model = cfg.model.build(
        prog_channels=input_steps * len(prognostic_names),
        boundary_channels=input_steps * len(boundary_names),
        out_channels=output_steps * len(prognostic_names),
        input_steps=input_steps,
        grid_sizes=[splits.train.grid_size],
    ).to(device)
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    model_state = checkpoint_data["model"]
    model.load_state_dict(
        {key.removeprefix("module."): value for key, value in model_state.items()},
        strict=True,
    )
    print("Checkpoint loaded. Building inference dataset...", flush=True)
    dataset = InferenceDataset(
        source=source,
        prognostic_var_names=prognostic_names,
        boundary_var_names=boundary_names,
        input_steps=input_steps,
        output_steps=output_steps,
        normalize_before_mask=cfg.data.normalize_before_mask,
        masked_fill_value=cfg.data.masked_fill_value,
        long_rollout=True,
    )
    dataset.to(device)
    init_seconds = time.perf_counter() - wall_start
    total_steps = len(dataset)
    run_steps = min(total_steps, args.max_steps) if args.max_steps else total_steps
    if run_steps < 1:
        raise RuntimeError("The configured inference window has no rollout steps")
    print(
        f"Starting CPU forward-only rollout: {run_steps} of {total_steps} steps.",
        flush=True,
    )

    model.eval()
    prediction_path = args.predictions.resolve()
    if prediction_path.name != "predictions.zarr":
        raise ValueError("--predictions path must end with predictions.zarr")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ZarrWriter(
        output_dir=prediction_path.parent,
        coords=source.coordinates(),
        output_steps=output_steps,
        model_path=checkpoint,
        time_chunk_size=10,
        preprocessor=dataset.preprocessor,
        data_layout=source.data_layout,
    )
    prognostic_read_start = time.perf_counter()
    prognostic = dataset.initial_prognostic
    prognostic_read_seconds = time.perf_counter() - prognostic_read_start
    boundary_read_seconds = 0.0
    zarr_write_seconds = 0.0
    written_model_steps = 0
    forward_times_ms: list[float] = []
    model_start = time.perf_counter()
    with torch.inference_mode():
        for step in range(run_steps):
            read_start = time.perf_counter()
            boundary = dataset.get_boundary(step).to(device=prognostic.device)
            boundary_read_seconds += time.perf_counter() - read_start

            forward_start = time.perf_counter()
            decoded = model.forward_once(prognostic, boundary, dataset.ctx)
            prediction = model._assemble_prediction(prognostic, decoded)
            forward_times_ms.append((time.perf_counter() - forward_start) * 1000)
            prognostic = model._advance_prognostic_history(prognostic, prediction)
            writer.record_batch(
                SimpleNamespace(
                    prediction=prediction,
                    time=dataset.get_target_time(step, 1),
                )
            )
            progress_interval = max(1, run_steps // 10)
            if (
                (step + 1) % 5 == 0
                or step + 1 == run_steps
            ):
                write_start = time.perf_counter()
                writer.write()
                zarr_write_seconds += time.perf_counter() - write_start
                written_model_steps = step + 1
            if (step + 1) % progress_interval == 0 or step + 1 == run_steps:
                print(
                    f"Completed {step + 1}/{run_steps} rollout steps; "
                    f"wrote {written_model_steps * output_steps} forecast records.",
                    flush=True,
                )
    model_seconds = time.perf_counter() - model_start
    if not torch.isfinite(prognostic).all().item():
        raise RuntimeError("Final autoregressive state has non-finite values")
    expected_shape = (1, input_steps * len(prognostic_names), *source.grid_size)
    if tuple(prognostic.shape) != expected_shape:
        raise RuntimeError(f"Unexpected final state shape: {tuple(prognostic.shape)}")

    total_seconds = time.perf_counter() - wall_start
    report = {
        "status": "passed",
        "kind": "cpu_forward_only_rollout",
        "start": args.start,
        "end": args.end,
        "steps_completed": run_steps,
        "steps_in_full_window": total_steps,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "samudra_commit": git_commit(samudra_root),
        "data_source": f"{OSN_ENDPOINT}/m2lines-pubs/{DATASET_PREFIX}/OM4.zarr",
        "device": "cpu",
        "output_saved": True,
        "prediction_store": str(prediction_path),
        "forecast_time_records_written": run_steps * output_steps,
        "metrics_or_visualizations_run": False,
        "initialization_seconds": init_seconds,
        "initial_state_read_seconds": prognostic_read_seconds,
        "boundary_reads_seconds": boundary_read_seconds,
        "zarr_write_seconds": zarr_write_seconds,
        "model_forward_latency_ms_median": statistics.median(forward_times_ms),
        "model_forward_latency_ms_p90": sorted(forward_times_ms)[
            max(0, int(0.9 * len(forward_times_ms)) - 1)
        ],
        "model_forward_latency_ms_max": max(forward_times_ms),
        "rollout_loop_seconds": model_seconds,
        "total_wall_seconds": total_seconds,
        "final_state_shape": list(prognostic.shape),
        "final_state_finite": True,
        "note": "CPU timing only. No forecast scoring or plotting. Model-forward time excludes boundary reads and Zarr writes; both are reported separately.",
    }
    report["model_forward_seconds"] = sum(forward_times_ms) / 1000
    manifest_path = (
        args.prediction_manifest.resolve()
        if args.prediction_manifest is not None
        else prediction_path.parent / "predictions.manifest.json"
    )
    prediction_manifest = {
        **prediction_store_manifest(prediction_path),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "samudra_commit": report["samudra_commit"],
        "start": args.start,
        "end": args.end,
        "steps_completed": run_steps,
        "forecast_time_records_written": run_steps * output_steps,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(prediction_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    report["prediction_manifest"] = str(manifest_path)
    report_text = json.dumps(report, indent=2)
    print(report_text, flush=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Samudra CPU Forward-Only Rollout\n\n"
        "This run performs an autoregressive forward rollout and writes forecast "
        "fields to `predictions.zarr`. It does not calculate forecast metrics or "
        "create visualizations.\n\n"
        "```json\n"
        f"{report_text}\n"
        "```\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
