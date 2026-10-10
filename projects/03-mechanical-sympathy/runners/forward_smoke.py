#!/usr/bin/env python3
"""Run one checkpoint-backed Samudra model forward pass on a synthetic grid.

This is a compute and checkpoint compatibility smoke test. It does not test
data loading, normalization, ocean masks, or forecast skill. Use Samudra's eval
pipeline for those checks after this passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samudra-root",
        type=Path,
        required=True,
        help="Path to the Samudra checkout, not its src directory.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/samudra/configs/samudra_om4_v2/eval.yaml"),
    )
    parser.add_argument("--device", choices=("cpu", "auto"), default="auto")
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must be a Samudra training checkpoint with key 'model'.")
    return {
        name.removeprefix("module."): value
        for name, value in checkpoint["model"].items()
    }


def main() -> int:
    args = parse_args()
    samudra_root = args.samudra_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (samudra_root / config_path).resolve()

    for path, label in (
        (samudra_root / "src", "Samudra source"),
        (checkpoint_path, "Checkpoint"),
        (config_path, "Config"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    sys.path.insert(0, str(samudra_root / "src"))

    from samudra.config import EvalConfig
    from samudra.utils.ctx import BatchGrid

    cfg = EvalConfig.from_yaml(config_path)
    data_source = cfg.data.sources[0]
    layout = data_source.build_layout()
    n_prog = len(layout.prognostic_var_names)
    n_boundary = len(layout.boundary_var_names)
    input_prog_channels = cfg.data.input_steps * n_prog
    input_boundary_channels = cfg.data.input_steps * n_boundary
    output_channels = cfg.data.output_steps * n_prog

    if args.height <= 0 or args.width <= 0 or args.batch_size <= 0:
        raise ValueError("height, width, and batch size must be positive")
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    )
    model = cfg.model.build(
        prog_channels=input_prog_channels,
        boundary_channels=input_boundary_channels,
        out_channels=output_channels,
        input_steps=cfg.data.input_steps,
        grid_sizes=[(args.height, args.width)],
    ).to(device)
    model.load_state_dict(load_state_dict(checkpoint_path, device), strict=True)
    model.eval()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    prognostic = torch.randn(
        args.batch_size,
        input_prog_channels,
        args.height,
        args.width,
        generator=generator,
    ).to(device)
    boundary = torch.randn(
        args.batch_size,
        input_boundary_channels,
        args.height,
        args.width,
        generator=generator,
    ).to(device)
    mask = torch.ones(1, 1, args.height, args.width, dtype=torch.bool, device=device)
    lat = torch.linspace(-89.5, 89.5, args.height)
    lon = torch.linspace(0.5, 359.5, args.width)
    ctx = BatchGrid(mask, (lat, lon), (lat, lon))

    def forward() -> torch.Tensor:
        decoded = model.forward_once(prognostic, boundary, ctx)
        return model._assemble_prediction(prognostic, decoded)

    with torch.inference_mode():
        for _ in range(args.warmup):
            output = forward()
        if device.type == "cuda":
            torch.cuda.synchronize()

        timings_ms: list[float] = []
        for _ in range(args.repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = forward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings_ms.append((time.perf_counter() - start) * 1000)

    expected_shape = (
        args.batch_size,
        output_channels,
        args.height,
        args.width,
    )
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(f"Output shape {tuple(output.shape)} != {expected_shape}")
    if not torch.isfinite(output).all().item():
        raise RuntimeError("Forward pass returned NaN or infinite values")

    report = {
        "status": "passed",
        "test_kind": "synthetic_checkpoint_forward",
        "samudra_config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "device": str(device),
        "input_shape_prognostic": list(prognostic.shape),
        "input_shape_boundary": list(boundary.shape),
        "output_shape": list(output.shape),
        "output_finite": True,
        "latency_ms": timings_ms,
        "latency_median_ms": statistics.median(timings_ms),
        "note": "Synthetic inputs only. This does not measure data preprocessing or forecast accuracy.",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
