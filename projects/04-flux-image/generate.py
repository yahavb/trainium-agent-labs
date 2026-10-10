"""Generate images with FLUX.1 on this seat pod's Trainium chip, using AWS's NxD Inference Flux code.

Compiles once per (model, size, tensor parallel degree) into /workspace/flux-compiled/ and reuses it after.

    python generate.py --prompt "A robot named trn2"
"""

import argparse
import json
import os
import statistics
import shutil
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from neuronx_distributed_inference.models.diffusers.flux.application import (
    NeuronFluxApplication,
    create_flux_config,
    get_flux_parallelism_config,
)

# Only the diffusers-format folders; the repo also holds a single-file copy of the weights we don't need.
DIFFUSERS_FILES = ["model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*",
                   "tokenizer/*", "tokenizer_2/*", "transformer/*", "vae/*"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="A cat holding a sign that says hello world")
    p.add_argument("--model", default=os.environ.get("MODEL", "black-forest-labs/FLUX.1-dev"))
    p.add_argument("--size", type=int, default=1024, help="square image side; changing it recompiles")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--tp", type=int, default=int(os.environ.get("TP", 4)), help="NeuronCores to use; changing it recompiles")
    p.add_argument("--num", type=int, default=1, help="images to generate")
    p.add_argument("--out", default="out")
    p.add_argument("--metrics", default="out/metrics.json", help="write timing metrics here ('' to skip)")
    args = p.parse_args()

    t_start = time.perf_counter()
    print(f"Downloading {args.model} (about 34 GB the first time) ...", flush=True)
    ckpt = snapshot_download(args.model, allow_patterns=DIFFUSERS_FILES)

    t_download = time.perf_counter()
    world_size = get_flux_parallelism_config(args.tp)
    clip, t5, backbone, decoder = create_flux_config(ckpt, world_size, args.tp, torch.bfloat16, args.size, args.size)
    app = NeuronFluxApplication(
        model_path=ckpt,
        text_encoder_config=clip,
        text_encoder2_config=t5,
        backbone_config=backbone,
        decoder_config=decoder,
        height=args.size,
        width=args.size,
    )

    # NxDI skips any part whose folder exists, even one left half-written by a failed compile.
    # Treat a part as compiled only if its model.pt is there, so a failure gets retried.
    compiled = Path(f"/workspace/flux-compiled/{args.model.split('/')[-1]}-{args.size}-tp{args.tp}")
    for part in ["text_encoder", "text_encoder_2", "transformer", "decoder"]:
        if not (compiled / part / "model.pt").exists():
            shutil.rmtree(compiled / part, ignore_errors=True)
    print(f"Compiling into {compiled} (parts already there are skipped).", flush=True)
    app.compile(str(compiled))
    t_compiled = time.perf_counter()
    app.load(str(compiled))
    t_loaded = time.perf_counter()

    Path(args.out).mkdir(exist_ok=True)
    latencies = []
    for i in range(args.num):
        start = time.perf_counter()
        image = app(
            args.prompt,
            height=args.size,
            width=args.size,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
        ).images[0]
        latencies.append(time.perf_counter() - start)
        path = f"{args.out}/image_{int(time.time())}_{i + 1}.png"
        image.save(path)
        print(f"Wrote {path} in {latencies[-1]:.2f} s ({latencies[-1] / args.steps * 1000:.0f} ms/step)")

    # The first image can include one-off warmup; with several images, report steady state without it.
    steady = latencies[1:] if len(latencies) > 1 else latencies
    mean = statistics.mean(steady)
    metrics = {
        "model": args.model, "size": args.size, "steps": args.steps, "tp": args.tp, "num_images": args.num,
        "download_s": t_download - t_start,
        "compile_s": t_compiled - t_download,
        "load_s": t_loaded - t_compiled,
        "latency_s": latencies,
        "first_image_s": latencies[0],
        "latency_mean_s": mean,
        "latency_p50_s": statistics.median(steady),
        "latency_max_s": max(steady),
        "ms_per_step": mean / args.steps * 1000,
        "images_per_s": 1 / mean,
        "images_per_min": 60 / mean,
        "steady_state_excludes_first": len(latencies) > 1,
    }
    print("\nMetrics" + (" (steady state excludes the first image)" if len(latencies) > 1 else ""))
    print(f"  compile {metrics['compile_s']:.1f} s, load {metrics['load_s']:.1f} s")
    print(f"  latency  mean {mean:.2f} s  p50 {metrics['latency_p50_s']:.2f} s  max {metrics['latency_max_s']:.2f} s"
          f"  (first {latencies[0]:.2f} s)")
    print(f"  {metrics['ms_per_step']:.0f} ms/step, {metrics['images_per_s']:.3f} images/s ({metrics['images_per_min']:.1f}/min)")
    if args.metrics:
        Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics).write_text(json.dumps(metrics, indent=2))
        print(f"  wrote {args.metrics}")

if __name__ == "__main__":
    main()
