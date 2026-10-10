"""Minimal sd-turbo img2img pipeline on Neuron: TAESD encoder -> UNet (1 step) -> TAESD decoder.

Each component is compiled separately with torch_neuronx.trace (bf16 weights, fixed shapes,
batch 1). No diffusers pipeline object exists at runtime; the scheduler math is done on the host.

  python pipeline_neuron.py compile --component {taesd_enc,unet,taesd_dec}
  python pipeline_neuron.py prepare      # scheduler constants + cached style embeddings (CPU)
  python pipeline_neuron.py check        # compare against tier1/out/reference (reference_cpu.py)
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from common import (ART_DIR, ARTIFACTS, CONSTS_FILE, INPUT_DIR, LATENT, MODEL_ID, OUT_DIR, REF_DIR,
                    REPORT_DIR, SEED, SIZE, STRENGTH, STYLES, STYLES_V1, TAESD_ID, cosine, frame_to_tensor, psnr_u8,
                    strength_to_timestep, tensor_to_frame, turbo_schedule, write_json)

BF16 = torch.bfloat16


# Graph inputs/outputs stay float32 so the host never handles bf16 (and the timestep, which
# bf16 cannot represent exactly, stays exact); the casts are the first and last ops in the graph.
class EncoderWrap(nn.Module):
    def __init__(self, taesd):
        super().__init__()
        self.taesd = taesd

    def forward(self, image):
        return self.taesd.encode(image.to(BF16), return_dict=False)[0].float()


class DecoderWrap(nn.Module):
    def __init__(self, taesd):
        super().__init__()
        self.taesd = taesd

    def forward(self, latents):
        return self.taesd.decode(latents.to(BF16), return_dict=False)[0].float()


class VaeEncoderWrap(nn.Module):
    """Full SD VAE encoder; returns the posterior mean already multiplied by the scaling factor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.scale = vae.config.scaling_factor

    def forward(self, image):
        mean = self.vae.encode(image.to(BF16), return_dict=False)[0].mode()
        return (mean * self.scale).float()


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        out = self.unet(sample.to(BF16), timestep, encoder_hidden_states.to(BF16), return_dict=False)[0]
        return out.float()


def build_component(component):
    """Returns (bf16 wrapper to trace, CPU fp32 output on the example, its seconds, example inputs)."""
    torch.manual_seed(0)
    if component == "unet":
        from diffusers import UNet2DConditionModel
        fp32 = UNet2DConditionModel.from_pretrained(
            MODEL_ID, subfolder="unet", variant="fp16", torch_dtype=torch.float32).eval()
        wrap = UNetWrap
        ref_forward = lambda s, t, e: fp32(s, t, e, return_dict=False)[0]
        example = (torch.randn(1, 4, LATENT, LATENT), torch.tensor([499.0]), torch.randn(1, 77, 1024))
    elif component == "vae_enc":
        from diffusers import AutoencoderKL
        fp32 = AutoencoderKL.from_pretrained(
            MODEL_ID, subfolder="vae", variant="fp16", torch_dtype=torch.float32).eval()
        wrap = VaeEncoderWrap
        ref_forward = lambda x: fp32.encode(x, return_dict=False)[0].mode() * fp32.config.scaling_factor
        example = (torch.rand(1, 3, SIZE, SIZE) * 2 - 1,)
    else:
        from diffusers import AutoencoderTiny
        fp32 = AutoencoderTiny.from_pretrained(TAESD_ID, torch_dtype=torch.float32).eval()
        if component == "taesd_enc":
            wrap = EncoderWrap
            ref_forward = lambda x: fp32.encode(x, return_dict=False)[0]
            example = (torch.rand(1, 3, SIZE, SIZE) * 2 - 1,)
        else:
            wrap = DecoderWrap
            ref_forward = lambda z: fp32.decode(z, return_dict=False)[0]
            example = (torch.randn(1, 4, LATENT, LATENT),)
    with torch.no_grad():
        t0 = time.perf_counter()
        ref = ref_forward(*example)
        cpu_fp32_s = time.perf_counter() - t0
    return wrap(fp32.to(BF16)).eval(), ref, cpu_fp32_s, example


def compile_component(component, force=False, compiler_args=None, tag="", core=0):
    """Trace one component to ART_DIR (skipped if the artifact exists), then validate it on `core`."""
    import torch_neuronx

    # Without this the runtime claims all 4 cores and fails if another process holds any of them.
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", str(core))
    ART_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ART_DIR / (ARTIFACTS[component] if not tag else ARTIFACTS[component].replace(".pt", f"_{tag}.pt"))
    meta_path = out_path.with_suffix(".meta.json")
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    model, ref, cpu_fp32_s, example = build_component(component)

    if out_path.exists() and not force:
        print(f"skip compile of {component}: {out_path} exists (use --force to recompile)")
    else:
        if compiler_args is None:
            compiler_args = ["--auto-cast=none"]
            if component == "unet":
                compiler_args.append("--model-type=unet-inference")
        workdir = ART_DIR / "work" / out_path.stem
        workdir.mkdir(parents=True, exist_ok=True)
        print(f"compiling {component} args={compiler_args} workdir={workdir}", flush=True)
        t0 = time.perf_counter()
        traced = torch_neuronx.trace(model, example, compiler_workdir=str(workdir), compiler_args=compiler_args)
        compile_s = time.perf_counter() - t0
        print(f"compile_s={compile_s:.1f}", flush=True)
        tmp = out_path.with_suffix(".pt.tmp")
        torch.jit.save(traced, str(tmp))
        os.replace(tmp, out_path)
        meta = {"component": component, "artifact": str(out_path), "compiler_args": compiler_args,
                "compile_s": round(compile_s, 1)}
        write_json(meta_path, meta)

    if "random_input_cosine_vs_cpu_fp32" in meta:
        return out_path
    loaded = torch.jit.load(str(out_path))
    with torch.no_grad():
        for _ in range(5):
            out = loaded(*example)
        lat = []
        for _ in range(20):
            t = time.perf_counter()
            out = loaded(*example)
            lat.append((time.perf_counter() - t) * 1e3)
    meta.update({
        "component": component, "artifact": str(out_path), "artifact_bytes": out_path.stat().st_size,
        "input_shapes": [list(e.shape) for e in example], "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
        "random_input_cosine_vs_cpu_fp32": round(cosine(out, ref), 6),
        "random_input_max_abs_err": round((out - ref).abs().max().item(), 5),
        "latency_ms_p50": round(float(np.median(lat)), 3),
        "cpu_fp32_ms": round(cpu_fp32_s * 1e3, 1),
        "torch": torch.__version__, "torch_neuronx": torch_neuronx.__version__,
    })
    write_json(meta_path, meta)
    print("META " + json.dumps(meta), flush=True)
    return out_path


def prepare():
    """Scheduler constants and one CPU text-encoder pass per style prompt."""
    from huggingface_hub import hf_hub_download
    from transformers import CLIPTextModel, CLIPTokenizer

    cfg = json.load(open(hf_hub_download(MODEL_ID, "scheduler/scheduler_config.json")))
    timestep, sigma, sigmas = turbo_schedule(cfg)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float32).eval()

    def embed(prompts):
        out = {}
        for style, prompt in prompts.items():
            ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors="pt").input_ids
            with torch.no_grad():
                out[style] = text_encoder(ids)[0].float().contiguous()
            print(f"{style}: {tuple(out[style].shape)}")
        return out

    torch.manual_seed(SEED)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"timestep": timestep, "sigma": sigma, "sigmas": sigmas, "prediction_type": cfg["prediction_type"],
                "prompts": dict(STYLES), "embeds": embed(STYLES),
                "prompts_v1": dict(STYLES_V1), "embeds_v1": embed(STYLES_V1),
                "noise": torch.randn(1, 4, LATENT, LATENT)}, ART_DIR / CONSTS_FILE)
    print(f"timestep={timestep} sigma={sigma:.6f} prediction_type={cfg['prediction_type']}")
    print(f"wrote {ART_DIR / CONSTS_FILE}")


class NeuronTurbo:
    """Runtime pipeline. Set NEURON_RT_VISIBLE_CORES before constructing to pin it to a core.

    One seeded noise tensor is reused for every frame of a stream (fresh noise per frame flickers).
    strength picks the single timestep; smooth blends each predicted x0 with the previous output
    latent; encoder is "taesd" or "vae" (the full SD VAE encoder).
    """

    def __init__(self, art_dir=ART_DIR, style="anime", strength=STRENGTH, smooth=0.0, encoder="taesd",
                 prompts="current"):
        import torch_neuronx  # noqa: F401  (registers the Neuron ops needed by torch.jit.load)

        consts = torch.load(art_dir / CONSTS_FILE)
        if consts["prediction_type"] != "epsilon":
            raise ValueError(f"unsupported prediction_type {consts['prediction_type']}")
        self.sigmas = consts["sigmas"]
        self.embeds = consts["embeds" if prompts == "current" else "embeds_v1"]
        self.noise = consts["noise"]
        self.encoder = encoder
        self.enc = torch.jit.load(str(art_dir / ARTIFACTS["taesd_enc" if encoder == "taesd" else "vae_enc"]))
        self.unet = torch.jit.load(str(art_dir / ARTIFACTS["unet"]))
        self.dec = torch.jit.load(str(art_dir / ARTIFACTS["taesd_dec"]))
        self.smooth = smooth
        self.prev_x0 = None
        self.set_strength(strength)
        self.set_style(style)

    def set_style(self, style):
        self.style = style
        self.ehs = self.embeds[style]

    def set_strength(self, strength):
        self.strength = strength
        t = strength_to_timestep(strength, len(self.sigmas))
        self.sigma = float(self.sigmas[t])
        self.timestep = torch.tensor([float(t)])
        self.input_scale = 1.0 / (self.sigma ** 2 + 1) ** 0.5

    def encode(self, image):
        # Both encoders return latents in the UNet's (scaled) latent space: TAESD natively (its
        # scaling_factor is 1.0), the VAE graph multiplies by 0.18215 internally.
        return self.enc(image)

    def denoise(self, latents, noise=None, return_eps=False, prev_x0=None):
        """Euler img2img with a single step ending at sigma=0, epsilon prediction."""
        noisy = latents + (self.noise if noise is None else noise) * self.sigma
        eps = self.unet(noisy * self.input_scale, self.timestep, self.ehs)
        x0 = noisy - self.sigma * eps
        prev = self.prev_x0 if prev_x0 is None else prev_x0
        if self.smooth > 0 and prev is not None:
            x0 = torch.lerp(x0, prev, self.smooth)
        self.prev_x0 = x0
        return (x0, eps) if return_eps else x0

    def decode(self, latents):
        return self.dec(latents)

    def warmup(self, n=5):
        frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        for _ in range(n):
            self(frame)
        self.prev_x0 = None

    @torch.no_grad()
    def __call__(self, frame_u8, timings=None):
        """HWC uint8 RGB frame -> restyled HWC uint8 RGB frame."""
        t0 = time.perf_counter()
        x = frame_to_tensor(frame_u8)
        t1 = time.perf_counter()
        z = self.encode(x)
        t2 = time.perf_counter()
        x0 = self.denoise(z)
        t3 = time.perf_counter()
        y = self.decode(x0)
        t4 = time.perf_counter()
        out = tensor_to_frame(y)
        t5 = time.perf_counter()
        if timings is not None:
            for k, v in (("pre", t1 - t0), ("encode", t2 - t1), ("unet", t3 - t2),
                         ("decode", t4 - t3), ("post", t5 - t4), ("total", t5 - t0)):
                timings.setdefault(k, []).append(v * 1e3)
        return out


@torch.no_grad()
def check():
    from PIL import Image, ImageDraw

    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
    pipe = NeuronTurbo(prompts="v1")  # the reference was generated with the v1 prompts
    check_dir = OUT_DIR / "check"
    check_dir.mkdir(parents=True, exist_ok=True)
    ref_summary = json.load(open(REF_DIR / "reference.json"))

    rows, sheets = [], {}
    for ref_path in sorted(REF_DIR.glob("*.pt")):
        ref = torch.load(ref_path)
        name = ref_path.stem
        pipe.set_style(ref["style"])
        gt = ref["image_u8"].numpy()
        src = np.asarray(Image.open(INPUT_DIR / ref["image"]).convert("RGB"))

        # 1) UNet alone, fed exactly the tensors the CPU reference UNet saw
        eps_n = pipe.unet(ref["unet_in"], ref["timestep"].reshape(1).float(), ref["embeds"])
        noisy_ref = ref["init_latents"] + ref["noise"] * ref["sigma"]
        x0_n = noisy_ref - ref["sigma"] * eps_n
        # 2) our full pipeline: TAESD encode -> same noise -> UNet -> TAESD decode
        z = pipe.encode(frame_to_tensor(src))
        x0_e2e, eps_e2e = pipe.denoise(z, noise=ref["noise"], return_eps=True)
        img_e2e = tensor_to_frame(pipe.decode(x0_e2e))
        # 3) attribution: reference x0 through the TAESD decoder only
        img_dec_only = tensor_to_frame(pipe.decode(ref["x0"]))

        row = {
            "case": name,
            "embeds_max_abs_diff": (pipe.ehs - ref["embeds"]).abs().max().item(),
            "host_unet_input_max_abs_diff": (noisy_ref * pipe.input_scale - ref["unet_in"]).abs().max().item(),
            "unet_eps_cosine": cosine(eps_n, ref["eps"]),
            "unet_eps_max_abs_err": (eps_n - ref["eps"]).abs().max().item(),
            "unet_x0_cosine": cosine(x0_n, ref["x0"]),
            "taesd_vs_vae_latent_cosine": cosine(z, ref["init_latents"]),
            "taesd_vs_vae_latent_std_ratio": (z.std() / ref["init_latents"].std()).item(),
            "e2e_x0_cosine": cosine(x0_e2e, ref["x0"]),
            "e2e_psnr_db": psnr_u8(img_e2e, gt),
            "e2e_image_cosine": cosine(torch.from_numpy(img_e2e).float(), torch.from_numpy(gt).float()),
            "taesd_decoder_only_psnr_db": psnr_u8(img_dec_only, gt),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

        strip = np.concatenate([src, gt, img_e2e], axis=1)
        Image.fromarray(strip).save(check_dir / f"{name}.png")
        sheets.setdefault(ref["image"], []).append((ref["style"], strip, row))

    for image_name, items in sheets.items():
        sheet = Image.fromarray(np.concatenate([s for _, s, _ in items], axis=0))
        draw = ImageDraw.Draw(sheet)
        for i, (style, _, row) in enumerate(items):
            for j, label in enumerate(("input", f"CPU diffusers fp32 ({style})",
                                       f"Neuron bf16 TAESD  PSNR {row['e2e_psnr_db']:.1f} dB")):
                draw.rectangle([j * SIZE, i * SIZE, j * SIZE + 8 + 6 * len(label), i * SIZE + 16], fill=(0, 0, 0))
                draw.text((j * SIZE + 4, i * SIZE + 3), label, fill=(255, 255, 255))
        sheet.save(check_dir / f"sheet_{image_name}")

    keys = [k for k in rows[0] if k != "case"]
    summary = {
        "timestep": float(pipe.timestep), "sigma": pipe.sigma,
        "reference_timestep": ref_summary["scheduler"]["timestep_used_at_strength_0.5"],
        "reference_sigma": ref_summary["scheduler"]["sigma_used"],
        "min": {k: min(r[k] for r in rows) for k in keys},
        "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
        "max": {k: max(r[k] for r in rows) for k in keys},
        "unet_cosine_pass_0.999": all(r["unet_eps_cosine"] >= 0.999 for r in rows),
        "cases": rows,
    }
    write_json(check_dir / "check.json", summary)
    write_json(REPORT_DIR / "tier1_check.json", summary)
    print("SUMMARY " + json.dumps({k: summary[k] for k in ("min", "mean", "unet_cosine_pass_0.999")}))
    print("CHECK DONE")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--component", required=True, choices=list(ARTIFACTS))
    c.add_argument("--force", action="store_true")
    c.add_argument("--compiler-args", default=None, help="space separated; overrides the defaults")
    c.add_argument("--tag", default="", help="suffix for the artifact name (experiments)")
    c.add_argument("--core", type=int, default=0, help="NeuronCore used to validate the artifact")
    sub.add_parser("prepare")
    sub.add_parser("check")
    args = ap.parse_args()

    if args.cmd == "compile":
        cargs = args.compiler_args.split() if args.compiler_args is not None else None
        compile_component(args.component, args.force, cargs, args.tag, args.core)
    elif args.cmd == "prepare":
        prepare()
    else:
        check()


if __name__ == "__main__":
    main()
