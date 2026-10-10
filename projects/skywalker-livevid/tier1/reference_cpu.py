"""CPU fp32 ground truth: diffusers AutoPipelineForImage2Image with sd-turbo, one real step.

For every (test image, style) pair this saves the output PNG plus the intermediate tensors
(VAE init latents, noise, UNet input/output, x0 latents) so the Neuron pipeline can be
checked with exactly the same noise and compared stage by stage.
"""
import json
import time

import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image
from huggingface_hub import hf_hub_download
from PIL import Image

from common import (INPUT_DIR, MODEL_ID, NUM_INFERENCE_STEPS, REF_DIR, SEED, STRENGTH, STYLES_V1 as STYLES,
                    turbo_schedule, write_json)


def main():
    REF_DIR.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(hf_hub_download(MODEL_ID, "scheduler/scheduler_config.json")))
    t_expected, sigma_expected, _ = turbo_schedule(cfg)

    pipe = AutoPipelineForImage2Image.from_pretrained(MODEL_ID, variant="fp16", torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)

    cap = {}
    orig_add_noise = pipe.scheduler.add_noise

    def add_noise(original_samples, noise, timesteps):
        cap["init_latents"] = original_samples.detach().clone()
        cap["noise"] = noise.detach().clone()
        return orig_add_noise(original_samples, noise, timesteps)

    pipe.scheduler.add_noise = add_noise

    def unet_hook(_module, args, kwargs, output):
        cap["unet_calls"] = cap.get("unet_calls", 0) + 1
        cap["unet_in"] = args[0].detach().clone()
        cap["timestep"] = torch.as_tensor(args[1]).detach().clone()
        cap["embeds"] = kwargs["encoder_hidden_states"].detach().clone()
        cap["eps"] = output[0].detach().clone()

    pipe.unet.register_forward_hook(unet_hook, with_kwargs=True)

    cases = []
    for img_path in sorted(INPUT_DIR.glob("*.png")):
        image = Image.open(img_path).convert("RGB")
        for style, prompt in STYLES.items():
            cap.clear()
            generator = torch.Generator("cpu").manual_seed(SEED)
            t0 = time.perf_counter()
            x0 = pipe(prompt=prompt, image=image, strength=STRENGTH,
                      num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=0.0,
                      generator=generator, output_type="latent").images
            with torch.no_grad():
                decoded = pipe.vae.decode(x0 / pipe.vae.config.scaling_factor, return_dict=False)[0]
            out = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
            dt = time.perf_counter() - t0

            name = f"{img_path.stem}__{style}"
            out.save(REF_DIR / f"{name}.png")
            sigma_used = float(pipe.scheduler.sigmas[pipe.scheduler.timesteps.tolist().index(
                float(cap["timestep"]))])
            torch.save({"image": img_path.name, "style": style, "prompt": prompt,
                        "sigma": sigma_used, "x0": x0.detach().clone(),
                        "image_u8": torch.from_numpy(np.asarray(out).copy()),
                        **{k: v for k, v in cap.items() if k != "unet_calls"}},
                       REF_DIR / f"{name}.pt")
            cases.append({"name": name, "seconds": round(dt, 3), "unet_calls": cap["unet_calls"],
                          "timestep": float(cap["timestep"]), "sigma": sigma_used})
            print(f"{name}: {dt:.2f}s unet_calls={cap['unet_calls']} "
                  f"t={float(cap['timestep'])} sigma={sigma_used:.6f}", flush=True)

    summary = {
        "model": MODEL_ID, "dtype": "float32", "weights_variant": "fp16", "device": "cpu",
        "strength": STRENGTH, "num_inference_steps": NUM_INFERENCE_STEPS, "guidance_scale": 0.0,
        "seed": SEED,
        "scheduler": {
            "class": cfg["_class_name"],
            "prediction_type": cfg["prediction_type"],
            "timestep_spacing": cfg["timestep_spacing"],
            "steps_offset": cfg["steps_offset"],
            "beta_schedule": cfg["beta_schedule"],
            "full_timesteps_for_2_steps": pipe.scheduler.timesteps.tolist(),
            "full_sigmas_for_2_steps": pipe.scheduler.sigmas.tolist(),
            "timestep_used_at_strength_0.5": cases[0]["timestep"],
            "sigma_used": cases[0]["sigma"],
            "host_math_timestep": t_expected,
            "host_math_sigma": sigma_expected,
        },
        "vae_scaling_factor": pipe.vae.config.scaling_factor,
        "cases": cases,
    }
    write_json(REF_DIR / "reference.json", summary)
    print("SUMMARY " + json.dumps(summary["scheduler"]))
    print("REFERENCE DONE")


if __name__ == "__main__":
    main()
