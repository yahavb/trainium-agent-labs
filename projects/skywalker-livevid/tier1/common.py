"""Shared constants and host-side helpers for the tier1 sd-turbo img2img pipeline."""
import json
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "stabilityai/sd-turbo"
TAESD_ID = "madebyollin/taesd"

TIER1_DIR = Path(__file__).resolve().parent
REPO_DIR = TIER1_DIR.parent
OUT_DIR = TIER1_DIR / "out"
INPUT_DIR = OUT_DIR / "inputs"
REF_DIR = OUT_DIR / "reference"
ART_DIR = Path("/workspace/livevid/artifacts/tier1")
REPORT_DIR = REPO_DIR / "reports"

SIZE = 512
LATENT = 64
STRENGTH = 0.5
NUM_INFERENCE_STEPS = 2  # with strength 0.5 this is exactly one UNet evaluation
SEED = 1234

TEST_IMAGES = {
    "cat": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
    "person": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png",
    "landscape": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png",
}

STYLES = {
    "anime": "anime style portrait, cel shading, clean line art, vibrant colors, soft lighting, "
             "detailed expressive eyes, 2D animation film still",
    "claymation": "claymation portrait, stop-motion clay figure, sculpted plasticine, soft rounded features, "
                  "matte clay texture, handmade miniature set, warm studio lighting",
    "oil_painting": "oil painting portrait, visible thick brush strokes, impasto texture on canvas, "
                    "rich warm colors, soft dramatic lighting, classical fine art",
    "pixel_art": "pixel art portrait, 16-bit retro game sprite, large blocky pixels, limited color palette, "
                 "hard edges, dithered shading",
}

# Demo defaults picked from the quality sweep on a real webcam clip (reports/tier1_quality.json):
# one strength per style (the non-anime styles need more noise to show), and each predicted x0
# latent is blended with the previous output latent to calm frame-to-frame flicker.
STYLE_STRENGTH = {"anime": 0.55, "oil_painting": 0.65, "claymation": 0.75, "pixel_art": 0.75}
SMOOTH = 0.35

# First version of the prompts; the CPU reference (tier1/out/reference) was generated with these.
STYLES_V1 = {
    "anime": "anime key visual, cel shaded, clean bold line art, vibrant flat colors, "
             "expressive eyes, studio anime film still, highly detailed",
    "claymation": "claymation stop-motion still, handmade plasticine clay figures, soft rounded shapes, "
                  "visible fingerprints in the clay, miniature set, warm studio lighting",
    "oil_painting": "classical oil painting on canvas, thick impasto brush strokes, rich saturated pigments, "
                    "dramatic chiaroscuro lighting, fine art masterpiece",
    "pixel_art": "16-bit pixel art, retro video game sprite style, crisp chunky pixels, "
                 "limited color palette, dithering shading, sharp edges",
}

ARTIFACTS = {
    "taesd_enc": f"taesd_enc_1x3x{SIZE}x{SIZE}_bf16.pt",
    "unet": f"sdturbo_unet_1x4x{LATENT}x{LATENT}_t1_ehs1x77x1024_bf16.pt",
    "taesd_dec": f"taesd_dec_1x4x{LATENT}x{LATENT}_bf16.pt",
    "vae_enc": f"sdturbo_vae_enc_1x3x{SIZE}x{SIZE}_bf16.pt",
}
CONSTS_FILE = "sdturbo_consts_embeds_1x77x1024_fp32.pt"


def turbo_schedule(cfg, num_inference_steps=NUM_INFERENCE_STEPS, strength=STRENGTH):
    """Single (timestep, sigma) that EulerDiscreteScheduler img2img uses for this config."""
    if cfg["timestep_spacing"] != "trailing" or cfg["beta_schedule"] != "scaled_linear":
        raise ValueError(f"unsupported scheduler config: {cfg}")
    n = cfg["num_train_timesteps"]
    betas = torch.linspace(cfg["beta_start"] ** 0.5, cfg["beta_end"] ** 0.5, n, dtype=torch.float32) ** 2
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    timesteps = np.round(np.arange(n, 0, -n / num_inference_steps)) - 1
    init = min(int(num_inference_steps * strength), num_inference_steps)
    timesteps = timesteps[num_inference_steps - init:]
    if len(timesteps) != 1:
        raise ValueError(f"expected exactly one denoising step, got timesteps {timesteps}")
    t = int(timesteps[0])
    sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    return t, float(sigmas[t]), sigmas


def strength_to_timestep(strength, num_train_timesteps=1000):
    """Timestep of a single trailing-spaced step at this img2img strength (0.5 -> 499, as diffusers
    uses for strength 0.5 with 2 inference steps)."""
    return min(max(int(round(num_train_timesteps * strength)) - 1, 0), num_train_timesteps - 1)


def frame_to_tensor(frame_u8):
    """HWC uint8 RGB -> [1,3,H,W] float32 in [-1,1]."""
    x = torch.from_numpy(np.ascontiguousarray(frame_u8)).permute(2, 0, 1).unsqueeze(0)
    return x.float().div_(127.5).sub_(1.0)


def tensor_to_frame(x):
    """[1,3,H,W] float in [-1,1] -> HWC uint8 RGB."""
    x = x[0].float().add(1.0).mul_(127.5).round_().clamp_(0, 255)
    return x.permute(1, 2, 0).to(torch.uint8).contiguous().numpy()


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().double(), b.flatten().double(), dim=0).item()


def psnr_u8(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
