"""Shared pieces for the G2 scripts: paths, the test clip, and access to the original repo."""
import os
import sys
from pathlib import Path

import numpy as np
import torch

TIER2 = Path(__file__).resolve().parent
REPO = TIER2.parent / "repos" / "StreamDiffusionV2"
SDV2_ROOT = Path("/workspace/livevid/sdv2_root")  # wan_models/ and ckpts/ in the layout the repo expects
ART = Path("/workspace/livevid/artifacts/tier2")
G2 = ART / "g2"
OUT = TIER2 / "out"
CLIP = TIER2 / "data" / "test_clip.mp4"

SEED = 1234
HEIGHT = WIDTH = 512
FRAME_TOKENS = 1024
STEPS = [700, 500]          # demo: --step 2 on [700, 500, 400, 200, 0], v2v drops the 0
NOISE_SCALE = 0.8           # demo/config.py default
NUM_STREAM_CHUNKS = 6       # after the 5-frame session start
CLIP_STRIDE = 2             # 30 fps clip -> 15 fps, close to the model's 16 fps
PROMPT_NAME = "anime"
PROMPT = ("anime key visual, cel shaded, clean bold line art, vibrant flat colors, "
          "expressive eyes, studio anime film still, highly detailed")


def use_repo():
    """Make `import models...` resolve to the unmodified clone."""
    os.environ["STREAMDIFFUSIONV2_ROOT"] = str(SDV2_ROOT)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def load_clip(num_frames):
    """[1, 3, T, H, W] float32 in [-1, 1], every CLIP_STRIDE-th frame of the test clip."""
    import cv2
    cap = cv2.VideoCapture(str(CLIP))
    frames, i = [], 0
    while len(frames) < num_frames:
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"clip ended after {len(frames)} frames")
        if i % CLIP_STRIDE == 0:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    x = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0).float()
    return x / 127.5 - 1.0


def to_u8(video):
    """[T, 3, H, W] in [-1, 1] -> uint8 [T, H, W, 3]."""
    return ((video.float() * 0.5 + 0.5).clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()


def write_mp4(path, frames_u8, fps=15):
    import cv2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_u8.shape[1:3]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_u8:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def psnr(a_u8, b_u8):
    mse = np.mean((a_u8.astype(np.float64) - b_u8.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def noise_scale_and_step(prev_and_new, noise_scale, init_noise_scale):
    """streamv2v/inference.py:51-57 (compute_noise_scale_and_step) for one 4-frame chunk plus the previous frame."""
    d = (prev_and_new[:, :, 1:] - prev_and_new[:, :, :-1]) ** 2
    l2 = (torch.sqrt(d.mean(dim=(0, 1, 3, 4))).max() / 0.2).clamp(0, 1)
    new_scale = (init_noise_scale - 0.1 * l2.item()) * 0.9 + noise_scale * 0.1
    return new_scale, int(1000 * new_scale) - 100
