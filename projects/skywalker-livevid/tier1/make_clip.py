"""Synthetic stand-in for a 10 s webcam clip: 300 frames, 30 fps, 512x512.

A slow pan/zoom over a test photo with a bouncing ball and mild sensor noise, plus two
one-second holds where nothing moves (so the similarity filter has something to skip).
"""
import argparse
import subprocess

import cv2
import numpy as np

from common import INPUT_DIR, SIZE, TIER1_DIR

FPS = 30
HOLDS = ((90, 120), (210, 240))  # frame ranges where the scene is frozen


def write_mp4(path, frames, fps=FPS, crf=18):
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
           "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def read_mp4(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(INPUT_DIR / "person.png"))
    ap.add_argument("--out", default=str(TIER1_DIR / "test_clip.mp4"))
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()

    base = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    rng = np.random.default_rng(0)
    frames, motion_t = [], 0
    for i in range(args.frames):
        if not any(a <= i < b for a, b in HOLDS):
            motion_t += 1
        t = motion_t / FPS
        zoom = 1.25 + 0.15 * np.sin(2 * np.pi * t / 6.0)
        cx = SIZE / 2 + 45 * np.sin(2 * np.pi * t / 4.0)
        cy = SIZE / 2 + 30 * np.sin(2 * np.pi * t / 5.0 + 1.0)
        m = np.float32([[zoom, 0, SIZE / 2 - zoom * cx], [0, zoom, SIZE / 2 - zoom * cy]])
        frame = cv2.warpAffine(base, m, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        bx = int(SIZE / 2 + 190 * np.sin(2 * np.pi * t / 2.5))
        by = int(SIZE - 70 - 260 * abs(np.sin(2 * np.pi * t / 1.6)))
        cv2.circle(frame, (bx, by), 34, (235, 60, 50), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (bx - 10, by - 10), 9, (255, 190, 180), -1, lineType=cv2.LINE_AA)
        noisy = frame.astype(np.float32) + rng.normal(0, 1.0, frame.shape)
        frames.append(np.clip(noisy, 0, 255).astype(np.uint8))

    write_mp4(args.out, frames)
    print(f"wrote {args.out}: {len(frames)} frames, {FPS} fps, holds at {HOLDS}")


if __name__ == "__main__":
    main()
