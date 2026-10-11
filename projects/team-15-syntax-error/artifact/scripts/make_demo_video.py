#!/usr/bin/env python3
"""
make_demo_video.py -- render scripts/demo.py (real logged runs) as a ~50 s terminal video for the deck.

    python scripts/make_demo_video.py OUT_DIR      -> OUT_DIR/demo.mp4 (H.264) + OUT_DIR/demo_still.png

Scenes: the command being typed; screen 1 (same bug, two checkers); the code that does it
(verdicts._swapped_axes); screen 2 (the level-5 A/B); screen 3 (the scoreboard). Every line on screen is
demo.py's own output, captured with its colours. Needs Pillow and imageio-ffmpeg.
"""

import os
import re
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 10
BG, BAR, FG, DIM = (15, 23, 42), (30, 41, 66), (230, 233, 239), (124, 135, 155)
COL = {"31": (255, 123, 114), "32": (111, 208, 140), "33": (242, 193, 78)}
ACCENT = (232, 145, 90)
FONT = "C:/Windows/Fonts/consola.ttf"
FONTB = "C:/Windows/Fonts/consolab.ttf"
CAPF = "C:/Windows/Fonts/segoeuib.ttf"
SIZE, LH, X0, Y0, WRAP = 26, 40, 90, 150, 104

f_reg, f_bold = ImageFont.truetype(FONT, SIZE), ImageFont.truetype(FONTB, SIZE)
f_cap = ImageFont.truetype(CAPF, 34)
f_title = ImageFont.truetype(FONT, 24)


def parse(line):
    """ANSI line -> [(text, color, bold)]."""
    segs, color, bold = [], FG, False
    for part in re.split(r"(\x1b\[[0-9;]*m)", line):
        m = re.fullmatch(r"\x1b\[([0-9;]*)m", part)
        if m:
            code = m.group(1)
            if code in ("0", ""):
                color, bold = FG, False
            elif code == "1":
                bold = True
            elif code == "2":
                color = DIM
            elif code in COL:
                color = COL[code]
            continue
        if part:
            segs.append((part, color, bold))
    return segs


def wrap(segs):
    """Wrap a segment list at WRAP characters, preserving colours."""
    out, cur, n = [], [], 0
    for text, c, b in segs:
        for word in re.split(r"(\s+)", text):
            if not word:
                continue
            if n + len(word) > WRAP and n > 0 and word.strip():
                out.append(cur)
                cur, n = [], 0
                word = word.lstrip()
                cur.append(("      ", c, b)); n = 6
            cur.append((word, c, b))
            n += len(word)
    out.append(cur)
    return out


def frame(lines, caption, cmd=None, cursor=False, highlight=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 70], fill=BAR)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([30 + i * 34, 24, 52 + i * 34, 46], fill=c)
    d.text((W // 2, 35), "hack-the-chip-seat73 — python scripts/demo.py", font=f_title, fill=DIM, anchor="mm")
    y = Y0 - 50
    if cmd is not None:
        d.text((X0, y), "$ ", font=f_bold, fill=(156, 196, 255))
        d.text((X0 + 32, y), cmd + ("▌" if cursor else ""), font=f_reg, fill=FG)
    y = Y0
    for k, segs in enumerate(lines):
        if highlight is not None and k in highlight:
            d.rectangle([X0 - 16, y - 4, W - 70, y + LH - 6], fill=(58, 44, 30))
        x = X0
        for text, c, b in segs:
            f = f_bold if b else f_reg
            d.text((x, y), text, font=f, fill=c)
            x += d.textlength(text, font=f)
        y += LH
    d.rectangle([0, H - 110, W, H], fill=BAR)
    d.rectangle([0, H - 110, 10, H], fill=ACCENT)
    d.text((60, H - 55), caption, font=f_cap, fill=FG, anchor="lm")
    return img


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    raw = subprocess.run([sys.executable, "scripts/demo.py", "--fast"], capture_output=True,
                         text=True, encoding="utf-8", env=env).stdout
    screens = re.split(r"\n(?=\x1b\[1m[123]\. )", raw.strip("\n"))
    screens = [s.strip("\n").split("\n") for s in screens]

    src = open("projects/02-kernel-agent/verdicts.py", encoding="utf-8").read().split("\n")
    i = next(k for k, l in enumerate(src) if l.startswith("def _swapped_axes"))
    code = src[i:i + 12]
    code_lines = [[("verdicts.py · the change that took levels 3 and 4 from 0/5 to 5/5", (232, 145, 90), True)], []]
    code_lines += [[(l, FG, False)] for l in code]
    hl = {2 + k for k, l in enumerate(code) if "startswith(axes[1]" in l or "good = " in l}

    caps = ["1 · Same bug, two checkers: upstream's verdict vs our code-aware instruction",
            "The change: ~15 lines that find the swapped slice in the model's own code",
            "2 · Level 5, same seat, alternating runs: one sentence of feedback, 0/5 → 5/5",
            "3 · Same Qwen3-8B all day: baseline 0 of 4 levels → final agent 7 of 8"]

    frames = []

    def hold(img, sec):
        frames.extend([img] * int(sec * FPS))

    cmd = "python scripts/demo.py"
    for k in range(len(cmd) + 1):
        hold(frame([], caps[0], cmd[:k], cursor=True), 0.06)
    hold(frame([], caps[0], cmd), 0.4)

    def reveal(lines, cap, per, end_hold, highlight=None, cmd_text=cmd):
        for k in range(1, len(lines) + 1):
            hold(frame(lines[:k], cap, cmd_text, highlight=highlight), per)
        last = frame(lines, cap, cmd_text, highlight=highlight)
        hold(last, end_hold)
        return last

    s1 = [w for l in screens[0] for w in wrap(parse(l))]
    still = reveal(s1, caps[0], 0.22, 9.0)
    reveal(code_lines, caps[1], 0.12, 7.0, highlight=hl, cmd_text="code projects/02-kernel-agent/verdicts.py")
    s2 = [w for l in screens[1] for w in wrap(parse(l))]
    reveal(s2, caps[2], 0.25, 9.5)
    s3 = [w for l in screens[2] for w in wrap(parse(l))]
    reveal(s3, caps[3], 0.25, 8.0)

    still.save(os.path.join(out_dir, "demo_still.png"))
    import imageio_ffmpeg
    mp4 = os.path.join(out_dir, "demo.mp4")
    w = imageio_ffmpeg.write_frames(mp4, (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=1,
                                    output_params=["-crf", "24", "-preset", "medium", "-movflags", "+faststart"])
    w.send(None)
    for im in frames:
        w.send(im.tobytes())
    w.close()
    print(f"{mp4}: {len(frames) / FPS:.1f} s, {os.path.getsize(mp4) / 1e6:.2f} MB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo")
