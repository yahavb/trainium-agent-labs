"""tier2/out/block_latency.png: per-block latency before/after keeping the KV caches on the device.

Reads the measured JSONs in artifacts/tier2 (run in the pod). One series, one hue, direct value labels.
"""
import json

from PIL import Image, ImageDraw, ImageFont

from block_port import ART_DIR
from g2_common import OUT

TAG = "512x512_L1024_cache6144_sink3072_text512"
g1 = json.load(open(ART_DIR / f"g1_block0_{TAG}_bench.json"))["latency_ms"]
r1 = json.load(open(ART_DIR / "g2_resident_1layers.json"))
r8 = json.load(open(ART_DIR / "g2_resident_8layers.json"))
DEVICE_MS = 4.21  # neuron-explorer total_exec_time for the G1 block NEFF

bars = [
    ("K/V caches cross the host every call (G1)", g1["mean"], f"{g1['mean']:.1f} ms"),
    ("K/V caches resident on device, 1 layer per graph", r1["latency_ms_call_plus_y_to_host"]["mean"],
     f"{r1['latency_ms_call_plus_y_to_host']['mean']:.1f} ms  (p50 {r1['latency_ms_call_plus_y_to_host']['p50']:.1f})"),
    ("Resident caches, 8 layers in one graph (per layer)", r8["latency_ms_per_layer"],
     f"{r8['latency_ms_per_layer']:.1f} ms"),
]

S = 2  # supersample
W, H = 1100 * S, 430 * S
SURFACE, INK, INK2, GRID, BAR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
img = Image.new("RGB", (W, H), SURFACE)
d = ImageDraw.Draw(img)
f_title, f_lab, f_small = (ImageFont.load_default(size=s * S) for s in (22, 16, 13))

x0, x1, top = 40 * S, W - 230 * S, 96 * S
xmax = 12.0
px = lambda v: x0 + (x1 - x0) * v / xmax
d.text((x0, 22 * S), "Latency of one Wan2.1-1.3B causal block on one NeuronCore", font=f_title, fill=INK)
d.text((x0, 54 * S), "ms per block call, mean of 50 calls, 512x512, 1024 tokens, 6144-token KV cache, bf16",
       font=f_small, fill=INK2)

row_h, bar_h = 92 * S, 26 * S
bottom = top + row_h * len(bars)
for t in range(0, 13, 2):
    d.line([(px(t), top), (px(t), bottom)], fill=GRID, width=S)
    d.text((px(t), bottom + 8 * S), str(t), font=f_small, fill=INK2, anchor="ma")
d.text((x1, bottom + 28 * S), "ms", font=f_small, fill=INK2, anchor="ra")

for i, (label, value, text) in enumerate(bars):
    y = top + i * row_h
    d.text((x0, y + 8 * S), label, font=f_lab, fill=INK)
    by = y + 36 * S
    d.rounded_rectangle([x0, by, px(value), by + bar_h], radius=4 * S, fill=BAR)
    d.rectangle([x0, by, x0 + 4 * S, by + bar_h], fill=BAR)  # square at the baseline, rounded at the data end
    d.text((px(value) + 10 * S, by + bar_h // 2), text, font=f_lab, fill=INK, anchor="lm")

xd = px(DEVICE_MS)
for yy in range(top, bottom, 10 * S):  # dashed reference rule
    d.line([(xd, yy), (xd, yy + 5 * S)], fill=INK2, width=S)
d.text((xd + 6 * S, top - 4 * S), f"device execution only: {DEVICE_MS} ms (neuron-explorer)", font=f_small,
       fill=INK2, anchor="lb")

OUT.mkdir(parents=True, exist_ok=True)
img.resize((W // S, H // S), Image.LANCZOS).save(OUT / "block_latency.png")
print("CHART OK", [(b[0], round(b[1], 2)) for b in bars])
