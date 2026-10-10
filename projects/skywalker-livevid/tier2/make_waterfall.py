"""Two FPS waterfalls from measured JSONs (run in the pod).

    python make_waterfall.py [untuned_end_to_end_fps]

tier2/out/fps_waterfall_dit.png   DiT only: untuned full model (G2) -> resident caches + 8-layer graphs -> 4 cores
tier2/out/fps_waterfall_e2e.png   end to end incl. VAE: untuned baseline (Session C, 5.62 FPS by default) -> 4 cores
"""
import json
import sys

from PIL import Image, ImageDraw, ImageFont

from block_port import ART_DIR
from g2_common import G2, OUT

TAG = "512x512_L1024_cache6144_sink3072_text512"
S = 2
SURFACE, INK, INK2, GRID, BAR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"

# measured inputs
g2_timing = json.load(open(G2 / "parity_neuron.json"))["timing"]
steady = [t["wall_s"] for t in g2_timing[3:]]  # chunks with 2 passes, after the cache has filled
untuned = 4 / (sum(steady) / len(steady))
one = json.load(open(ART_DIR / "g3_pipeline_1core.json"))["steady_state_fps"]
four = json.load(open(ART_DIR / "g3_pipeline_4core.json"))["steady_state_fps"]
e2e = json.load(open(ART_DIR / "g4_e2e_4core.json"))["steady_state_fps"]
# estimated from block latency, footnote only
g1 = json.load(open(ART_DIR / f"g1_block0_{TAG}_bench.json"))["latency_ms"]["mean"]
r1 = json.load(open(ART_DIR / "g2_resident_1layers.json"))["latency_ms_call_plus_y_to_host"]["mean"]
est = lambda ms: 4 / (ms / 1e3 * 30 * 2)
# Untuned end-to-end baseline: Session C's looped-clip run on seat-230 (branch tier2-g2, commit 8b92545).
e2e_base = float(sys.argv[1]) if len(sys.argv) > 1 else 5.62


def draw(path, title, steps, caption, footnote, ymax):
    W, H = 1100 * S, 600 * S
    img = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(img)
    f_title, f_lab, f_small, f_val = (ImageFont.load_default(size=s * S) for s in (22, 16, 13, 20))
    d.text((40 * S, 22 * S), title, font=f_title, fill=INK)
    x0, x1, base, top = 90 * S, W - 40 * S, H - 170 * S, 100 * S
    py = lambda v: base - (base - top) * v / ymax
    for t in range(0, int(ymax) + 1, 10):
        d.line([(x0, py(t)), (x1, py(t))], fill=GRID, width=S)
        d.text((x0 - 10 * S, py(t)), str(t), font=f_small, fill=INK2, anchor="rm")
    d.text((x0 - 10 * S, top - 22 * S), "FPS", font=f_small, fill=INK2, anchor="rm")
    slot, bw, prev = (x1 - x0) / len(steps), 130 * S, None
    for i, (name, sub, fps) in enumerate(steps):
        cx = x0 + slot * (i + 0.5)
        left, right = cx - bw / 2, cx + bw / 2
        if fps is None:  # placeholder: dashed outline, no value
            ytop = py(ymax * 0.12)
            for xx in range(int(left), int(right), 12 * S):
                d.line([(xx, ytop), (min(xx + 6 * S, right), ytop)], fill=INK2, width=S)
            for yy in range(int(ytop), int(base), 12 * S):
                d.line([(left, yy), (left, min(yy + 6 * S, base))], fill=INK2, width=S)
                d.line([(right, yy), (right, min(yy + 6 * S, base))], fill=INK2, width=S)
            d.text((cx, ytop - 10 * S), "to be measured", font=f_lab, fill=INK2, anchor="mb")
        else:
            y = py(fps)
            d.rounded_rectangle([left, y, right, base], radius=4 * S, fill=BAR)
            d.rectangle([left, base - 4 * S, right, base], fill=BAR)  # square at the baseline
            d.text((cx, y - 10 * S), f"{fps:.1f}", font=f_val, fill=INK, anchor="mb")
            if prev:
                d.text((cx - slot / 2, py(max(fps, prev)) - 14 * S), f"x{fps / prev:.1f}", font=f_lab, fill=INK2, anchor="mb")
            prev = fps
        d.text((cx, base + 12 * S), name, font=f_lab, fill=INK, anchor="ma")
        for k, line in enumerate(sub.split("\n")):
            d.text((cx, base + (36 + 20 * k) * S), line, font=f_small, fill=INK2, anchor="ma")
    d.text((40 * S, H - 86 * S), caption, font=f_small, fill=INK)
    if footnote:
        d.text((40 * S, H - 58 * S), footnote, font=f_small, fill=INK2)
    OUT.mkdir(parents=True, exist_ok=True)
    img.resize((W // S, H // S), Image.LANCZOS).save(OUT / path)


draw("fps_waterfall_dit.png", "Causal Wan DiT on Trainium2: frames per second, DiT only",
     [("Untuned full model", "1 core, 30 graphs, caches via host", untuned),
      ("Resident KV caches + 8-layer graphs", "1 core", one),
      ("4-core layer pipeline", "one 8-layer graph per core", four)],
     "Same model, 512x512, 2 steps, steady state, cosine >= 0.999 vs CPU reference at every bar.",
     f"Estimated from block latency, not measured on the full model: {est(g1):.1f} FPS (one block, caches via host), "
     f"{est(r1):.1f} FPS (resident cache, 1 layer per graph).", 50.0)

draw("fps_waterfall_e2e.png", "Video to video on Trainium2: frames per second, end to end including the VAE",
     [("Untuned end to end", "untuned, 1 core, 30 per-layer graphs, caches through host;\n"
                             "clip looped to 25 chunks, first 5 excluded", e2e_base),
      ("4-core pipeline + TAEHV encode / decode", "pixels in to pixels out\n"
       "3 chunks in flight: 26.3 FPS, 458 ms mean / 505 ms p99 latency", e2e)],
     "Same model, 512x512, 2 steps, steady state. Parity: DiT cosine >= 0.999 vs CPU reference; TAEHV cosine >= 0.999 "
     "vs CPU TAEHV; end-to-end output checked visually.", None, 50.0)
print("WATERFALLS OK", {"untuned": round(untuned, 2), "one": one, "four": four, "e2e": e2e, "e2e_base": e2e_base})
