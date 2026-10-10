"""Visual quality sweep on a clip: strength, temporal smoothing, prompts and encoder.

  python quality.py sweep --configs taesd:0.45:0.2:anime:current ...   # encoder:strength:smooth:style:prompts
  python quality.py grid --configs <name> ... --out quality_grid.mp4

Each config writes tier1/out/quality/<name>.mp4, stills and a metrics json. Configs are spread
over the 4 NeuronCores, one process per core.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import cv2
import numpy as np

from common import OUT_DIR, SIZE, TIER1_DIR, write_json
from make_clip import read_mp4, write_mp4

QDIR = OUT_DIR / "quality"
STILL_FRAMES = (45, 150, 255)


def config_name(cfg):
    enc, strength, smooth, style, prompts = cfg
    return f"{enc}_s{float(strength):.2f}_a{float(smooth):.2f}_{style}" + ("" if prompts == "current" else f"_{prompts}")


def mad(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def run_configs(core, clip_path, configs, lag, result_q):
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    from pipeline_neuron import NeuronTurbo

    frames = read_mp4(clip_path)
    input_motion = float(np.mean([mad(frames[i], frames[i - 1]) for i in range(1, len(frames))]))
    pipes = {}
    for cfg in configs:
        enc, strength, smooth, style, prompts = cfg
        key = (enc, prompts)
        if key not in pipes:
            pipes[key] = NeuronTurbo(encoder=enc, prompts=prompts)
            pipes[key].warmup(5)
        pipe = pipes[key]
        pipe.set_style(style)
        pipe.set_strength(float(strength))
        pipe.smooth = float(smooth)
        pipe.prev_x0 = None
        timings, outs, history = {}, [], []
        for i, f in enumerate(frames):
            # lag > 1 emulates the live server, where the newest finished latent is a few frames old
            prev = history[-lag] if (pipe.smooth > 0 and len(history) >= lag) else None
            pipe.prev_x0 = prev
            outs.append(pipe(f, timings))
            history.append(pipe.prev_x0)
            history = history[-8:]
        name = config_name(cfg) + (f"_lag{lag}" if lag != 1 else "")
        QDIR.mkdir(parents=True, exist_ok=True)
        write_mp4(QDIR / f"{name}.mp4", outs)
        for i in STILL_FRAMES:
            if i < len(outs):
                cv2.imwrite(str(QDIR / f"{name}_f{i:03d}.jpg"), cv2.cvtColor(outs[i], cv2.COLOR_RGB2BGR))
        flicker = float(np.mean([mad(outs[i], outs[i - 1]) for i in range(1, len(outs))]))
        metrics = {
            "name": name, "encoder": enc, "strength": float(strength), "smooth": float(smooth), "style": style,
            "prompts": prompts, "lag": lag, "timestep": float(pipe.timestep), "sigma": round(pipe.sigma, 4),
            # frame-to-frame change of the output; input_motion is the same number for the input clip
            "flicker": round(flicker, 2), "input_motion": round(input_motion, 2),
            "flicker_ratio": round(flicker / input_motion, 2),
            # how far the output moved away from the input (higher = more restyling)
            "change_from_input": round(float(np.mean([mad(o, f) for o, f in zip(outs[::5], frames[::5])])), 2),
            "stage_ms_p50": {k: round(float(np.median(v)), 2) for k, v in timings.items()},
        }
        write_json(QDIR / f"{name}.json", metrics)
        print("METRICS " + json.dumps(metrics), flush=True)
        result_q.put(metrics)
    result_q.put(None)


def sweep(args):
    configs = [tuple(c.split(":")) for c in args.configs]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for core in range(4):
        mine = configs[core::4]
        if mine:
            p = ctx.Process(target=run_configs, args=(core, args.clip, mine, args.lag, q))
            p.start()
            procs.append(p)
    done, results = 0, []
    while done < len(procs):
        r = q.get(timeout=1800)
        if r is None:
            done += 1
        else:
            results.append(r)
    for p in procs:
        p.join()
    results.sort(key=lambda r: r["name"])
    print(f"{'name':44s} {'flicker':>8s} {'ratio':>6s} {'change':>7s} {'total ms':>9s}")
    for r in results:
        print(f"{r['name']:44s} {r['flicker']:8.2f} {r['flicker_ratio']:6.2f} {r['change_from_input']:7.2f} "
              f"{r['stage_ms_p50']['total']:9.1f}")
    print("SWEEP DONE")


def label(img, lines):
    y = 6
    for i, s in enumerate(lines):
        scale = 0.6 if i == 0 else 0.48
        (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        box = img[y:y + h + base + 6, 6:6 + w + 8]
        box[:] = box // 4
        cv2.putText(img, s, (10, y + h + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += h + base + 8


def grid(args):
    """Tiles the input clip and the chosen configs (3 columns) into one mp4 plus still images."""
    tiles = [("input", ["input (webcam)"], read_mp4(args.clip))]
    for spec in args.configs:
        name, _, text = spec.partition("=")
        m = json.load(open(QDIR / f"{name}.json"))
        lines = [text or name, f"strength {m['strength']}  smooth {m['smooth']}  {m['encoder']} enc",
                 f"flicker {m['flicker']:.1f} (input {m['input_motion']:.1f})"]
        tiles.append((name, lines, read_mp4(QDIR / f"{name}.mp4")))
    cols = args.cols
    rows = (len(tiles) + cols - 1) // cols
    t = args.tile
    n = min(len(frames) for _, _, frames in tiles)
    out = []
    for i in range(n):
        canvas = np.zeros((rows * t, cols * t, 3), np.uint8)
        for k, (_, lines, frames) in enumerate(tiles):
            tile = cv2.resize(frames[i], (t, t), interpolation=cv2.INTER_AREA)
            label(tile, lines)
            r, c = divmod(k, cols)
            canvas[r * t:(r + 1) * t, c * t:(c + 1) * t] = tile
        out.append(canvas)
    path = OUT_DIR / args.out
    write_mp4(path, out)
    stem = str(path).replace(".mp4", "")
    for i in STILL_FRAMES:
        if i < n:
            cv2.imwrite(f"{stem}_f{i:03d}.jpg", cv2.cvtColor(out[i], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {path} ({n} frames, {cols}x{rows} tiles) and stills")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--configs", nargs="+", required=True, help="encoder:strength:smooth:style:prompts")
    s.add_argument("--lag", type=int, default=1, help="frames between a frame and the latent it is blended with")
    g = sub.add_parser("grid")
    g.add_argument("--configs", nargs="+", required=True, help="config name, optionally name=label")
    g.add_argument("--out", default="quality_grid.mp4")
    g.add_argument("--cols", type=int, default=3)
    g.add_argument("--tile", type=int, default=448)
    for p in (s, g):
        p.add_argument("--clip", default=str(TIER1_DIR / "webcam_clip.mp4"))
    args = ap.parse_args()
    t0 = time.perf_counter()
    (sweep if args.cmd == "sweep" else grid)(args)
    print(f"elapsed {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
