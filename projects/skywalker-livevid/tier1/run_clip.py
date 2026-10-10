"""Runs the 4-core pipeline over a clip and writes a side-by-side (input | styled) mp4.

The overlay shows the rolling delivered FPS measured while the clip was processed (frames are
fed as fast as the pool accepts them, so this is throughput, not the clip's playback rate).
"""
import argparse
import json

import cv2
import numpy as np

from bench import DEFAULT_SKIP_THRESHOLD, CorePool
from common import OUT_DIR, STYLES, TIER1_DIR, write_json
from make_clip import FPS, read_mp4, write_mp4

WINDOW = 30  # frames in the rolling FPS window


def label(img, text, org, scale=0.6):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=str(TIER1_DIR / "test_clip.mp4"))
    ap.add_argument("--out", default=str(OUT_DIR / "test_clip_styled.mp4"))
    ap.add_argument("--styles", default=",".join(STYLES), help="comma separated; the clip is split evenly")
    ap.add_argument("--skip-threshold", type=float, default=DEFAULT_SKIP_THRESHOLD,
                    help="similarity filter threshold; negative disables it")
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()

    frames = read_mp4(args.clip)
    n = len(frames)
    names = args.styles.split(",")
    styles = [names[min(i * len(names) // n, len(names) - 1)] for i in range(n)]
    threshold = args.skip_threshold if args.skip_threshold >= 0 else None

    pool = CorePool((0, 1, 2, 3), style=names[0])
    pool.run_stream(frames[:40], depth=args.depth)  # warm the queues
    outputs, info, trace = pool.run_stream(frames, depth=args.depth, skip_threshold=threshold, styles=styles)
    pool.close()

    # frame i is deliverable once every processed frame up to its source has come back
    ready, latest = [], trace["t_start"]
    for src in trace["source"]:
        latest = max(latest, trace["done_t"][src])
        ready.append(latest)

    composed = []
    for i, (src_frame, out) in enumerate(zip(frames, outputs)):
        j = max(0, i - WINDOW)
        span = ready[i] - (ready[j] if i > 0 else trace["t_start"])
        fps = (i - j if i > 0 else 1) / max(span, 1e-6)
        canvas = np.concatenate([src_frame, out], axis=1)
        skipped = trace["source"][i] != i
        label(canvas, "input", (10, 24))
        label(canvas, f"sd-turbo 1 step | {styles[trace['source'][i]]} | 4 NeuronCores", (522, 24))
        label(canvas, f"{fps:5.1f} FPS" + ("  (skipped: reused)" if skipped else ""), (522, 50), 0.7)
        composed.append(canvas)

    write_mp4(args.out, composed, fps=FPS)
    info.update({"clip": args.clip, "out": args.out, "styles": names, "skip_threshold": threshold,
                 "pool_startup_s": round(pool.startup_s, 2)})
    write_json(args.out.replace(".mp4", ".json"), info)
    print("RESULT " + json.dumps(info))
    print("CLIP DONE")


if __name__ == "__main__":
    main()
