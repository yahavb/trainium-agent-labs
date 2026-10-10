"""Latency / FPS benchmark for the tier1 Neuron pipeline.

  1. single core (NEURON_RT_VISIBLE_CORES=0): per-stage latency and end-to-end FPS, 200 frames
  2. four cores: one worker process per core, round-robin dispatch through multiprocessing
     queues, outputs reordered by the dispatcher; aggregate FPS and p50/p99 frame latency
  3. similarity filter: skip rate on the test clip, and its effect on delivered FPS

Results go to reports/tier1_bench.json and reports/tier1_bench.md.
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import time
from collections import deque

import numpy as np

from common import ART_DIR, ARTIFACTS, INPUT_DIR, REPORT_DIR, TIER1_DIR, percentile, write_json

DIFF_STRIDE = 4          # the similarity filter looks at every 4th pixel in each dimension
DEFAULT_SKIP_THRESHOLD = 3.0  # mean abs difference on the 0..255 scale


def mean_abs_diff(a, b):
    return float(np.abs(a[::DIFF_STRIDE, ::DIFF_STRIDE].astype(np.int16)
                        - b[::DIFF_STRIDE, ::DIFF_STRIDE].astype(np.int16)).mean())


def stats(ms):
    return {"mean": round(float(np.mean(ms)), 3), "p50": round(percentile(ms, 50), 3),
            "p99": round(percentile(ms, 99), 3)}


def load_frames(n, clip=None):
    """n RGB uint8 frames: the test clip (looped) if present, else the test images."""
    clip = clip or TIER1_DIR / "test_clip.mp4"
    if os.path.exists(clip):
        from make_clip import read_mp4
        src, source = read_mp4(clip), str(clip)
    else:
        from PIL import Image
        src = [np.asarray(Image.open(p).convert("RGB")) for p in sorted(INPUT_DIR.glob("*.png"))]
        source = str(INPUT_DIR)
    return [src[i % len(src)] for i in range(n)], source


def _pin(core):
    # must happen before the Neuron runtime initialises, i.e. before the first model load
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
    import torch
    torch.set_num_threads(2)


def single_core_main(core, frames, warmup, style, result_q):
    _pin(core)
    from pipeline_neuron import NeuronTurbo
    t0 = time.perf_counter()
    pipe = NeuronTurbo(style=style)
    pipe.warmup(warmup)
    load_s = time.perf_counter() - t0
    timings = {}
    t0 = time.perf_counter()
    for f in frames:
        pipe(f, timings)
    wall = time.perf_counter() - t0
    result_q.put({"core": core, "frames": len(frames), "load_and_warmup_s": round(load_s, 2),
                  "wall_s": round(wall, 4), "fps": round(len(frames) / wall, 2),
                  "stage_ms": {k: stats(v) for k, v in timings.items()}})


def worker_main(idx, core, in_q, out_q, style):
    _pin(core)
    from pipeline_neuron import NeuronTurbo
    pipe = NeuronTurbo(style=style)
    pipe.warmup(10)
    out_q.put(("ready", idx, None, None, None))
    while True:
        item = in_q.get()
        if item is None:
            break
        seq, frame, frame_style = item
        t0 = time.perf_counter()
        if frame_style != pipe.style:
            pipe.set_style(frame_style)
        out = pipe(frame)
        out_q.put(("frame", idx, seq, out, (time.perf_counter() - t0) * 1e3))


class CorePool:
    """One worker process per NeuronCore, each holding its own copy of the three graphs."""

    def __init__(self, cores=(0, 1, 2, 3), style="anime", ready_timeout=600):
        ctx = mp.get_context("spawn")
        self.cores = list(cores)
        self.style = style
        self.out_q = ctx.Queue()
        self.in_qs = [ctx.Queue() for _ in self.cores]
        self.procs = [ctx.Process(target=worker_main, args=(i, c, self.in_qs[i], self.out_q, style), daemon=True)
                      for i, c in enumerate(self.cores)]
        t0 = time.perf_counter()
        for p in self.procs:
            p.start()
        for _ in self.procs:
            kind = self.out_q.get(timeout=ready_timeout)[0]
            assert kind == "ready", kind
        self.startup_s = time.perf_counter() - t0

    def close(self):
        for q in self.in_qs:
            q.put(None)
        for p in self.procs:
            p.join(timeout=30)

    def run_stream(self, frames, depth=1, skip_threshold=None, styles=None):
        """Dispatch frames round-robin (at most `depth` in flight per worker) and reorder outputs.

        A frame whose mean abs pixel difference from the last *processed* frame is below
        skip_threshold is not sent; it reuses that frame's output.
        Returns (outputs in input order, info dict).
        """
        n, workers = len(frames), len(self.cores)
        results, source = {}, [None] * n
        submit_t, latency_ms, service_ms, done_t = {}, {}, {}, {}
        inflight = [0] * workers
        per_worker = [0] * workers
        state = {"pending": 0, "max_reorder": 0, "next_out": 0}

        def recv():
            _, idx, seq, out, svc = self.out_q.get(timeout=120)
            now = time.perf_counter()
            results[seq] = out
            done_t[seq] = now
            latency_ms[seq] = (now - submit_t[seq]) * 1e3
            service_ms[seq] = svc
            inflight[idx] -= 1
            state["pending"] -= 1
            # frames that arrived ahead of an earlier, still missing frame wait in the reorder buffer
            while state["next_out"] < n and source[state["next_out"]] is not None \
                    and source[state["next_out"]] in results:
                state["next_out"] += 1
            waiting = sum(1 for s in results if s >= state["next_out"])
            state["max_reorder"] = max(state["max_reorder"], waiting)

        rr, last, last_idx, skipped = 0, None, None, 0
        t_start = time.perf_counter()
        for i, frame in enumerate(frames):
            if skip_threshold is not None and last is not None and mean_abs_diff(frame, last) < skip_threshold:
                source[i] = last_idx
                skipped += 1
                continue
            w = rr
            rr = (rr + 1) % workers
            while inflight[w] >= depth:
                recv()
            source[i] = i
            submit_t[i] = time.perf_counter()
            self.in_qs[w].put((i, frame, styles[i] if styles else self.style))
            inflight[w] += 1
            per_worker[w] += 1
            state["pending"] += 1
            last, last_idx = frame, i
        while state["pending"]:
            recv()
        wall = time.perf_counter() - t_start

        outputs = [results[s] for s in source]
        processed = n - skipped
        lat = list(latency_ms.values())
        info = {
            "frames": n, "processed": processed, "skipped": skipped,
            "skip_rate": round(skipped / n, 4), "depth_per_worker": depth,
            "wall_s": round(wall, 4),
            "fps_delivered": round(n / wall, 2), "fps_processed": round(processed / wall, 2),
            "latency_ms": stats(lat), "worker_service_ms": stats(list(service_ms.values())),
            "frames_per_worker": per_worker, "max_reorder_buffer": state["max_reorder"],
        }
        return outputs, info, {"source": source, "done_t": done_t, "t_start": t_start}


def neuron_env():
    def run(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"error: {e}"
    from importlib.metadata import version
    return {"hostname": run("hostname"), "instance_type": run("neuron-ls --json-output | grep -m1 instance_type"),
            "logical_nc_config": os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
            "torch": version("torch"), "torch_neuronx": version("torch-neuronx"),
            "neuronx_cc": version("neuronx-cc"), "diffusers": version("diffusers")}


def write_markdown(res, path):
    sc, st = res["single_core"], res["single_core"]["stage_ms"]
    lines = ["# Tier 1 benchmark: sd-turbo img2img on Neuron", "",
             f"stabilityai/sd-turbo, 512x512, 1 UNet step, TAESD encode/decode, bf16, batch 1. "
             f"{res['env']['hostname']}, {res['frames']} frames from `{os.path.basename(res['frame_source'])}`.", "",
             "## Compile", "", "| component | compile s | compiler args | cosine vs CPU fp32 (random input) |",
             "|---|---|---|---|"]
    for name, m in res["compile"].items():
        lines.append(f"| {name} | {m.get('compile_s')} | `{' '.join(m.get('compiler_args', []))}` | "
                     f"{m.get('random_input_cosine_vs_cpu_fp32')} |")
    lines += ["", "## Single core (core 0, in-process)", "", "| stage | mean ms | p50 ms | p99 ms |", "|---|---|---|---|"]
    for k in ("pre", "encode", "unet", "decode", "post", "total"):
        lines.append(f"| {k} | {st[k]['mean']} | {st[k]['p50']} | {st[k]['p99']} |")
    lines += ["", f"End-to-end: **{sc['fps']} FPS** ({sc['frames']} frames in {sc['wall_s']} s). "
              "`pre`/`post` are the host uint8<->float conversions.", "",
              "## Worker pool (round-robin dispatcher, reordered outputs)", "",
              "| cores | in flight per worker | FPS | latency p50 ms | latency p99 ms | worker service p50 ms |",
              "|---|---|---|---|---|---|"]
    for row in res["pool"]:
        lines.append(f"| {row['cores']} | {row['depth_per_worker']} | **{row['fps_delivered']}** | "
                     f"{row['latency_ms']['p50']} | {row['latency_ms']['p99']} | {row['worker_service_ms']['p50']} |")
    lines += ["", "Latency is dispatcher submit -> result received (queue wait + worker + IPC).", ""]
    sf = res["similarity_filter"]
    lines += ["## Similarity filter", "",
              f"Mean abs pixel difference (0..255, every {DIFF_STRIDE}th pixel) against the last processed frame, "
              f"on `{os.path.basename(sf['clip'])}` ({sf['clip_frames']} frames).", "",
              "| threshold | skip rate |", "|---|---|"]
    for thr, rate in sf["sweep"].items():
        lines.append(f"| {thr} | {rate * 100:.1f}% |")
    run = sf["four_core_run"]
    lines += ["", f"4 cores with threshold {sf['threshold']}: skipped {run['skipped']}/{run['frames']} "
              f"({run['skip_rate'] * 100:.1f}%), delivered **{run['fps_delivered']} FPS** "
              f"({run['fps_processed']} FPS actually processed).", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--style", default="anime")
    ap.add_argument("--skip-threshold", type=float, default=DEFAULT_SKIP_THRESHOLD)
    ap.add_argument("--out", default=str(REPORT_DIR / "tier1_bench"))
    args = ap.parse_args()

    frames, source = load_frames(args.frames)
    res = {"frames": args.frames, "frame_source": source, "style": args.style, "env": neuron_env(), "compile": {}}
    for name, fname in ARTIFACTS.items():
        meta = (ART_DIR / fname).with_suffix(".meta.json")
        res["compile"][name] = json.load(open(meta)) if meta.exists() else {}

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=single_core_main, args=(0, frames, args.warmup, args.style, q))
    p.start()
    res["single_core"] = q.get(timeout=900)
    p.join()
    print("single_core " + json.dumps(res["single_core"]), flush=True)

    res["pool"] = []
    for cores in ((0,), (0, 1, 2, 3)):
        pool = CorePool(cores, style=args.style)
        for depth in (1, 2):
            pool.run_stream(frames[:20 * len(cores)], depth=depth)  # settle queues
            _, info, _ = pool.run_stream(frames, depth=depth)
            info["cores"] = len(cores)
            res["pool"].append(info)
            print("pool " + json.dumps(info), flush=True)
        if len(cores) == 4:
            from make_clip import read_mp4
            clip_path = TIER1_DIR / "test_clip.mp4"
            clip = read_mp4(clip_path) if clip_path.exists() else frames
            diffs = [mean_abs_diff(clip[i], clip[i - 1]) for i in range(1, len(clip))]
            sweep = {}
            for thr in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
                last, skipped = clip[0], 0
                for f in clip[1:]:
                    if mean_abs_diff(f, last) < thr:
                        skipped += 1
                    else:
                        last = f
                sweep[str(thr)] = round(skipped / len(clip), 4)
            _, run, _ = pool.run_stream(clip, depth=2, skip_threshold=args.skip_threshold)
            res["similarity_filter"] = {
                "clip": str(clip_path), "clip_frames": len(clip), "threshold": args.skip_threshold,
                "pixel_stride": DIFF_STRIDE,
                "consecutive_frame_diff": {"min": round(min(diffs), 3), "p50": round(percentile(diffs, 50), 3),
                                           "max": round(max(diffs), 3)},
                "sweep": sweep, "four_core_run": run}
            print("similarity " + json.dumps(res["similarity_filter"]), flush=True)
        pool.close()

    one = next(r for r in res["pool"] if r["cores"] == 1 and r["depth_per_worker"] == 2)
    four = next(r for r in res["pool"] if r["cores"] == 4 and r["depth_per_worker"] == 2)
    res["summary"] = {"fps_1core_inprocess": res["single_core"]["fps"], "fps_1core_pool": one["fps_delivered"],
                      "fps_4core_pool": four["fps_delivered"],
                      "scaling_4_vs_1_pool": round(four["fps_delivered"] / one["fps_delivered"], 2)}
    write_json(args.out + ".json", res)
    write_markdown(res, args.out + ".md")
    print("SUMMARY " + json.dumps(res["summary"]))
    print("BENCH DONE")


if __name__ == "__main__":
    main()
