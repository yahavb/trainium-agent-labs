"""Quick profile -> reports/tier1_profile.md.

  1. per-core utilization (neuron-monitor) while the 4-core pool runs flat out
  2. per-engine active % and % of peak for one UNet call, from a neuron-explorer summary:
       neuron-explorer capture -n <unet workdir>/graph.neff -s unet.ntff
       neuron-explorer view -n ... -s unet.ntff --output-format summary-json > unet_summary.json
"""
import argparse
import json
import subprocess
import tempfile
import threading
import time

import numpy as np

from bench import CorePool, load_frames
from common import ART_DIR, REPORT_DIR, TIER1_DIR

MONITOR_INDEX_OFFSET = 2  # see server.py


def sample_monitor(samples, stop):
    cfg = {"period": "1s", "system_metrics": [],
           "neuron_runtimes": [{"tag_filter": ".*", "metrics": [{"type": "neuroncore_counters"}]}]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
    proc = subprocess.Popen(["neuron-monitor", "-c", f.name], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    for line in proc.stdout:
        if stop.is_set():
            break
        logical, physical = {}, {}
        for rt in json.loads(line).get("neuron_runtime_data") or []:
            cores = ((rt.get("report") or {}).get("neuroncore_counters") or {}).get("neuroncores_in_use") or {}
            for idx, core in cores.items():
                logical[int(idx)] = max(logical.get(int(idx), 0.0), core["neuroncore_utilization"])
                for name, sub in (core.get("v3d") or {}).items():
                    physical[name] = max(physical.get(name, 0.0), sub["neuroncore_utilization"])
        if logical:
            samples.append((logical, physical))
    proc.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--summary", default=str(ART_DIR / "profile" / "unet_summary.json"))
    args = ap.parse_args()

    frames, source = load_frames(200, TIER1_DIR / "webcam_clip.mp4")
    pool = CorePool((0, 1, 2, 3))
    samples, stop = [], threading.Event()
    threading.Thread(target=sample_monitor, args=(samples, stop), daemon=True).start()
    runs, t0 = [], time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        _, info, _ = pool.run_stream(frames, depth=2)
        runs.append(info)
    stop.set()
    pool.close()
    samples = samples[3:]  # the first periods include pool start-up
    fps = float(np.mean([r["fps_delivered"] for r in runs]))

    lines = ["# Tier 1 quick profile", "",
             f"sd-turbo img2img, 512x512, 1 step, TAESD encode/decode, bf16. Frames from `{source.split('/')[-1]}`.", "",
             "## Per-core utilization during the 4-core run", "",
             f"Pool running flat out (2 frames in flight per worker) for {args.seconds:.0f} s: **{fps:.1f} FPS**, "
             f"latency p50 {runs[-1]['latency_ms']['p50']} ms. neuron-monitor, 1 s periods, {len(samples)} samples. "
             f"Each logical core is two physical cores (logical-neuroncore-config 2); the logical figure is their mean.", "",
             "| visible core | monitor index | logical util mean % | min % | max % | physical cores mean % |",
             "|---|---|---|---|---|---|"]
    for core in range(4):
        idx = (core + MONITOR_INDEX_OFFSET) % 4
        vals = [s[0].get(idx, 0.0) for s in samples]
        phys = [f"{name} {np.mean([s[1].get(name, 0.0) for s in samples]):.1f}"
                for name in (f"nc_v3.{2 * idx}", f"nc_v3.{2 * idx + 1}")]
        lines.append(f"| {core} | {idx} | {np.mean(vals):.1f} | {np.min(vals):.1f} | {np.max(vals):.1f} | "
                     f"{', '.join(phys)} |")

    summary = next(iter(json.load(open(args.summary)).values()))
    pct = lambda key: f"{100 * summary[key]:.1f}"
    lines += ["", "## One UNet call (neuron-explorer device profile)", "",
              f"Total time {summary['total_time'] * 1e3:.1f} ms, of which some engine is active "
              f"{pct('total_active_time_percent')}%.", "",
              "| engine | active % of the call | instructions |", "|---|---|---|"]
    for name, key in (("tensor (matmul)", "tensor_engine"), ("vector", "vector_engine"), ("scalar", "scalar_engine"),
                      ("sync", "sync_engine"), ("gpsimd", "gpsimd_engine")):
        lines.append(f"| {name} | {pct(key + '_active_time_percent')} | {summary[key + '_instruction_count']} |")
    lines += ["", "| % of peak | value |", "|---|---|",
              f"| model FLOPs utilization (MFU) | {pct('mfu_estimated_percent')}% |",
              f"| max achievable MFU for this instruction mix | {pct('mfu_max_achievable_estimated_percent')}% |",
              f"| hardware FLOPs utilization (HFU, includes transposes) | {pct('hfu_estimated_percent')}% |",
              f"| memory bandwidth utilization (MBU) | {pct('mbu_estimated_percent')}% |", "",
              f"The profile also reports a utilization limit on the two physical cores: the 50% limit was active "
              f"{pct('throttle_activity_1_active_time_nc4_percent')}% of the call, average limit "
              f"{pct('throttle_avg_util_limit_nc4_percent')}%. "
              f"Spill traffic: {summary['spill_save_bytes'] / 1e9:.1f} GB saved, "
              f"{summary['spill_reload_bytes'] / 1e9:.1f} GB reloaded per call; weights {summary['weight_size_bytes'] / 1e9:.2f} GB.", ""]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "tier1_profile.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("PROFILE DONE")


if __name__ == "__main__":
    main()
