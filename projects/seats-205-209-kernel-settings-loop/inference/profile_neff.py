"""profile_neff.py -- on-chip time of a compiled model, via neuron-explorer (the team's method).

    python profile_neff.py path/to/graph.neff [reps]
Prints one JSON line: median on-chip time and the engine-busy figures of the last capture.
"""
import json, os, shutil, statistics, subprocess, sys, tempfile

neff = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
env = dict(os.environ, NEURON_RT_VISIBLE_CORES="0", NEURON_LOGICAL_NC_CONFIG="1",
           NEURON_RT_ENABLE_DGE_NOTIFICATIONS="1")
times, m = [], None
for i in range(reps):
    d = tempfile.mkdtemp(prefix="prof_")
    cap = subprocess.run(["neuron-explorer", "capture", "-n", neff, "-s", os.path.join(d, "profile.ntff"),
                          "--profile-nth-exec=2"], env=env, capture_output=True, text=True, timeout=600)
    traces = sorted(f for f in os.listdir(d) if f.endswith(".ntff"))
    if cap.returncode != 0 or not traces:
        print(json.dumps(dict(error="capture failed: " + (cap.stderr or cap.stdout)[-500:])))
        sys.exit(1)
    view = subprocess.run(["neuron-explorer", "view", "--output-format", "summary-json", "-n", neff,
                           "-s", os.path.join(d, traces[-1])], env=env, capture_output=True, text=True, timeout=600)
    if view.returncode != 0:
        print(json.dumps(dict(error="view failed: " + view.stderr[-500:])))
        sys.exit(1)
    m = json.loads(view.stdout)
    if len(m) == 1 and isinstance(next(iter(m.values())), dict):
        m = next(iter(m.values()))
    times.append(m["total_time"] * 1e6)
    shutil.rmtree(d, ignore_errors=True)
print(json.dumps(dict(chip_us_median=round(statistics.median(times), 1), chip_us_runs=[round(t, 1) for t in times],
                      te_busy=m.get("tensor_engine_active_time_percent"), dma_busy=m.get("dma_active_time_percent"),
                      bytes=m.get("dma_transfer_total_bytes"), transfers=m.get("dma_transfer_count"),
                      mfu=m.get("mfu_estimated_percent"))))
