#!/usr/bin/env python3
"""One forward pass of Samudra per grid, on CPU and on Trainium; prints a results table.

    nohup python sweep.py > sweep.log 2>&1 < /dev/null &       # everything
    python sweep.py --grids 2deg 1deg --devices cpu            # a subset
    python sweep.py --table                                    # print the table so far

Weights (--weights auto): the real Samudra2 checkpoint for the grid it was trained on
(1deg->onedeg, 0.5deg->halfdeg, 0.25deg->quarterdeg); random weights on 2deg (no 2deg checkpoint).
If a download fails, that grid falls back to random weights; the table says which was used.
Speed does not depend on the weight values; correctness is always Trainium vs CPU on the same weights.

Order: all CPU runs first (they also save the reference each Trainium run is checked against),
then Trainium runs one at a time on one NeuronCore. Before each phase it waits until no other
bench.py is running (e.g. a bf16 run), so timings are not shared with another job. Each run gets
its own log in sweep_logs/ and a timeout, so one huge compile cannot block the rest.
"One forward pass" = --warmup 0 --iters 1. On Trainium the first call (compile + run) is recorded
as compile_s, then one timed pass on the compiled graph.
"""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRIDS = ["2deg", "1deg", "0.5deg", "0.25deg"]
CKPT = {"1deg": "onedeg", "0.5deg": "halfdeg", "0.25deg": "quarterdeg"}


def bench_pids():
    out = subprocess.run(["pgrep", "-f", "bench.py"], capture_output=True, text=True).stdout
    return [int(p) for p in out.split()]


def wait_idle(poll=30):
    shown = False
    while pids := bench_pids():
        if not shown:
            print(f"  waiting for other bench.py runs to finish: pids {pids}", flush=True)
            shown = True
        time.sleep(poll)


def fetch(name):
    code = ("from huggingface_hub import hf_hub_download as d; "
            f"print(d('M2LInES/Samudra2', '{name}/ema_ckpt.pt'))")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return False, "download timed out (30 min)"
    if r.returncode == 0:
        return True, r.stdout.strip().splitlines()[-1]
    return False, (r.stderr.strip().splitlines() or ["?"])[-1][:200]


def run(cmd, log, timeout_min, env):
    print(f"  $ {' '.join(cmd[1:])}   (log: {log})", flush=True)
    t = time.time()
    with open(HERE / log, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             env=env, cwd=HERE, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout_min * 60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)      # also kills the neuronx-cc children
            p.wait()
            rc = "timeout"
    print(f"    -> {'ok' if rc == 0 else f'FAILED ({rc})'} in {time.time() - t:.0f} s", flush=True)
    if rc != 0:
        for l in (HERE / log).read_text().strip().splitlines()[-3:]:
            print("    " + l[:200], flush=True)
    return rc


def table(tag):
    rows = [json.loads(l) for l in open(HERE / "results.jsonl") if l.strip()]
    best = {(r["grid"], r["device"]): r for r in rows   # last run wins; CPU rows shared by all tags
            if r.get("tag") == tag or (r["device"] == "cpu" and str(r.get("tag", "")).startswith("sweep"))}
    print(f"\n[{tag}] one forward pass per grid")
    print("| grid | H x W | weights | CPU ms | Trainium ms | speedup | compile s | rel RMS err "
          "| pass | Trainium sim-years/day |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    f = lambda v, d=1: "-" if v is None else f"{v:,.{d}f}"
    for g in GRIDS:
        c, n = best.get((g, "cpu")), best.get((g, "neuron"))
        if not (c or n):
            continue
        r0 = c or n
        sp = f"{c['median_ms'] / n['median_ms']:.1f}x" if (c and n) else "-"
        err = format(n["rel_rms_err"], ".1e") if n and "rel_rms_err" in n else "-"
        print(f"| {g} | {r0['H']}x{r0['W']} | {r0.get('weights', '?')} | "
              f"{f(c['median_ms'] if c else None)} | {f(n['median_ms'] if n else None)} | {sp} | "
              f"{f(n.get('compile_s') if n else None, 0)} | {err} | "
              f"{n.get('pass', '-') if n else '-'} | {f(n['sim_years_per_day'] if n else None, 0)} |",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", default=GRIDS, choices=GRIDS)
    ap.add_argument("--devices", nargs="+", default=["cpu", "neuron"], choices=["cpu", "neuron"])
    ap.add_argument("--weights", default="auto", choices=["auto", "random"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"], help="Trainium dtype")
    ap.add_argument("--cores", default="2")
    ap.add_argument("--segments", type=int, default=0,
                    help="1 = compile each UNet layer separately (fits 0.5deg/0.25deg in 24 GB)")
    ap.add_argument("--timeout-min", type=float, default=120, help="per Trainium run")
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    tag = a.tag + ("" if a.dtype == "fp32" else "-bf16") + ("-seg" if a.segments else "")
    if a.table:
        return table(tag)
    (HERE / "sweep_logs").mkdir(exist_ok=True)
    env = dict(os.environ)
    env.pop("OMP_NUM_THREADS", None)   # bench.py sets threads from the cgroup quota
    env.pop("NEURON_RT_VISIBLE_CORES", None)   # bench.py --cores decides

    weights = {}
    for g in a.grids:
        hf = CKPT.get(g) if a.weights == "auto" else None
        if hf:
            print(f"[{g}] fetching checkpoint {hf} ...", flush=True)
            ok, msg = fetch(hf)
            print(f"    {'ok: ' if ok else 'FAILED, this grid uses random weights: '}{msg}", flush=True)
            hf = hf if ok else None
        weights[g] = hf

    def cmd(g, dev):
        c = [sys.executable, "bench.py", "--device", dev, "--grid", g, "--warmup", "0",
             "--iters", "1", "--tag", tag, "--cores", a.cores]
        if weights[g]:
            c += ["--hf", weights[g]]
        if dev == "neuron" and a.dtype == "bf16":
            c += ["--dtype", "bf16"]
        if dev == "neuron" and a.segments:
            c += ["--segments", "1"]
        return c

    if "cpu" in a.devices:
        print("== CPU (fp32; also saves the reference)", flush=True)
        wait_idle()
        for g in a.grids:
            run(cmd(g, "cpu"), f"sweep_logs/{g}_cpu.log", 120, env)
        table(tag)
    if "neuron" in a.devices:
        print(f"== Trainium ({a.dtype}, core {a.cores}, timeout {a.timeout_min:.0f} min each)",
              flush=True)
        for g in a.grids:
            wait_idle()
            run(cmd(g, "neuron"), f"sweep_logs/{g}_neuron_{a.dtype}{'_seg' if a.segments else ''}.log", a.timeout_min, env)
            table(tag)


if __name__ == "__main__":
    main()
