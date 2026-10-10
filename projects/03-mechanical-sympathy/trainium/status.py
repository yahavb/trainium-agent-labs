#!/usr/bin/env python3
"""What is running, and did each run finish properly?  (run in /workspace/samudra)

    python status.py                         # what's running + the last 10 finished runs
    python status.py autocast-1deg unet-1deg # verdict for specific --tag names

How a verdict is decided:
  bench.py writes its line to results.jsonl only at the very END of a run, after timing and the
  correctness check. So a tag in results.jsonl = the run finished. Anything else is either still
  running (a bench.py process with that --tag exists) or it crashed (no process, no result line);
  for a crash we show the error from the log file that mentions the tag.
"""
import glob, json, subprocess, sys
from pathlib import Path


def procs():
    out = subprocess.run(["ps", "-eo", "pid,etime,args"], capture_output=True, text=True).stdout
    keep = ("bench.py", "sweep.py", "agent_loop.py", "probe_ops.py", "neuronx-cc", "vllm serve")
    rows = [l.strip() for l in out.splitlines()[1:] if any(k in l for k in keep)
            and "status.py" not in l and "ps -eo" not in l]
    return rows


def results():
    p = Path("results.jsonl")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def fmt(r):
    ok = r.get("pass")
    verdict = ("PASS" if ok else "FAIL (check)") if r["device"] == "neuron" else "cpu ref"
    if r["device"] == "neuron" and "pass" not in r:
        verdict = "no CPU reference"
    err = r.get("rel_rms_err")
    return (f"{r.get('time','?'):8s}  {r.get('tag') or '-':16.16s} {r['device']:6s} {r['grid']:7s} "
            f"b{r.get('batch',1)} {r.get('dtype','fp32'):4s}  {r['median_ms']:9.1f} ms  "
            f"{'' if err is None else f'err {err:.1e}':12s} {verdict}")


def error_for(tag):
    for log in sorted(glob.glob("*.log") + glob.glob("sweep_logs/*.log"), key=lambda f: Path(f).stat().st_mtime):
        txt = Path(log).read_text(errors="ignore")
        if f"tag={tag} " in txt:                       # a shared log: take only this run's part
            txt = txt.split(f"START tag={tag} ")[-1].split("START tag=")[0]
        elif tag not in log and tag not in txt:
            continue
        if True:
            errs = [l.strip() for l in txt.splitlines()
                    if ("Error" in l or "NCC_" in l) and "nrt_infodump" not in l]
            if errs:
                return f"{log}: {errs[-1][:220]}"
    return None


def main():
    running = procs()
    print("== RUNNING NOW")
    for r in running:
        print("  " + r[:160])
    ours = [r for r in running if "vllm" not in r]
    if not ours:
        print("  (none of our jobs; only the vLLM server, if listed)")
    res = results()
    tags = sys.argv[1:]
    if not tags:
        print("\n== LAST 10 FINISHED RUNS (from results.jsonl; time = when the run STARTED, UTC)")
        for r in res[-10:]:
            print("  " + fmt(r))
        print("\nA run is finished only when it appears here. Neuron runs must say PASS.")
        return
    print("\n== VERDICT PER TAG")
    for t in tags:
        done = [r for r in res if r.get("tag") == t]
        live = [r for r in ours if f"--tag {t}" in r and "bench.py" in r and not r.split()[2] == "sh"]
        if done:
            print(f"  {t:18s} FINISHED  {fmt(done[-1])}")
        elif live:
            print(f"  {t:18s} RUNNING   for {live[0].split()[1]} (compile + run)")
        else:
            e = error_for(t)
            print(f"  {t:18s} {'CRASHED' if e else 'NOT STARTED / unknown'}" + (f"\n      {e}" if e else ""))


if __name__ == "__main__":
    main()
