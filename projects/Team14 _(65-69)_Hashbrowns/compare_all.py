#!/usr/bin/env python3
"""
compare_all.py -- every attention kernel version, on the device, on every level-8 shape, repeated,
written up as RESULTS.md: correctness, latency with its spread, where the time goes, and what each
version changes relative to the one it was built from.

    python compare_all.py                       # all versions, 3 shapes, 5 profiled runs each
    python compare_all.py --repeats 10
    python compare_all.py --shapes 128x64       # one shape, quicker
    python compare_all.py --render results/compare_<stamp>/results.json   # rewrite the .md only

Per kernel and shape: one device run checked against a float64 CPU reference (a kernel that is
WRONG on the device is not timed), then --repeats separate neuron-explorer captures of the same
NEFF, each profiling its own 2nd execution. The latency is neuron-explorer's total_time.

A version is called FASTER or SLOWER than V0 only when every one of its runs beats, or loses to,
every one of V0's. Anything else is reported as within noise.

Needs the nkivenv (torch-xla with the Neuron plugin) and the NeuronCores free: vLLM holds both
logical cores on this pod, so stop it first with /workspace/serve.sh --stop.
"""

import argparse
import ast
import difflib
import json
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# (label, file, built from, what it changes). The parent decides which diff the report shows.
KERNELS = [
    ("V0", "my_attention_v0.py", None,
     "Baseline. Q and K transposed on the Tensor engine and copied out of PSUM on Vector; "
     "S = Q K^T; row max; bias = -scale * max; exp; row sum; transpose P; P V; normalise by "
     "reciprocal-and-multiply."),
    ("V1", "my_attention.py", "V0",
     "V0 rescheduled, same maths. The softmax scale rides on Q's PSUM copy (Scalar engine), "
     "tensor_reduce(negate=True) yields the exp bias directly, and the exp instruction accumulates "
     "the row sum (activation_reduce), taking work off the in-order Vector queue."),
    ("V2", "my_attention_v2_bf16.py", "V1",
     "V1 with the P transpose and P @ V in bfloat16 instead of float32. Q K^T stays float32."),
    ("V3", "my_attention_v3_dmaT.py", "V1",
     "V1 with Q and K transposed by the DMA engine on the way in (dma_transpose), removing both "
     "Tensor-engine transposes and both PSUM copies."),
    ("V4", "my_attention_v4_kchunk.py", "V1",
     "V1 with the back half (transpose P, copy, P @ V) in chunks of 32 keys, accumulating P @ V "
     "in PSUM."),
]
SHAPES = [(128, 64), (64, 128), (96, 32)]
ENTRY = "nki_attention_"

# neuron-explorer 2.32 summary-json: times in seconds, *_percent fields are fractions.
ENGINES = ["tensor", "vector", "scalar", "gpsimd"]


# ---------------------------------------------------------------- source analysis

def kernel_source(path):
    """The @nki.jit function's source, decorators included and module docstring excluded."""
    src = open(path).read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == ENTRY:
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            return "\n".join(src.splitlines()[start - 1:node.end_lineno])
    return ""


def instructions(path):
    """Each nisa.* call in source order, as written, with '[loop]' when it sits inside a for."""
    tree = ast.parse(open(path).read())
    in_loop = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            for sub in ast.walk(node):
                in_loop.add(id(sub))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
             and n.func.value.id == "nisa"]
    calls.sort(key=lambda n: (n.lineno, n.col_offset))
    return [("[loop] " if id(c) in in_loop else "") + re.sub(r"\s+", " ", ast.unparse(c))
            for c in calls]


# ---------------------------------------------------------------- device runs

def failure_reason(text):
    hits = [l.strip() for l in text.splitlines()
            if re.search(r"error[:\]]|Error:|unsupported|not available|cached failed", l)]
    return " | ".join(dict.fromkeys(hits))[-400:] or text.strip()[-400:]


def measure(pd, exe, path, seq, dim, root, repeats):
    out_dir = os.path.join(root, "output")
    os.makedirs(out_dir, exist_ok=True)
    rec = {"status": None, "error_vs_rms": None, "runs": [], "notes": []}
    result, proc = pd.run_child(path, seq, dim, out_dir)
    if not result or "error" in result:
        rec["status"] = "failed to run"
        rec["reason"] = result["error"] if result else failure_reason(proc.stdout + proc.stderr)
        return rec
    rec["error_vs_rms"] = result["error_vs_rms"]
    if result["error_vs_rms"] > pd.TOL:
        rec["status"] = "WRONG on device"
        return rec
    neff, listing = pd.find_neff(out_dir)
    if not neff:
        rec["status"] = "correct, no NEFF found"
        rec["reason"] = listing[-400:]
        return rec
    for r in range(repeats):
        metrics, note = pd.capture_and_view(exe, neff, os.path.join(root, f"rep{r}"))
        if metrics is None:
            rec["notes"].append(note[-300:])
            continue
        if "RUNTIME trace" in note:
            rec["notes"].append("a run fell back to the runtime trace (1st execution, includes warm-up)")
        run = {"total_us": pd.pick(metrics, "total_time") * 1e6,
               "dma_pct": 100 * (pd.pick(metrics, "dma_active_time_percent") or 0),
               "mfu_pct": pd.pick(metrics, "mfu_estimated_percent"),
               "hbm_read": pd.pick(metrics, "hbm_read_bytes"),
               "hbm_write": pd.pick(metrics, "hbm_write_bytes")}
        for e in ENGINES:
            run[f"{e}_pct"] = 100 * (pd.pick(metrics, f"{e}_engine_active_time_percent") or 0)
            run[f"{e}_instr"] = pd.pick(metrics, f"{e}_engine_instruction_count")
        rec["runs"].append(run)
        print(f"    run {r + 1}/{repeats}: {run['total_us']:.2f} us")
    rec["status"] = "measured" if rec["runs"] else "correct, profile failed"
    return rec


# ---------------------------------------------------------------- report

def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def stats(rec):
    t = [r["total_us"] for r in rec.get("runs", [])]
    if not t:
        return None
    return {"median": statistics.median(t), "min": min(t), "max": max(t), "n": len(t)}


def verdict(s, base):
    if not s or not base:
        return "-"
    if s["max"] < base["min"]:
        return "**faster** (every run)"
    if s["min"] > base["max"]:
        return "**slower** (every run)"
    return "within noise"


def med(rec, key):
    vals = [r[key] for r in rec.get("runs", []) if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def render(res, md_path):
    shapes = [tuple(s) for s in res["shapes"]]
    labels = [k[0] for k in KERNELS if k[0] in res["kernels"]]
    info = {k[0]: k for k in KERNELS}
    L = []
    w = L.append
    w("# Attention kernel comparison\n")
    w(f"Measured on **{res['host']}** at {res['when']}. "
      f"NKI version {res['versions'].get('nki', '?')}, neuronx-cc {res['versions'].get('neuronx-cc', '?')}, "
      f"{res['versions'].get('neuron-explorer', 'neuron-explorer ?')}.  ")
    w(f"Each kernel: 1 correctness run on the device, then **{res['repeats']} profiled runs** "
      f"(separate neuron-explorer captures, 2nd execution each). Latency = neuron-explorer "
      f"`total_time`. Reproduce: `python compare_all.py --repeats {res['repeats']}`.\n")
    w("**How to read the verdicts:** a version is *faster* or *slower* than V0 only if every one of "
      "its runs beats, or loses to, every V0 run. Otherwise the ranges overlap and the difference "
      "is not distinguishable from run-to-run noise.\n")

    # headline: the first shape
    s0 = "x".join(map(str, shapes[0]))
    w(f"## Headline: seq x dim = {s0}\n")
    w("| Version | What it changes | Correct on device (error / RMS) | Median us | Range us | vs V0 | Verdict |")
    w("|---|---|---|---|---|---|---|")
    base = stats(res["results"].get(s0, {}).get("V0", {}))
    for lab in labels:
        rec = res["results"].get(s0, {}).get(lab, {})
        s = stats(rec)
        corr = (f"yes ({rec['error_vs_rms']:.1e})" if rec.get("error_vs_rms") is not None
                and rec.get("status") != "WRONG on device"
                else ("**NO** " + fmt(rec.get('error_vs_rms'), 3) if rec.get("status") == "WRONG on device"
                      else rec.get("status", "not run")))
        short = info[lab][3].split(". ")[0].rstrip(".") + "."
        w(f"| **{lab}** | {short} | {corr} | {fmt(s and s['median'])} | "
          f"{fmt(s and s['min'])} – {fmt(s and s['max'])} | "
          f"{(f'{base['median'] / s['median']:.3f}x' if s and base else '-')} | "
          f"{verdict(s, base) if lab != 'V0' else 'baseline'} |")
    w("")

    # all shapes
    w("## Median latency on every shape (us)\n")
    w("| Version | " + " | ".join("x".join(map(str, sh)) for sh in shapes) + " |")
    w("|---|" + "---|" * len(shapes))
    for lab in labels:
        cells = []
        for sh in shapes:
            key = "x".join(map(str, sh))
            rec = res["results"].get(key, {}).get(lab, {})
            s, b = stats(rec), stats(res["results"].get(key, {}).get("V0", {}))
            if s:
                tag = "" if lab == "V0" or not b else f" ({b['median'] / s['median']:.2f}x, {verdict(s, b)})"
                cells.append(f"{s['median']:.2f}{tag}")
            else:
                cells.append(rec.get("status", "not run"))
        w(f"| **{lab}** | " + " | ".join(cells) + " |")
    w("\nSpeedups are V0 median / version median: above 1.00x is faster.\n")

    # where the time goes
    w(f"## Where the time goes ({s0}, medians)\n")
    w("Busy % is the share of the kernel's total time each engine was executing. "
      "Instruction counts are what the hardware ran after compilation, not the nisa calls in the source.\n")
    w("| Version | Total us | Tensor % | Vector % | Scalar % | GpSimd % | DMA % | "
      "HW instr T / V / S / G | HBM read / write B | MFU % |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for lab in labels:
        rec = res["results"].get(s0, {}).get(lab, {})
        if not rec.get("runs"):
            w(f"| **{lab}** | {rec.get('status', 'not run')} |" + " |" * 8)
            continue
        r0 = rec["runs"][0]
        instr = " / ".join(str(r0.get(f"{e}_instr", "-")) for e in ENGINES)
        w(f"| **{lab}** | {fmt(med(rec, 'total_us'))} | "
          + " | ".join(fmt(med(rec, f"{e}_pct"), 0) for e in ENGINES)
          + f" | {fmt(med(rec, 'dma_pct'), 0)} | {instr} | {r0.get('hbm_read')} / {r0.get('hbm_write')} | "
          f"{fmt(med(rec, 'mfu_pct'), 4)} |")
    w("")

    # what each version changes
    w("## What each version changes\n")
    for lab in labels:
        _, fname, parent, desc = info[lab]
        ins = res["instructions"][lab]
        w(f"### {lab} — `{fname}`\n")
        w(f"{desc}\n")
        if parent is None:
            w(f"Its {len(ins)} nisa instructions, in source order:\n")
            w("```text")
            w("\n".join(ins))
            w("```\n")
            continue
        pins = res["instructions"].get(parent, [])
        w(f"Built from **{parent}**. Instructions in source order: {len(pins)} → {len(ins)} "
          f"(`[loop]` = inside a loop, so issued more than once).\n")
        d = list(difflib.unified_diff(pins, ins, parent, lab, lineterm="", n=1))
        w("```diff")
        w("\n".join(d[2:]) if d else "(identical instruction sequence)")
        w("```\n")
        cd = list(difflib.unified_diff(res["sources"].get(parent, "").splitlines(),
                                       res["sources"][lab].splitlines(),
                                       f"{parent}", f"{lab}", lineterm="", n=2))
        w(f"<details><summary>Full code diff {parent} → {lab} ({len(cd)} lines)</summary>\n")
        w("```diff")
        w("\n".join(cd) if cd else "(identical)")
        w("```\n</details>\n")

    # problems
    probs = [(sh, lab, rec) for sh, by in res["results"].items() for lab, rec in by.items()
             if rec.get("status") != "measured" or rec.get("notes")]
    if probs:
        w("## Failures and warnings\n")
        for sh, lab, rec in probs:
            msg = rec.get("reason") or "; ".join(dict.fromkeys(rec.get("notes", [])))
            w(f"- **{lab} @ {sh}** — {rec.get('status')}: `{msg}`")
        w("")

    w("## Caveats\n")
    w("- Correctness is one random input per shape on the device (scale 1). The hostile cases "
      "(large values, identical rows, ragged lengths) are covered in the simulator by `verify_sdk.py`, "
      "not here.")
    w("- Repeats re-execute the same compiled NEFF, so they measure execution noise, not compile variation.")
    w("- These shapes are tiny; at this size the kernels are bound by the dependency chain between "
      "instructions, not by any engine's throughput, so expect small differences.")
    # The report says "NKI version N"; V0..V4 stay as the internal keys (folder names, --kernels).
    open(md_path, "w").write(re.sub(r"\bV([0-4])\b", r"NKI version \1", "\n".join(L) + "\n"))


# ---------------------------------------------------------------- main

def versions():
    v = {}
    for mod, key in (("nki", "nki"), ("neuronxcc", "neuronx-cc")):
        try:
            v[key] = __import__(mod).__version__.split("+")[0]
        except Exception:
            pass
    try:
        out = subprocess.run(["neuron-explorer", "--version"], capture_output=True, text=True).stdout
        v["neuron-explorer"] = out.strip().splitlines()[0].split(" built")[0] if out.strip() else ""
    except OSError:
        pass
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--shapes", default=",".join(f"{a}x{b}" for a, b in SHAPES))
    ap.add_argument("--kernels", default=",".join(k[0] for k in KERNELS),
                    help="labels to run, e.g. V0,V1,V2")
    ap.add_argument("--render", help="rewrite RESULTS.md from a saved results.json; runs nothing")
    a = ap.parse_args()

    if a.render:
        res = json.load(open(a.render))
        render(res, os.path.join(HERE, "RESULTS.md"))
        print(f"wrote {os.path.join(HERE, 'RESULTS.md')}")
        return

    import profile_device as pd
    try:
        import torch_xla  # noqa: F401
    except ImportError:
        sys.exit(f"{sys.executable} has no torch_xla. Run: source /workspace/nkivenv/bin/activate")
    if subprocess.run(["pgrep", "-f", "vllm serve"], capture_output=True).returncode == 0:
        sys.exit("vLLM is running and holds both logical NeuronCores. Stop it first: "
                 "/workspace/serve.sh --stop   (restart afterwards with /workspace/serve.sh)")
    exe = pd.profiler()
    if not exe:
        sys.exit("neither neuron-explorer nor neuron-profile is on PATH")

    want = a.kernels.split(",")
    kernels = [k for k in KERNELS if k[0] in want]
    shapes = [tuple(int(x) for x in s.split("x")) for s in a.shapes.split(",")]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(HERE, "results", f"compare_{stamp}")
    res = {"host": os.uname().nodename, "when": time.strftime("%Y-%m-%d %H:%M"),
           "versions": versions(), "repeats": a.repeats, "shapes": shapes, "results": {},
           "kernels": [k[0] for k in kernels],
           "instructions": {k[0]: instructions(os.path.join(HERE, k[1])) for k in KERNELS},
           "sources": {k[0]: kernel_source(os.path.join(HERE, k[1])) for k in KERNELS}}
    print(f"{len(kernels)} kernels x {len(shapes)} shapes x {a.repeats} profiled runs -> {run_dir}")

    for seq, dim in shapes:
        key = f"{seq}x{dim}"
        res["results"][key] = {}
        for lab, fname, _, _ in kernels:
            print(f"== {lab} ({fname}) seq={seq} dim={dim}")
            rec = measure(pd, exe, os.path.join(HERE, fname), seq, dim,
                          os.path.join(run_dir, key, lab), a.repeats)
            s = stats(rec)
            print(f"   {rec['status']}" + (f": median {s['median']:.2f} us "
                                            f"[{s['min']:.2f} - {s['max']:.2f}]" if s else
                                            f" -- {rec.get('reason', '')[:200]}"))
            res["results"][key][lab] = rec
            json.dump(res, open(os.path.join(run_dir, "results.json"), "w"), indent=1)

    md = os.path.join(HERE, "RESULTS.md")
    render(res, md)
    render(res, os.path.join(run_dir, "RESULTS.md"))
    print(f"\nwrote {md}\n  (copy kept in {run_dir})")


if __name__ == "__main__":
    main()
