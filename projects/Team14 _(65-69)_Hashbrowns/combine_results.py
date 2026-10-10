#!/usr/bin/env python3
"""
combine_results.py -- every measurement in results/, from both seats, in one file and one summary.

    python combine_results.py          # writes results/combined.json and COMBINED.md; runs nothing

Two sources, measured differently, so every record carries its metric:

  seat-65  results/compare_*.json, from compare_all.py: per kernel and shape, 5 separate neuron-explorer
           captures of one execution each. The figure is neuron-explorer's total_time.
  seat-68  results/seat68/<suite>/run-*/results.json, from bench_suites.py (then bench_device.py) and
           bench_qwen_attention.py: the Neuron runtime's device-side benchmark, mean over 100-2,000
           timed iterations after a warmup.

The two are not comparable with each other. The single profiled execution runs several microseconds
above the benchmark mean on these kernels, and it ranks hardware-DMA-descriptor kernels wrongly: the
trace records every hardware descriptor, which inflates that execution (V6 profiled at 22.66 us
against V1's 21.99 us while the benchmark had V6 1.13x faster). Speedups are only ever taken against a
baseline from the same source and the same run.

Labels are mapped to this folder's numbering. seat-68 wrote its own V3 and V4 before this folder's
existed, so its "V3 hwdge" is V5 here and its "V4 hwdge+bf16" is V6; its "V0" was rebuilt from V1's
docstring before the original my_attention_v0.py was available (same algorithm, both PSUM copies
pinned to the Vector engine).
"""

import glob
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

SEAT65_FILES = {"V0": "my_attention_v0.py", "V1": "my_attention.py", "V2": "my_attention_v2_bf16.py",
                "V3": "my_attention_v3_dmaT.py", "V4": "my_attention_v4_kchunk.py"}

# seat-68 label -> (version here, kernel file:entry, note)
SEAT68 = {
    "attention": {
        "V0": ("V0", "my_attention_v0.py:nki_attention_",
               "seat-68 reconstruction of V0 (same algorithm, PSUM copies pinned to Vector); not this folder's file"),
        "V1": ("V1", "my_attention.py:nki_attention_", ""),
        "V2 bf16": ("V2", "my_attention_v2_bf16.py:nki_attention_", ""),
        "V3 hwdge": ("V5", "my_attention_v5_hwdge.py:nki_attention_", "seat-68 called it V3"),
        "V4 hwdge+bf16": ("V6", "my_attention_v6_hwdge_bf16.py:nki_attention_", "seat-68 called it V4"),
    },
    "matmul": {
        "level-4 reference": ("matmul reference", "reference_level4.py:nki_matmul_tiled_", ""),
        "parallel": ("matmul parallel", "matmul_parallel.py:nki_matmul_hoist_load_", ""),
    },
    "mha": {
        "M0 V0 per head": ("M0", "mha_v0_loop.py:nki_mha_", ""),
        "M4 V4 per head": ("M6", "mha_v6_loop.py:nki_mha_", "seat-68 called it M4 / V4 per head"),
        "fast, 1 core": ("fast", "mha_fast.py:nki_mha_", ""),
        "fast, 2 cores": ("fast", "mha_fast.py:nki_mha_", ""),
        "fast balanced, 2 cores": ("fast balanced", "mha_fast.py:nki_mha_balanced_", ""),
        "fast tf32 PV, 2 cores": ("fast tf32 P@V", "mha_fast.py:nki_mha_tf32_pv_", ""),
        "fast tf32 all, 2 cores": ("fast tf32 all", "mha_fast.py:nki_mha_tf32_all_", ""),
    },
    "qwen3_attention": {
        "nkilib original": ("nkilib attention_cte", "nkilib.core.attention.attention_cte:attention_cte", ""),
        "hwdge patched": ("attention_cte + hwdge loads", "attention_cte_hwdge.py:attention_cte", ""),
    },
}
# In this run "fast balanced" was the first form, Q^T copied out of PSUM on GpSimd; it did not compile.
GPSIMD_BALANCED_RUN = "run-20261010-213726"
MHA_LNC = {"M0 V0 per head": 1, "M4 V4 per head": 1, "fast, 1 core": 1}


def shape_of(text):
    """'seq=128 dim=64' -> '128x64'; 'seq=128 heads=16 dim=64' -> '128x16x64'; 'K=.. M=.. N=..' as KxMxN."""
    nums = dict(re.findall(r"(\w+)=(\d+)", text))
    for keys in (("seq", "heads", "dim"), ("seq", "dim"), ("K", "M", "N")):
        if all(k in nums for k in keys):
            return "x".join(nums[k] for k in keys)
    return text


def seat65():
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "compare_*.json"))):
        d = json.load(open(path))
        for shape, by_version in d["results"].items():
            for v, r in by_version.items():
                runs = [x["total_us"] for x in r.get("runs", [])]
                rec = dict(source="seat-65", suite="attention", run=os.path.basename(path),
                           metric="neuron-explorer total_time, median of separate profiled executions",
                           label=v, version=v, kernel=SEAT65_FILES.get(v, "") + ":nki_attention_", lnc=1,
                           shape=shape, samples=len(runs), status=r.get("status"),
                           correct=r.get("status") == "measured",
                           error_vs_rms=r.get("error_vs_rms"))
                if runs:
                    rec.update(us=round(sorted(runs)[len(runs) // 2], 3), min_us=round(min(runs), 3),
                               max_us=round(max(runs), 3))
                    rec["profile_median"] = {k: sorted(x[k] for x in r["runs"])[len(runs) // 2]
                                             for k in r["runs"][0] if k != "total_us"}
                out.append(rec)
    return out


def seat68():
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "seat68", "*", "run-*", "results.json"))):
        suite = path.split(os.sep)[-3]
        run = path.split(os.sep)[-2]
        d = json.load(open(path))
        iters = d.get("iterations")
        for r in d["results"]:
            label = r.get("version") or r.get("variant")
            version, kernel, note = SEAT68[suite][label]
            if suite == "mha" and label == "fast balanced, 2 cores" and run == GPSIMD_BALANCED_RUN:
                version, kernel = "fast balanced (GpSimd form)", "mha_fast.py:nki_mha_balanced_ with Q^T copy on GpSimd"
                note = "first form of the balanced variant; GpSimd cannot read PSUM, so it did not compile"
            if suite == "mha":
                lnc = MHA_LNC.get(label, 2)
            elif suite == "qwen3_attention":
                lnc = 2
            else:
                lnc = d.get("lnc", 1)
            shape = shape_of(r["shape"]) if "shape" in r else f"seq={r['seqlen']}"
            rec = dict(source="seat-68", suite=suite, run=run,
                       metric=f"device benchmark mean over {iters} timed iterations",
                       label=label, version=version, kernel=kernel, lnc=lnc, shape=shape, note=note)
            if "error" in r:
                rec.update(status="failed", correct=False, error=r["error"].strip().splitlines()[-1][:300])
            else:
                correct = r.get("correct", r.get("identical_to_original", True))
                rec.update(status="measured", correct=bool(correct), us=round(r["mean_us"], 3),
                           std_us=round(r["std_us"], 3) if r.get("std_us") is not None else None,
                           min_us=round(r["min_us"], 3) if r.get("min_us") is not None else None)
                if r.get("mismatch"):
                    rec["mismatch"] = r["mismatch"].splitlines()[0]
                for k in ("max_err_vs_ref", "rel_rms_vs_ref", "identical_to_original"):
                    if k in r:
                        rec[k] = r[k]
                if r.get("profile"):
                    rec["profile"] = r["profile"]
            out.append(rec)
    return out


# ---------------------------------------------------------------- summary

def geo(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def speedups(records, base_version, versions, key=lambda r: r["version"]):
    """Per version: geometric mean over shapes of base / version, within one run. Correct rows only."""
    out = {}
    for v in versions:
        ratios = []
        for r in records:
            if key(r) != v or not r.get("correct") or "us" not in r:
                continue
            b = next((x for x in records if key(x) == base_version and x["shape"] == r["shape"]
                      and x["run"] == r["run"] and x.get("correct") and "us" in x), None)
            if b:
                ratios.append(b["us"] / r["us"])
        out[v] = geo(ratios)
    return out


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt_us(r):
    if r is None:
        return "—"
    if r.get("status") == "failed":
        return "does not compile"
    s = f"{r['us']:.2f}"
    return s if r.get("correct") else s + " (wrong)"


def render(records):
    md = ["# Combined results", "",
          "Generated by `combine_results.py` from `results/` -- every number below is in "
          "`results/combined.json` with its run, metric and kernel. Rerun it after adding results; "
          "it measures nothing itself.", "",
          "**Two metrics, never mixed.** seat-65 (`compare_all.py`) reports neuron-explorer's "
          "`total_time` for one profiled execution, the median of 5 captures. seat-68 "
          "(`bench_suites.py`, `bench_qwen_attention.py`) reports the runtime's device-benchmark mean "
          "over many iterations. Each speedup below divides by a baseline from the same source and the "
          "same run.", ""]

    # single-head attention
    s65 = [r for r in records if r["source"] == "seat-65"]
    s68a = [r for r in records if r["source"] == "seat-68" and r["suite"] == "attention"
            and r["run"] in ("run-20261010-204410", "run-20261010-204707")]
    shapes = ["128x64", "64x128", "96x32"]
    versions = ["V0", "V1", "V2", "V3", "V4", "V5", "V6"]
    sp65 = speedups(s65, "V0", versions)
    sp68 = {run: speedups([r for r in s68a if r["run"] == run], "V0", versions)
            for run in ("run-20261010-204410", "run-20261010-204707")}
    rows = []
    for v in versions:
        def cell(recs, shape, run=None):
            r = next((x for x in recs if x["version"] == v and x["shape"] == shape
                      and (run is None or x["run"] == run)), None)
            return fmt_us(r) if r else "—"
        f65 = sp65.get(v)
        f68 = [sp68[run].get(v) for run in sp68]
        rows.append([f"**{v}**", next((x["kernel"].split(":")[0] for x in records if x["version"] == v), ""),
                     cell(s65, "128x64"), f"{f65:.3f}x" if f65 else "—",
                     cell(s68a, "128x64", "run-20261010-204410"),
                     " / ".join(f"{x:.3f}x" for x in f68 if x) or "—"])
    md += ["## Single-head attention (nkibench level 8)", "",
           table(["Version", "Kernel", "seat-65 total_time, 128x64 (us)", "seat-65 vs V0",
                  "seat-68 benchmark mean, 128x64 (us)", "seat-68 vs V0, run 1 / run 2"], rows), "",
           "seat-65: geometric mean over the three shapes of V0 median / version median; its "
           "verdicts, with ranges, are in RESULTS.md. seat-68: the same over the device-benchmark "
           "means of two separate runs; its V0 is the reconstruction described in "
           "combine_results.py. V3 and V4 were measured only on seat-65, V5 and V6 only on seat-68.", ""]

    # multi-head attention
    mha = [r for r in records if r["suite"] == "mha"]
    main_run = "run-20261010-213836"
    mrun = [r for r in mha if r["run"] == main_run]
    labels = ["M0 V0 per head", "M4 V4 per head", "fast, 1 core", "fast, 2 cores", "fast balanced, 2 cores"]
    names = {"M0 V0 per head": "M0 (V0 per head)", "M4 V4 per head": "M6 (V6 per head)",
             "fast, 1 core": "fast, LNC 1", "fast, 2 cores": "fast, LNC 2",
             "fast balanced, 2 cores": "fast balanced, LNC 2"}
    mshapes = sorted({r["shape"] for r in mrun}, key=lambda s: [int(x) for x in s.split("x")])
    rows = []
    for lab in labels:
        cells = [fmt_us(next((r for r in mrun if r["label"] == lab and r["shape"] == s), None)) for s in mshapes]
        sp = speedups(mrun, "M0 V0 per head", [lab], key=lambda r: r["label"])[lab]
        rows.append([f"**{names[lab]}**"] + cells + [f"{sp:.3f}x" if sp else "—"])
    every = []
    for run in sorted({r["run"] for r in mha}):
        rr = [r for r in mha if r["run"] == run]
        sp = speedups(rr, "M0 V0 per head", ["fast, 2 cores", "fast, 1 core"], key=lambda r: r["label"])
        if sp["fast, 2 cores"]:
            every.append(f"{run}: LNC 2 {sp['fast, 2 cores']:.3f}x, LNC 1 {sp['fast, 1 core']:.3f}x")
    md += ["## Multi-head attention (nkibench level 9), seat-68", "",
           f"Run {main_run}, device-benchmark mean (us) per shape (seq x heads x dim). "
           "\"(wrong)\" is timed but misses nkibench's tolerance; \"vs M0\" counts only shapes where "
           "both are correct.", "",
           table(["Version"] + mshapes + ["vs M0"], rows), "",
           "`mha_fast.py` against M0 in every run: " + "; ".join(every) + ".", "",
           "Measured and not kept as a speedup: TF32 entries (`nki_mha_tf32_pv_`, "
           "`nki_mha_tf32_all_`) do not compile for the device; the first balanced form (Q^T copy on "
           "GpSimd) does not compile because GpSimd cannot read PSUM. Both are in combined.json as "
           "status `failed`.", ""]

    # matmul
    mm = [r for r in records if r["suite"] == "matmul" and r["run"] == "run-20261010-202005"]
    rows = []
    for s in sorted({r["shape"] for r in mm}, key=lambda s: [int(x) for x in s.split("x")]):
        ref = next(r for r in mm if r["shape"] == s and r["version"] == "matmul reference")
        par = next(r for r in mm if r["shape"] == s and r["version"] == "matmul parallel")
        rows.append([s, f"{ref['us']:.2f}", f"{par['us']:.2f}", f"{ref['us'] / par['us']:.3f}x"])
    md += ["## Tiled matmul (nkibench level 4), seat-68", "",
           table(["K x M x N", "reference_level4 (us)", "matmul_parallel (us)", "parallel vs reference"], rows),
           "", "On the device both read and write exactly the minimum bytes, so the parallel version has "
           "no traffic to save; see `matmul_parallel.py` and the per-version metrics.json.", ""]

    # qwen3
    qw = [r for r in records if r["suite"] == "qwen3_attention"]
    rows = []
    for s in sorted({r["shape"] for r in qw}, key=lambda s: int(s.split("=")[1])):
        o = next(r for r in qw if r["shape"] == s and r["label"] == "nkilib original")
        p = next(r for r in qw if r["shape"] == s and r["label"] == "hwdge patched")
        rows.append([s.split("=")[1], f"{o['us']:.2f}", f"{p['us']:.2f}", f"{o['us'] / p['us']:.3f}x",
                     "yes" if p.get("identical_to_original") else "no"])
    md += ["## Qwen3-8B prefill attention kernel, seat-68", "",
           "nkilib `attention_cte` as vLLM-neuron runs it for Qwen3-8B (16 heads x 128 per core, bf16, "
           "causal, LNC 2), against `attention_cte_hwdge.py`. One run per prompt length.", "",
           table(["Prompt tokens", "original (us)", "hwdge loads (us)", "patched vs original",
                  "bit-identical"], rows), ""]

    md += ["## Every run", "",
           table(["Source", "Suite", "Run", "Records", "Failed", "Metric"],
                 [[src, suite, run, sum(1 for r in records if (r["source"], r["suite"], r["run"]) == (src, suite, run)),
                   sum(1 for r in records if (r["source"], r["suite"], r["run"]) == (src, suite, run)
                       and r.get("status") == "failed"),
                   next(r["metric"] for r in records if (r["source"], r["suite"], r["run"]) == (src, suite, run))]
                  for src, suite, run in sorted({(r["source"], r["suite"], r["run"]) for r in records})]), ""]
    return "\n".join(md)


def main():
    records = seat65() + seat68()
    with open(os.path.join(RESULTS, "combined.json"), "w") as f:
        json.dump(dict(generated_by="combine_results.py", records=records), f, indent=1)
    with open(os.path.join(HERE, "COMBINED.md"), "w") as f:
        f.write(render(records) + "\n")
    print(f"{len(records)} records -> results/combined.json, COMBINED.md")


if __name__ == "__main__":
    main()
