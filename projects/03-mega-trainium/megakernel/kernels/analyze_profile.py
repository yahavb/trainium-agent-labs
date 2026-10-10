"""Bounds analysis of a megakernel profile captured by profile_mega.py.

Follows AWS's neuron-nki-profile-querying performance-bounds method (memory / compute / pipeline
families, per physical core), plus a phase breakdown: every instruction is labelled attention / mlp /
glue by the NKI source file it came from, the per-core timeline is cut where the label changes, and
DMA bytes, DMA-active time and engine-active time are reported per phase. DMA bytes are attributed
by packet start time, so they mean "issued during this phase": the scheduler prefetches the next
block's weights, so MLP weight bytes can land in attention windows. "glue" is everything in
our own kernel file or transformer_tkg.py (residual adds, barriers, inter-block layout).

  python kernels/analyze_profile.py base_L1 [base_L2 ...]   # tags under results/profiles/
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.join(HERE, "..", "results", "profiles")
ENGINES = ["tensor", "vector", "scalar", "gpsimd", "sync"]


def merge_ns(starts, ends, t0, t1):
    """Wall-clock ns covered by the union of [start, end) intervals, clipped to [t0, t1)."""
    s = np.maximum(np.asarray(starts, dtype=np.int64), t0)
    e = np.minimum(np.asarray(ends, dtype=np.int64), t1)
    keep = s < e
    s, e = s[keep], e[keep]
    if len(s) == 0:
        return 0
    order = np.argsort(s)
    total, cur = 0, -1
    for a, b in zip(s[order], e[order]):
        if a >= cur:
            total += b - a
            cur = b
        elif b > cur:
            total += b - cur
            cur = b
    return int(total)


def phase_of(loc):
    if not isinstance(loc, str):
        return None
    if "/mlp/" in loc:
        return "mlp"
    if any(k in loc for k in ("attention", "output_projection", "/qkv/", "rope", "rmsnorm_tkg", "norm_tkg_utils")):
        return "attn"
    if "qwen3_megakernel" in loc or "transformer_tkg.py" in loc or "decode_step" in loc:
        return "glue"
    return None


def phases(inst_core, t_end):
    """[(label, t0, t1)] contiguous runs of the per-core instruction timeline."""
    lab = inst_core.assign(ph=inst_core.nki_source_location.map(phase_of)).dropna(subset=["ph"])
    lab = lab.sort_values("start_ts")
    runs = []
    for ph, ts in zip(lab.ph.values, lab.start_ts.values):
        if not runs or runs[-1][0] != ph:
            runs.append([ph, int(ts)])
    out = [("start", int(inst_core.start_ts.min()), runs[0][1])] if runs else []
    for k, (ph, ts) in enumerate(runs):
        te = runs[k + 1][1] if k + 1 < len(runs) else t_end
        out.append((ph, ts, te))
    return out


def analyze(tag):
    pdir = os.path.join(PROFILES, tag)
    meta = json.load(open(os.path.join(pdir, "meta.json")))
    d = os.path.join(pdir, "parquet", "profiles", "global", f"{tag}@latest")
    rd = lambda t: pd.read_parquet(os.path.join(d, f"{t}.parquet"))
    md, active, inst = rd("Metadata").iloc[0], rd("ActiveTime"), rd("Instruction")
    # DmaPacket is sampled on this profiler build (it held 145 of 195 MB per core in base_L1);
    # DmaPacketAggregated matched the Summary's dma_transfer_total_bytes exactly, so use it.
    pkts = rd("DmaPacketAggregated")
    pkts = pkts[(pkts.queue_type != "instruction") & (pkts.transfer_bytes > 4)]
    dma_bw = float(md.dma_ddr_bandwidth)                     # per physical core
    te_peak = md.tensor_engine_num_rows * md.tensor_engine_num_cols * 2.0 * md.tensor_engine_clock_freq * 1e9

    L, H, I, q, kv, dh, S_ctx = (meta[k] for k in ("layers", "H", "I", "q", "kv", "d", "S_ctx"))
    weight_bytes = 2 * L * (H * (q + 2 * kv) * dh + q * dh * H + 3 * H * I)
    kv_bytes = 2 * L * kv * S_ctx * dh * 2
    if "head_parts" in meta["kernel"]:          # LM head weights + one embedding row
        weight_bytes, kv_bytes = 2 * (151936 * H + H), 0
    necessary = weight_bytes + kv_bytes                       # whole logical core; each physical core ~half

    rep = dict(tag=tag, kernel=meta["kernel"], layers=L, necessary_MB=round(necessary / 1e6, 1), cores={})
    for c in sorted(inst.pcore_idx.unique()):
        ic, pc, ac = inst[inst.pcore_idx == c], pkts[pkts.pcore_idx == c], active[active.pcore_idx == c]
        t0 = int(min(ic.start_ts.min(), pc.start_ts.min()))
        t1 = int(max(ic.end_ts.max(), pc.end_ts.max()))
        total = (t1 - t0) / 1e3
        eng = {e: merge_ns(ac[ac.engine == e].start_ts, ac[ac.engine == e].end_ts, t0, t1) / 1e3 for e in ENGINES}
        eng["dma"] = merge_ns(pc.start_ts, pc.end_ts, t0, t1) / 1e3
        mm = ic[(ic.engine == "Tensor") & (ic.opcode == "MATMUL")]
        hw_flops = mm.adjusted_flops.fillna(0).sum()
        tr_flops = mm[mm.tensor_instruction_type == "TRANSPOSE"].adjusted_flops.fillna(0).sum()
        moved = pc.transfer_bytes.sum()
        bounds = dict(
            total_us=total,
            memory_bound_us=eng["dma"],
            memory_bound_ideal_us=moved / dma_bw * 1e6,
            memory_no_reloads_us=(necessary / 2) / dma_bw * 1e6,
            compute_bound_us=eng["tensor"],
            compute_ideal_flops_us=hw_flops / te_peak * 1e6,
            compute_ideal_useful_us=(hw_flops - tr_flops) / te_peak * 1e6,
            perfect_pipeline_us=max(eng[e] for e in ("tensor", "vector", "scalar", "gpsimd", "dma")),
        )
        ph_rows = {}
        for ph, a, b in phases(ic, t1):
            r = ph_rows.setdefault(ph, dict(wall_us=0.0, dma_active_us=0.0, dma_MB=0.0, n_windows=0,
                                            **{f"{e}_us": 0.0 for e in ENGINES}))
            r["wall_us"] += (b - a) / 1e3
            r["n_windows"] += 1
            w = pc[(pc.start_ts >= a) & (pc.start_ts < b)]
            r["dma_active_us"] += merge_ns(pc.start_ts, pc.end_ts, a, b) / 1e3
            r["dma_MB"] += w.transfer_bytes.sum() / 1e6
            for e in ENGINES:
                r[f"{e}_us"] += merge_ns(ac[ac.engine == e].start_ts, ac[ac.engine == e].end_ts, a, b) / 1e3
        for r in ph_rows.values():
            r["achieved_GBps"] = r["dma_MB"] / r["wall_us"] * 1e3 if r["wall_us"] else 0.0  # MB/us = 1000 GB/s
        idle = dma_idle_gaps(pc, ic, t0, t1)
        by_var = (pc.groupby("variable").transfer_bytes.agg(["sum", "count"])
                  .sort_values("sum", ascending=False).head(15))
        rep["cores"][int(c)] = dict(
            bounds={k: round(v, 1) for k, v in bounds.items()},
            engine_active_us={k: round(v, 1) for k, v in eng.items()},
            bottleneck=max(("tensor", "vector", "scalar", "gpsimd", "dma"), key=eng.get),
            dma_moved_MB=round(moved / 1e6, 1), dma_packets=int(len(pc)),
            dma_avg_packet_B=int(moved / max(len(pc), 1)),
            achieved_GBps=round(moved / (total * 1e3), 1),
            phases={k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in ph_rows.items()},
            dma_idle=idle,
            top_dma_variables={str(k): dict(MB=round(v["sum"] / 1e6, 2), packets=int(v["count"])) for k, v in by_var.iterrows()},
        )
    return rep


def short(loc):
    return loc.replace("/opt/conda/lib/python3.13/site-packages/", "").replace("/workspace/projects/03-megakernel/", "") \
        if isinstance(loc, str) else "(no source)"


def dma_idle_gaps(pc, ic, t0, t1, min_ns=500):
    """Intervals inside [t0, t1) where no DMA is in flight, and which source lines run during them."""
    order = np.argsort(pc.start_ts.values)
    s, e = pc.start_ts.values[order], pc.end_ts.values[order]
    gaps, cur = [], t0
    for a, b in zip(s, e):
        if a > cur:
            gaps.append((cur, a))
        cur = max(cur, b)
    if t1 > cur:
        gaps.append((cur, t1))
    gaps = [(a, b) for a, b in gaps if b - a >= min_ns]
    by_loc = {}
    ist, ien, iloc = ic.start_ts.values, ic.end_ts.values, ic.nki_source_location.map(short).values
    for a, b in gaps:
        ov = np.minimum(ien, b) - np.maximum(ist, a)
        sel = ov > 0
        if not sel.any():
            by_loc["(nothing running)"] = by_loc.get("(nothing running)", 0) + (b - a)
            continue
        # charge the gap to the source line with the most instruction overlap inside it
        locs = pd.Series(ov[sel], index=iloc[sel]).groupby(level=0).sum()
        top = locs.idxmax()
        by_loc[top] = by_loc.get(top, 0) + (b - a)
    top = sorted(by_loc.items(), key=lambda kv: -kv[1])[:12]
    return dict(total_us=round(sum(b - a for a, b in gaps) / 1e3, 1), n_gaps=len(gaps),
                largest_us=sorted([round((b - a) / 1e3, 1) for a, b in gaps], reverse=True)[:10],
                by_source_us={k: round(v / 1e3, 1) for k, v in top})


def print_report(rep):
    print(f"\n=== {rep['tag']}  ({rep['kernel']}, L={rep['layers']}, necessary {rep['necessary_MB']} MB per logical core)")
    for c, r in rep["cores"].items():
        b = r["bounds"]
        print(f" pcore {c}: total {b['total_us']:.0f} us | bottleneck {r['bottleneck']} | moved {r['dma_moved_MB']} MB in "
              f"{r['dma_packets']} pkts (avg {r['dma_avg_packet_B']} B) = {r['achieved_GBps']} GB/s")
        print("   memory : total {total_us:.0f} -> dma_active {memory_bound_us:.0f} -> ideal(moved) {memory_bound_ideal_us:.0f} "
              "-> ideal(necessary/2) {memory_no_reloads_us:.0f} us".format(**b))
        print("   compute: total {total_us:.0f} -> te_active {compute_bound_us:.0f} -> ideal_flops {compute_ideal_flops_us:.1f} "
              "-> useful {compute_ideal_useful_us:.1f} us | perfect pipeline {perfect_pipeline_us:.0f} us".format(**b))
        print("   engines active (us): " + ", ".join(f"{k} {v:.0f}" for k, v in r["engine_active_us"].items()))
        for ph, p in sorted(r["phases"].items()):
            print(f"   phase {ph:5s}: wall {p['wall_us']:7.1f} us in {p['n_windows']:3d} windows | dma active {p['dma_active_us']:7.1f} "
                  f"| issued {p['dma_MB']:7.2f} MB | {p['achieved_GBps']:6.1f} GB/s | te {p['tensor_us']:6.1f} ve {p['vector_us']:6.1f} "
                  f"act {p['scalar_us']:6.1f} pool {p['gpsimd_us']:6.1f}")
        di = r["dma_idle"]
        print(f"   DMA idle >=0.5us: {di['total_us']} us in {di['n_gaps']} gaps; largest {di['largest_us']}")
        for k, v in di["by_source_us"].items():
            print(f"      {v:7.1f} us  {k}")
        print("   top DMA variables: " + "; ".join(f"{k} {v['MB']} MB/{v['packets']}p" for k, v in list(r["top_dma_variables"].items())[:8]))


if __name__ == "__main__":
    reps = [analyze(t) for t in sys.argv[1:]]
    for r in reps:
        print_report(r)
        with open(os.path.join(PROFILES, r["tag"], "bounds.json"), "w") as f:
            json.dump(r, f, indent=1)
