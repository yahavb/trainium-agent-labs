"""End to end on 4 NeuronCores: pixels -> TAEHV encode -> 30-block causal DiT (2 passes) -> TAEHV decode -> pixels.

    python g3_pipeline.py warm        (once; fills /dev/shm/g3 with the warmed KV caches)
    python g4_e2e.py --chunks 120

Same 4 stage processes as g3_pipeline.py (layers 0-7, 8-15, 16-22, 23-29, one core each, KV caches resident).
The TAEHV graphs traced by tier2/vae_neuron.py live inside the two least-loaded stage processes: the encoder with
stage 2 (core 2), the decoder with stage 3 (core 3). No extra processes; VAE temporal state (9 tensors each) is
carried on the host side of those processes.

Per chunk: 4 pixel frames -> encoder -> latent (rescaled from TAEHV's normalised latents to the raw latents the
pipeline's default encode produces: lat * std + mean) -> noise mix -> DiT row 0 -> re-noise -> DiT row 1 ->
decoder -> 4 pixel frames. TAEHV chunk f is pixel frames 4f..4f+3 (3 frames of lookahead relative to the Wan
encoder); decoded latent f is pixel frames 4f-3..4f, so the decoder's first 3 frames are dropped.

Warm-up (not timed): KV caches from g3_pipeline.py warm; encoder state from pixel chunks 0-2; decoder state from
the 3 warm-up latents. The timed loop starts at latent frame 3, like g3_pipeline.py.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from block_port import ART_DIR, MASK_NEG, BlockCfg, rope_cos_sin
from g2_common import G2, OUT, STEPS, to_u8, write_mp4
from g3_pipeline import SHM, load_stage, monitor_parse, monitor_start
from g3_stage import STAGES

BF16 = torch.bfloat16
VAE_DIR = ART_DIR / "vae"
VAE_ROLE = {2: "enc", 3: "dec"}  # stage index -> TAEHV half hosted in that stage's process
# Wan VAE latent statistics (repos/StreamDiffusionV2/models/wan/wan_wrapper.py:82-89)
MEAN = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                     0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]).view(1, 16, 1, 1)
STD = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]).view(1, 16, 1, 1)


def worker(stage, core, a, b, q_in, q_out, ready):
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
    import torch_neuronx  # noqa: F401
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    ctx = torch.load(SHM / "ctx.pt")
    rows = [load_stage(stage, a, b, r) for r in (0, 1)]
    role, vae, mems = VAE_ROLE.get(stage), None, None
    if role:
        vae = torch.jit.load(str(VAE_DIR / f"taehv_{role}_512x512_bf16.pt"))
        shapes = json.loads((VAE_DIR / f"taehv_{role}_compile.json").read_text())["input_shapes"][1:]
        mems = [torch.zeros(s, dtype=BF16) for s in shapes]
    last = stage == len(STAGES) - 1
    ready.put(stage)
    while True:
        job = q_in.get()
        if job is None:
            q_out.put(None)
            return
        if job.get("done") or job["kind"] not in ("dit", role):
            q_out.put(job)  # someone else's result or job: pass it down the chain
            continue
        t = time.perf_counter()
        if job["kind"] == "dit":
            job["x"] = rows[job["row"]](job["x"], job["e0"], ctx, job["cos"], job["sin"], job["bias"])[0].cpu()
            job["done"] = last
        else:
            res = vae(job["x"], *mems)
            mems = list(res[1:])
            job["x"], job["done"] = res[0].cpu(), True
        job["times"].append((stage, job["kind"], time.perf_counter() - t))
        q_out.put(job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=120)
    ap.add_argument("--inflight", type=int, default=6, help="chunks in flight")
    ap.add_argument("--name", default="e2e_4core")
    args = ap.parse_args()
    from g2_neuron import Host
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    cfg = BlockCfg()
    ref = torch.load(G2 / "ref.pt")
    calls = ref["calls"]
    host = Host(cfg, torch.load(G2 / "prompt_anime.pt")["prompt_embeds"])
    warm = torch.load(SHM / "warm.pt")
    x01 = ref["frames"][0].permute(1, 0, 2, 3) * 0.5 + 0.5  # [29, 3, H, W] in [0, 1]
    n_pix = x01.shape[0]

    def pix_chunk(f):  # TAEHV chunk f = pixel frames 4f..4f+3; the clip is looped
        return torch.stack([x01[(4 * f + k) % n_pix] for k in range(4)]).to(BF16)

    ctxm = mp.get_context("spawn")
    qs = [ctxm.Queue() for _ in range(len(STAGES) + 1)]
    ready = ctxm.Queue()
    procs = [ctxm.Process(target=worker, args=(s, s, a, b, qs[s], qs[s + 1], ready), daemon=True)
             for s, (a, b) in enumerate(STAGES)]
    t0 = time.time()
    for p in procs:
        p.start()
    for _ in procs:
        ready.get(timeout=900)
    print(f"4 stage processes ready in {time.time() - t0:.0f}s", flush=True)

    def vae_job(kind, x, tag):
        qs[2 if kind == "enc" else 3].put({"kind": kind, "x": x, "tag": tag, "times": []})

    # ---- warm-up, not timed: encoder state from chunks 0-2, decoder state from the 3 warm-up latents ----
    out_frames = []
    for f in range(3):
        vae_job("enc", pix_chunk(f), ("warm", f))
        qs[-1].get(timeout=300)
    warm_lat = torch.cat([warm["finals"][0], warm["finals"][1]], dim=1)[0]  # [3, 16, 64, 64]
    for f in range(3):
        vae_job("dec", warm_lat[f:f + 1].to(BF16), ("warm", f))
        fr = qs[-1].get(timeout=300)["x"].float()
        out_frames.append(fr[3:] if f == 0 else fr)

    def bias_for(call_idx):
        b = torch.zeros(cfg.num_kv_cache)
        b[3:3 + max(0, 2 - call_idx)] = MASK_NEG
        return b.repeat_interleave(cfg.frame_tokens).view(1, 1, 1, -1).to(BF16)

    meta = {}

    def submit_dit(row, j, lat, step):
        e, e0 = host.time(step, 1)
        cos_, sin_ = (t.to(BF16) for t in rope_cos_sin(cfg, 3 + j))
        meta[(row, j)] = {"lat": lat, "step": step, "e": e}
        qs[0].put({"kind": "dit", "row": row, "tag": j, "x": host.patch(lat).to(BF16), "e0": e0.to(BF16),
                   "cos": cos_, "sin": sin_, "bias": bias_for(j), "times": []})

    n = args.chunks
    t_submit, t_done, per_kind = {}, {}, {}
    active, next_j, done = 0, 0, 0
    mon_path = SHM / f"monitor_{args.name}.jsonl"
    mon = monitor_start(mon_path)
    time.sleep(2.5)
    t_start_wall = time.time()
    while done < n:
        while active < args.inflight and next_j < n:
            t_submit[next_j] = time.perf_counter()
            vae_job("enc", pix_chunk(3 + next_j), next_j)
            active, next_j = active + 1, next_j + 1
        job = qs[-1].get(timeout=300)
        j, kind = job["tag"], job["kind"]
        for stage, k, dt in job["times"]:
            per_kind.setdefault(f"stage{stage}_{k}", []).append(dt)
        c = calls[1 + (j % (len(calls) - 1))]
        if kind == "enc":
            lat = (job["x"].float() * STD + MEAN).unsqueeze(0)  # raw-latent convention of the pipeline's encode
            ns = c["noise_scale"]
            submit_dit(0, j, torch.randn_like(lat) * ns + lat * (1 - ns), c["current_step"])
        elif kind == "dit":
            m = meta.pop((job["row"], j))
            x0 = host.to_x0(host.head(job["x"].float(), m["e"], 1), m["lat"], m["step"])
            if job["row"] == 0:
                submit_dit(1, j, host.renoise(x0, torch.randn_like(x0), STEPS[1]), STEPS[1])
            else:
                vae_job("dec", x0[0].to(BF16), j)
        else:
            t_done[j] = time.perf_counter()
            if j < 5:
                out_frames.append(job["x"].float())
            active, done = active - 1, done + 1
    t_end_wall = time.time()
    time.sleep(1.5)
    mon.terminate()
    qs[0].put(None)

    skip = min(5, n // 4)
    order = sorted(t_done.values())
    steady_s = order[-1] - order[skip - 1]
    lat_ms = [(t_done[j] - t_submit[j]) * 1e3 for j in range(skip, n)]
    res = {"name": args.name, "chunks": n, "chunks_in_flight": args.inflight, "skipped_for_steady_state": skip,
           "steady_state_fps": round(4 * (n - skip) / steady_s, 2),
           "steady_state_s_per_chunk": round(steady_s / (n - skip), 4),
           "chunk_latency_ms": {"mean": round(float(np.mean(lat_ms)), 1), "p50": round(float(np.median(lat_ms)), 1),
                                "p99": round(float(np.percentile(lat_ms, 99)), 1), "max": round(max(lat_ms), 1)},
           "call_ms_mean": {k: round(1e3 * float(np.mean(v)), 2) for k, v in sorted(per_kind.items())},
           "vae_placement": {"enc": "stage 2 / core 2", "dec": "stage 3 / core 3"},
           "monitor": monitor_parse(mon_path, t_start_wall, t_end_wall)}
    (ART_DIR / f"g4_{args.name}.json").write_text(json.dumps(res, indent=2) + "\n")
    print("E2E " + json.dumps(res), flush=True)

    if args.name != "e2e_4core":  # only the default run writes the video
        for p in procs:
            p.join(timeout=20)
        return
    out = to_u8(torch.cat(out_frames).mul(2).sub(1))  # decoder output is [0, 1]
    inp = to_u8(ref["frames"][0].permute(1, 0, 2, 3)[:len(out)])
    write_mp4(OUT / "e2e_4core.mp4", np.concatenate([inp, out], axis=2))
    import cv2
    idx = [0, 8, 16, len(out) - 1]
    sheet = np.concatenate([np.concatenate([v[i] for i in idx], axis=1) for v in (inp, out)], axis=0)
    cv2.imwrite(str(OUT / "e2e_contact.png"), cv2.cvtColor(sheet[::2, ::2], cv2.COLOR_RGB2BGR))
    print(f"VIDEO OK {len(out)} frames", flush=True)
    for p in procs:
        p.join(timeout=20)


if __name__ == "__main__":
    main()
