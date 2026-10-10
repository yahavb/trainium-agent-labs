"""End to end on one NeuronCore: TAEHV encoder -> 30-layer causal Wan DiT -> TAEHV decoder, all traced graphs.

    python e2e.py            run tier2/data/test_clip.mp4, write tier2/out/e2e_trainium.mp4, e2e_compare.png, e2e.json

Reuses Track 2's G2 host code unchanged (g2_neuron.Runner: schedule, per-row KV caches, sink handling, cached T5
embedding) and the TAEHV graphs from vae_neuron.py.

Encode side, the two fixes from reports/vae_neuron.md:
  * lookahead: TAEHV latent k is pixel frames 4k..4k+3 (Wan latent k is frames 4k-3..4k), so the clip is padded
    with 3 copies of its last frame at the END and every encoder call is one plain 4-frame chunk;
  * scale: TAEHV latents are mean/std normalised, the pipeline encodes with is_scale=False
    (streamv2v/inference.py:148), so they are mapped back with lat * std + mean (wan_wrapper.py:82-91).
Decode side: the DiT's x0 is already in the normalised space TAEHV decodes; the first 3 decoded frames are dropped.

One-off: the session start (2 latent frames in one 2048-token chunk) runs the DiT on CPU with the fp32 port, as
in Track 2's G2 run; the traced block graph is fixed at 1 frame. Every stream chunk is Neuron only.

Noise: if artifacts/tier2/g2/ref.pt exists (CPU reference with the Wan VAE, same seed as Track 2's G2 run), its
noise tensors are replayed so the output is comparable with that reference; otherwise torch.manual_seed(SEED).
"""
import argparse
import json
import time

import cv2
import numpy as np
import torch

from g2_common import G2, NOISE_SCALE, NUM_STREAM_CHUNKS, OUT, SEED, STEPS, TIER2, load_clip, noise_scale_and_step, psnr, to_u8, write_mp4
from g2_neuron import Runner
from vae_neuron import BF16, VAE_DIR, artifact, u8_01

WAN_MEAN = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                         0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]).view(1, 16, 1, 1)
WAN_STD = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]).view(1, 16, 1, 1)


class StreamGraph:
    """A traced TAEHV half plus its 9 memories, fed back on every call."""

    def __init__(self, which):
        self.mod = torch.jit.load(str(artifact(which)))
        shapes = json.loads((VAE_DIR / f"taehv_{which}_compile.json").read_text())["input_shapes"][1:]
        self.mems = [torch.zeros(s, dtype=BF16) for s in shapes]
        self.seconds = 0.0

    def __call__(self, x):
        t = time.perf_counter()
        res = self.mod(x.to(BF16), *self.mems)
        self.seconds += time.perf_counter() - t
        self.mems = list(res[1:])
        return res[0].float()


def label(frames_u8, text):
    out = frames_u8.copy()
    for f in out:
        cv2.rectangle(f, (0, 0), (f.shape[1] - 1, 26), (0, 0, 0), -1)
        cv2.putText(f, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def read_panel(path, panel, n):
    """Frames of one 512-wide panel of a side-by-side mp4, RGB uint8."""
    cap, out = cv2.VideoCapture(str(path)), []
    while len(out) < n:
        ok, bgr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(bgr[:, 512 * panel:512 * (panel + 1)], cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=NUM_STREAM_CHUNKS,
                    help="stream chunks; above 6 the 29-frame clip is looped and only timing is reported")
    ap.add_argument("--skip", type=int, default=5, help="looped mode: chunks excluded from the steady state")
    args = ap.parse_args()
    loop = args.chunks > NUM_STREAM_CHUNKS
    torch.set_grad_enabled(False)
    import torch_neuronx  # noqa: F401
    n_frames = 5 + 4 * NUM_STREAM_CHUNKS
    frames = load_clip(n_frames)                                   # [1, 3, 29, H, W] in [-1, 1]
    x01 = frames[0].permute(1, 0, 2, 3) * 0.5 + 0.5
    x32 = torch.cat([x01, x01[-1:].repeat(3, 1, 1, 1)])            # end padding: 3-frame encoder lookahead
    ref = torch.load(G2 / "ref.pt") if (G2 / "ref.pt").exists() else None
    replay = ref is not None and "enc" in ref and not loop
    print(f"noise: {'replayed from the CPU reference' if replay else f'seed {SEED}'}", flush=True)
    torch.manual_seed(SEED)

    run = Runner("neuron", None)
    enc, dec = StreamGraph("enc"), StreamGraph("dec")
    host = run.host

    def encode(k):
        """TAEHV latent k (frames 4k..4k+3) in the pipeline's un-normalised latent space: [1, 1, 16, 64, 64]."""
        px = x01[[(4 * k + j) % n_frames for j in range(4)]] if loop else x32[4 * k:4 * k + 4]  # loop: wrap the clip
        return (enc(px) * WAN_STD + WAN_MEAN).unsqueeze(1)

    def decode(lat):
        """One finished latent [1, 1, 16, 64, 64] -> 4 uint8 frames."""
        return u8_01(dec(lat[0]))

    out, chunks = [], []
    # ---- session start: latents 0 and 1 (8 input frames with the lookahead), DiT on the CPU port ----
    t0 = time.perf_counter()
    lat = torch.cat([encode(0), encode(1)], dim=1)
    enc_lats = [lat]
    n0 = ref["enc"][0]["noise"] if replay else torch.randn_like(lat)
    noisy = n0 * NOISE_SCALE + lat * (1 - NOISE_SCALE)
    start_noises = ref["start"]["noises"] if replay else [torch.randn(2, 16, 64, 64)]
    t1 = time.perf_counter()
    x0s = run.start(noisy, start_noises)
    t2 = time.perf_counter()
    out += [decode(x0s[-1][:, :1])[3:], decode(x0s[-1][:, 1:2])]   # 1 + 4 frames
    start = {"encode_s": enc.seconds, "dit_cpu_s": t2 - t1, "decode_s": dec.seconds, "wall_s": time.perf_counter() - t0}
    print(f"start: {start}", flush=True)

    # ---- stream chunks: everything on the NeuronCore ----
    hidden_prev, scale, last = None, NOISE_SCALE, frames[:, :, [4]]
    for c in range(args.chunks):
        enc.seconds = dec.seconds = 0.0
        run.t = {"dit": 0.0, "host": 0.0}
        tc = time.perf_counter()
        images = frames[:, :, [(5 + 4 * c + j) % n_frames for j in range(4)]]
        scale, step = noise_scale_and_step(torch.cat([last, images], dim=2), scale, NOISE_SCALE)
        lat = encode(2 + c)
        enc_lats.append(lat)
        n = ref["enc"][1 + c]["noise"][:, :1] if replay else torch.randn_like(lat)
        noisy = n * scale + lat * (1 - scale)
        x0_r0 = run.forward_row(0, noisy, step, 2 + c)
        emitted = 0
        if hidden_prev is not None:
            x0_r1 = run.forward_row(1, hidden_prev, STEPS[1], 1 + c)
            out.append(decode(x0_r1))
            emitted = 4
        n2 = ref["calls"][c]["noise"].view_as(x0_r0) if replay else torch.randn_like(x0_r0)
        hidden_prev = host.renoise(x0_r0, n2, STEPS[1])
        wall = time.perf_counter() - tc
        row = {"chunk": c, "current_step": step, "noise_scale": scale, "frames_out": emitted, "wall_s": wall,
               "encode_s": enc.seconds, "dit_s": run.t["dit"], "decode_s": dec.seconds,
               "host_s": wall - enc.seconds - run.t["dit"] - dec.seconds, "fps": emitted / wall}
        chunks.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        last = images[:, :, [-1]]

    if loop:  # same accounting as Track 2's g4_e2e.py: looped clip, first `skip` chunks excluded
        st = chunks[args.skip:]
        walls = sorted(r["wall_s"] for r in st)
        res = {"measured": True, "core": "one NeuronCore (NEURON_RT_VISIBLE_CORES=0)", "chunks": args.chunks,
               "skipped_for_steady_state": args.skip, "noise": f"seed {SEED}", "start": start,
               "steady_state_fps": 4 * len(st) / sum(walls),
               "chunk_s": {"mean": float(np.mean(walls)), "p50": float(np.median(walls)),
                           "p99": float(np.percentile(walls, 99)), "max": walls[-1], "min": walls[0]},
               "steady_state_mean_s": {k: float(np.mean([r[k] for r in st]))
                                       for k in ("encode_s", "dit_s", "decode_s", "host_s")},
               "chunk_rows": chunks}
        (G2 / f"e2e_loop{args.chunks}.json").write_text(json.dumps(res, indent=2) + "\n")
        print("E2E LOOP " + json.dumps({k: v for k, v in res.items() if k != "chunk_rows"}), flush=True)
        return

    video = np.concatenate(out)                                    # 25 frames: 1 + 4 * 6
    n_out = video.shape[0]
    inp = to_u8(frames[0].permute(1, 0, 2, 3)[:n_out])
    steady = [r for r in chunks if r["frames_out"]][2:]            # drop the first two full chunks (still warming up)
    mean = {k: float(np.mean([r[k] for r in steady])) for k in ("wall_s", "encode_s", "dit_s", "decode_s", "host_s")}
    res = {"measured": True, "core": "one NeuronCore (NEURON_RT_VISIBLE_CORES=0)", "noise": "replay" if replay else "seed",
           "frames_out": n_out, "start": start, "chunks": chunks, "steady_state_chunks": [r["chunk"] for r in steady],
           "steady_state_mean_s": mean, "fps_measured": 4.0 / mean["wall_s"]}

    if replay:  # TAEHV (Neuron) latents after rescaling vs the Wan encoder's un-normalised latents
        cs = torch.nn.functional.cosine_similarity
        res["encoder_latent_vs_wan"] = [
            {"cos": cs(a.flatten().double(), e["lat"].flatten().double(), dim=0).item(),
             "rms_taehv": a.pow(2).mean().sqrt().item(), "rms_wan": e["lat"].pow(2).mean().sqrt().item(),
             "rms_diff": (a - e["lat"]).pow(2).mean().sqrt().item()} for a, e in zip(enc_lats, ref["enc"])]
        print("encoder latents vs Wan:", [round(d["cos"], 4) for d in res["encoder_latent_vs_wan"]],
              [round(d["rms_diff"], 3) for d in res["encoder_latent_vs_wan"]], flush=True)

    # ---- quality against the G2 outputs ----
    bounds = [(0, 5)] + [(5 + 4 * i, 9 + 4 * i) for i in range((n_out - 5) // 4)]
    comps = {}
    t2_video = TIER2 / "out" / "g2_side_by_side.mp4"
    if t2_video.exists():   # Track 2's committed G2 video: input | CPU reference | Neuron DiT, all decoded by the Wan VAE
        comps["track2_g2_neuron_dit_wan_vae_mp4"] = read_panel(t2_video, 2, n_out)
        comps["track2_g2_cpu_ref_wan_vae_mp4"] = read_panel(t2_video, 1, n_out)
    if ref is not None:     # this seat's CPU reference latents (Wan VAE encode, CPU DiT) through the Neuron TAEHV decoder
        from g2_decode import final_latents_ref
        d2 = StreamGraph("dec")
        fr = np.concatenate([u8_01(d2(l)) for l in final_latents_ref(ref)[0].split(1)])[3:]
        comps["cpu_ref_latents_taehv_decode_neuron"] = fr[:n_out]
    res["quality_psnr_db"] = {name: {"all": psnr(video, v), "per_chunk": [psnr(video[a:b], v[a:b]) for a, b in bounds]}
                              for name, v in comps.items() if len(v) == n_out}
    res["psnr_output_vs_input"] = psnr(video, inp)
    for k, v in res["quality_psnr_db"].items():
        print(k, round(v["all"], 2), [round(p, 2) for p in v["per_chunk"]], flush=True)
    (G2 / "e2e.json").write_text(json.dumps(res, indent=2) + "\n")

    OUT.mkdir(parents=True, exist_ok=True)
    side = np.concatenate([label(inp, "input"), label(video, "all stages on Trainium2 (1 core)")], axis=2)
    write_mp4(OUT / "e2e_trainium.mp4", side)
    rows = [("input", inp), ("all stages on Trainium2", video)]
    if "track2_g2_neuron_dit_wan_vae_mp4" in res["quality_psnr_db"]:
        rows.insert(1, ("Track 2 G2 (Wan VAE on CPU)", comps["track2_g2_neuron_dit_wan_vae_mp4"]))
    idx = [0, 8, 16, n_out - 1]
    sheet = np.concatenate([np.concatenate([label(v[idx], name)[i] for i in range(len(idx))], axis=1)
                            for name, v in rows], axis=0)
    cv2.imwrite(str(OUT / "e2e_compare.png"), cv2.cvtColor(sheet[::2, ::2], cv2.COLOR_RGB2BGR))
    print(f"E2E OK steady-state {mean}  MEASURED FPS {res['fps_measured']:.2f}", flush=True)


if __name__ == "__main__":
    main()
