"""TAEHV (taew2_1.pth) encoder and decoder on Neuron as fixed-shape, one-chunk streaming graphs.

    python vae_neuron.py compile enc|dec    trace one graph -> artifacts/tier2/vae/taehv_<enc|dec>_512x512_bf16.pt
    python vae_neuron.py ref                CPU fp32 references: TAEHV (repo code) and the Wan VAE round trips
    python vae_neuron.py eval               Neuron parity, quality, latency, tier2/out/vae_compare.png

One chunk = 4 pixel frames [4, 3, 512, 512] in [0, 1]  <->  1 latent frame [1, 16, 64, 64].

Temporal state: every MemBlock takes the previous timestep's input as `past` (taehv_wrapper.py:57-72). The graphs
take those memories as explicit inputs and return the updated ones (9 per graph), so chunk-by-chunk calls starting
from zero memories reproduce the repo's sequential pass over the whole video (`parallel=False`,
taehv_wrapper.py:116-153). TPool / TGrow strides divide the 4-frame chunk evenly, so they need no state.

Alignment with the Wan VAE (1 + 4k frames): TAEHV chunk i is pixel frames 4i..4i+3 and its latent lines up with
Wan latent i (frames 4i-3..4i); TAEHV decode correspondingly drops its first 3 output frames (`frames_to_trim`,
taehv_wrapper.py:186). So the encoder needs 3 frames of lookahead: a clip of 1 + 4k frames is padded with 3 copies
of its last frame at the END (as upstream taehv does), and a live stream simply runs 3 frames behind. Padding at
the front instead shifts the round trip by 3 frames (measured: latent cosine vs Wan 0.92 instead of 0.997).
"""
import json
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from g2_common import ART, OUT, SDV2_ROOT, load_clip, psnr, to_u8, use_repo

VAE_DIR = ART / "vae"
BF16 = torch.bfloat16
COMPILER_ARGS = ["--auto-cast=none"]
N_FRAMES = 29   # 5-frame session start + 6 stream chunks, same frames as the G2 run
GATE = 0.999


def artifact(which):
    return VAE_DIR / f"taehv_{which}_512x512_bf16.pt"


def load_taehv():
    use_repo()
    from models.wan.taehv_wrapper import TAEHV
    return TAEHV(checkpoint_path=str(SDV2_ROOT / "ckpts" / "taew2_1.pth")).eval().requires_grad_(False)


class StreamTAE(nn.Module):
    """One chunk through a TAEHV encoder or decoder: (x, *mems) -> (y, *new_mems). x is [T, C, H, W]."""

    def __init__(self, blocks):
        super().__init__()
        from models.wan.taehv_wrapper import MemBlock
        self.blocks = blocks
        self.is_mem = [isinstance(b, MemBlock) for b in blocks]

    def forward(self, x, *mems):
        new, i = [], 0
        for block, is_mem in zip(self.blocks, self.is_mem):
            if is_mem:
                first = mems[i] if mems else torch.zeros_like(x[:1])
                past = first if x.shape[0] == 1 else torch.cat([first, x[:-1]], dim=0)
                new.append(x[-1:].clone())
                x = block(x, past)
                i += 1
            else:
                x = block(x)
        return (x, *new)


def example_inputs(mod, which):
    x = torch.rand(4, 3, 512, 512) if which == "enc" else torch.randn(1, 16, 64, 64)
    return (x, *[torch.zeros_like(m) for m in mod(x)[1:]])


def stream(mod, shapes, chunks, dtype):
    """Feed chunks one at a time, carrying the memories. Returns the list of fp32 outputs."""
    mems, outs = None, []
    for x in chunks:
        if mems is None:
            mems = [torch.zeros(s, dtype=dtype) for s in shapes]
        res = mod(x.to(dtype), *mems)
        outs.append(res[0].float())
        mems = list(res[1:])
    return outs


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def u8_01(v):
    """[T, 3, H, W] in [0, 1] -> uint8 [T, H, W, 3]."""
    return (v.float().clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()


def padded_clip():
    frames = load_clip(N_FRAMES)                          # [1, 3, T, H, W] in [-1, 1]
    x01 = (frames[0].permute(1, 0, 2, 3) * 0.5 + 0.5)     # [T, 3, H, W] in [0, 1]
    return frames, torch.cat([x01, x01[-1:].repeat(3, 1, 1, 1)])  # 32 frames = 8 chunks


def compile_one(which):
    import torch_neuronx
    tae = load_taehv()
    mod = StreamTAE(tae.encoder if which == "enc" else tae.decoder).eval()
    ex = example_inputs(mod, which)
    shapes = [list(t.shape) for t in ex]
    VAE_DIR.mkdir(parents=True, exist_ok=True)
    work = VAE_DIR / "work" / which
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    traced = torch_neuronx.trace(mod.to(BF16), tuple(t.to(BF16) for t in ex), compiler_workdir=str(work),
                                 compiler_args=COMPILER_ARGS)
    dt = time.perf_counter() - t0
    torch.jit.save(traced, str(artifact(which)))
    (VAE_DIR / f"taehv_{which}_compile.json").write_text(json.dumps(
        {"which": which, "compile_s": round(dt, 1), "compiler_args": COMPILER_ARGS, "input_shapes": shapes,
         "state_mb_bf16": round(sum(int(np.prod(s)) for s in shapes[1:]) * 2 / 1e6, 1)}) + "\n")
    print(f"{which}: COMPILE OK {dt:.1f}s  inputs {shapes}", flush=True)


def ref():
    """Everything that needs only the CPU: repo TAEHV, the chunked port in fp32, and the Wan VAE round trips."""
    from g2_decode import decode_wan
    tae = load_taehv()
    from models.wan.wan_wrapper import WanVAEWrapper
    frames, x32 = padded_clip()
    t = time.perf_counter()
    lat_tae = tae.encode_video(x32.unsqueeze(0), parallel=False)          # [1, 8, 16, 64, 64]
    tae_enc_s = (time.perf_counter() - t) / 8
    t = time.perf_counter()
    dec_tae = tae.decode_video(lat_tae, parallel=False)                   # [1, 32, 3, 512, 512]
    tae_dec_s = (time.perf_counter() - t) / 8

    port = {}
    for which, seq in (("enc", tae.encoder), ("dec", tae.decoder)):
        m = StreamTAE(seq).eval()
        port[which] = (m, [t.shape for t in example_inputs(m, which)[1:]])
    lat_port = torch.stack(stream(*port["enc"], x32.split(4), torch.float32), dim=1)
    dec_port = torch.cat(stream(*port["dec"], lat_tae[0].split(1), torch.float32)).unsqueeze(0)
    port_check = {"enc_cos": cos(lat_port, lat_tae), "enc_max_abs": (lat_port - lat_tae).abs().max().item(),
                  "dec_cos": cos(dec_port, dec_tae), "dec_max_abs": (dec_port - dec_tae).abs().max().item()}
    print("chunked port vs repo sequential (fp32):", port_check, flush=True)

    vae = WanVAEWrapper().eval().requires_grad_(False)
    vae.model.first_encode = True
    lats, enc_s = [], []
    old = torch.load(VAE_DIR / "ref.pt") if (VAE_DIR / "ref.pt").exists() else None  # Wan-only parts are reusable
    for sl in [] if old else [slice(0, 5)] + [slice(5 + 4 * i, 9 + 4 * i) for i in range((N_FRAMES - 5) // 4)]:
        t = time.perf_counter()
        lats.append(vae.stream_encode(frames[:, :, sl], is_scale=True).transpose(2, 1).float())
        enc_s.append(time.perf_counter() - t)
        print(f"wan encode {sl}: {enc_s[-1]:.1f}s", flush=True)
    if old:
        lat_wan, wan_rt, wan_s = old["lat_wan"], old["wan_rt"], {k: v for k, v in old["cpu_s"].items() if k.startswith("wan")}
    else:
        lat_wan = torch.cat(lats, dim=1)                                  # [1, 8, 16, 64, 64], mean/std normalised
        wan_rt, dec_s = decode_wan(vae, lat_wan)                          # [29, 3, H, W] in [-1, 1]
        wan_s = {"wan_encode_first_5_frames": enc_s[0], "wan_encode_per_chunk": float(np.mean(enc_s[1:])),
                 "wan_decode_first_2_latents": dec_s[0], "wan_decode_per_chunk": float(np.mean(dec_s[1:]))}
    print(f"wan round trip ready: {wan_s}", flush=True)
    wan_from_tae, _ = decode_wan(vae, lat_tae)
    VAE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"frames": frames, "x32": x32, "lat_tae": lat_tae, "dec_tae": dec_tae, "lat_wan": lat_wan,
                "wan_rt": wan_rt, "wan_from_tae": wan_from_tae, "port_check": port_check,
                "cpu_s": {"taehv_encode_per_chunk": tae_enc_s, "taehv_decode_per_chunk": tae_dec_s, **wan_s,
                          "threads": torch.get_num_threads()}}, VAE_DIR / "ref.pt")
    print("REF OK", flush=True)


def bench(mod, shapes, x, iters=100, warmup=5):
    mems = [torch.zeros(s, dtype=BF16) for s in shapes]
    x, ms = x.to(BF16), []
    for i in range(warmup + iters):
        t = time.perf_counter()
        res = mod(x, *mems)
        if i >= warmup:
            ms.append((time.perf_counter() - t) * 1e3)
        mems = list(res[1:])
    ms.sort()
    return {"iters": iters, "mean_ms": statistics.mean(ms), "p50_ms": ms[len(ms) // 2], "p99_ms": ms[int(len(ms) * 0.99) - 1]}


def evaluate():
    import cv2
    import torch_neuronx  # noqa: F401
    r = torch.load(VAE_DIR / "ref.pt")
    x32, lat_tae, dec_tae, lat_wan = r["x32"], r["lat_tae"], r["dec_tae"], r["lat_wan"]
    mods = {}
    for which in ("enc", "dec"):
        mods[which] = (torch.jit.load(str(artifact(which))),
                       json.loads((VAE_DIR / f"taehv_{which}_compile.json").read_text())["input_shapes"][1:])
    n = x32.shape[0] // 4

    # ---- parity: Neuron bf16 (chunked, own state fed back) vs repo TAEHV on CPU fp32 (one sequential pass) ----
    lat_n = stream(*mods["enc"], x32.split(4), BF16)                       # 8 x [1, 16, 64, 64]
    dec_n = stream(*mods["dec"], lat_tae[0].split(1), BF16)                # 8 x [4, 3, 512, 512], CPU latents in
    parity = [{"chunk": i, "enc_cos": cos(lat_n[i], lat_tae[0, i]),
               "enc_max_abs": (lat_n[i][0] - lat_tae[0, i]).abs().max().item(),
               "dec_cos": cos(dec_n[i], dec_tae[0, 4 * i:4 * i + 4]),
               "dec_psnr": psnr(u8_01(dec_n[i]), u8_01(dec_tae[0, 4 * i:4 * i + 4]))} for i in range(n)]
    for p in parity:
        print(p, flush=True)
    min_cos = min(min(p["enc_cos"], p["dec_cos"]) for p in parity)

    # ---- quality: round trips against the input clip (frames 0..28; TAEHV decode drops its first 3 frames) ----
    inp = u8_01(x32[:N_FRAMES])
    full_n = u8_01(torch.cat(stream(*mods["dec"], lat_n, BF16))[3:])       # Neuron encode -> Neuron decode
    dec_of_wan = u8_01(torch.cat(stream(*mods["dec"], lat_wan[0].split(1), BF16))[3:])  # Wan encode -> Neuron decode
    tae_cpu = u8_01(dec_tae[0, 3:])
    wan_rt, wan_from_tae = to_u8(r["wan_rt"]), to_u8(r["wan_from_tae"])
    bounds = [(0, 5)] + [(5 + 4 * i, 9 + 4 * i) for i in range(n - 2)]
    lat_n_t = torch.cat(lat_n).unsqueeze(0)
    quality = [{"chunk": i, "frames": [a, b],
                "psnr_wan_enc_wan_dec": psnr(wan_rt[a:b], inp[a:b]),
                "psnr_taehv_enc_taehv_dec_neuron": psnr(full_n[a:b], inp[a:b]),
                "psnr_taehv_enc_taehv_dec_cpu": psnr(tae_cpu[a:b], inp[a:b]),
                "psnr_wan_enc_taehv_dec_neuron": psnr(dec_of_wan[a:b], inp[a:b]),
                "psnr_taehv_enc_wan_dec": psnr(wan_from_tae[a:b], inp[a:b]),
                "latent_cos_taehv_vs_wan": cos(lat_n_t[:, sl], lat_wan[:, sl]),
                "latent_rms_taehv": lat_n_t[:, sl].pow(2).mean().sqrt().item(),
                "latent_rms_wan": lat_wan[:, sl].pow(2).mean().sqrt().item(),
                "latent_rms_diff": (lat_n_t[:, sl] - lat_wan[:, sl]).pow(2).mean().sqrt().item()}
               for i, (a, b) in enumerate(bounds) for sl in [slice(0, 2) if i == 0 else slice(i + 1, i + 2)]]
    for q in quality:
        print(q, flush=True)
    names = ["psnr_wan_enc_wan_dec", "psnr_taehv_enc_taehv_dec_neuron", "psnr_taehv_enc_taehv_dec_cpu",
             "psnr_wan_enc_taehv_dec_neuron", "psnr_taehv_enc_wan_dec"]
    arrs = [wan_rt, full_n, tae_cpu, dec_of_wan, wan_from_tae]
    overall = {k: psnr(v, inp) for k, v in zip(names, arrs)}
    overall["latent_cos_taehv_vs_wan"] = cos(lat_n_t, lat_wan)

    # ---- latency on one core: state fed back every call, host call to host return ----
    latency = {"enc": bench(*mods["enc"], x32[4:8]), "dec": bench(*mods["dec"], lat_tae[0, 1:2])}
    print(latency, flush=True)

    res = {"gate": GATE, "min_cosine": min_cos, "pass": bool(min_cos >= GATE), "parity": parity,
           "quality": quality, "quality_all_frames": overall, "latency_ms": latency, "cpu_s": r["cpu_s"],
           "port_check_fp32": r["port_check"],
           "compile": {w: json.loads((VAE_DIR / f"taehv_{w}_compile.json").read_text()) for w in ("enc", "dec")}}
    (VAE_DIR / "vae_neuron.json").write_text(json.dumps(res, indent=2) + "\n")

    OUT.mkdir(parents=True, exist_ok=True)
    cols = [("input", inp), ("Wan enc + Wan dec (CPU)", wan_rt), ("TAEHV enc + dec (Neuron)", full_n),
            ("TAEHV enc (Neuron) + Wan dec", wan_from_tae), ("Wan enc + TAEHV dec (Neuron)", dec_of_wan)]
    rows = []
    for f in (2, 14, 26):
        tiles = []
        for name, v in cols:
            tile = cv2.resize(v[f], (384, 384), interpolation=cv2.INTER_AREA).copy()
            cv2.rectangle(tile, (0, 0), (383, 22), (0, 0, 0), -1)
            cv2.putText(tile, f"{name}  f{f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        rows.append(np.concatenate(tiles, axis=1))
    cv2.imwrite(str(OUT / "vae_compare.png"), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
    print(f"VAE EVAL min_cosine={min_cos:.6f} gate={GATE} -> {'PASS' if res['pass'] else 'FAIL'}", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    cmd = sys.argv[1]
    if cmd == "compile":
        compile_one(sys.argv[2])
    else:
        {"ref": ref, "eval": evaluate}[cmd]()
