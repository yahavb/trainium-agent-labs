"""Decode the reference and Neuron latents with the same VAE, compare frames, write the side-by-side video.

    python g2_decode.py [--backend neuron]

Wan VAE: the repo's WanVAEWrapper stream decode (same call pattern as the pipeline: 2 latents first, then 1 at a
time). TAEHV: the repo's TAEHV class with ckpts/taew2_1.pth, fp32, all latents in one causal pass.
Outputs in tier2/out/: g2_side_by_side.mp4 (input | CPU ref | Neuron, Wan VAE), g2_side_by_side_taehv.mp4,
g2_contact.png, and artifacts/tier2/g2/decode_<backend>.json.
"""
import argparse
import json
import time

import numpy as np
import torch

from g2_common import G2, OUT, SDV2_ROOT, load_clip, psnr, to_u8, use_repo, write_mp4


def final_latents_ref(ref):
    """Clean latents in output order: 2 from the session start, then row 1 of every stream call after the first."""
    return torch.cat([ref["start"]["denoised"]] + [c["x0"][1:2] for c in ref["calls"][1:]], dim=1)


def decode_wan(vae, lat):
    """lat [1, T, 16, h, w] -> ([T', 3, H, W] in [-1, 1], seconds per call). wan_wrapper.py:143-151 in fp32."""
    vae.model.first_decode = True
    scale = [vae.mean, 1.0 / vae.std]
    outs, secs = [], []
    for sl in [slice(0, 2)] + [slice(i, i + 1) for i in range(2, lat.shape[1])]:
        t = time.perf_counter()
        v = vae.model.stream_decode(lat[:, sl].permute(0, 2, 1, 3, 4), scale).float().clamp_(-1, 1)
        secs.append(time.perf_counter() - t)
        outs.append(v[0].permute(1, 0, 2, 3))
    return torch.cat(outs), secs


def decode_taehv(taehv, lat):
    t = time.perf_counter()
    v = taehv.decode_video(lat, parallel=False)[0]  # [4T, 3, H, W] in [0, 1]
    return v[3:].mul(2).sub(1).clamp(-1, 1), time.perf_counter() - t  # first latent stands for 1 frame, not 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="neuron")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    use_repo()
    from models.wan.taehv_wrapper import TAEHV
    from models.wan.wan_wrapper import WanVAEWrapper

    ref = torch.load(G2 / "ref.pt")
    lat_ref = final_latents_ref(ref)
    lat_new = torch.load(G2 / f"latents_{args.backend}.pt")["finals"].float()
    assert lat_ref.shape == lat_new.shape, (lat_ref.shape, lat_new.shape)
    n_frames = 1 + 4 * (lat_ref.shape[1] - 1)
    inp = to_u8(ref["frames"][0].permute(1, 0, 2, 3)[:n_frames])

    vae = WanVAEWrapper().eval().requires_grad_(False)
    ref_wan, secs_ref = decode_wan(vae, lat_ref)
    new_wan, secs_new = decode_wan(vae, lat_new)
    taehv = TAEHV(checkpoint_path=str(SDV2_ROOT / "ckpts" / "taew2_1.pth")).eval().requires_grad_(False)
    ref_tae, tae_s = decode_taehv(taehv, lat_ref)
    new_tae, _ = decode_taehv(taehv, lat_new)
    ref_wan, new_wan, ref_tae, new_tae = (to_u8(v) for v in (ref_wan, new_wan, ref_tae, new_tae))

    # chunk 0 = the 5 start frames (2 latents); chunk i = the next 4 frames
    bounds = [(0, 5)] + [(5 + 4 * i, 9 + 4 * i) for i in range(lat_ref.shape[1] - 2)]
    chunks = [{"chunk": i, "frames": [a, b], "psnr_wan_neuron_vs_ref": psnr(new_wan[a:b], ref_wan[a:b]),
               "psnr_taehv_neuron_vs_ref": psnr(new_tae[a:b], ref_tae[a:b]),
               "latent_cos": torch.nn.functional.cosine_similarity(
                   lat_new[:, max(i + 1, 0) if i else slice(0, 2)].flatten().double(),
                   lat_ref[:, max(i + 1, 0) if i else slice(0, 2)].flatten().double(), dim=0).item()}
              for i, (a, b) in enumerate(bounds)]
    res = {"backend": args.backend, "frames": n_frames, "chunks": chunks,
           "psnr_wan_all": psnr(new_wan, ref_wan), "psnr_taehv_all": psnr(new_tae, ref_tae),
           "psnr_ref_taehv_vs_ref_wan": psnr(ref_tae, ref_wan),
           "psnr_ref_wan_vs_input": psnr(ref_wan, inp),
           "vae_cpu_seconds": {"wan_decode_first_call_2_latents": secs_ref[0],
                               "wan_decode_per_chunk_mean": float(np.mean(secs_ref[1:])),
                               "taehv_decode_all_latents": tae_s,
                               "taehv_decode_per_chunk": tae_s / lat_ref.shape[1],
                               "wan_encode_start_5_frames": ref["start"]["encode_s"],
                               "wan_encode_per_chunk_mean": float(np.mean([c["encode_s"] for c in ref["calls"]]))}}
    (G2 / f"decode_{args.backend}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))

    OUT.mkdir(parents=True, exist_ok=True)
    write_mp4(OUT / "g2_side_by_side.mp4", np.concatenate([inp, ref_wan, new_wan], axis=2))
    write_mp4(OUT / "g2_side_by_side_taehv.mp4", np.concatenate([inp, ref_tae, new_tae], axis=2))
    import cv2
    idx = [0, 8, 16, n_frames - 1]
    sheet = np.concatenate([np.concatenate([v[i] for i in idx], axis=1) for v in (inp, ref_wan, new_wan, new_tae)], axis=0)
    cv2.imwrite(str(OUT / "g2_contact.png"), cv2.cvtColor(sheet[::2, ::2], cv2.COLOR_RGB2BGR))
    print("DECODE OK")


if __name__ == "__main__":
    main()
