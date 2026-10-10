"""Why are TAEHV-encoded latents poor? CPU fp32 probe of frame alignment and of a static video (no motion)."""
import json

import torch

from vae_neuron import VAE_DIR, cos, load_taehv, psnr, u8_01

torch.set_grad_enabled(False)
tae = load_taehv()
r = torch.load(VAE_DIR / "ref.pt")
x29, lat_wan = r["x32"][:29], r["lat_wan"]         # [29, 3, H, W] in [0, 1]; [1, 8, 16, 64, 64]
out = {}


def roundtrip(x):
    lat = tae.encode_video(x.unsqueeze(0), parallel=False)
    return lat, tae.decode_video(lat, parallel=False)[0]


for name, x in {"pad_front_3": torch.cat([x29[:1].repeat(3, 1, 1, 1), x29]),
                "pad_end_3": torch.cat([x29, x29[-1:].repeat(3, 1, 1, 1)]),
                "pad_front_2_end_1": torch.cat([x29[:1].repeat(2, 1, 1, 1), x29, x29[-1:]]),
                "pad_front_1_end_2": torch.cat([x29[:1], x29, x29[-1:].repeat(2, 1, 1, 1)])}.items():
    lat, dec = roundtrip(x)
    res = {"latent_cos_vs_wan": cos(lat, lat_wan)}
    for shift in range(-3, 8):                       # decoded frame j is compared with input frame j + shift
        a, b = max(0, -shift), min(32, 29 - shift)
        res[f"psnr_dec[3+j]_vs_input[j{shift:+d}]"] = psnr(u8_01(dec[3 + a:b + 3 if b + 3 <= 32 else 32]),
                                                         u8_01(x29[a + shift:a + shift + len(dec[3 + a:b + 3])]))
    out[name] = res
    print(name, json.dumps(res, indent=1), flush=True)

static = x29[14:15].repeat(32, 1, 1, 1)
lat, dec = roundtrip(static)
out["static_frame_14_x32"] = {"psnr_per_latent": [psnr(u8_01(dec[4 * i:4 * i + 4]), u8_01(static[:4])) for i in range(8)],
                              "latent_cos_vs_wan_latent_4": cos(lat[0, 4], lat_wan[0, 4])}
print("static", json.dumps(out["static_frame_14_x32"]), flush=True)
out["input_frame_to_frame_psnr"] = psnr(u8_01(x29[1:]), u8_01(x29[:-1]))
out["input_psnr_at_lag_4"] = psnr(u8_01(x29[4:]), u8_01(x29[:-4]))
print("motion", out["input_frame_to_frame_psnr"], out["input_psnr_at_lag_4"])
(VAE_DIR / "enc_probe.json").write_text(json.dumps(out, indent=2) + "\n")
print("PROBE OK")
