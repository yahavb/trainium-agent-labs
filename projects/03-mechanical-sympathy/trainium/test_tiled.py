"""CPU check: the tiled Samudra UNet must match the untiled one.

    python test_tiled.py --ckpt /workspace/samudra/onedeg/ema_ckpt.pt
"""
import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from samudra_core import SamudraNet, load_checkpoint  # noqa: E402
from samudra_tiled import tile_large_blocks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    H, W = 180, 360
    model = SamudraNet(154, 8, 154, norm="batch", use_bfloat16=False).eval()
    missing, unexpected = load_checkpoint(model, args.ckpt)
    assert not missing and not unexpected, (missing, unexpected)

    g = torch.Generator().manual_seed(1)
    prog = torch.randn(1, 154, H, W, generator=g)
    bnd = torch.randn(1, 8, H, W, generator=g)
    mask = (torch.rand(1, 1, H, W, generator=g) >= 0.3).expand(1, 154, H, W)

    with torch.inference_mode():
        ref = model(prog, bnd, mask)
        ref_rms = ref[mask].pow(2).mean().sqrt().item()
        failed = False
        # min_pixels 64800 tiles level 0 only; 16200 also tiles level 1 (dilation 2).
        for band_rows, min_pixels in [(10, 64800), (20, 64800), (45, 64800), (90, 64800),
                                      (15, 16200), (45, 16200)]:
            tiled = copy.deepcopy(model)
            names = tile_large_blocks(tiled, (H, W), band_rows, min_pixels)
            out = tiled(prog, bnd, mask)
            d = (out - ref)[mask]
            rel = d.pow(2).mean().sqrt().item() / ref_rms
            max_abs = d.abs().max().item()
            land_zero = bool((out[~mask] == 0).all())
            ok = rel < 1e-5 and land_zero
            failed |= not ok
            print(f"band_rows={band_rows:3d} min_pixels={min_pixels:6d} tiled={len(names)} "
                  f"rel_rms={rel:.2e} max_abs={max_abs:.2e} land_zero={land_zero} "
                  f"{'OK' if ok else 'FAIL'}", flush=True)
            if band_rows == 10:
                for n in names:
                    print("   ", n)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
