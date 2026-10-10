# Checks samudra_core.SamudraNet against the ORIGINAL Samudra modules (loaded from a clone without
# installing the package): same parameter names, and identical output for the same weights.
import importlib.util, sys, types, itertools
from pathlib import Path
import torch

SRC = Path(sys.argv[1]) / "src" / "samudra"

def stub(name, **attrs):
    m = types.ModuleType(name); m.__path__ = []; m.__dict__.update(attrs); sys.modules[name] = m

def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, SRC / rel)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m

def pairwise(it):
    a, b = itertools.tee(it); next(b, None); return zip(a, b)

for p in ["samudra", "samudra.models", "samudra.models.modules", "samudra.utils"]:
    stub(p)
stub("samudra.utils.train", pairwise=pairwise)
act = load("samudra.models.modules.activations", "models/modules/activations.py")
blocks = load("samudra.models.modules.blocks", "models/modules/blocks.py")
unetmod = load("samudra.models.modules.unet_backbone", "models/modules/unet_backbone.py")

sys.path.insert(0, str(Path(__file__).parent))
from samudra_core import SamudraNet

torch.manual_seed(0)
P, B_, O = 154, 8, 154
mine = SamudraNet(P, B_, O, use_bfloat16=False).eval()

def create_block(in_channels, out_channels, dilation, n_layers, pad, checkpoint_simple):
    return blocks.ConvNeXtBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                dilation=dilation, n_layers=n_layers, activation=act.CappedGELU,
                                pad=pad, upscale_factor=2, norm="instance")

orig_unet = unetmod.UNetBackbone(P + B_, [280, 380, 480, 520], [1, 2, 4, 8], [1, 1, 1, 1], "circular",
                                 create_block, blocks.AvgPool(),
                                 lambda in_channels, out_channels: blocks.ZonallyPeriodicBilinearUpsample(),
                                 None).eval()
orig_decoder = torch.nn.Conv2d(280, O, 3)

orig = torch.nn.Module(); orig.unet = orig_unet; orig.decoder = orig_decoder
ko, km = set(orig.state_dict()), set(mine.state_dict())
print("param/buffer names identical:", ko == km, f"({len(km)} tensors)")
assert ko == km, (sorted(ko - km)[:5], sorted(km - ko)[:5])
print("parameters:", sum(p.numel() for p in mine.parameters()) / 1e6, "M")
orig.load_state_dict(mine.state_dict())

H, W = 90, 180
x = torch.randn(1, P, H, W); b = torch.randn(1, B_, H, W); mask = torch.rand(1, O, H, W) > 0.3
with torch.inference_mode():
    y_mine = mine(x, b, mask)
    f = orig_unet(torch.cat((x, b), 1))
    f = torch.nn.functional.pad(f, (1, 1, 0, 0), mode="circular")
    f = torch.nn.functional.pad(f, (0, 0, 1, 1), mode="constant")
    y_orig = torch.where(mask, orig_decoder(f), 0.0)
err = (y_mine - y_orig).abs().max().item()
print("output shape", tuple(y_mine.shape), " max |diff| vs original:", err)
assert err == 0.0 or err < 1e-5
print("VERIFIED: samudra_core matches the original Samudra UNet")
