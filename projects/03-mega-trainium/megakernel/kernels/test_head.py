"""Device test of the decode-step bookends: embedding row gather and final norm + LM head + argmax.

Random weights at Qwen3-8B shapes (V=151936, H=4096) against a float32 numpy reference.

  NEURON_RT_VISIBLE_CORES=3 python kernels/test_head.py [--pad 1] [--V 151936]
"""
import argparse, os, sys, time
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "3")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, ml_dtypes, torch, nki
from qwen3_decode_step import qwen3_head_parts, qwen3_head_parts_tiled, pack_lm_head, pack_lm_head_tiled

ap = argparse.ArgumentParser()
ap.add_argument("--V", type=int, default=151936)
ap.add_argument("--H", type=int, default=4096)
ap.add_argument("--pad", type=int, default=1, help="pad the vocab to a multiple of this")
ap.add_argument("--token", type=int, default=12345)
ap.add_argument("--tiled", action="store_true", help="test lm_head_tiled_argmax (our own LM-head matmul)")
args = ap.parse_args()
V, H, eps = args.V, args.H, 1e-6
lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
bf = ml_dtypes.bfloat16
g = torch.Generator().manual_seed(0)
embed = (torch.randn(V, H, generator=g) * 0.02).bfloat16()
lm = (torch.randn(V, H, generator=g) * 0.02).bfloat16()
hidden = torch.randn(1, 1, H, generator=g).bfloat16()
g_final = (1 + 0.1 * torch.randn(1, H, generator=g)).bfloat16()
def np_bf16(x): return x.float().numpy().astype(bf)
inputs = dict(token_ids=np.array([[args.token]], dtype=np.uint32), embed=np_bf16(embed), hidden=np_bf16(hidden),
              g_final=np_bf16(g_final))
t0 = time.time()
if args.tiled:
    W_tiled, _, Vp = pack_lm_head_tiled(lm, lnc)
    X, logits, token = nki.jit(qwen3_head_parts_tiled)[lnc](**inputs, W_tiled=np_bf16(W_tiled), V=V, eps=eps)
else:
    W_lm, lm_bias, _ = pack_lm_head(lm, lnc, args.pad)
    X, logits, token = nki.jit(qwen3_head_parts)[lnc](**inputs, W_lm=np_bf16(W_lm),
                                                       lm_bias=None if lm_bias is None else np_bf16(lm_bias), eps=eps)
print(f"compile+run {time.time() - t0:.0f}s; logits {np.asarray(logits).shape}")

X = np.asarray(X).astype(np.float32).reshape(H)
ok_emb = np.array_equal(X, embed[args.token].float().numpy())
print(f"[embed] row {args.token} gathered exactly: {ok_emb}")

h = hidden.float().reshape(H)
hn = h * torch.rsqrt(h.pow(2).mean() + eps) * g_final.float().reshape(H)
ref = (lm.float() @ hn).numpy()
got = np.asarray(logits).astype(np.float32).reshape(-1)[:V]
rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
tok = int(np.asarray(token).reshape(-1)[0])
print(f"[lm_head] logits rel-L2 vs fp32 ref: {rel:.3e}; kernel argmax {tok} (logit {got[tok]:.4f}), "
      f"ref argmax {int(ref.argmax())} (logit {ref.max():.4f}), kernel-logits argmax {int(got.argmax())}")
if rel >= 2e-2:   # where is the error: per 128-column tile
    nt = V // 128
    e = np.array([np.linalg.norm(got[i*128:(i+1)*128] - ref[i*128:(i+1)*128]) / (np.linalg.norm(ref[i*128:(i+1)*128]) + 1e-9) for i in range(nt)])
    bad = np.where(e > 2e-2)[0]
    print(f"[debug] bad tiles {len(bad)}/{nt}: first {bad[:12].tolist()} last {bad[-6:].tolist()}; median tile err {np.median(e):.2e}")
    i = int(bad[0]) if len(bad) else 0
    g_, r_ = got[i*128:(i+1)*128], ref[i*128:(i+1)*128]
    print(f"[debug] tile {i}: corr {np.corrcoef(g_, r_)[0,1]:.3f}, ratio {np.linalg.norm(g_)/np.linalg.norm(r_):.3f}, got[:4] {g_[:4]}, ref[:4] {r_[:4]}")
ok = ok_emb and rel < 2e-2 and tok == int(got.argmax())
print("PASS" if ok else "FAIL")
