"""CPU half of the LM-head walkthrough: the heuristics queries and the nkiequiv proofs, run live.

For each kernel version it prints what the agent asks the rules corpus (heuristics/) given the symptom the
device showed, and what nkiequiv says about the version's matmul core against a NumPy spec. The device
half (correctness and profiles) is run_device.sh. No Trainium device is needed here.

  NKIEQUIV=/path/to/symnki python examples/lm-head-walkthrough/replay_cpu.py   # from the repo root
"""
import os, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from heuristics import heuristics as H

POC = os.path.join(ROOT, "megakernel", "symnki_poc")
NKIEQUIV = os.environ.get("NKIEQUIV")


def rules(query, k=3):
    print(f"  heuristics.retrieve({query!r})")
    for r in H.retrieve(query, k=k):
        print(f"    [{r['id']}] {r['rule'][:150]}")
        print(f"        fix: {r['fix'][:150]}")


def errors(text, k=3):
    print(f"  heuristics.for_error({text!r})")
    for r in H.for_error(text, k=k):
        print(f"    [{r['id']}] {r['rule'][:150]}")


def prove(kernel, profile="small"):
    if not NKIEQUIV:
        print("  nkiequiv: skipped (set NKIEQUIV to a symnki checkout, commit b9b6b2a)")
        return
    cmd = [sys.executable, "-m", "nkiequiv", "check", "numpy:lm_head_poc.py::ref_lm_head",
           f"lm_head_poc.py::{kernel}", "--spec", "lm_head_poc.py", "--profile", profile]
    t0 = time.time()
    out = subprocess.run(cmd, cwd=POC, env={**os.environ, "PYTHONPATH": NKIEQUIV}, capture_output=True, text=True).stdout
    dt = time.time() - t0
    keep = [l for l in out.splitlines() if l.startswith(("verdict", "  output", "  all ", "numerics"))]
    print(f"  nkiequiv check {kernel} --profile {profile}   ({dt:.2f} s wall)")
    for l in keep[:3]:
        print(f"    {l.strip()[:170]}")


print("v0  nkilib output_projection_tkg (the library head)")
print("  device symptom: W_lm streamed by software DGE, 4,129 GpSimd DMA_DIRECT2D issued one at a time, ~230 GB/s per core")
rules("software DGE gpsimd dma_direct2d slow weight streaming")

print("\nv1  same head, weight DMAs on hardware DGE, two queues")
print("  device symptom: still DMA-bound; each transfer moves ~1.2 KB per partition")
rules("dma bytes per partition contiguous small transfers")

print("\nv2a own head, first version: host-pre-tiled weights (32 KB contiguous per partition), weights stationary,")
print("    4 vocab tiles accumulating in one [P, 4] PSUM tile")
prove("lm_head_core_shared_psum")
print("  device symptom: 14.5% rel-L2, tiles 0-2 of every 4-tile block lost one k-contribution")
rules("psum accumulation groups share one psum tile wrong result")

print("\nv2  own head, one PSUM tile per vocab tile (the fix)")
prove("lm_head_core")
prove("lm_head_core", profile="full")

print("\ncontrol: v2 with one k-tile skipped (a deliberate bug the proof must reject)")
prove("lm_head_core_skip_k")

print("\nv3  v2 inside the full one-launch step, in the generate loop")
print("  device symptom: out-of-bounds indirect DMA (scalar DGE, GpSimd), nrta 1006, on the second token")
errors("nrta 1006 out-of-bounds indirect DMA, scalar DGE on GpSimd, hardware DGE queues in the same kernel")
