"""Samudra next-step inference: CPU baseline vs Trainium (torch-xla + Neuron PJRT), with a correctness check.

  python bench.py --device cpu    --grid 2deg            # CPU baseline; saves the reference answer
  python bench.py --device neuron --grid 2deg            # compile + time on a free NeuronCore, compare
  python bench.py --device neuron --grid 1deg --batch 2  # bigger runs once 2deg works

Every run appends one JSON line to results.jsonl. Weights are random (fixed seed) unless --ckpt is
given: cost per step does not depend on the weight values, and the checker compares the two devices
on the SAME weights and inputs, so no download is needed to measure speed or correctness.
"""
import argparse, json, os, time
from pathlib import Path

GRIDS = {"2deg": (90, 180), "1deg": (180, 360), "0.5deg": (360, 720), "0.25deg": (720, 1440)}
DAYS_PER_CALL = 10  # output_steps=2 raw steps of 5 days each


def make_inputs(torch, args, H, W):
    g = torch.Generator().manual_seed(1)
    prog = torch.randn(args.batch, args.prog_ch, H, W, generator=g)
    bnd = torch.randn(args.batch, args.bnd_ch, H, W, generator=g)
    # A fixed "land" pattern: ~30% land, same for every channel, like the real ocean mask.
    land = torch.rand(1, 1, H, W, generator=g) < 0.3
    mask = (~land).expand(1, args.prog_ch, H, W).contiguous()
    return prog, bnd, mask


def build(torch, args):
    from samudra_core import SamudraNet, load_checkpoint, read_state_dict
    torch.manual_seed(0)
    norm = args.norm
    if args.ckpt:
        # Read the architecture off the checkpoint instead of guessing it.
        sd = read_state_dict(args.ckpt)
        in_ch = sd["unet.layers.0.convblock.0.weight"].shape[1]
        args.prog_ch = sd["decoder.weight"].shape[0]
        args.bnd_ch = in_ch - args.prog_ch
        norm = "batch" if any("running_mean" in k for k in sd) else "instance"
        print(f"checkpoint: {args.prog_ch} prognostic + {args.bnd_ch} boundary channels, norm={norm}")
    m = SamudraNet(args.prog_ch, args.bnd_ch, args.prog_ch, norm=norm,
                   use_bfloat16=bool(args.bf16)).eval()
    if args.ckpt:
        missing, unexpected = load_checkpoint(m, args.ckpt)
        print(f"checkpoint loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
        if missing or unexpected:
            print("  missing e.g.", missing[:5], " unexpected e.g.", unexpected[:5])
            raise SystemExit("checkpoint does not match the architecture; paste this output to Claude")
    return m


def timeit(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    ts.sort()
    return {"mean_ms": 1000 * sum(ts) / len(ts), "median_ms": 1000 * ts[len(ts) // 2],
            "min_ms": 1000 * ts[0], "max_ms": 1000 * ts[-1], "iters": iters}


def compare(torch, out, ref, mask):
    out, ref = out.float(), ref.float()
    m = mask.expand_as(ref)
    d = (out - ref)[m]
    rms = ref[m].pow(2).mean().sqrt().item()
    return {"max_abs_err": d.abs().max().item(), "rms_err": d.pow(2).mean().sqrt().item(),
            "ref_rms": rms, "rel_rms_err": d.pow(2).mean().sqrt().item() / max(rms, 1e-30),
            "land_is_zero": bool((out[~m] == 0).all().item())}


def cpu_quota():
    """CPUs this container may use (cgroup quota), not the host's core count (192 vs 11 on seat-212)."""
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            return max(1, int(q) // int(p))
    except Exception:
        pass
    return os.cpu_count() or 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "neuron"], required=True)
    ap.add_argument("--grid", choices=list(GRIDS), default="2deg")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--prog-ch", type=int, default=154)   # 77 variables x 2 input steps
    ap.add_argument("--bnd-ch", type=int, default=8)      # 4 forcings x 2 input steps (om4 preset)
    ap.add_argument("--bf16", type=int, default=0, help="1 = CPU autocast bf16 like the original")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--ckpt", default=None, help="path to a Samudra checkpoint (.pt)")
    ap.add_argument("--hf", default=None, choices=["onedeg", "halfdeg", "quarterdeg"],
                    help="download the real Samudra 2 weights from huggingface M2LInES/Samudra2")
    ap.add_argument("--norm", default="instance", choices=["instance", "batch"],
                    help="random-weight runs only; a checkpoint sets this itself")
    ap.add_argument("--cores", default="2", help="NEURON_RT_VISIBLE_CORES; vLLM holds 0-1 (probed)")
    ap.add_argument("--cc-args", default="", help="extra neuronx-cc flags, e.g. '--auto-cast=none'")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"],
                    help="neuron only: run the whole model in bf16 (the original trains in bf16)")
    ap.add_argument("--segments", type=int, default=0,
                    help="neuron only: 1 = compile each UNet layer as its own graph (cuts device "
                         "memory; needed for 0.5deg fp32 and 0.25deg on a 24 GB logical core)")
    ap.add_argument("--tol", type=float, default=3e-2, help="pass if rel RMS error <= tol")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    # first line of every log: lets status.py tie a crash in a shared log to its --tag
    print(f"START tag={args.tag or '-'} device={args.device} grid={args.grid} batch={args.batch} "
          f"dtype={args.dtype} NEURON_CC_FLAGS={os.environ.get('NEURON_CC_FLAGS', '')}", flush=True)

    if args.device == "neuron":
        os.environ.setdefault("NEURON_RT_VISIBLE_CORES", args.cores)
        os.environ.setdefault("PJRT_DEVICE", "NEURON")
        if args.cc_args:
            os.environ["NEURON_CC_FLAGS"] = (os.environ.get("NEURON_CC_FLAGS", "") + " "
                                             + args.cc_args).strip()
    import torch
    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", cpu_quota())))
    if args.hf:
        from huggingface_hub import hf_hub_download
        args.ckpt = hf_hub_download("M2LInES/Samudra2", f"{args.hf}/ema_ckpt.pt")
        print("downloaded", args.ckpt)

    H, W = GRIDS[args.grid]
    model = build(torch, args)
    prog, bnd, mask = make_inputs(torch, args, H, W)
    n_params = sum(p.numel() for p in model.parameters())
    refdir = Path("refs"); refdir.mkdir(exist_ok=True)
    ref_path = refdir / f"ref_{args.grid}_b{args.batch}{'_ckpt' if args.ckpt else ''}.pt"
    rec = {"device": args.device, "grid": args.grid, "H": H, "W": W, "batch": args.batch,
           "params_M": round(n_params / 1e6, 2), "tag": args.tag, "time": time.strftime("%H:%M:%S"),
           "weights": args.hf or (Path(args.ckpt).name if args.ckpt else "random"),
           "prog_ch": args.prog_ch, "bnd_ch": args.bnd_ch}

    if args.device == "cpu":
        with torch.inference_mode():
            out = model(prog, bnd, mask)
            rec.update(timeit(lambda: model(prog, bnd, mask), args.iters, args.warmup))
        if not args.bf16:
            torch.save({"out": out}, ref_path)
            print(f"reference saved to {ref_path} (fp32 CPU)")
        rec["threads"] = torch.get_num_threads()
        rec["bf16"] = bool(args.bf16)
    else:
        # Trainium through torch-xla (the pod has torch-xla 2.11 + neuronx-cc; no torch-neuronx).
        # The first call records the graph and neuronx-cc compiles it (minutes); later calls with
        # the same shapes reuse the compiled graph from the cache.
        # The image lacks torch-xla's Neuron PJRT plugin (libneuronxla); we pip-installed it
        # into a side folder so the vLLM environment is untouched.
        import sys
        plugin = os.environ.get("XLA_PLUGIN_DIR", "/workspace/xla_plugin")
        if os.path.isdir(plugin) and plugin not in sys.path:
            sys.path.insert(0, plugin)
        import torch_xla
        import torch_xla.core.xla_model as xm
        dev = torch_xla.device()
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        m_dev = model.to(dev).to(dtype)
        p_d, b_d = prog.to(dev).to(dtype), bnd.to(dev).to(dtype)
        k_d = mask.to(dev)

        cut = torch_xla.sync if args.segments else None

        def step():
            y = m_dev(p_d, b_d, k_d, cut=cut)
            torch_xla.sync()          # cut the graph and launch it on the chip
            xm.wait_device_ops()      # wait until the chip has finished
            return y

        print(f"device {dev}, cores={os.environ['NEURON_RT_VISIBLE_CORES']}, dtype={args.dtype}, "
              f"segments={bool(args.segments)}, "
              f"NEURON_CC_FLAGS={os.environ.get('NEURON_CC_FLAGS', '(default)')}")
        print("compiling for Trainium on the first call ... this can take several minutes")
        t = time.perf_counter()
        with torch.no_grad():
            out = step().cpu()
            rec["compile_s"] = round(time.perf_counter() - t, 1)
            print(f"first call (compile + run) took {rec['compile_s']} s")
            rec.update(timeit(step, args.iters, args.warmup))
        rec["dtype"] = args.dtype
        rec["segments"] = bool(args.segments)
        rec["cc_flags"] = os.environ.get("NEURON_CC_FLAGS", "")
        if ref_path.exists():
            rec.update(compare(torch, out, torch.load(ref_path)["out"], mask))
            rec["pass"] = rec["rel_rms_err"] <= args.tol and rec["land_is_zero"]
        else:
            print(f"no CPU reference at {ref_path}: run --device cpu with the same --grid/--batch first")

    per_sample_ms = rec["median_ms"] / args.batch
    rec["ms_per_step_per_sample"] = round(per_sample_ms, 3)
    rec["sim_years_per_day"] = round(DAYS_PER_CALL / 365 * 86400 / (per_sample_ms / 1000), 1)
    print(json.dumps(rec, indent=1))
    with open("results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
