"""Profile the megakernel on the device: one standalone launch with runtime inspection on, then
neuron-explorer capture + summary-json, following AWS's neuron-nki-profiling workflow.

Random bf16 weights at Qwen3-8B shapes (timing does not depend on values). Output lands in
results/profiles/<tag>/: the NEFF, the NTFF of the 2nd execution, summary.json, and parquet tables
ingested by neuron-explorer (for the bounds analysis in analyze_profile.py).

The kernel run happens in a child process: neuron-explorer capture needs the core, and the runtime
does not release it until the process that initialised it exits.

  NEURON_RT_VISIBLE_CORES=2 python kernels/profile_mega.py --layers 1 --tag base_L1
  NEURON_RT_VISIBLE_CORES=2 python kernels/profile_mega.py --layers 2 --tag sbuf_L2 \
      --kernel qwen3_megakernel:qwen3_decode_layers_sbuf
"""
import argparse, glob, importlib, json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=1)
ap.add_argument("--tag", required=True)
ap.add_argument("--kernel", default="qwen3_megakernel:qwen3_decode_layers")
ap.add_argument("--S_ctx", type=int, default=1024)
ap.add_argument("--S_max", type=int, default=2048)
ap.add_argument("--tiny", action="store_true")
ap.add_argument("--head", action="store_true", help="profile qwen3_decode_step:qwen3_head_parts (embedding + LM head + argmax) instead")
ap.add_argument("--head-tiled", action="store_true", help="profile qwen3_head_parts_tiled (our own LM-head matmul)")
ap.add_argument("--hwdge-oproj", action="store_true", help="qwen3_decode_step.use_hwdge_output_projection() before tracing")
ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
args = ap.parse_args()

out_dir = os.path.abspath(os.path.join(HERE, "..", "results", "profiles", args.tag))
inspect_dir = os.path.join(out_dir, "inspect")
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "2")
if args.head_tiled:
    args.head = True
    args.kernel = "qwen3_decode_step:qwen3_head_parts_tiled"
elif args.head:
    args.kernel = "qwen3_decode_step:qwen3_head_parts"
if args.tiny:
    H, I, q, kv, d = 512, 1024, 4, 2, 128
else:  # Qwen3-8B
    H, I, q, kv, d = 4096, 12288, 32, 8, 128
L, B, S_ctx, S_max, eps = args.layers, 1, args.S_ctx, args.S_max, 1e-6


def run_kernel():
    os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
    os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
    os.environ["NEURON_RT_INSPECT_SYSTEM_PROFILE"] = "0"
    os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = inspect_dir
    os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
    sys.path.insert(0, HERE)
    import numpy as np, ml_dtypes, torch, nki
    from qwen3_ref import rope_tables, decode_mask

    t = S_ctx - 64
    bf = ml_dtypes.bfloat16
    rng = np.random.default_rng(0)
    def rn(*shape, scale=0.02): return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(bf)
    pos = torch.full((B, 1), t, dtype=torch.int64)
    cos, sin = rope_tables(torch.full((B,), t), d, 1e6)
    inputs = dict(
        X=rn(B, 1, H, scale=1.0),
        W_qkv=rn(L, H, (q + 2 * kv) * d), W_out=rn(L, q * d, H),
        W_gate=rn(L, H, I), W_up=rn(L, H, I), W_down=rn(L, I, H),
        g_attn=np.ones((L, 1, H), bf), g_mlp=np.ones((L, 1, H), bf),
        g_q=np.ones((L, 1, d), bf), g_k=np.ones((L, 1, d), bf),
        K_cache=rn(L, B, kv, d, S_max, scale=1.0), V_cache=rn(L, B, kv, S_max, d, scale=1.0),
        cos=cos.numpy().astype(bf), sin=sin.numpy().astype(bf),
        mask=decode_mask(pos, S_ctx, q).numpy().astype(np.uint8), pos_ids=pos.numpy().astype(np.uint32),
    )
    if args.hwdge_oproj:
        from qwen3_decode_step import use_hwdge_output_projection
        use_hwdge_output_projection()
    mod_name, fn_name = args.kernel.split(":")
    kernel = nki.jit(getattr(importlib.import_module(mod_name), fn_name))
    lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
    kw = dict(num_layers=L, eps=eps)
    if args.head:
        from qwen3_decode_step import pack_lm_head, pack_lm_head_tiled
        V = 151936
        lm = torch.from_numpy(rn(V, H).astype(np.float32))
        inputs = dict(token_ids=np.array([[1234]], np.uint32), embed=rn(V, H), hidden=rn(1, 1, H, scale=1.0),
                      g_final=np.ones((1, H), bf))
        if args.head_tiled:
            inputs["W_tiled"] = pack_lm_head_tiled(lm, lnc)[0].numpy().astype(bf)
            kw = dict(V=V, eps=eps)
        else:
            inputs.update(W_lm=pack_lm_head(lm, lnc)[0].numpy().astype(bf), lm_bias=None)
            kw = dict(eps=eps)
    t0 = time.time()
    kernel[lnc](**inputs, **kw)
    print(f"compile+run {time.time() - t0:.0f}s", flush=True)


def capture():
    neffs = [p for p in glob.glob(os.path.join(inspect_dir, "i-*", "**", "*.neff"), recursive=True)]
    assert len(neffs) == 1, f"expected one kernel NEFF under {inspect_dir}, found {neffs}"
    neff = os.path.join(out_dir, "kernel.neff")
    shutil.copy(neffs[0], neff)
    env = dict(os.environ, NEURON_RT_ENABLE_DGE_NOTIFICATIONS="1")
    ntff_arg = os.path.join(out_dir, "profile.ntff")
    subprocess.run(["neuron-explorer", "capture", "-n", neff, "-s", ntff_arg, "--profile-nth-exec=2"],
                   env=env, check=True)
    ntff = (glob.glob(os.path.join(out_dir, "profile_exec_2.ntff")) or [ntff_arg])[0]
    summary = subprocess.run(["neuron-explorer", "view", "--output-format", "summary-json", "-n", neff, "-s", ntff],
                             capture_output=True, text=True, check=True).stdout
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        f.write(summary)
    subprocess.run(["neuron-explorer", "view", "-n", neff, "-s", ntff, "--ingest-only",
                    "--data-path", os.path.join(out_dir, "parquet"), "--display-name", args.tag], check=True)
    meta = dict(tag=args.tag, kernel=args.kernel, layers=L, H=H, I=I, q=q, kv=kv, d=d, S_ctx=S_ctx, S_max=S_max,
                neff=neff, ntff=ntff)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if args.child:
    run_kernel()
else:
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(inspect_dir)
    subprocess.run([sys.executable, os.path.abspath(__file__), *sys.argv[1:], "--child"], check=True)
    capture()
